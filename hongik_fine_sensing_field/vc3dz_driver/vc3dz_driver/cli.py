"""CLI: python -m vc3dz_driver.cli <read-params|backup|set-params|restore|record|stats|monitor> --ip <sensor ip>

- read-params : print the full ID 44 text (+ number of parsed items, MainVersion)
- backup      : save the full ID 44 text as backup_<date>.txt
- set-params  : send 'Name=value' items with ID 43, then read back and compare
- restore     : send the writable items of a backup_<date>.txt with ID 43 and verify by reading ID 44 again
- record      : initialise (section 3.2 sequence) -> start -> receive N s -> stop -> save .npz (or .h5)
                (x, z, ts_ms, result_counter, t_rx_ns, global_counter, error, n_points, params_text)
- stats       : receive N s, then print fps, the dts histogram, counter gaps, valid point ratio, NumberOfPoints (static scan stats)
- monitor     : while streaming, read ID 44 again at 1 Hz (AutoTriggerError watch) - test of STRING reads while streaming
"""
# (한국어 설명, 코드 주석 — 도움말은 위 영어, 사용자 2026-10-10)
# CLI: python -m vc3dz_driver.cli <read-params|backup|set-params|record|stats|monitor> --ip 192.168.0.19
#
# - read-params : ID 44 전문 출력(+ 파싱된 항목 수, MainVersion)
# - backup      : ID 44 전문을 backup_<date>.txt 로 저장
# - set-params  : 'Name=value' 항목을 ID 43 으로 전송 후 재읽기 비교
# - record      : §3.2 시퀀스로 초기화 → start → N 초 수신 → stop → .npz(또는 .h5) 저장
#                 (x, z, ts_ms, result_counter, t_rx_ns, global_counter, error, n_points, params_text)
# - stats       : N 초 수신 후 fps·Δts 히스토그램·카운터 갭·유효점 비율·NumberOfPoints 출력(정지 스캔 통계 §3.6-3)
# - monitor     : 스트리밍 중 1 Hz 로 ID 44 재읽기(AutoTriggerError 감시) — 스트리밍 중 STRING 읽기 시험
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import params as P
from .client import ProfileRecord, VC3DZClient
from .valid import valid_mask

log = logging.getLogger('vc3dz_driver.cli')


def _parse_kv(items: Sequence[str]) -> Dict[str, P.Value]:
    kv: Dict[str, P.Value] = {}
    for it in items:
        if '=' not in it:
            raise SystemExit(f'bad item {it!r}; use Name=value')
        k, v = it.split('=', 1)
        kv[k.strip()] = P.parse_value(v)
    return kv


def _collect(c: VC3DZClient, seconds: float, quiet: bool = False) -> List[ProfileRecord]:
    recs: List[ProfileRecord] = []
    t_end = time.monotonic() + seconds
    t_rep = time.monotonic() + 1.0
    while time.monotonic() < t_end:
        try:
            recs.append(c.get_frame(timeout=0.2))
        except Exception:
            if not c.connected:
                log.error('connection lost: %s', c.last_error)
                break
        if not quiet and time.monotonic() >= t_rep:
            t_rep += 1.0
            print(f'  frames={len(recs)} N={c.n_points} stats={ {k: v for k, v in c.stats.items() if v} }', file=sys.stderr)
    return recs


def summarize(recs: List[ProfileRecord], fps_nominal: Optional[float] = None) -> Dict[str, object]:
    if not recs:
        return {'frames': 0}
    rc = np.array([r.result_counter for r in recs], dtype=np.int64)
    ts = np.array([r.ts_ms for r in recs], dtype=np.int64)
    trx = np.array([r.t_rx_ns for r in recs], dtype=np.int64) * 1e-9
    n = np.array([r.n_points for r in recs])
    out: Dict[str, object] = {
        'frames': len(recs),
        'n_points_min': int(n.min()), 'n_points_max': int(n.max()),
        'result_counter_first': int(rc[0]), 'result_counter_last': int(rc[-1]),
        'result_counter_gaps': int((np.diff(rc) != 1).sum()) if len(rc) > 1 else 0,
        'span_s_host': float(trx[-1] - trx[0]),
        'span_ms_sensor': int(ts[-1] - ts[0]),
    }
    if len(recs) > 1 and trx[-1] > trx[0]:
        out['fps_host'] = float((len(recs) - 1) / (trx[-1] - trx[0]))
        out['fps_sensor_ts'] = float(1000.0 * (len(recs) - 1) / max(ts[-1] - ts[0], 1))
        d = np.diff(ts)
        vals, cnts = np.unique(d, return_counts=True)
        out['dts_histogram_ms'] = {int(v): int(c) for v, c in zip(vals, cnts)}
        # 시각 모델 예비값(§3.4): T_est = OLS(t_rx ~ k), a_lo = p5(residual)
        k = (rc - rc[0]).astype(float)
        T = float(np.polyfit(k, trx - trx[0], 1)[0])
        e = (trx - trx[0]) - T * k
        out['T_est_ms'] = T * 1e3
        out['residual_p5_ms'] = float(np.percentile(e, 5) * 1e3)
        out['residual_p95_minus_p5_ms'] = float((np.percentile(e, 95) - np.percentile(e, 5)) * 1e3)
        if fps_nominal:
            out['slope_ppm'] = (T * fps_nominal - 1.0) * 1e6
            # ts 무결성: 고정 앵커(ts0) 대신 잔차 중앙값 앵커 -- 실기(1.9.7)에서 초기 짧은 증분(5~6 ms)이
            # 누적돼 ts 가 카운터 예측보다 최대 6 ms 뒤처지므로 허용치는 0.8 프레임(100 Hz 에서 8 ms).
            cad = (ts - ts[0]).astype(float) - k * 1000.0 / fps_nominal
            cad -= float(np.median(cad))
            tol = max(1.5, 0.8 * 1000.0 / fps_nominal)
            out['ts_cadence_tol_ms'] = tol
            out['ts_cadence_violations'] = int((np.abs(cad) > tol).sum())
            out['ts_cadence_max_dev_ms'] = float(np.abs(cad).max())
    vr = np.array([float(valid_mask(r.x, r.z).mean()) if r.n_points else 0.0 for r in recs])
    out['valid_ratio_mean'] = float(vr.mean())
    out['valid_ratio_min'] = float(vr.min())
    if n.min() == n.max() and n.min() > 0:
        Z = np.stack([r.z.astype(np.float64) for r in recs])
        X = np.stack([r.x.astype(np.float64) for r in recs])
        m = valid_mask(X, Z)
        Zm = np.where(m, Z, np.nan)
        col_sigma = np.nanstd(Zm, axis=0)
        if np.isfinite(col_sigma).any():
            out['z_sigma_per_column_median_mm'] = float(np.nanmedian(col_sigma))
    return out


def save_record(path: str, recs: List[ProfileRecord], params_text: str = '', meta: Optional[dict] = None) -> str:
    n = np.array([r.n_points for r in recs]) if recs else np.zeros(0, int)
    nmax = int(n.max()) if n.size else 0
    X = np.full((len(recs), nmax), np.nan, np.float32)
    Z = np.full((len(recs), nmax), np.nan, np.float32)
    for i, r in enumerate(recs):
        X[i, :r.n_points] = r.x
        Z[i, :r.n_points] = r.z
    arrays = dict(
        x=X, z=Z, n_points=n,
        ts_ms=np.array([r.ts_ms for r in recs], np.uint64),
        result_counter=np.array([r.result_counter for r in recs], np.uint32),
        t_rx_ns=np.array([r.t_rx_ns for r in recs], np.int64),
        global_counter=np.array([r.global_counter for r in recs], np.uint32),
        error=np.array([r.error for r in recs], np.int32),
        checksum_ok=np.array([r.checksum_ok for r in recs], bool),
        line_counter=np.array([(-1 if r.line_counter is None else r.line_counter) for r in recs], np.int64),
    )
    meta = dict(meta or {})
    meta['params_text'] = params_text
    if path.endswith('.h5') or path.endswith('.hdf5'):
        try:
            import h5py  # type: ignore
        except ImportError:
            path = path.rsplit('.', 1)[0] + '.npz'
            log.warning('h5py not available -> writing %s', path)
        else:
            with h5py.File(path, 'w') as f:
                for k, v in arrays.items():
                    f.create_dataset(k, data=v, compression='gzip' if v.ndim == 2 else None)
                f.attrs['meta_json'] = json.dumps(meta)
            return path
    np.savez_compressed(path, meta_json=json.dumps(meta), **arrays)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog='vc3dz_driver.cli', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ip', default='192.168.3.15', help='lab sensor 192.168.3.15 (factory IP; 2026-09-30 confirmed)')
    ap.add_argument('--port', type=int, default=1096)
    ap.add_argument('--timeout', type=float, default=2.0, help='STRING reply timeout [s]')
    ap.add_argument('--no-quickack', action='store_true')
    ap.add_argument('-v', '--verbose', action='store_true')
    sub = ap.add_subparsers(dest='cmd', required=True)

    sub.add_parser('read-params')
    b = sub.add_parser('backup'); b.add_argument('--dir', default='.')
    s = sub.add_parser('set-params'); s.add_argument('items', nargs='+', help='Name=value ...')
    rs = sub.add_parser('restore', help='send the writable items of backup_<date>.txt with ID 43, then verify by reading ID 44 again')
    rs.add_argument('--file', required=True, help='backup text (the full ID 44 text)')
    rs.add_argument('--dry-run', action='store_true', help='print the items to send only')

    def add_stream_args(p):
        p.add_argument('--seconds', type=float, default=10.0)
        p.add_argument('--init', choices=sorted(P.INIT_SETS), default='standard')
        p.add_argument('--set', nargs='*', default=[], help='extra Name=value overrides')
        p.add_argument('--no-init', action='store_true', help='do not send init parameters (only NbrLines)')
        p.add_argument('--backup-dir', default=None, help='write backup_<date>.txt here before init')
        p.add_argument('--restore', action='store_true', help='re-send backed-up parameters at end')

    r = sub.add_parser('record'); add_stream_args(r); r.add_argument('--out', required=True, help='.npz or .h5')
    st = sub.add_parser('stats'); add_stream_args(st)
    mo = sub.add_parser('monitor'); add_stream_args(mo); mo.add_argument('--period', type=float, default=1.0)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    with VC3DZClient(a.ip, a.port, recv_timeout=5.0, quickack=False if a.no_quickack else None) as c:
        if a.cmd == 'read-params':
            text, kv = c.read_params(a.timeout)
            print(text)
            print(f'# parsed {len(kv)} items; MainVersion={kv.get("MainVersion", kv.get("Mainversion"))}; '
                  f'unknown={P.unknown_params(kv)}')
            return 0
        if a.cmd == 'backup':
            print(c.backup_params_to_file(directory=a.dir, timeout=a.timeout))
            return 0
        if a.cmd == 'restore':
            with open(a.file, encoding='utf-8', errors='replace') as f:
                btext = f.read()
            bvals, bmeta = P.parse_params_text_with_meta(btext)
            restore = P.writable_params(bvals, bmeta)
            print(f'restoring {len(restore)} writable params from {a.file}')
            if a.dry_run:
                print(P.format_params(restore))
                return 0
            c.stop(a.timeout)                      # 스트리밍 중이면 먼저 정지 (ID 0)
            rep = c.set_params(restore, a.timeout, check=False)
            _, after = c.read_params(a.timeout)
            mism = P.diff_params(restore, after)
            print(f'reply error={rep.error} ({rep.error_name}); verified {len(restore) - len(mism)}/{len(restore)}'
                  + (f'; NOT restored: {mism}' if mism else ' (fully restored)'))
            print('LaserMode =', after.get('LaserMode'), '; DataMode =', after.get('DataMode'), '; NbrLines =', after.get('NbrLines'))
            return 0 if (rep.error == 0 and not mism) else 1
        if a.cmd == 'set-params':
            kv = _parse_kv(a.items)
            rep = c.set_params(kv, a.timeout, check=False)
            print(f'reply error={rep.error} ({rep.error_name})')
            _, actual = c.read_params(a.timeout)
            print('mismatch:', P.diff_params(kv, actual))
            return 0 if rep.error == 0 else 1

        # 스트리밍 계열
        fps_nominal = None
        if a.no_init:
            text, kv = c.read_params(a.timeout)
            c.param_backup_text, c.param_backup = text, kv
        else:
            init = dict(P.INIT_SETS[a.init])
            init.update(_parse_kv(a.set))
            mism = c.init_session(init, backup_dir=a.backup_dir, timeout=a.timeout)
            if mism:
                print(f'WARNING parameters not applied as requested: {mism}', file=sys.stderr)
            fps_nominal = float(init.get('AutoTriggerFPS', 0)) or None
        try:
            fps_nominal = fps_nominal or float(c.param_backup.get('AutoTriggerFPS', 0)) or None
        except (TypeError, ValueError):
            pass
        c.start(a.timeout)
        print(f'streaming for {a.seconds:.1f}s ...', file=sys.stderr)
        try:
            if a.cmd == 'monitor':
                recs: List[ProfileRecord] = []
                t_end = time.monotonic() + a.seconds
                while time.monotonic() < t_end:
                    recs += _collect(c, a.period, quiet=True)
                    try:
                        _, kv = c.read_params(a.timeout)
                        print(f'  frames={len(recs)} AutoTriggerError={kv.get("AutoTriggerError")} '
                              f'stats={ {k: v for k, v in c.stats.items() if v} }')
                    except TimeoutError as e:
                        print(f'  ID44 during stream: {e}')
            else:
                recs = _collect(c, a.seconds)
        finally:
            c.stop()
            time.sleep(0.2)
            recs += c.drain()
        summ = summarize(recs, fps_nominal)
        summ['client_stats'] = dict(c.stats)
        print(json.dumps(summ, indent=2, default=str))
        if a.cmd == 'record':
            meta = {'ip': a.ip, 'port': a.port, 'init': a.init if not a.no_init else None,
                    'fps_nominal': fps_nominal, 'summary': summ, 'time_model_note':
                    't_acq(k) = a_lo + T_est*k, k=result_counter; T_est from OLS over full span, a_lo = p5(residual). ts_ms integrity only.'}
            print('written:', save_record(a.out, recs, c.param_backup_text or '', meta))
        if a.restore and c.param_backup:
            # 2026-09-30 실기: 접미사(Min/Max/Step)가 섞인 값을 재전송하면 error=0 이어도 일부만 반영된다.
            # → 값 토큰만, 쓰기 가능 항목만 보내고 반드시 ID 44 재읽기로 검증한다.
            _, bmeta = P.parse_params_text_with_meta(c.param_backup_text or '')
            restore = P.writable_params(c.param_backup, bmeta)
            rep = c.set_params(restore, a.timeout, check=False)
            _, after = c.read_params(a.timeout)
            mism = P.diff_params(restore, after)
            print(f'restore reply error={rep.error}; verified {len(restore) - len(mism)}/{len(restore)} params'
                  + (f'; NOT restored: {mism}' if mism else ' (fully restored)'))
            if mism:
                print('WARNING: sensor parameters differ from backup; run set-params manually', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
