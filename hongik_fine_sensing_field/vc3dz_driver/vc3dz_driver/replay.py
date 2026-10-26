"""오프라인 재생: SmartShape TXT 로더, DataMode 4 인코딩, pcap/pcapng 파서 + TCP 재조립, 검증표 CLI.

- SmartShape `profile_mm_*.txt`: 'X; Y; Z;' 행, Y = 정수 프레임 인덱스(시간·거리 아님). 모드 0(유효점만) 기록물.
- pcap: libpcap(.pcap, 마이크로/나노초, 양 엔디안) + pcapng(SHB/IDB/EPB/SPB). 링크 타입 Ethernet(1),
  NULL/loopback(0, Npcap 루프백), Raw IP(101), Linux SLL(113). IPv4/TCP 만. 802.1Q VLAN 1단 처리.
- 재조립: (src,sport,dst,dport) 흐름별 seq 기반. 재전송·중복은 첫 값 유지, 갭은 기록 후 건너뜀.

CLI: python -m vc3dz_driver.replay --pcap capture.pcap [--port 1096] [--max-frames 5]
     python -m vc3dz_driver.replay --txt profile_mm_x.txt [--max-frames 5]
"""
from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .protocol import (GH_SIZE, ParsedMessage, ResultFrame, StreamParser, StringCommand, StringReply,
                       build_result_message, message_offsets_table, parse_global_header)
from .valid import reflection_filter, valid_mask


# ---------------------------------------------------------------------------
# SmartShape TXT
# ---------------------------------------------------------------------------
@dataclass
class TxtFrame:
    index: int
    x: np.ndarray
    z: np.ndarray


def load_smartshape_txt(path: str, max_frames: Optional[int] = None, dtype=np.float32) -> List[TxtFrame]:
    """'X; Y; Z;' 텍스트 → 프레임별 (x, z). 프레임 순서는 Y 오름차순(파일 순서 유지)."""
    a = np.loadtxt(path, delimiter=';', usecols=(0, 1, 2), ndmin=2)
    if a.size == 0:
        return []
    y = a[:, 1].astype(np.int64)
    # 파일 순서대로 연속 구간 분할(같은 Y 가 떨어져 다시 나오면 별도 프레임으로 취급)
    brk = np.flatnonzero(np.diff(y) != 0) + 1
    starts = np.concatenate([[0], brk])
    ends = np.concatenate([brk, [len(y)]])
    frames: List[TxtFrame] = []
    for s, e in zip(starts, ends):
        frames.append(TxtFrame(int(y[s]), a[s:e, 0].astype(dtype), a[s:e, 2].astype(dtype)))
        if max_frames is not None and len(frames) >= max_frames:
            break
    return frames


def frames_to_messages(frames: Iterable[Tuple[np.ndarray, np.ndarray]], fps: float = 100.0,
                       start_counter: int = 0, start_result_counter: int = 0, ts0_ms: int = 1_000_000,
                       data_mode: int = 4, line_counter: bool = False) -> List[bytes]:
    """프레임 목록 → DataMode 4/5 메시지 바이트 목록(테스트·fake sensor 용)."""
    out: List[bytes] = []
    ts = float(ts0_ms)
    c, rc = start_counter, start_result_counter
    for x, z in frames:
        out.append(build_result_message(x, z, rc, int(ts), c, data_mode, rc if line_counter else None))
        c += 1; rc += 1; ts += 1000.0 / fps
    return out


# ---------------------------------------------------------------------------
# pcap / pcapng
# ---------------------------------------------------------------------------
@dataclass
class Packet:
    ts: float            # epoch seconds
    linktype: int
    data: bytes


_PCAP_MAGICS = {
    0xA1B2C3D4: ('<', 1e-6), 0xD4C3B2A1: ('>', 1e-6),
    0xA1B23C4D: ('<', 1e-9), 0x4D3CB2A1: ('>', 1e-9),
}
PCAPNG_SHB = 0x0A0D0D0A


def read_pcap_packets(path: str) -> Iterator[Packet]:
    with open(path, 'rb') as f:
        buf = f.read()
    if len(buf) < 4:
        return
    magic_le = struct.unpack_from('<I', buf, 0)[0]
    if magic_le == PCAPNG_SHB:
        yield from _iter_pcapng(buf)
        return
    if magic_le in _PCAP_MAGICS:
        yield from _iter_pcap(buf)
        return
    raise ValueError(f'not a pcap/pcapng file (magic 0x{magic_le:08X})')


def _iter_pcap(buf: bytes) -> Iterator[Packet]:
    magic = struct.unpack_from('<I', buf, 0)[0]
    endian, res = _PCAP_MAGICS[magic]
    _, vmaj, vmin, tz, sig, snap, linktype = struct.unpack_from(endian + 'IHHiIII', buf, 0)
    linktype &= 0x0FFFFFFF
    off = 24
    rec = struct.Struct(endian + 'IIII')
    while off + 16 <= len(buf):
        ts_s, ts_f, incl, orig = rec.unpack_from(buf, off)
        off += 16
        data = buf[off:off + incl]
        off += incl
        yield Packet(ts_s + ts_f * res, linktype, data)


def _iter_pcapng(buf: bytes) -> Iterator[Packet]:
    off = 0
    endian = '<'
    ifaces: List[Tuple[int, float]] = []   # (linktype, ts_resolution)
    while off + 12 <= len(buf):
        btype = struct.unpack_from('<I', buf, off)[0]
        if btype == PCAPNG_SHB:
            bom = struct.unpack_from('<I', buf, off + 8)[0]
            endian = '<' if bom == 0x1A2B3C4D else '>'
            ifaces = []
        blen = struct.unpack_from(endian + 'I', buf, off + 4)[0]
        if blen < 12 or off + blen > len(buf):
            break
        body = buf[off + 8:off + blen - 4]
        if btype == 1:      # IDB
            linktype = struct.unpack_from(endian + 'H', body, 0)[0]
            res = 1e-6
            o = 8
            while o + 4 <= len(body):
                code, ln = struct.unpack_from(endian + 'HH', body, o)
                o += 4
                val = body[o:o + ln]
                o += (ln + 3) & ~3
                if code == 0:
                    break
                if code == 9 and ln >= 1:   # if_tsresol
                    v = val[0]
                    res = 2.0 ** -(v & 0x7F) if v & 0x80 else 10.0 ** -v
            ifaces.append((linktype, res))
        elif btype == 6:    # EPB
            iid, ts_hi, ts_lo, cap, orig = struct.unpack_from(endian + 'IIIII', body, 0)
            lt, res = ifaces[iid] if iid < len(ifaces) else (1, 1e-6)
            ts = ((ts_hi << 32) | ts_lo) * res
            yield Packet(ts, lt, body[20:20 + cap])
        elif btype == 3:    # SPB
            orig = struct.unpack_from(endian + 'I', body, 0)[0]
            lt = ifaces[0][0] if ifaces else 1
            yield Packet(0.0, lt, body[4:4 + orig])
        off += blen


@dataclass
class TcpSegment:
    ts: float
    src: str
    sport: int
    dst: str
    dport: int
    seq: int
    flags: int
    payload: bytes


def _ip4(b: bytes) -> str:
    return '.'.join(str(c) for c in b)


def decode_tcp(pkt: Packet) -> Optional[TcpSegment]:
    d = pkt.data
    lt = pkt.linktype
    if lt == 1:                       # Ethernet
        if len(d) < 14:
            return None
        eth = struct.unpack_from('>H', d, 12)[0]
        o = 14
        while eth in (0x8100, 0x88A8) and len(d) >= o + 4:
            eth = struct.unpack_from('>H', d, o + 2)[0]
            o += 4
        if eth != 0x0800:
            return None
    elif lt == 0:                     # NULL/loopback: 4 B AF (host order)
        if len(d) < 4:
            return None
        af = struct.unpack_from('<I', d, 0)[0]
        if af not in (2, 0x02000000):
            af = struct.unpack_from('>I', d, 0)[0]
            if af != 2:
                return None
        o = 4
    elif lt == 101:                   # raw IP
        o = 0
    elif lt == 113:                   # Linux SLL
        if len(d) < 16 or struct.unpack_from('>H', d, 14)[0] != 0x0800:
            return None
        o = 16
    else:
        return None
    if len(d) < o + 20 or (d[o] >> 4) != 4:
        return None
    ihl = (d[o] & 0x0F) * 4
    total = struct.unpack_from('>H', d, o + 2)[0]
    proto = d[o + 9]
    if proto != 6:
        return None
    src, dst = _ip4(d[o + 12:o + 16]), _ip4(d[o + 16:o + 20])
    t = o + ihl
    if len(d) < t + 20:
        return None
    sport, dport, seq, ack, off_flags = struct.unpack_from('>HHIIH', d, t)
    doff = (off_flags >> 12) * 4
    flags = off_flags & 0x1FF
    end = min(len(d), o + total) if total >= ihl + doff else len(d)
    payload = d[t + doff:end]
    return TcpSegment(pkt.ts, src, sport, dst, dport, seq, flags, payload)


@dataclass
class TcpStream:
    key: Tuple[str, int, str, int]
    data: bytes = b''
    segments: int = 0
    gaps: int = 0
    duplicates: int = 0
    first_ts: float = 0.0
    last_ts: float = 0.0
    payload_offsets: List[Tuple[int, float]] = field(default_factory=list)   # (stream offset, ts) per segment

    @property
    def src(self) -> str:
        return f'{self.key[0]}:{self.key[1]}'

    @property
    def dst(self) -> str:
        return f'{self.key[2]}:{self.key[3]}'


def reassemble_streams(segments: Iterable[TcpSegment]) -> Dict[Tuple[str, int, str, int], TcpStream]:
    """흐름별 seq 재조립. SYN 이 있으면 ISN+1, 없으면 최소 seq 를 시작으로. 갭은 카운트 후 건너뜀."""
    flows: Dict[Tuple[str, int, str, int], List[TcpSegment]] = {}
    for s in segments:
        flows.setdefault((s.src, s.sport, s.dst, s.dport), []).append(s)
    out: Dict[Tuple[str, int, str, int], TcpStream] = {}
    for key, segs in flows.items():
        st = TcpStream(key)
        isn = None
        for s in segs:
            if s.flags & 0x02:      # SYN
                isn = (s.seq + 1) & 0xFFFFFFFF
                break
        payload_segs = [s for s in segs if s.payload]
        if not payload_segs:
            if segs:
                st.first_ts = st.last_ts = segs[0].ts
            out[key] = st
            continue
        if isn is None:
            isn = min(s.seq for s in payload_segs)
        # 32-bit 순환을 감안한 상대 seq
        def rel(seq: int) -> int:
            return (seq - isn) & 0xFFFFFFFF
        payload_segs.sort(key=lambda s: (rel(s.seq), s.ts))
        chunks: List[bytes] = []
        expect = 0
        st.first_ts, st.last_ts = payload_segs[0].ts, payload_segs[-1].ts
        for s in payload_segs:
            r = rel(s.seq)
            end = r + len(s.payload)
            if end <= expect:
                st.duplicates += 1
                continue
            if r > expect:
                st.gaps += 1
                expect = r
            if r < expect:              # 부분 중복
                s_payload = s.payload[expect - r:]
                st.duplicates += 1
            else:
                s_payload = s.payload
            st.payload_offsets.append((sum(len(c) for c in chunks), s.ts))
            chunks.append(s_payload)
            expect = end
            st.segments += 1
        st.data = b''.join(chunks)
        out[key] = st
    return out


@dataclass
class PcapSession:
    sensor_to_host: Optional[TcpStream]
    host_to_sensor: Optional[TcpStream]
    messages: List[ParsedMessage]          # 센서→호스트
    commands: List[ParsedMessage]          # 호스트→센서
    parser_stats: dict
    frame_rx_times: List[float] = field(default_factory=list)   # 결과 프레임의 pcap 시각(세그먼트 기준)


def load_pcap_session(path: str, port: int = 1096, sensor_ip: Optional[str] = None) -> PcapSession:
    segs = [t for t in (decode_tcp(p) for p in read_pcap_packets(path)) if t is not None]
    streams = reassemble_streams(segs)
    s2h = h2s = None
    for key, st in streams.items():
        src, sport, dst, dport = key
        if sport == port and (sensor_ip is None or src == sensor_ip):
            if s2h is None or len(st.data) > len(s2h.data):
                s2h = st
        elif dport == port and (sensor_ip is None or dst == sensor_ip):
            if h2s is None or len(st.data) > len(h2s.data):
                h2s = st
    p = StreamParser()
    msgs = p.feed(s2h.data) if s2h else []
    q = StreamParser()
    cmds = q.feed(h2s.data) if h2s else []
    # 프레임별 pcap 시각: 메시지 끝 오프셋을 포함하는 세그먼트의 ts
    rx_times: List[float] = []
    if s2h and s2h.payload_offsets:
        offs = np.array([o for o, _ in s2h.payload_offsets])
        tss = np.array([t for _, t in s2h.payload_offsets])
        for m in msgs:
            end = m.offset + GH_SIZE + m.header.data_length - 1
            i = int(np.searchsorted(offs, end, side='right') - 1)
            rx_times.append(float(tss[max(i, 0)]))
    return PcapSession(s2h, h2s, msgs, cmds, p.stats.__dict__, rx_times)


# ---------------------------------------------------------------------------
# 검증표 출력
# ---------------------------------------------------------------------------
def _fmt_rows(rows: Sequence[Tuple[int, str, str]]) -> str:
    return '\n'.join(f'  @{o:3d}  {k:<24s} {v}' for o, k, v in rows)


def describe_session(sess: PcapSession, max_frames: int = 5, fps_hint: Optional[float] = None) -> str:
    L: List[str] = []
    if sess.host_to_sensor:
        h = sess.host_to_sensor
        L.append(f'[host->sensor] {h.src} -> {h.dst}: {len(h.data)} B, {h.segments} seg, gaps {h.gaps}, dup {h.duplicates}')
        for pm in sess.commands:
            m = pm.message
            if isinstance(m, StringCommand):
                L.append(f'  cmd @{pm.offset:<8d} counter={pm.header.counter:<6d} wait={pm.header.data_wait_ms:<5d} '
                         f'cks=({pm.header.cks_hdr},{pm.header.cks_data}) param_id={m.param_id:<3d} hid={m.host_cmd_id:<5d} '
                         f'text={m.text[:70]!r}')
            else:
                L.append(f'  cmd @{pm.offset} (unparsed: {pm.error.splitlines()[0] if pm.error else type(m).__name__})')
    else:
        L.append('[host->sensor] stream not found')
    if not sess.sensor_to_host:
        L.append('[sensor->host] stream not found')
        return '\n'.join(L)
    s = sess.sensor_to_host
    L.append(f'[sensor->host] {s.src} -> {s.dst}: {len(s.data)} B, {s.segments} seg, gaps {s.gaps}, dup {s.duplicates}, '
             f'{s.last_ts - s.first_ts:.3f} s')
    st = sess.parser_stats
    L.append(f'  parser: messages={st["messages"]} results={st["results"]} string_replies={st["string_replies"]} '
             f'resyncs={st["resyncs"]} skipped={st["skipped_bytes"]} format_errors={st["format_errors"]} '
             f'cks_mismatch(hdr/data)={st["checksum_mismatch_hdr"]}/{st["checksum_mismatch_data"]} '
             f'counter_gaps={st["counter_gaps"]} error_nonzero={st["error_nonzero"]}')
    results = [pm for pm in sess.messages if isinstance(pm.message, ResultFrame)]
    replies = [pm for pm in sess.messages if isinstance(pm.message, StringReply)]
    for pm in replies[:3]:
        m = pm.message
        assert isinstance(m, StringReply)
        L.append(f'  reply @{pm.offset:<8d} param_id={m.param_id} hid={m.host_cmd_id} error={m.error} len={len(m.text)} '
                 f'text={m.text[:100]!r}')
    if results:
        n = np.array([pm.message.n_points for pm in results])
        rc = np.array([pm.message.result_counter for pm in results], dtype=np.int64)
        ts = np.array([pm.message.ts_ms for pm in results], dtype=np.int64)
        modes = sorted(set(int(pm.message.data_mode) for pm in results))
        lc = sum(1 for pm in results if pm.message.has_line_counter)
        L.append(f'  results: {len(results)} frames, data_mode={modes}, N min/max={n.min()}/{n.max()}, '
                 f'line_counter_variant={lc}, result_counter {rc[0]}->{rc[-1]} '
                 f'(gaps {int((np.diff(rc) != 1).sum())}), ts_ms {ts[0]}->{ts[-1]}')
        if len(ts) > 1:
            d = np.diff(ts)
            vals, cnts = np.unique(d, return_counts=True)
            hist = ', '.join(f'{int(v)}ms x{int(c)}' for v, c in zip(vals, cnts) if c >= max(1, len(d) // 200))
            fps_meas = 1000.0 * (len(ts) - 1) / max(ts[-1] - ts[0], 1)
            L.append(f'  dts histogram: {hist}   fps(ts)={fps_meas:.2f}')
            if sess.frame_rx_times and len(sess.frame_rx_times) == len(sess.messages):
                rx = np.array([sess.frame_rx_times[sess.messages.index(pm)] for pm in results])
                if rx[-1] > rx[0]:
                    L.append(f'  fps(pcap rx)={(len(rx) - 1) / (rx[-1] - rx[0]):.2f}')
        # 유효점 통계
        vr = [float(valid_mask(pm.message.x, pm.message.z).mean()) for pm in results[:200]]
        L.append(f'  valid ratio (first {len(vr)}): mean {np.mean(vr):.3f} min {np.min(vr):.3f}')
        for pm in results[:max_frames]:
            raw = s.data[pm.offset:pm.offset + GH_SIZE + pm.header.data_length]
            L.append(f'  --- frame @{pm.offset} (message offsets) ---')
            L.append(_fmt_rows(message_offsets_table(raw)))
    return '\n'.join(L)


def describe_txt(path: str, max_frames: int = 5) -> str:
    frames = load_smartshape_txt(path, max_frames=None)
    L = [f'{path}: {len(frames)} frames']
    n = np.array([f.x.size for f in frames])
    L.append(f'  points/frame min/max/mean = {n.min()}/{n.max()}/{n.mean():.1f}')
    z0 = sum(int((f.z == 0).sum()) for f in frames)
    L.append(f'  z=0 sentinel rows: {z0}')
    for f in frames[:max_frames]:
        r = reflection_filter(f.x, f.z)
        L.append(f'  frame {f.index}: N={f.x.size} valid={int(r.valid.sum())} refl_x={int(r.reflection_x.sum())} '
                 f'iso_z={int(r.isolated_z.sum())} x[{np.nanmin(f.x):.3f},{np.nanmax(f.x):.3f}] '
                 f'z[{np.nanmin(f.z[r.valid]) if r.valid.any() else float("nan"):.3f},{np.nanmax(f.z):.3f}]')
    msgs = frames_to_messages([(f.x, f.z) for f in frames[:1]])
    L.append('  encoded DataMode 4 message of frame 0:')
    L.append(_fmt_rows(message_offsets_table(msgs[0])))
    return '\n'.join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='VC nano 3D-Z offline replay / pcap verification')
    ap.add_argument('--pcap', help='Wireshark capture (.pcap or .pcapng) of a SmartShape session')
    ap.add_argument('--txt', help='SmartShape profile_mm_*.txt')
    ap.add_argument('--port', type=int, default=1096)
    ap.add_argument('--sensor-ip', default=None)
    ap.add_argument('--max-frames', type=int, default=5)
    a = ap.parse_args(argv)
    if not a.pcap and not a.txt:
        ap.error('--pcap or --txt required')
    if a.pcap:
        sess = load_pcap_session(a.pcap, a.port, a.sensor_ip)
        print(describe_session(sess, a.max_frames))
    if a.txt:
        print(describe_txt(a.txt, a.max_frames))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
