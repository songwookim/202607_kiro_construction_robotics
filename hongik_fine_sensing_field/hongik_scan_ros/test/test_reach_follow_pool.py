"""MoveIt 어댑터의 도달 후보 추종 (MoveItPlatform._follow — reach_candidates·follow_from) 은 spawn 프로세스 풀 (dense_tracks) 에서 돌지만
결과는 서버 스레드에서 차례로 한 계산 (trajectory.follow_candidate) 과 비트까지 같다 (2026-10-10: 실기 g1c_04 획득 계획 30 s → 풀). 풀이 답하지
못한 후보 (멈춘 풀) 도 여기서 같은 계산으로 채운다 — 순위 (rank_reach_candidates) 는 후보 무리 전체의 상대 비교라 후보 하나도 빠지면 안 된다.
ROS 그래프 없이 돈다 (체인은 시험 URDF, IK 없음)."""
import io
import math
import os
from types import SimpleNamespace

import numpy as np

from hongik_scan_ros import moveit_adapter as MA
from scan_core.fk_urdf import chain_from_joints, parse_urdf_joints
from scan_core import seam_features as SF
from scan_core.trajectory import fast_fk, follow_candidate

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(os.path.dirname(os.path.dirname(HERE)), "scan_l0", "tests", "data", "rb20_1900es_u_eoat.urdf")
# 연구실 1G 시작 자세 (2026-10-08 손 교시, ~/scan_data/2026-10-09/demo_b/job_rough_g1.yaml start_pose)
Q_START = [2.754425756423491, 0.0469306991049455, -2.5981306910011384, -0.9163067123076475, 0.4618459288737595, 5.142267921429792]
KEYS = ("q", "ok", "reason", "limit_margin_rad", "min_sigma", "n_track", "dist_rad")


def _platform():
    """ROS 노드 없는 어댑터 (추종에 쓰는 것만: 체인·설정)."""
    chain = chain_from_joints(parse_urdf_joints(io.StringIO(open(URDF, encoding="utf-8").read())), "link0", "tcp")
    plat = object.__new__(MA.MoveItPlatform)
    plat._chain, plat._fk = chain, None
    plat.mj = SimpleNamespace(cartesian_step_m=0.002, max_joint_step_rad=0.05)
    return plat, chain


def _case(chain):
    """후보 관절 (따라가는 것·한계로 거부되는 것) + 따라갈 자세 (판 법선으로 내려가는 직선 → 거친 직선, 획득 계획과 같은 모양)."""
    fk = fast_fk(chain)
    q0 = np.array(Q_START)
    T_pre = fk(q0)
    T_in = T_pre.copy()
    T_in[2, 3] -= 0.04
    T_out = T_in.copy()
    T_out[:3, 3] += np.array([0.08, 0.01, 0.0])
    lo = np.array([j.lower for j in chain.movable_joints])
    hi = np.array([j.upper for j in chain.movable_joints])
    qs = [q0, q0 + np.array([0.3, -0.2, 0.25, 0.4, -0.9, -0.5]), q0 + np.array([-0.2, 0.1, 0.3, -0.2, 0.3, 0.4])]
    q_lim = q0.copy()
    q_lim[5] = hi[5] - 1e-3                                       # 따라가다 손목3 한계에 닿는다 (거부되는 후보도 같은 기록)
    qs.append(q_lim)
    return qs, [T_in, T_out], lo, hi


def _ref(chain, q, follow, lo, hi, sigma, q_ref):
    rec = follow_candidate(chain, fast_fk(chain), q, follow, step_m=0.002, lower=lo, upper=hi, max_joint_step_rad=0.05,
                           sigma_radius_m=sigma, q_ref=q_ref)
    rec.pop("q_end")
    return rec


def test_reach_follow_in_the_pool_equals_the_sequential_computation(monkeypatch):
    plat, chain = _platform()
    qs, follow, lo, hi = _case(chain)
    q_ref = np.array(Q_START) + 0.01
    sigma = 0.317
    ref = [_ref(chain, q, follow, lo, hi, sigma, q_ref) for q in qs]
    assert any(r["ok"] for r in ref) and any(not r["ok"] for r in ref), [(r["ok"], r["reason"][:60]) for r in ref]
    SF.shutdown_feature_pool()
    try:
        got = plat._follow(qs, follow, sigma, q_ref)
        assert SF._POOL is not None and SF._POOL.children(), "the candidates were not followed in the process pool"
        assert [{k: g[k] for k in KEYS} for g in got] == [{k: r[k] for k in KEYS} for r in ref]
        assert all("q_end" not in g for g in got)
        # follow_from (IK 없이 아는 관절에서 — yaw 띠·사전검사): 같은 계산, dict {이름: 값} 도
        names = list(chain.joint_names)
        ff = plat.follow_from(dict(zip(names, qs[1])), follow, sigma_radius_m=sigma)
        assert ff["names"] == names and {k: ff[k] for k in KEYS} == {k: v for k, v in _ref(chain, qs[1], follow, lo, hi, sigma, None).items()
                                                                       if k in KEYS}
        # 따라갈 자세 없음 (획득 비용 하한: IK 해의 조건만)
        got0 = plat._follow(qs, [], sigma, q_ref)
        assert [{k: g[k] for k in KEYS} for g in got0] == [{k: v for k, v in _ref(chain, q, [], lo, hi, sigma, q_ref).items() if k in KEYS}
                                                           for q in qs]
        # 풀이 답하지 못한 후보 (멈춘 풀 → dense_tracks 의 '확인하지 않음') 도 여기서 같은 계산으로 — 빠지는 후보 없음
        real = MA.dense_tracks

        def stalled_second(arglists, fn=None, workers="auto"):
            tracks, _unchecked, note = real(arglists, fn=fn, workers=workers)
            tracks[1] = None
            return tracks, {1: "no answer from the process pool (test)"}, note
        monkeypatch.setattr(MA, "dense_tracks", stalled_second)
        got1 = plat._follow(qs, follow, sigma, q_ref)
        assert [{k: g[k] for k in KEYS} for g in got1] == [{k: r[k] for k in KEYS} for r in ref]
    finally:
        SF.shutdown_feature_pool()
    assert math.isfinite(ref[0]["min_sigma"]) and ref[0]["n_track"] > 2
