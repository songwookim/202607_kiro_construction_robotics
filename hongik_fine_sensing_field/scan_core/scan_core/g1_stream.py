"""1G core: the second start of a pass profile on the PLANNED straight leg (2026-10-10, streamed leg fits).

A streamed leg's profiles are fitted while the leg is recorded, so everything expensive in a profile's fit must be fixed before the
reconstructed poses exist and must not depend on when a frame arrives. The profile prior (c, sigma_c, phi) is a per-leg constant from
the planned attitude (teach_1g). The second start of a pass profile (theta0 of the seam estimate's prediction) comes from here:

* the planned leg = the straight move the server sends: flange from T_from (the measured pose when the leg starts) to T_to (the
  target), position linear, attitude slerp (the Cartesian path the executor follows);
* a grid along the leg, uniform from the start to the end with step <= the estimate's station step (WPS station_step_mm);
* the start at grid point j = seam_estimate.profile_predictor(estimate, planned optical pose at j) — a pure function of (plan, estimate, j);
* a profile's grid index = its along-leg position (projection of its flange position on the leg), rounded to the grid, inside the leg;
  by definition from its RECONSTRUCTED pose. While streaming the worker uses the pose the server paired with the frame (a hint); a
  streamed row is kept only when the hint gives the same grid index, otherwise it is fitted again after the leg.

The start choice and the seam-position evidence (theta_pred, Sigma_R) stay on the reconstructed poses (seam_estimate.profile_predictor
as before — the planned leg ignores the robot's path deviation, 0.2-0.4 mm on g1c_02, which tilted the seam evidence): both are cheap
and are applied after the row fits (groove_model.finish_candidates / rows_select), identically with or without the stream.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

__all__ = ["PlannedLeg", "LegPredictor"]


class PlannedLeg:
    """The straight leg the server plans: flange T_from -> T_to (position linear, attitude slerp), optical frame = flange @ T_fo."""

    def __init__(self, T_from, T_to, T_flange_opt):
        self.T_from = np.asarray(T_from, dtype=float).reshape(4, 4).copy()
        self.T_to = np.asarray(T_to, dtype=float).reshape(4, 4).copy()
        self.T_fo = np.asarray(T_flange_opt, dtype=float).reshape(4, 4).copy()
        self.p0 = self.T_from[:3, 3]
        self.d = self.T_to[:3, 3] - self.p0
        self.length_m = float(np.linalg.norm(self.d))

    def to_dict(self) -> dict:
        return {"T_from": self.T_from.tolist(), "T_to": self.T_to.tolist(), "T_flange_opt": self.T_fo.tolist(),
                "length_m": self.length_m}

    @classmethod
    def from_dict(cls, d: dict) -> "PlannedLeg":
        return cls(d["T_from"], d["T_to"], d["T_flange_opt"])

    def flange_at(self, f: float) -> np.ndarray:
        """Planned flange pose at the along-leg fraction f in [0, 1]."""
        from .frames import slerp_transform
        f = min(max(float(f), 0.0), 1.0)
        return slerp_transform(self.T_from, self.T_to, f)

    def opt_at(self, f: float) -> np.ndarray:
        return self.flange_at(f) @ self.T_fo

    def fraction(self, T_base_flange) -> np.ndarray:
        """Along-leg fraction of measured flange poses (N,4,4): projection of the flange position on the leg (unclipped)."""
        return self.fraction_of_positions(np.asarray(T_base_flange, dtype=float).reshape(-1, 4, 4)[:, :3, 3])

    def fraction_of_positions(self, P) -> np.ndarray:
        """Along-leg fraction of flange positions (N,3)."""
        P = np.asarray(P, dtype=float).reshape(-1, 3)
        dd = float(self.d @ self.d)
        if not dd > 0.0:
            return np.zeros(P.shape[0])
        return ((P - self.p0) @ self.d) / dd


class LegPredictor:
    """Prediction on a grid along a planned leg (module docstring): the worker uses its theta0 as the second start of pass profiles.

    index(T_base_flange (N,4,4)) -> grid index j (N,) int;  __call__(j) -> seam_estimate.profile_predictor(...)(j) at the planned pose j.
    The grid poses are built once (the same array in every process that builds the predictor from the same inputs)."""

    def __init__(self, est, leg: PlannedLeg, step_m: float, bevel_x_sign: Optional[int], spec, cls: str = "pass",
                 profile_predictor: Optional[Callable] = None):
        if profile_predictor is None:
            from .seam_estimate import profile_predictor
        self.leg = leg
        step = float(step_m)
        if not step > 0.0:
            raise ValueError(f"LegPredictor: grid step must be > 0, got {step_m!r}")
        self.J = max(1, int(math.ceil(leg.length_m / step))) if leg.length_m > 0.0 else 1
        self.T_grid = np.stack([leg.opt_at(j / self.J) for j in range(self.J + 1)])
        if bevel_x_sign is None and est is not None:
            # the leg's sensor-x side of the bevel plate from the fused side and the PLANNED laser-line axis (not the measured poses)
            from .seam_estimate import _section_basis
            _, _, w_ = _section_basis(est.u, est.n_plate)
            bevel_x_sign = 1 if float(np.median(self.T_grid[:, :3, 0], axis=0) @ w_) * int(est.bevel_sign_w) >= 0 else -1
        self.bevel_x_sign = bevel_x_sign
        self._pred = profile_predictor(est, self.T_grid, bevel_x_sign, spec, cls)
        self._cache: dict = {}

    def index(self, T_base_flange) -> np.ndarray:
        return self.index_of_positions(np.asarray(T_base_flange, dtype=float).reshape(-1, 4, 4)[:, :3, 3])

    def index_of_positions(self, P) -> np.ndarray:
        """Grid index of flange positions (N,3): the along-leg position rounded to the grid, inside the leg."""
        f = self.leg.fraction_of_positions(P)
        return np.clip(np.rint(f * self.J), 0, self.J).astype(np.int64)

    def __call__(self, j: int):
        j = int(j)
        if j not in self._cache:
            self._cache[j] = self._pred(j)
        return self._cache[j]

    def for_rows(self, T_base_flange) -> Callable:
        """callable(i) -> prediction of row i of a leg whose measured flange poses are T_base_flange (N,4,4)."""
        jj = self.index(T_base_flange)
        return lambda i: self(int(jj[int(i)]))
