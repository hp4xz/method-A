import math
import numpy as np

import physics
from method_a_fit import MethodAFit


class FastMethodAFit(MethodAFit):
    """
    Drop-in replacement for MethodAFit with a non-recursive FillChannels implementation.

    The physics/mapping equations are unchanged; only control flow is optimized to avoid
    Python recursion overhead in deep channel splitting.
    """

    def FillChannels(
        self,
        i0: int,
        cosMin: float,
        cosMax: float,
        pp2: float,
        intens: float,
        parA: np.ndarray,
        *,
        max_depth: int = 60,
    ) -> None:
        self.g_nFillChan += 1

        def floor_nint(x: float) -> int:
            return int(math.floor(x + 0.5))

        # Stack items: (cos_lo, cos_hi, depth_remaining)
        # Push right first so left branch is processed first (same order as recursion).
        stack = [(float(cosMin), float(cosMax), int(max_depth))]

        while stack:
            cmin, cmax, depth = stack.pop()
            if depth <= 0:
                continue

            tMax = self.TOF_MethodA(cmax, pp2, parA)
            if tMax < 4.0:
                continue
            Et2_max = physics.t2factor * pp2 / (tMax * tMax)
            rBnd = self.Et2_to_di(Et2_max)

            tMin = self.TOF_MethodA(cmin, pp2, parA)
            if tMin < 4.0:
                continue
            Et2_min = physics.t2factor * pp2 / (tMin * tMin)
            lBnd = self.Et2_to_di(Et2_min)

            ilBnd = floor_nint(lBnd)
            irBnd = floor_nint(rBnd)

            if rBnd <= lBnd:
                continue
            if ilBnd >= self.Et2_npts:
                continue
            if irBnd < 0:
                continue

            if ((rBnd - lBnd) > 0.05) and (ilBnd != irBnd):
                cosMid = (3.0 * cmin + cmax) / 4.0
                tMid = self.TOF_MethodA(cosMid, pp2, parA)
                if tMid < 4.0:
                    continue
                # Keep same split strategy as original recursive implementation
                stack.append((cosMid, cmax, depth - 1))
                stack.append((cmin, cosMid, depth - 1))
                continue

            dcos = cmax - cmin
            dt2 = 1.0 / (rBnd - lBnd)

            if ilBnd == irBnd:
                dx = rBnd - lBnd
                if 0 <= ilBnd <= (self.Et2_npts - 1):
                    self.chan[i0 + ilBnd] += intens * dt2 * dx * dcos
                continue

            if 0 <= ilBnd <= (self.Et2_npts - 1):
                dx = float(ilBnd + 1) - lBnd
                self.chan[i0 + ilBnd] += intens * dt2 * dx * dcos

            lo = max(ilBnd + 1, 0)
            hi = min(irBnd - 1, self.Et2_npts - 1)
            if hi >= lo:
                self.chan[i0 + lo : i0 + hi + 1] += intens * dt2 * dcos

            if irBnd < self.Et2_npts:
                dx = rBnd - float(irBnd)
                if 0 <= irBnd <= (self.Et2_npts - 1):
                    self.chan[i0 + irBnd] += intens * dt2 * dx * dcos
