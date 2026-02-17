from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fast_method_a_fit import FastMethodAFit


def _require_iminuit():
    try:
        from iminuit import Minuit
        return Minuit
    except Exception as e:
        raise ImportError(
            "iminuit is required for fitting. Install it with: pip install iminuit"
        ) from e


def _default_fit_parameters() -> np.ndarray:
    p0 = np.zeros(23, dtype=float)
    p0[0] = -0.10
    p0[1] = 0.00
    p0[2] = 0.91
    p0[3] = 0.76

    p0[4] = 0.09
    p0[5] = 0.60
    p0[6] = -0.05
    p0[7] = 2.10
    p0[8] = 0.056

    p0[9] = -0.132
    p0[10] = 0.08
    p0[11] = 0.01
    p0[12] = 0.00
    p0[13] = 0.13

    p0[14] = -0.0217
    p0[15] = 0.0
    p0[16] = 0.507
    p0[17] = -0.960
    p0[18] = 0.105
    p0[19] = -0.0457

    p0[20] = 1.0
    p0[21] = 0.0
    p0[22] = 1.5
    return p0


@dataclass
class FitResult:
    csv_path: str
    best: np.ndarray
    fval: float | None
    valid: bool | None
    quick_sanity: bool


def fit_noE_fast(
    csv_path: str,
    *,
    verbose: bool = False,
    quick_sanity: bool = False,
    migrad_ncall: int | None = None,
) -> tuple[FastMethodAFit, np.ndarray, Any | None]:
    """
    Fast version of fit_noE using FastMethodAFit + optional quick sanity mode.
    """
    methodA = FastMethodAFit(verbose=verbose)
    methodA.load_teardrop_csv_long(csv_path, allow_resize=True)
    methodA.build_fit_region_from_data()

    p0 = _default_fit_parameters()

    if quick_sanity:
        # One simulation only for max speed. Caller can evaluate chi2 if needed.
        methodA.simulateET2SpecMethodA(p0)
        return methodA, p0.copy(), None

    Minuit = _require_iminuit()

    def fcn(a_ev, b_Fierz, log10N,
            costhetamin, LNabM5, alpha, beta, gamma, eta,
            z0_center, z0_width, missdet, tailfrac, tailVal,
            hvMapMin1, hvMap0, hvMap1, hvMap2, hvMap3, hvMap4,
            calEe, EeNonLinearity, sigmaEe_keV):
        pars = np.array([
            a_ev, b_Fierz, log10N,
            costhetamin, LNabM5, alpha, beta, gamma, eta,
            z0_center, z0_width, missdet, tailfrac, tailVal,
            hvMapMin1, hvMap0, hvMap1, hvMap2, hvMap3, hvMap4,
            calEe, EeNonLinearity, sigmaEe_keV,
        ], dtype=float)
        return methodA.chi2(pars)

    m = Minuit(
        fcn,
        a_ev=p0[0], b_Fierz=p0[1], log10N=p0[2], costhetamin=p0[3],
        LNabM5=p0[4], alpha=p0[5], beta=p0[6], gamma=p0[7], eta=p0[8],
        z0_center=p0[9], z0_width=p0[10], missdet=p0[11], tailfrac=p0[12], tailVal=p0[13],
        hvMapMin1=p0[14], hvMap0=p0[15], hvMap1=p0[16], hvMap2=p0[17], hvMap3=p0[18], hvMap4=p0[19],
        calEe=p0[20], EeNonLinearity=p0[21], sigmaEe_keV=p0[22],
    )

    m.limits["a_ev"] = (-1.0, 1.0)
    m.limits["b_Fierz"] = (-0.5, 0.5)
    m.limits["costhetamin"] = (0.0, 1.0)
    m.limits["LNabM5"] = (-0.2, 0.7)
    m.limits["sigmaEe_keV"] = (0.0, 10.0)

    for name in [
        "costhetamin", "gamma", "z0_center", "z0_width", "missdet", "tailfrac", "tailVal",
        "hvMapMin1", "hvMap0", "hvMap1", "hvMap2", "hvMap3", "hvMap4",
        "calEe", "EeNonLinearity", "sigmaEe_keV",
    ]:
        m.fixed[name] = True

    m.errordef = 1.0
    m.print_level = 1 if verbose else 0

    kwargs = {}
    if migrad_ncall is not None:
        kwargs["ncall"] = int(migrad_ncall)
    m.migrad(**kwargs)

    best = np.array([
        m.values["a_ev"], m.values["b_Fierz"], m.values["log10N"], m.values["costhetamin"],
        m.values["LNabM5"], m.values["alpha"], m.values["beta"], m.values["gamma"], m.values["eta"],
        m.values["z0_center"], m.values["z0_width"], m.values["missdet"], m.values["tailfrac"], m.values["tailVal"],
        m.values["hvMapMin1"], m.values["hvMap0"], m.values["hvMap1"], m.values["hvMap2"], m.values["hvMap3"], m.values["hvMap4"],
        m.values["calEe"], m.values["EeNonLinearity"], m.values["sigmaEe_keV"],
    ], dtype=float)

    methodA.simulateET2SpecMethodA(best)
    return methodA, best, m


def _fit_one_for_pool(args: tuple[str, bool, bool, int | None]) -> FitResult:
    csv_path, verbose, quick_sanity, migrad_ncall = args
    _, best, m = fit_noE_fast(
        csv_path,
        verbose=verbose,
        quick_sanity=quick_sanity,
        migrad_ncall=migrad_ncall,
    )
    return FitResult(
        csv_path=csv_path,
        best=best,
        fval=(None if m is None else float(m.fval)),
        valid=(None if m is None else bool(m.fmin.is_valid)),
        quick_sanity=quick_sanity,
    )


def fit_noE_batch(
    csv_paths: list[str] | tuple[str, ...],
    *,
    workers: int = 1,
    quick_sanity: bool = True,
    verbose: bool = False,
    migrad_ncall: int | None = None,
) -> list[FitResult]:
    """
    Batch process many CSVs.

    - workers=1: serial
    - workers>1: multiprocess fan-out
    """
    csv_paths = [str(Path(p)) for p in csv_paths]
    if workers <= 1:
        return [_fit_one_for_pool((p, verbose, quick_sanity, migrad_ncall)) for p in csv_paths]

    with ProcessPoolExecutor(max_workers=int(workers)) as ex:
        return list(ex.map(
            _fit_one_for_pool,
            [(p, verbose, quick_sanity, migrad_ncall) for p in csv_paths],
        ))
