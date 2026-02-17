import numpy as np
from method_a_fit import MethodAFit

def _require_iminuit():
    try:
        from iminuit import Minuit
        return Minuit
    except Exception as e:
        raise ImportError(
            "iminuit is required for fitting. Install it with: pip install iminuit"
        ) from e

def fit_noE(argv, *, verbose: bool = True):
    """
    Mirror of C++ fit_noE:
      - load teardrop (long-form CSV iE,itp,content)
      - scale like ROOT
      - build fit region from data
      - run minimization with iminuit
    Returns: (methodA, best_params_array, minuit_object)
    """
    if len(argv) < 1:
        raise ValueError("fit_noE expects argv[0] = teardrop CSV path")

    teardrop_csv = argv[0]

    methodA = MethodAFit(verbose=verbose)
    methodA.load_teardrop_csv_long(teardrop_csv, allow_resize=True)

    # Build fit region like C++
    methodA.build_fit_region_from_data()

    # Default parameters copied from methodAFit.cpp::fit_noE()
    p0 = np.zeros(23, dtype=float)
    p0[0] = -0.10   # a
    p0[1] = 0.00    # b
    p0[2] = 0.91    # log10N (matches C++ fit_noE defaults)
    p0[3] = 0.76    # cosThetaMin (matches C++ fit_noE defaults)

    p0[4] = 0.09    # LNabM5 (so A_L = p0[4] + 5)
    p0[5] = 0.60    # alpha
    p0[6] = -0.05   # beta
    p0[7] = 2.10    # gamma
    p0[8] = 0.056   # eta

    p0[9]  = -0.132 # z0_center
    p0[10] = 0.08   # z0_width
    p0[11] = 0.01   # missdet
    p0[12] = 0.00   # tailfrac
    p0[13] = 0.13   # tailVal

    # Fixed HV mapping values from C++ (issue 130 note-16)
    p0[14] = -0.0217
    p0[15] = 0.0
    p0[16] = 0.507
    p0[17] = -0.960
    p0[18] = 0.105
    p0[19] = -0.0457

    p0[20] = 1.0
    p0[21] = 0.0
    p0[22] = 1.5

    if verbose:
        methodA.sanity_checks()

    # iminuit setup
    Minuit = _require_iminuit()

    # Define FCN with explicit parameters for iminuit
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
            calEe, EeNonLinearity, sigmaEe_keV
        ], dtype=float)
        return methodA.chi2(pars)

    m = Minuit(
        fcn,
        a_ev=p0[0],
        b_Fierz=p0[1],
        log10N=p0[2],
        costhetamin=p0[3],
        LNabM5=p0[4],
        alpha=p0[5],
        beta=p0[6],
        gamma=p0[7],
        eta=p0[8],
        z0_center=p0[9],
        z0_width=p0[10],
        missdet=p0[11],
        tailfrac=p0[12],
        tailVal=p0[13],
        hvMapMin1=p0[14],
        hvMap0=p0[15],
        hvMap1=p0[16],
        hvMap2=p0[17],
        hvMap3=p0[18],
        hvMap4=p0[19],
        calEe=p0[20],
        EeNonLinearity=p0[21],
        sigmaEe_keV=p0[22],
    )

    # Parameter limits matching the C++ SetParameter calls where present
    m.limits["a_ev"] = (-1.0, 1.0)
    m.limits["b_Fierz"] = (-0.5, 0.5)
    # log10N had no explicit limits in C++ but keep reasonable
    m.limits["costhetamin"] = (0.0, 1.0)
    m.limits["LNabM5"] = (-0.2, 0.7)
    m.limits["sigmaEe_keV"] = (0.0, 10.0)

    # Fix parameters to match C++ fit_noE configuration
    m.fixed["costhetamin"] = True
    m.fixed["gamma"] = True
    m.fixed["z0_center"] = True
    m.fixed["z0_width"] = True
    m.fixed["missdet"] = True
    m.fixed["tailfrac"] = True
    m.fixed["tailVal"] = True
    m.fixed["hvMapMin1"] = True
    m.fixed["hvMap0"] = True
    m.fixed["hvMap1"] = True
    m.fixed["hvMap2"] = True
    m.fixed["hvMap3"] = True
    m.fixed["hvMap4"] = True
    m.fixed["calEe"] = True
    m.fixed["EeNonLinearity"] = True
    m.fixed["sigmaEe_keV"] = True

    # Make Minuit more verbose initially
    m.errordef = 1.0
    m.print_level = 2 if verbose else 0

    if verbose:
        # Force one evaluation at the starting point
        print("[fit_noE] Pre-migrad sanity eval")

        # actually build model at the start point
        methodA.simulateET2SpecMethodA(p0)

        # optional: compute chi2 directly
        chi2_0 = methodA.chi2(p0)
        print(f"[fit_noE] chi2(start) = {chi2_0}")

        # Slice compare at ix450 (your fit-region uses this)
        ix = getattr(methodA, "ix450", 45)
        d = methodA.h_data[ix, :]
        f = methodA.h_fit2D[ix, :]

        print(f"[slice dbg] ix={ix}  data sum={d.sum():.6e}  fit sum={f.sum():.6e}")
        print(f"[slice dbg] ix={ix}  argmax ybin: data={int(d.argmax())}  fit={int(f.argmax())}")

        dnz = np.where(d > 0)[0]
        fnz = np.where(f > 0)[0]
        if dnz.size > 0:
            print(f"[slice dbg] ix={ix}  data nonzero ybin range: {int(dnz.min())}..{int(dnz.max())}")
        else:
            print(f"[slice dbg] ix={ix}  data has no nonzero bins (unexpected)")

        if fnz.size > 0:
            print(f"[slice dbg] ix={ix}  fit  nonzero ybin range: {int(fnz.min())}..{int(fnz.max())}")
        else:
            print(f"[slice dbg] ix={ix}  fit has no nonzero bins (bad)")

        print("[fit_noE] Starting migrad")
    m.migrad()

    if verbose:
        print("[fit_noE] migrad done")
        print(m)

    # Extract best params in the same vector order
    best = np.array([
        m.values["a_ev"],
        m.values["b_Fierz"],
        m.values["log10N"],
        m.values["costhetamin"],
        m.values["LNabM5"],
        m.values["alpha"],
        m.values["beta"],
        m.values["gamma"],
        m.values["eta"],
        m.values["z0_center"],
        m.values["z0_width"],
        m.values["missdet"],
        m.values["tailfrac"],
        m.values["tailVal"],
        m.values["hvMapMin1"],
        m.values["hvMap0"],
        m.values["hvMap1"],
        m.values["hvMap2"],
        m.values["hvMap3"],
        m.values["hvMap4"],
        m.values["calEe"],
        m.values["EeNonLinearity"],
        m.values["sigmaEe_keV"],
    ], dtype=float)

    return methodA, best, m
