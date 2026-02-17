import numpy as np
import math
import physics as physics

class MethodAFit:
    """
    Python port of methodAFit.cpp/methodAFit.hh.
    Designed to match the C++ logic as closely as possible while keeping the same CSV input
    format you currently use: long-form CSV with header "iE,itp,content" (1-based indices).
    """

    def __init__(self, *, numBinsX: int = 80, numBinsY: int = 80,
                 xmin: float = 0.01e6, xmax: float = 0.81e6,
                 ymin: float = 0.0, ymax: float = 0.0065e12,
                 cosThetaMax: float = 1.0, npos: int = 5,
                 lenSpec: int = 999,
                 verbose: bool = True):
        self.verbose = bool(verbose)

        # FitOpt-like settings
        self.numBinsX = int(numBinsX)
        self.numBinsY = int(numBinsY)
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)

        self.xBinWidth = (self.xmax - self.xmin) / float(self.numBinsX)
        self.yBinWidth = (self.ymax - self.ymin) / float(self.numBinsY)

        # Axis mapping par used by C++
        self.e2_start = self.xmin
        self.e2_step = self.xBinWidth
        self.e2_npts = self.numBinsX

        self.Et2_start = self.ymin
        self.Et2_step = self.yBinWidth
        self.Et2_npts = self.numBinsY

        # Other C++ members
        self.cosThetaMax = float(cosThetaMax)
        self.npos = int(npos)

        # Data/model holders
        self.h_data = np.zeros((self.numBinsX, self.numBinsY), dtype=float)
        self.h_err  = np.zeros((self.numBinsX, self.numBinsY), dtype=float)

        self.h_fit2D = np.zeros_like(self.h_data)
        self.h_residual = np.zeros_like(self.h_data)

        # Fit region curves (per X bin): low/up (outer/inner)
        self.h_low_outer = np.zeros(self.numBinsX, dtype=float)
        self.h_low_inner = np.zeros(self.numBinsX, dtype=float)
        self.h_up_outer  = np.zeros(self.numBinsX, dtype=float)
        self.h_up_inner  = np.zeros(self.numBinsX, dtype=float)

        # Global switches like the C++ globals
        self.FitRegionIncludesEdges = True
        self.UseHVCorrection = False

        # Tail struct (brems)
        self.tailStruct = np.zeros(3, dtype=float)

        # HV mapping coefficients
        self.hvMap = np.zeros(6, dtype=float)

        # Counters for sanity checks
        self.g_nFillChan = 0
        self.g_nSimulateET2Spec = 0

        # Spectrum tables
        self.lenSpec = int(lenSpec)
        self.eESpec_raw = np.zeros(self.lenSpec + 1, dtype=float)
        self.eESpec_me  = np.zeros(self.lenSpec + 1, dtype=float)
        self.specNorm = 1.0

        self._build_electron_spectrum_tables()

    # -----------------------
    # Axis conversions
    # -----------------------
    def e2_to_di(self, e2: float) -> float:
        return (float(e2) - self.e2_start) / self.e2_step

    def di_to_e2(self, di: float) -> float:
        return self.e2_start + float(di) * self.e2_step

    def Et2_to_di(self, Et2: float) -> float:
        return (float(Et2) - self.Et2_start) / self.Et2_step

    def di_to_Et2(self, di: float) -> float:
        return self.Et2_start + float(di) * self.Et2_step

    # -----------------------
    # Data loading (long-form CSV: iE,itp,content)
    # -----------------------
    def load_teardrop_csv_long(self, path: str, *, allow_resize: bool = True) -> None:
        """
        Load CSV formatted as:
            iE,itp,content
        with 1-based indices. Matches the C++ file reading intent (bin indices).

        C++ behavior:
          - h_data bin content = N
          - h_data bin error = sqrt(N+1)
          - then h_data->Scale(1/totEvents) (errors scale too)

        This function mirrors that: after reading, we normalize both h_data and h_err by totEvents.
        """
        arr = np.genfromtxt(path, delimiter=",", skip_header=1)
        if arr.ndim == 1 and arr.size == 0:
            raise ValueError(f"Empty CSV: {path}")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 3:
            raise ValueError(f"Expected columns [iE,itp,content], got shape {arr.shape}")

        iE = arr[:, 0].astype(int)
        itp = arr[:, 1].astype(int)
        content = arr[:, 2].astype(float)

        max_iE = int(iE.max()) if iE.size else 0
        max_itp = int(itp.max()) if itp.size else 0

        if (max_iE != self.numBinsX) or (max_itp != self.numBinsY):
            msg = (f"CSV extents (max iE={max_iE}, max itp={max_itp}) "
                   f"do not match current (numBinsX={self.numBinsX}, numBinsY={self.numBinsY}).")
            if not allow_resize:
                raise ValueError(msg)
            # Resize to match file, recompute axis steps consistently with current xmin/xmax/ymin/ymax
            if self.verbose:
                print("[load_teardrop_csv_long] WARNING:", msg, "Resizing to match CSV.")
            self._resize_bins(max_iE, max_itp)

        self.h_data.fill(0.0)
        self.h_err.fill(0.0)

        # Fill
        for ix1, iy1, val in zip(iE, itp, content):
            ix = int(ix1) - 1
            iy = int(iy1) - 1
            if 0 <= ix < self.numBinsX and 0 <= iy < self.numBinsY:
                self.h_data[ix, iy] = float(val)
                self.h_err[ix, iy] = math.sqrt(float(val) + 1.0)

        tot = float(np.sum(self.h_data))
        if tot <= 0.0:
            raise ValueError("Total counts in loaded teardrop is zero.")

        self.h_data /= tot
        self.h_err  /= tot

        if self.verbose:
            print(f"[load_teardrop_csv_long] Loaded {path}")
            print(f"  shape: {self.h_data.shape}")
            print(f"  total counts (pre-scale): {tot}")
            print(f"  sum(h_data) after scale: {np.sum(self.h_data)}")
            print(f"  min/max bin content: {self.h_data.min()} {self.h_data.max()}")

    def _resize_bins(self, newX: int, newY: int) -> None:
        self.numBinsX = int(newX)
        self.numBinsY = int(newY)
        self.xBinWidth = (self.xmax - self.xmin) / float(self.numBinsX)
        self.yBinWidth = (self.ymax - self.ymin) / float(self.numBinsY)

        self.e2_step = self.xBinWidth
        self.e2_npts = self.numBinsX
        self.Et2_step = self.yBinWidth
        self.Et2_npts = self.numBinsY

        self.h_data = np.zeros((self.numBinsX, self.numBinsY), dtype=float)
        self.h_err  = np.zeros_like(self.h_data)
        self.h_fit2D = np.zeros_like(self.h_data)
        self.h_residual = np.zeros_like(self.h_data)

        self.h_low_outer = np.zeros(self.numBinsX, dtype=float)
        self.h_low_inner = np.zeros(self.numBinsX, dtype=float)
        self.h_up_outer  = np.zeros(self.numBinsX, dtype=float)
        self.h_up_inner  = np.zeros(self.numBinsX, dtype=float)

    # -----------------------
    # Electron spectrum (C++ faithful)
    # -----------------------
    def _eEspec(self, x: float) -> float:
        """
        C++ eEspec(x):
          (E0-x)^2 * pe(x) * (x+me) * corr(beta)
        with corr = (2*pi*alpha/beta) / (1 - exp(-2*pi*alpha/beta))
        """
        if x < 0.0:
            return 0.0
        if x > physics.E0:
            return 0.0
        E0m = (physics.E0 - x)
        p = physics.pe(x)
        E = x + physics.me
        b = physics.beta(x)
        if b <= 0.0:
            return 0.0
        eta = 2.0 * math.pi * physics.alpha_fs / b
        denom = 1.0 - math.exp(-eta)
        corr = eta / denom if denom != 0.0 else 1.0
        return (E0m*E0m) * p * E * corr

    def _build_electron_spectrum_tables(self) -> None:
        # Mirrors C++ constructor loop over lenSpec with midpoint integration.
        raw = self.eESpec_raw
        mew = self.eESpec_me

        raw.fill(0.0)
        mew.fill(0.0)

        raw[0] = 0.0
        mew[0] = 0.0

        for ie in range(1, self.lenSpec + 1):
            lVal = (ie - 1) * physics.E0 / float(self.lenSpec)
            uVal = ie * physics.E0 / float(self.lenSpec)
            midVal = 0.5 * (lVal + uVal)
            dE = (uVal - lVal)

            specVal = self._eEspec(midVal)
            raw[ie] = raw[ie - 1] + specVal * dE
            mew[ie] = mew[ie - 1] + specVal * (physics.me / (physics.me + midVal)) * dE

        self.specNorm = float(raw[self.lenSpec]) if raw[self.lenSpec] != 0 else 1.0

        # Normalize so getSpecFierz returns cumulative fraction like C++ (dividing by specNorm).
        raw /= self.specNorm
        mew /= self.specNorm

        if self.verbose:
            print("[_build_electron_spectrum_tables] Spectrum tables built")
            print(f"  lenSpec={self.lenSpec} specNorm={self.specNorm}")
            print(f"  raw[end]={raw[-1]} me[end]={mew[-1]} (should be 1.0 for raw)")

    def getSpecFierz(self, spec_bin: int, b: float) -> float:
        # C++: cumulative raw + b*cumulative_me
        i = int(spec_bin)
        if i < 0:
            i = 0
        if i > self.lenSpec:
            i = self.lenSpec
        return float(self.eESpec_raw[i] + float(b) * self.eESpec_me[i])

    def numElectrons(self, eMin: float, eMax: float, b: float) -> float:
        """
        C++ numElectrons(eMin,eMax,b):
          interpolate cumulative at bin indices and return difference (fraction).
        """
        if eMax < eMin:
            eMin, eMax = eMax, eMin
        eMin = physics.EnsureRange(eMin, 0.0, physics.E0)
        eMax = physics.EnsureRange(eMax, 0.0, physics.E0)
        if eMax <= eMin:
            return 0.0

        # Convert to fractional bin in [0,lenSpec]
        uBinMin = self.lenSpec * eMin / physics.E0
        uBinMax = self.lenSpec * eMax / physics.E0

        lBinMin = int(math.floor(uBinMin))
        lBinMax = int(math.floor(uBinMax))

        # Linear interpolation of cumulative table like C++
        def interp(uBin: float, lBin: int) -> float:
            frac = uBin - float(lBin)
            v0 = self.getSpecFierz(lBin, b)
            v1 = self.getSpecFierz(lBin + 1, b)
            return v0 + frac * (v1 - v0)

        valMin = interp(uBinMin, lBinMin)
        valMax = interp(uBinMax, lBinMax)
        return float(valMax - valMin)

    # -----------------------
    # Energy calibration (C++ eECal)
    # -----------------------
    def eECal(self, eE_guess: float, calibration: np.ndarray) -> float:
        # C++ eECal(e2, offset, gain, Nonlin):
        #   energyADC = offset + gain*e2 + (Nonlin/1e6)*gain*gain*e2*e2
        # Here, eE_guess is the dimensionless e2 index-like coordinate.
        calEe = float(calibration[0])
        nonlin = float(calibration[1])
        offset = self.e2_start
        gain = calEe * self.e2_step
        energy_adc = offset + gain * eE_guess + (nonlin / 1e6) * gain * gain * eE_guess * eE_guess
        if energy_adc < 0.0 or energy_adc > physics.E0:
            return 0.0
        return energy_adc

    # -----------------------
    # HV mapping (present in C++ but optional)
    # -----------------------
    def sethvMapping(self, hvMapMin1: float, hvMap0: float, hvMap1: float,
                     hvMap2: float, hvMap3: float, hvMap4: float) -> None:
        self.hvMap[:] = [hvMapMin1, hvMap0, hvMap1, hvMap2, hvMap3, hvMap4]

    def HVCorrection(self, p_p_MeV: float) -> float:
        # C++ uses pp2/1e12 (so pp in MeV). This polynomial is used only if UseHVCorrection=True.
        x = float(p_p_MeV)
        c = self.hvMap
        return (1.0 +
                c[0] / x +
                c[1] +
                c[2] * x +
                c[3] * x * x +
                c[4] * x * x * x +
                c[5] * x * x * x * x) if x != 0.0 else 1.0

    # -----------------------
    # TOF mapping (C++ methodAFit::TOF_MethodA)
    # -----------------------
    def TOF_MethodA(self, cosTh: float, pp2: float, parA: np.ndarray) -> float:
        """
        Mirrors C++ methodAFit::TOF_MethodA(cosTh, pp2, parA[0..5]).
        parA = [cosThetaMin, L_zDV, alpha, beta, gamma, eta]
        """
        A_cosThetaMin = float(parA[0])
        L_zDV = float(parA[1])
        A_alpha = float(parA[2])
        A_beta  = float(parA[3])
        A_gamma = float(parA[4])
        A_eta   = float(parA[5])

        cosmid = 0.5 * (1.0 + A_cosThetaMin)

        # C++ nudges cosTh by 1e-6 if cosTh <= min; keeps printouts.
        if (cosTh <= A_cosThetaMin) or (A_cosThetaMin >= 1.0) or (cosTh > 1.0):
            if self.verbose:
                print(f"[TOF_MethodA] WARNING cosTh={cosTh} <= cosThetaMin={A_cosThetaMin} (diff={cosTh - A_cosThetaMin})")
            cosTh = cosTh + 1e-6

        if ((L_zDV < 4.0) or (L_zDV > 6.0) or
            (A_alpha < -5.0) or (A_alpha > 5.0) or
            (A_beta  < -1.0) or (A_beta  > 1.0) or
            (A_gamma < -5.0) or (A_gamma > 5.0) or
            (A_eta   < -1.0) or (A_eta   > 1.0)):
            if self.verbose:
                print(f"[TOF_MethodA] WARNING Illegal par L={L_zDV} alpha={A_alpha} beta={A_beta} gamma={A_gamma} eta={A_eta}")
            return 5.0

        cosTSub = (cosTh - cosmid)

        if self.UseHVCorrection:
            hvCorr = self.HVCorrection(pp2 / 1e12)
            res = hvCorr * (L_zDV
                            - A_eta * math.log((cosTh - A_cosThetaMin) / (1.0 - A_cosThetaMin))
                            - A_alpha * cosTSub
                            + A_beta * cosTSub * cosTSub
                            - A_gamma * cosTSub * cosTSub * cosTSub)
        else:
            res = (L_zDV
                   - A_eta * math.log((cosTh - A_cosThetaMin) / (1.0 - A_cosThetaMin))
                   - A_alpha * cosTSub
                   + A_beta * cosTSub * cosTSub
                   - A_gamma * cosTSub * cosTSub * cosTSub)

        return float(res)

    # -----------------------
    # FillChannels (C++ faithful angular integral)
    # -----------------------
    def _fill_bin_range_uniform(self, ioffset: int, yL: float, yR: float, weight: float) -> None:
        """
        Fill chan bins uniformly in Et2 between yL and yR with total weight 'weight'.
        This matches the C++ tail of FillChannels where it fills partial edge bins and full interior bins.
        """
        if yR < yL:
            yL, yR = yR, yL
        if yR <= self.ymin or yL >= self.ymax:
            return

        # Clamp to histogram range
        yL = max(yL, self.ymin)
        yR = min(yR, self.ymax)

        lDi = self.Et2_to_di(yL)
        rDi = self.Et2_to_di(yR)

        il = int(math.floor(lDi))
        ir = int(math.floor(rDi))

        if il < 0:
            il = 0
        if ir > self.numBinsY - 1:
            ir = self.numBinsY - 1

        if ir < il:
            return

        width = (yR - yL)
        if width <= 0:
            return

        # Uniform density over [yL,yR]
        density = weight / width

        # Left partial
        left_edge = yL
        right_edge = yR

        # Bin edges in Et2:
        def bin_lo(i): return self.ymin + i * self.yBinWidth
        def bin_hi(i): return self.ymin + (i + 1) * self.yBinWidth

        if il == ir:
            frac = (right_edge - left_edge)
            self.chan[ioffset + il] += density * frac
            return

        # Left bin portion
        frac_left = bin_hi(il) - left_edge
        if frac_left > 0:
            self.chan[ioffset + il] += density * frac_left

        # Middle full bins
        for i in range(il + 1, ir):
            self.chan[ioffset + i] += density * self.yBinWidth

        # Right bin portion
        frac_right = right_edge - bin_lo(ir)
        if frac_right > 0:
            self.chan[ioffset + ir] += density * frac_right

    def FillChannels(self, i0: int, cosMin: float, cosMax: float,
                     pp2: float, intens: float, parA: np.ndarray,
                     *, max_depth: int = 60) -> None:
        """
        Faithful port of C++ FillChannels (recursion version).

        C++ computes:
            Et2 = physics::t2factor * pp2 / (tof^2)
            lBnd = Et2_to_di(Et2_min)
            rBnd = Et2_to_di(Et2_max)

        Recurses if:
            (rBnd - lBnd > 0.05) && (ilBnd != irBnd)

        Uses:
            cosMid = (3*cosMin + cosMax)/4

        Does NOT split intens between branches (the dcos factor is applied in the terminal fill).
        """
        self.g_nFillChan += 1

        # Guard against pathological recursion
        if max_depth <= 0:
            return

        # Helper matching ROOT TMath::FloorNint (nearest int via floor(x+0.5) for x>=0)
        def floor_nint(x: float) -> int:
            return int(math.floor(x + 0.5))

        # Compute bounds in "di" space exactly like C++
        tMax = self.TOF_MethodA(cosMax, pp2, parA)
        if tMax < 4.0:
            return
        Et2_max = physics.t2factor * pp2 / (tMax * tMax)
        rBnd = self.Et2_to_di(Et2_max)

        tMin = self.TOF_MethodA(cosMin, pp2, parA)
        if tMin < 4.0:
            return
        Et2_min = physics.t2factor * pp2 / (tMin * tMin)
        lBnd = self.Et2_to_di(Et2_min)

        ilBnd = floor_nint(lBnd)
        irBnd = floor_nint(rBnd)

        # Optional debug (first few calls)
        if getattr(self, "_dbg_fill", 0) < 8:
            print(
                f"[FillChannels dbg] tMin={tMin:.6f} tMax={tMax:.6f}  "
                f"Et2_min={Et2_min:.3e} Et2_max={Et2_max:.3e}  "
                f"lBnd={lBnd:.3f} rBnd={rBnd:.3f}  il={ilBnd} ir={irBnd}"
            )
            self._dbg_fill = getattr(self, "_dbg_fill", 0) + 1

        # Error handling (match C++ early returns)
        if rBnd <= lBnd:
            return
        if ilBnd >= self.Et2_npts:
            return
        if irBnd < 0:
            return

        # Recursion criterion: uses (rBnd - lBnd) in di-space, not cos-space
        if ((rBnd - lBnd) > 0.05) and (ilBnd != irBnd):
            cosMid = (3.0 * cosMin + cosMax) / 4.0

            tMid = self.TOF_MethodA(cosMid, pp2, parA)
            if tMid < 4.0:
                return
            Et2_mid = physics.t2factor * pp2 / (tMid * tMid)
            mBnd = self.Et2_to_di(Et2_mid)

            # Recurse WITHOUT splitting intens (matches C++)
            self.FillChannels(i0, cosMin, cosMid, pp2, intens, parA, max_depth=max_depth - 1)
            self.FillChannels(i0, cosMid, cosMax, pp2, intens, parA, max_depth=max_depth - 1)
            return

        # Terminal fill: copy the C++ logic
        dcos = cosMax - cosMin
        dt2 = 1.0 / (rBnd - lBnd)
        if getattr(self, "_dbg_term_cnt", 0) < 3:
            print(f"[terminal dbg] dcos={dcos:.6f} dt2={dt2:.6e} il={ilBnd} ir={irBnd}")
            self._dbg_term_cnt = getattr(self, "_dbg_term_cnt", 0) + 1

        # Single-bin case
        if ilBnd == irBnd:
            dx = rBnd - lBnd
            if 0 <= ilBnd <= (self.Et2_npts - 1):
                self.chan[i0 + ilBnd] += intens * dt2 * dx * dcos
            return

        # Multi-bin case
        if 0 <= ilBnd <= (self.Et2_npts - 1):
            dx = float(ilBnd + 1) - lBnd
            self.chan[i0 + ilBnd] += intens * dt2 * dx * dcos

        lo = max(ilBnd + 1, 0)
        hi = min(irBnd - 1, self.Et2_npts - 1)
        for it2 in range(lo, hi + 1):
            self.chan[i0 + it2] += intens * dt2 * dcos

        if irBnd < self.Et2_npts:
            dx = rBnd - float(irBnd)
            if 0 <= irBnd <= (self.Et2_npts - 1):
                self.chan[i0 + irBnd] += intens * dt2 * dx * dcos


    # -----------------------
    # Simulation (C++ simulateET2SpecMethodA)
    # -----------------------
    def simulateET2SpecMethodA(self, par: np.ndarray) -> None:
        self.g_nSimulateET2Spec += 1

        a_ev = float(par[0])
        b_F  = float(par[1])
        intens = 10.0 ** float(par[2])

        # TOF mapping nuisance
        A_cosThetaMin = float(par[3])
        A_L = float(par[4]) + 5.0
        A_alpha = float(par[5])
        A_beta  = float(par[6])
        A_gamma = float(par[7])
        A_eta   = float(par[8])

        parA = np.array([A_cosThetaMin, A_L, A_alpha, A_beta, A_gamma, A_eta], dtype=float)

        # Beam profile
        z0_center = float(par[9])
        z0_width  = float(par[10])

        # Tail / missdet etc (brems)
        self.tailStruct[0] = float(par[11])
        self.tailStruct[1] = float(par[12]) * 1e-4
        self.tailStruct[2] = float(par[13])

        # HV mapping
        self.sethvMapping(par[14], par[15], par[16], par[17], par[18], par[19])

        # Energy recon
        calEe = float(par[20])
        EeNonLinearity = float(par[21])
        calibration = np.array([calEe, EeNonLinearity], dtype=float)
        sigmaEe_keV = float(par[22])

        # Determine n_dE like C++
        n_dE = 1
        if sigmaEe_keV > 0:
            n_dE = int(math.floor(self.e2_step / 1000.0 / sigmaEe_keV + 0.5))
            if n_dE < 1:
                n_dE = 1

        # Channels (flattened)
        self.chan = np.zeros(self.numBinsX * self.numBinsY, dtype=float)

        norm_e2 = 0.0

        for ie in range(self.e2_npts):
            ioffset = ie * self.numBinsY

            for ide in range(n_dE):
                # C++ passes an index-like variable into eECal:
                #   ie + (ide+0.5)/n_dE
                e2_guess = float(ie) + (ide + 0.5) / float(n_dE)

                # calibrated energy
                e2_cal = self.eECal(e2_guess, calibration)
                if (not np.isfinite(e2_cal)) or (e2_cal <= 0.0) or (e2_cal >= physics.E0):
                    if getattr(self, "_dbg_bad_e2cal", 0) < 10:
                        print(f"[bad e2_cal] ie={ie} ide={ide} e2_guess={e2_guess} e2_cal={e2_cal} cal={calibration}")
                        self._dbg_bad_e2cal = getattr(self, "_dbg_bad_e2cal", 0) + 1
                    continue

                # probability weight in this energy slice (spectrum integral)
                eMin = self.eECal(float(ie) + float(ide) / float(n_dE), calibration)
                eMax = self.eECal(float(ie) + (float(ide) + 1.0) / float(n_dE), calibration)
                e2prob = self.numElectrons(eMin, eMax, b_F)

                # pp2 bounds at this energy
                pp2min = physics.ppmin(e2_cal) ** 2
                pp2max = physics.ppmax(e2_cal) ** 2
                # ---------------- DEBUG: pp2 and Et2 ranges ----------------
                # Print once for the Ee ~ 450 keV bin (ix450), first time we hit it
                if ie == getattr(self, "ix450", -999) and getattr(self, "_dbg_pp2_done", False) is False:
                    self._dbg_pp2_done = True
                    ppmin = physics.ppmin(e2_cal)
                    ppmax = physics.ppmax(e2_cal)
                    print(f"[pp2 dbg] ie={ie} Ee={e2_cal:.1f} eV")
                    print(f"          ppmin={ppmin:.6e}  ppmax={ppmax:.6e}  (eV/c)")
                    print(f"          pp2min={pp2min:.6e}  pp2max={pp2max:.6e}  ((eV/c)^2)")

                    # Use representative TOFs near what your TOF grid prints (microseconds)
                    for tof_us in [6.0, 5.5, 5.2]:
                        et2_lo = physics.t2factor * pp2min / (tof_us * tof_us)
                        et2_hi = physics.t2factor * pp2max / (tof_us * tof_us)
                        print(f"          [Et2range dbg] tof={tof_us:.2f} us  "
                              f"Et2(pp2min)={et2_lo:.3e}  Et2(pp2max)={et2_hi:.3e}  (s^-2)")
# -----------------------------------------------------------

                # skip if no phase space
                if pp2max <= pp2min:
                    continue

                # C++: npp = 5*FloorNint(t2factor*(pp2max-pp2min)/(A_L*A_L)/Et2_step + 0.5), capped at 1000
                npp_base = physics.t2factor * (pp2max - pp2min) / (A_L * A_L) / self.Et2_step + 0.5
                npp = 5 * int(math.floor(npp_base + 0.5))
                npp = max(1, min(npp, 1000))

                for ipos in range(self.npos):
                    # z smear: midpoint samples across [-width/2, +width/2]
                    zTmp = z0_center + (ipos + 0.5) * z0_width / float(self.npos) - 0.5 * z0_width
                    parA_tmp = parA.copy()
                    parA_tmp[1] = A_L - zTmp  # L_zDV = A_L - zTmp

                    # Uniform sample pp2
                    for ipp in range(npp):
                        # midpoint sample
                        pp2 = pp2min + (ipp + 0.5) * (pp2max - pp2min) / float(npp)

                        pp2mid = physics.ppmid2(e2_cal)
                        pp2denom = 2.0 * (e2_cal + physics.me) * (physics.delta - physics.me - e2_cal)
                        if pp2denom == 0:
                            continue

                        # P_p2 in C++
                        fierz_fac = physics.me / (e2_cal + physics.me) if (e2_cal + physics.me) != 0 else 0.0
                        numer = 1.0 + a_ev * (pp2 - pp2mid) / pp2denom + b_F * fierz_fac
                        denom_f = 1.0 + b_F * fierz_fac
                        P_p2 = numer / denom_f if denom_f != 0 else 0.0

                        intensTmp = intens * (P_p2 * e2prob) / (self.npos * npp) / 2.0

                        # Integrate over acceptance angle cosTheta0
                        # cosThetaMin is parA_tmp[0], cosThetaMax is self.cosThetaMax
                        cosMin = A_cosThetaMin + 1e-6
                        cosMax = self.cosThetaMax

                        # Guard: must have cosMin < cosMax (otherwise dcos=0 and model is identically zero)
                        if cosMax <= cosMin:
                            # C++ effectively would produce nothing; but print once so we know
                            if getattr(self, "_dbg_cos_bad", 0) < 5:
                                print(f"[cos dbg] BAD cos range: cosMin={cosMin} cosMax={cosMax}  (skipping)")
                                self._dbg_cos_bad = getattr(self, "_dbg_cos_bad", 0) + 1
                            continue

                        cosMin = float(A_cosThetaMin) + 1.0e-6
                        cosMax = 1.0
                        self.FillChannels(ioffset, cosMin, cosMax, pp2, intensTmp, parA_tmp)



                norm_e2 += e2prob

        # Reshape channels into 2D prediction
        self.h_fit2D = self.chan.reshape(self.numBinsX, self.numBinsY)

        if self.verbose:
            print("[simulateET2SpecMethodA] done")
            print(f"  chan sum={self.chan.sum()} h_fit2D sum={self.h_fit2D.sum()}")
            print(f"  norm_e2={norm_e2}")
            print(f"  fill calls={self.g_nFillChan} simulate calls={self.g_nSimulateET2Spec}")

    # -----------------------
    # Fit region setup (C++ fit_noE block)
    # -----------------------
    def build_fit_region_from_data(self) -> None:
        ix450 = int(math.floor(self.e2_to_di(450000.0) + 0.5)) + 1  # C++: FloorNint +1
        ix450 = max(1, min(ix450, self.numBinsX))
        ix = ix450 - 1
        self.ix450 = ix

        events_in_slice = float(np.sum(self.h_data[ix, :]))
        if events_in_slice <= 0.0:
            raise ValueError("No events in 450 keV slice; cannot build fit region.")

        cum = 0.0
        Et2_450_low_outer = 0.0
        Et2_450_low_inner = 0.0
        Et2_450_up_inner = 0.0
        Et2_450_up_outer = 0.0

        for iy in range(1, self.numBinsY + 1):
            cum += float(self.h_data[ix, iy - 1])
            if cum < 0.02 * events_in_slice:
                Et2_450_low_outer = self.di_to_Et2(iy - 0.5)
            if cum < 0.13 * events_in_slice:
                Et2_450_low_inner = self.di_to_Et2(iy - 0.5)
            if cum < 0.87 * events_in_slice:
                Et2_450_up_inner = self.di_to_Et2(iy - 0.5)
            if cum < 0.98 * events_in_slice:
                Et2_450_up_outer = self.di_to_Et2(iy - 0.5)

        lo_clip = (1.0 / 40e-6) ** 2
        hi_clip = (1.0 / 10e-6) ** 2

        for ix1 in range(1, self.numBinsX + 1):
            e2_guess = self.di_to_e2(ix1 - 1)
            if e2_guess < 100000.0:
                continue

            low_outer = (physics.ppmin(e2_guess) / physics.ppmin(450000.0)) ** 2 * Et2_450_low_outer
            low_inner = (physics.ppmin(e2_guess) / physics.ppmin(450000.0)) ** 2 * Et2_450_low_inner
            up_outer  = (physics.ppmax(e2_guess) / physics.ppmax(450000.0)) ** 2 * Et2_450_up_outer
            up_inner  = (physics.ppmax(e2_guess) / physics.ppmax(450000.0)) ** 2 * Et2_450_up_inner

            self.h_low_outer[ix1 - 1] = physics.EnsureRange(low_outer, lo_clip, hi_clip)
            self.h_low_inner[ix1 - 1] = physics.EnsureRange(low_inner, lo_clip, hi_clip)
            self.h_up_outer[ix1 - 1]  = physics.EnsureRange(up_outer,  lo_clip, hi_clip)
            self.h_up_inner[ix1 - 1]  = physics.EnsureRange(up_inner,  lo_clip, hi_clip)

        if self.verbose:
            print("[build_fit_region_from_data] built")
            print(f"  ix450={ix450} events_in_slice={events_in_slice}")
            print(f"  Et2_450_low_outer={Et2_450_low_outer} low_inner={Et2_450_low_inner}")
            print(f"  Et2_450_up_inner={Et2_450_up_inner} up_outer={Et2_450_up_outer}")

    # -----------------------
    # Chi2 (C++ faithful)
    # -----------------------
    def chi2(self, par: np.ndarray) -> float:
        a_ev = float(par[0])
        b_F  = float(par[1])
        logN = float(par[2])

        # Parameter bounds like C++ chi2 checks
        if (a_ev < -1.0) or (a_ev > 1.0) or (b_F < -0.5) or (b_F > 0.5) or (logN < 0.0) or (logN > 20.0):
            return 1e18

        # TOF parameter sanity check grid like C++ (coarse)
        A_cosThetaMin = float(par[3])

        A_L     = float(par[4]) + 5.0
        A_alpha = float(par[5])
        A_beta  = float(par[6])
        A_gamma = float(par[7])
        A_eta   = float(par[8])


        parA = np.array([A_cosThetaMin, A_L, A_alpha, A_beta, A_gamma, A_eta], dtype=float)

        # Check monotonic TOF and range (C++ uses 10 steps)
        test_cos = np.linspace(A_cosThetaMin + 1e-5, 1.0, 10)
        prev = None
        for c in test_cos:
            t = self.TOF_MethodA(float(c), 0.0, parA)
            if (t < 4.5) or (t > 5.8):
                return 1e18
            if prev is not None and t >= prev:
                return 1e18
            prev = t

        # Simulate
        self.simulateET2SpecMethodA(par)

        # Compute chi2 in fit region
        chi2 = 0.0
        nfit = 0

        for ix in range(self.numBinsX):
            # Determine region bounds for this ix
            # If curves not built, default full range.
            low = self.h_low_outer[ix] if self.h_low_outer[ix] != 0 else self.ymin
            up  = self.h_up_outer[ix]  if self.h_up_outer[ix]  != 0 else self.ymax

            ilow = int(math.floor(self.Et2_to_di(low) + 0.5))
            iup  = int(math.floor(self.Et2_to_di(up) + 0.5))

            if self.FitRegionIncludesEdges:
                ilow = max(ilow, 0)
                iup  = min(iup, self.numBinsY - 1)
            else:
                ilow = max(ilow + 1, 0)
                iup  = min(iup - 1, self.numBinsY - 1)

            if iup < ilow:
                continue

            for iy in range(ilow, iup + 1):
                d = float(self.h_data[ix, iy])
                m = float(self.h_fit2D[ix, iy])
                e = float(self.h_err[ix, iy])

                if e != 0.0:
                    chi2 += (m - d) * (m - d) / (e * e)
                else:
                    chi2 += (m - d) * (m - d)
                nfit += 1

        if self.verbose:
            print(f"[chi2] chi2={chi2} nfitbins={nfit}")
        return float(chi2)

    # -----------------------
    # Sanity checks
    # -----------------------
    def sanity_checks(self) -> None:
        print("---- SANITY CHECKS ----")
        print(f"Bins: X={self.numBinsX} Y={self.numBinsY}")
        print(f"X range: {self.xmin}..{self.xmax} step={self.xBinWidth}")
        print(f"Y range: {self.ymin}..{self.ymax} step={self.yBinWidth}")
        print(f"Data sum: {np.sum(self.h_data)} min/max: {self.h_data.min()} {self.h_data.max()}")
        print(f"Err  min/max: {self.h_err.min()} {self.h_err.max()}")
        print(f"Spectrum raw[end]={self.eESpec_raw[-1]} (should be ~1)")
        print(f"Spectrum me[end]={self.eESpec_me[-1]}")
        # Quick TOF check with typical par
        parA = np.array([0.76, 5.09, 0.6, -0.05, 2.1, 0.056], dtype=float)
        cgrid = np.linspace(parA[0]+1e-4, 1.0, 6)
        tof = [self.TOF_MethodA(float(c), 0.0, parA) for c in cgrid]
        print("TOF grid:", tof)
        print("-----------------------")
