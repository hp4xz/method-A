import numpy as np

# Mirror of physics namespace in methodAFit.cpp/methodAFit.hh
me = 510999.06            # eV
mn = 939565641.8          # eV
delta = 1293331.8         # eV (mn - mp)
E0 = delta - me           # eV
mp = mn - delta           # eV
alpha_fs = 1.0 / 137.036

c_SI = 2.99792e8          # m/s
t2factor = (c_SI * c_SI) / (mp * mp)


def EnsureRange(val: float, lo: float, hi: float) -> float:
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val


def pe(Ee: float) -> float:
    return np.sqrt((Ee + me) * (Ee + me) - (me * me))


def pe2(Ee: float) -> float:
    return (Ee + me) * (Ee + me) - (me * me)


def pv(Ee: float) -> float:
    # C++: delta - me - Ee
    return delta - me - Ee


def ppmin(Ee: float) -> float:
    return pe(Ee) - pv(Ee)


def ppmax(Ee: float) -> float:
    return pe(Ee) + pv(Ee)


def pp2diff(Ee: float) -> float:
    return 4.0 * pe(Ee) * pv(Ee)


def ppmid(Ee: float) -> float:
    return np.sqrt(pe2(Ee) + pv(Ee) * pv(Ee))


def ppmid2(Ee: float) -> float:
    return pe2(Ee) + pv(Ee) * pv(Ee)


def beta(Ee: float) -> float:
    if Ee > 0:
        return pe(Ee) / (me + Ee)
    return 0.0


def costheta_ev(Ee: float, pp2: float) -> float:
    den = 2.0 * pe(Ee) * pv(Ee)
    if den == 0:
        return 1.0
    return (pp2 - pe2(Ee) - (pv(Ee) * pv(Ee))) / den


def gamma_C(Ee: float) -> float:
    b = beta(Ee)
    return 1.0 / np.sqrt(1.0 - b * b)
