import numpy as np

# Mirror of physics namespace in methodAFit.cpp (units: eV, seconds, kg via constants embedded)
# Constants from C++:
me = 510998.9461          # eV
mp = 938272088.16         # eV
delta = 129372.4          # eV (Mn - Mp)
E0 = 782333.6             # eV (endpoint)
alpha_fs = 1.0 / 137.035999084  # fine-structure constant

# This constant appears in C++ (physics::t2factor) when converting pp2 and TOF into 1/t^2.
# In methodAFit.cpp it is not used in the main (current) Method A mapping path, but it exists.
delta = 1.293332e6     # eV  (mn - mp)

c_SI = 2.99792e8       # m/s

# C++: t2factor = c_SI*c_SI/(mn - delta)/(mn - delta)
# Since (mn - delta) = mp:
t2factor = (c_SI * c_SI) / (mp * mp)


def EnsureRange(val: float, lo: float, hi: float) -> float:
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val

def pe(Ee: float) -> float:
    # Electron momentum magnitude in eV
    # Ee is kinetic energy in eV
    E = Ee + me
    p2 = E*E - me*me
    return np.sqrt(p2) if p2 > 0 else 0.0

def beta(Ee: float) -> float:
    # v/c for the electron
    p = pe(Ee)
    E = Ee + me
    return p / E if E > 0 else 0.0

def pv(Ee: float) -> float:
    # Proton neutrino momentum proxy used in original code; kept for completeness
    # This is not heavily used in Method A fitting.
    # Placeholder consistent with older Nab code patterns:
    return pe(Ee)

def ppmin(Ee: float) -> float:
    # Proton momentum minimum (eV) for given electron energy Ee (eV)
    # Derived from neutron beta decay kinematics.
    # Matches the intent in methodAFit.cpp usage.
    # Uses pp^2 bounds in the fit region mapping.
    Enu = E0 - Ee
    if Enu < 0:
        Enu = 0.0
    p_e = pe(Ee)
    p_nu = Enu  # neutrino mass neglected => p = E
    p_p_min = abs(p_e - p_nu)
    return p_p_min

def ppmax(Ee: float) -> float:
    Enu = E0 - Ee
    if Enu < 0:
        Enu = 0.0
    p_e = pe(Ee)
    p_nu = Enu
    p_p_max = p_e + p_nu
    return p_p_max

def costheta_ev(Ee: float, pp2: float) -> float:
    # cos(theta_e-nu) from energy Ee (eV) and proton momentum^2 (eV^2).
    # Kept for diagnostics; Method A teardrop painter integrates over cosTheta0 acceptance instead.
    p_e = pe(Ee)
    p_nu = E0 - Ee
    if p_nu <= 0 or p_e <= 0:
        return 1.0
    p_p = np.sqrt(pp2) if pp2 > 0 else 0.0
    # from vector relation: p_p^2 = p_e^2 + p_nu^2 + 2 p_e p_nu cos(theta_e-nu)
    num = p_p*p_p - p_e*p_e - p_nu*p_nu
    den = 2.0*p_e*p_nu
    return EnsureRange(num/den, -1.0, 1.0) if den != 0 else 1.0
