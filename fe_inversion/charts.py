"""Digitized Schlumberger chart relations shared by the projects (2009 Log Interpretation Charts)."""
import numpy as np

# Chart Por-5 (former Por-13b), thermal neutron NPHI, porosity equivalence: true porosity of quartz sandstone and
# dolomite against the apparent limestone reading (v/v). Digitized from the curve colours (about ±0.005); the
# sandstone value at zero porosity was read by eye.
POR5_PHI = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
POR5_SS_LS = np.array([-0.024, 0.0148, 0.0578, 0.1042, 0.1531, 0.2015, 0.2525, 0.3031, 0.3524])
POR5_DOL_LS = np.array([0.0323, 0.1192, 0.1798, 0.2312, 0.2780, 0.3202, 0.3611, 0.3971, 0.4330])


def _interp_ext(x, xp, fp):
    """Linear interpolation, extended linearly beyond the last chart point (shales read about 0.5)."""
    x = np.asarray(x, float)
    y = np.interp(x, xp, fp)
    hi = x > xp[-1]
    return np.where(hi, fp[-1] + (x - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]), y)


def ss_to_ls(nphi_ss):
    """Sandstone- to limestone-unit neutron porosity from chart Por-5 (about -0.047 above 0.15)."""
    return _interp_ext(nphi_ss, POR5_PHI, POR5_SS_LS)


def ls_to_ss(nphi_ls):
    """Limestone- to sandstone-unit neutron porosity (inverse of ss_to_ls)."""
    return _interp_ext(nphi_ls, POR5_SS_LS, POR5_PHI)


def nd_lines(ax, units="sandstone"):
    """Water-filled sandstone, limestone and dolomite lines (fresh water, as chart Por-11) in the given neutron
    units, with the neutron responses of chart Por-5."""
    phi = np.linspace(0, 0.40, 21)
    ls = {"Sandstone": ss_to_ls(phi), "Limestone": phi, "Dolomite": np.interp(phi, POR5_PHI, POR5_DOL_LS)}
    for name, rho, col in [("Sandstone", 2.65, "goldenrod"), ("Limestone", 2.71, "steelblue"), ("Dolomite", 2.87, "purple")]:
        x = ls[name] if units == "limestone" else ls_to_ss(ls[name])
        ax.plot(x, rho - phi * (rho - 1.0), color=col, lw=1.2, label=name)
