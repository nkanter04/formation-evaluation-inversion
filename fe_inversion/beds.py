"""Bed boundaries and thick-bed center points.

Readings near a bed boundary average the beds on both sides (shoulder-bed
effect), so calibration and cross-checks use only samples near the middle of
beds that are thick compared with the tools' vertical resolution.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def boundaries(depth, curves, thresholds, smooth_ft=0.75, min_spacing_ft=1.0):
    """Indices of bed boundaries: peaks in |d(curve)/dz| above each curve's threshold.

    `curves` and `thresholds` are parallel lists (threshold in curve units per ft).
    The first and last samples are always boundaries.
    """
    depth = np.asarray(depth, float)
    step = float(np.median(np.diff(depth)))
    picks = [0, len(depth) - 1]
    for curve, thr in zip(curves, thresholds):
        c = np.asarray(curve, float)
        c = np.interp(depth, depth[np.isfinite(c)], c[np.isfinite(c)])  # fill gaps for smoothing
        grad = np.abs(np.gradient(gaussian_filter1d(c, smooth_ft / step), step))
        peaks, _ = find_peaks(grad, height=thr, distance=max(1, int(min_spacing_ft / step)))
        picks.extend(peaks.tolist())
    picks = np.unique(picks)
    kept = [picks[0]]
    for i in picks[1:]:
        if depth[i] - depth[kept[-1]] >= min_spacing_ft:
            kept.append(i)
    if kept[-1] != len(depth) - 1:
        kept[-1] = len(depth) - 1
    return np.array(kept)


def bed_table(depth, idx):
    """(top, bottom, thickness) of every bed between consecutive boundaries."""
    depth = np.asarray(depth, float)
    tops, bots = depth[idx[:-1]], depth[idx[1:]]
    return np.column_stack([tops, bots, bots - tops])


def thick_bed_centers(depth, idx, min_thickness_ft, edge_ft):
    """Mask of samples inside beds >= min thickness and >= edge_ft from both boundaries."""
    depth = np.asarray(depth, float)
    mask = np.zeros(depth.size, bool)
    for top, bot, thick in bed_table(depth, idx):
        if thick >= min_thickness_ft:
            mask |= (depth >= top + edge_ft) & (depth <= bot - edge_ft)
    return mask


def thickest_bed(depth, idx, select, min_fraction=0.8):
    """(top, bottom) of the thickest bed in which at least `min_fraction` of samples satisfy `select`."""
    depth, select = np.asarray(depth, float), np.asarray(select, bool)
    best = None
    for top, bot, thick in bed_table(depth, idx):
        m = (depth >= top) & (depth < bot)
        if m.any() and select[m].mean() >= min_fraction and (best is None or thick > best[2]):
            best = (top, bot, thick)
    return None if best is None else (best[0], best[1])
