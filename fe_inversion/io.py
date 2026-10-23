"""Loading of configuration, well logs (LAS) and formation-pressure data."""
import tomllib
from pathlib import Path

import lasio
import numpy as np
import pandas as pd


def load_config(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_well(path, curves, top=None, bottom=None):
    """Read a LAS file and return a DataFrame indexed by depth with standard names.

    `curves` maps standard names (GR, RHOB, ...) to LAS mnemonics. Curves that
    are missing from the file are filled with NaN so later code can test for them.
    """
    las = lasio.read(path)
    raw = las.df()
    raw.index = raw.index.astype(float)
    raw.index.name = "DEPTH"
    out = pd.DataFrame(index=raw.index)
    for name, mnemonic in curves.items():
        out[name] = raw[mnemonic].astype(float) if mnemonic in raw.columns else np.nan
    if top is not None or bottom is not None:
        out = out.loc[top:bottom]
    return out, las


def load_pressures(path):
    """Formation-tester pressures. Rows without a valid formation pressure are dropped."""
    las = lasio.read(path)
    df = las.df().reset_index()
    df.columns = ["DEPTH", "FM_PRESS", "HYD_PRESS", "MOB", "TEMP"][: len(df.columns)]
    df = df.replace(-999.25, np.nan)
    df = df[df.FM_PRESS.notna() & (df.FM_PRESS > 0)]
    return df.sort_values("DEPTH").reset_index(drop=True)


def quality_flags(df, max_drho):
    """Boolean flags for samples whose measurements should not be trusted."""
    flags = pd.DataFrame(index=df.index)
    flags["BAD_RHOB"] = df.DRHO.abs() > max_drho
    flags["NO_RHRV"] = ~((df.RH > 0) & (df.RV > 0))
    # Bad-hole flag relative to the median caliper (bit size is not in the LAS).
    flags["WASHOUT"] = (df.CALI - df.CALI.median()) > 1.0
    return flags


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
