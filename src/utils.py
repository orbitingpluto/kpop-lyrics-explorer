import numpy as np
import pandas as pd
import streamlit as st
# map language codes in csv to labels
LANG_LABELS = {
    "kor": "Korean only",
    "koreng": "Korean & English mix",
    "en": "English only",
    "jpn": "Japanese only",
    "ch": "Chinese only",
}

# assign hex to each label for charts
LANG_COLORS = {
    "Korean only": "#fa688c",
    "Korean & English mix": "#af83f9",
    "English only": "#6bb5d7",
    "Japanese only": "#dfac6d",
    "Chinese only": "#5ac865",
}

# big 4 agencies used by the big4-only filter toggle below.
BIG4_AGENCIES = ["SM", "JYP", "YG", "HYBE"]


def padded_range(*series_list, pad_frac=0.08, floor=None, ceil=None):
    """
    Tight axis range around actual data instead of a fixed full-scale range.
    :param series_list: one or more Series/arrays
    :param pad_frac: how much padding to apply
    :param floor: minimum value
    :param ceil: maximum value
    """
    vals = np.concatenate([pd.Series(s).dropna().values for s in series_list])
    lo, hi = vals.min(), vals.max()
    span = max(hi - lo, 1e-9)
    pad = span * pad_frac
    lo, hi = lo - pad, hi + pad
    if floor is not None:
        lo = max(lo, floor)
    if ceil is not None:
        hi = min(hi, ceil)
    return [lo, hi]


def mobile_layout(**extra):
    """
    Returns Plotly layout kwargs tuned for narrow mobile screens.
    """
    if not st.session_state.get("is_mobile", False):
        return extra  # desktop: only pass through explicit overrides like height
    base = dict(
        font=dict(size=11),
        margin=dict(l=8, r=8, t=36, b=48),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0,
            font=dict(size=10),
        ),
        title_font=dict(size=13),
    )
    base.update(extra)
    return base
