import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import linregress

# thresholds used everywhere we decide whether a trend is "solid" (real,
# consistent) vs just noisy bouncing that happens to average out to a slope.
# pulled out as shared constants so the intro, movers tab, heatmap sort, and
# artist lookup can't quietly drift out of sync with each other again.
SOLID_TREND_P_MAX = 0.1
SOLID_TREND_R2_MIN = 0.2


def fit_trend(series):
    """
    Fit an OLS regression of a value against time (index position) and
    return everything needed to narrate it *consistently*.

    Returns a dict with:
      - fitted_start, fitted_end: the regression LINE's values at the first
        and last point (not the raw/noisy actual values there). Use these
        two numbers together in a sentence and their direction can never
        contradict the reported change or trend direction, since they come
        from the exact same fit.
      - change: fitted_end - fitted_start (same sign as slope, by construction)
      - r_squared, p_value: how much to trust the fit
      - is_solid: whether this trend clears the shared "real trend" bar
      - n: number of valid (non-NaN) points used

    Returns None if fewer than 2 valid points are available.

    :param series: pandas Series indexed by year (or any ordered sequence),
        with NaNs allowed (they're dropped before fitting)
    """
    s = series.dropna()
    if len(s) < 2:
        return None
    x = np.arange(len(s))
    slope, intercept, r, p, _ = linregress(x, s.values)
    fitted_start = intercept
    fitted_end = intercept + slope * (len(s) - 1)
    r_squared = r ** 2
    return {
        "fitted_start": fitted_start,
        "fitted_end": fitted_end,
        "change": fitted_end - fitted_start,
        "slope": slope,
        "r_squared": r_squared,
        "p_value": p,
        "is_solid": (p < SOLID_TREND_P_MAX) and (r_squared > SOLID_TREND_R2_MIN),
        "n": len(s),
    }


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