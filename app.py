import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import json
import re
from scipy.stats import linregress
from src.utils import (
    LANG_LABELS, LANG_COLORS, BIG4_AGENCIES, mobile_layout,
    fit_trend, SOLID_TREND_P_MAX, SOLID_TREND_R2_MIN,
)

# use full browser width
st.set_page_config(
    layout="wide",
    page_title="kpop lyrics explorer ✦",
    page_icon="✦",
    menu_items={"About": "13,000+ kpop songs tagged by language, 2010–2025."}
)
# ── mobile detection ──────────────────────────────────────────────────────
_ua = st.context.headers.get("User-Agent", "")
st.session_state["is_mobile"] = bool(
    re.search(r"Mobi|Android|iPhone|iPad|iPod", _ua, re.IGNORECASE)
)

def _is_big4(agency):
    """
    Uppercase agency string and check if it starts with one of the four
    :param agency: agency value from csv
    :returns: True if big 4 agency
    """
    a = str(agency).upper()
    return any(a.startswith(k) for k in BIG4_AGENCIES)

def slope_change(series):
    """
    Fit a linear regression (OLS) of a value against time (in index position,
    e.g. years) and return the *fitted* total change across the series'
    span i.e. slope * (n_points - 1) instead of the naive
    last-value-minus-first-value difference.

    This is far less sensitive to noisy endpoints: a single unusually high
    or low first/last year won't dominate the "change" the way it would with
    a raw endpoint subtraction, because the slope is fit using every point
    in between.

    :param series: pandas Series indexed by year (or any ordered sequence),
        with NaNs allowed (they're dropped before fitting)
    :returns: float total change over the series' span fit via OLS slope,
        or np.nan if fewer than 2 valid points are available
    """
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    x = np.arange(len(s))
    slope, intercept, _, _, _ = linregress(x, s.values)
    return slope * (len(s) - 1)

def slope_per_step(series):
    """
    Same fit as slope_change, but returns the raw per-step slope rather than
    the total change across the span. Useful when comparing series of
    different lengths (e.g. artists with different numbers of active years).

    :param series: pandas Series indexed by year (or any ordered sequence)
    :returns: float slope (change per index step), or np.nan if fewer than
        2 valid points are available
    """
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    x = np.arange(len(s))
    slope, intercept, _, _, _ = linregress(x, s.values)
    return slope

# ── data loading ──────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    """
    load csv data, agency map, generation map, group type map

    """
    df = pd.read_csv("data/processed/kpop_lyrics_clean.csv")
    df["year"] = df["date"].str[:4]
    df["date"] = pd.to_datetime(df["date"])
    # drop pre-2010 data. upper cap at 2025 is intentional (not a leftover)
    # 2026 data is still too sparse this early in the year to be reliable.
    # bump this cap manually once there's enough 2026 data to trust.
    df = df[df["year"] >= "2010"]
    df = df[df["year"] <= "2025"]

    with open("data/processed/agency_map.json") as f:
        agency_map = json.load(f)
    with open("data/processed/gen_map.json") as f:
        gen_map = json.load(f)
    with open("data/processed/group_type_map_ccl.json") as f:
        group_type_map = json.load(f)

    # map lang release codes to labels
    df["language_release"] = df["language_release"].map(LANG_LABELS)

    # fill na with unknown labels
    df["group_type"] = df["artist"].map(group_type_map).fillna("unknown")
    df["agency"] = df["artist"].map(agency_map).fillna("Other")
    df["generation"] = df["artist"].map(gen_map).fillna("Unknown")

    song_counts = df.groupby("artist")["url"].count()
    # only include artists with 10+ songs
    eligible_artists = song_counts[song_counts >= 10].index
    df = df[df["artist"].isin(eligible_artists)].copy()

    # remove artists whose songs are exclusively Japanese or Chinese
    non_jpn_ch = df[~df["language_release"].isin(["Japanese only", "Chinese only"])]
    artists_with_korean_or_english = non_jpn_ch["artist"].unique()
    df = df[df["artist"].isin(artists_with_korean_or_english)].copy()

    return df

@st.cache_data
def load_trends():
    return pd.read_csv("data/processed/artist_trends_without_voc.csv")

df = load_data()
trends_df = load_trends()

# ── scrollytelling intro ──────────────────────────────────────────────────


@st.cache_data
def compute_intro_stats(df, trends_df):
    """
    
    """
    pct_col = "pct_english_without_voc"

    # 1. overall industry trend average, by year
    overall = (df.groupby("year")[pct_col].mean() * 100)
    first_year, last_year = overall.index[0], overall.index[-1]
    start_overall, end_overall = overall.iloc[0], overall.iloc[-1]

    # 2. release-type composition (% english only, korean & english, etc) by year
    lang_counts = df.groupby(["year", "language_release"]).size().unstack(fill_value=0)
    for col in LANG_LABELS.values():
        if col not in lang_counts.columns:
            lang_counts[col] = 0
    totals = lang_counts[list(LANG_LABELS.values())].sum(axis=1)
    lang_pct = lang_counts.div(totals, axis=0) * 100

    # 3. find first year vs last year % in korean and english only songs
    # (endpoint values, used for the "it went from X to Y" narrative text)
    kor_only_start = lang_pct.loc[first_year, "Korean only"]
    kor_only_recent = lang_pct.loc[last_year, "Korean only"]
    eng_only_start = lang_pct.loc[first_year, "English only"]
    eng_only_recent = lang_pct.loc[last_year, "English only"]
    # change is the OLS-fitted slope's total movement across the whole
    # series, not just the recent-year minus first-year endpoints, so one
    # unusually high/low year at either edge doesn't dominate the number
    kor_only_change = slope_change(lang_pct["Korean only"])
    eng_only_change = slope_change(lang_pct["English only"])

    # and first and last english % within mixed-language songs
    mixed = df[df["language_release"] == "Korean & English mix"]
    mixed_eng = mixed.groupby("year")[pct_col].mean() * 100
    mixed_start = mixed_eng.iloc[0]
    mixed_recent = mixed_eng.iloc[-1]
    mixed_change = slope_change(mixed_eng)

    # 4. find inflection point in graph: fit two separate lines (before/after
    # a candidate split year) and pick the split that minimizes the combined
    # fit error across both segments
    # uses sum of squared residuals (SSE)
    overall_vals = overall.values
    overall_years = overall.index.tolist()
    best_year = None
    best_sse = np.inf
    for i in range(2, len(overall_vals) - 2):
        x1, y1 = np.arange(i), overall_vals[:i]
        x2, y2 = np.arange(len(overall_vals) - i), overall_vals[i:]
        slope1, intercept1, _, _, _ = linregress(x1, y1)
        slope2, intercept2, _, _, _ = linregress(x2, y2)
        sse = np.sum((y1 - (slope1 * x1 + intercept1)) ** 2) + np.sum((y2 - (slope2 * x2 + intercept2)) ** 2)
        if sse < best_sse:
            best_sse = sse
            best_year = overall_years[i]
    inflection_year = best_year

    # 5. find artists with most influence in each direction
    t = trends_df.copy()
    t["weighted_slope"] = t["slope_per_year"] * np.log1p(t["n_songs"])
    # slope-fitted change
    t["fitted_change"] = t["slope_per_year"] * (t["n_years_active"] - 1)
    median_change = t["fitted_change"].median() * 100

    # same "still active" + "solid trend" bar the movers section uses below,
    # so the intro's headline mover can't contradict what that section shows
    # a few scrolls later. computed off the full (unfiltered) df since the
    # intro renders before the filter bar exists.
    _intro_recent_cutoff = str(int(df["year"].max()) - 2)
    _intro_recent_counts = (
        df[df["year"] >= _intro_recent_cutoff]
        .groupby("artist")["url"].count()
    )
    _intro_active = _intro_recent_counts[_intro_recent_counts > 5].index
    _is_intro_solid = (t["p_value"] < SOLID_TREND_P_MAX) & (t["r_squared"] > SOLID_TREND_R2_MIN) & (t["artist"].isin(_intro_active))
    t_solid = t[_is_intro_solid].copy()

    # flag whether we had to fall back to the full unfiltered pool (rather
    # than crashing), since that pool includes noisy/insignificant trends
    # the narrative text needs to caveat this so it doesn't narrate a shaky
    # trend with the same confidence as a real one
    used_solid_fallback = t_solid.empty
    if used_solid_fallback:
        t_solid = t

    top_mover = t_solid.sort_values("weighted_slope", ascending=False).iloc[0]

    # only treat this as a genuine "toward korean" mover if it actually has a
    # negative slope otherwise "bottom_mover" is just the weakest
    # toward-english mover, and shouldn't be narrated as "the other way"
    _toward_korean = t_solid[t_solid["weighted_slope"] < 0]
    if not _toward_korean.empty:
        bottom_mover = _toward_korean.sort_values("weighted_slope", ascending=True).iloc[0]
        has_korean_mover = True
    else:
        # fall back to the weakest english-ward mover just so we have *a*
        # second name to reference, but flag it so the narrative text
        # doesn't misrepresent its direction
        bottom_mover = t_solid.sort_values("weighted_slope", ascending=True).iloc[0]
        has_korean_mover = False

    # top movers for the stage-3 bar chart — mirrors the homepage's movers
    # chart: x-axis is the weighted shift score (trend strength × how much
    # data backs it up), not the raw pp change, so it's ranked consistently
    # with the homepage view
    movers_df = t_solid.sort_values("weighted_slope", ascending=False).head(10).copy()
    movers_df["change_pct_pts"] = movers_df["fitted_change"] * 100
    movers_df["direction"] = movers_df["weighted_slope"].apply(
        lambda s: "toward english" if s > 0 else "toward korean"
    )
    movers_df = movers_df.sort_values("weighted_slope", ascending=True)

    return {
        "first_year": first_year,
        "last_year": last_year,
        "start_overall": round(start_overall),
        "end_overall": round(end_overall),
        "inflection_year": inflection_year,
        "overall_series": overall,
        "kor_only_start": round(kor_only_start, 1),
        "kor_only_recent": round(kor_only_recent, 1),
        "kor_only_change": round(kor_only_change, 1),
        "eng_only_start": round(eng_only_start, 1),
        "eng_only_recent": round(eng_only_recent, 1),
        "eng_only_change": round(eng_only_change, 1),
        "mixed_start": round(mixed_start),
        "mixed_recent": round(mixed_recent),
        "mixed_change": round(mixed_change),
        "median_change": round(median_change, 1),
        "top_mover_name": top_mover["artist"],
        "top_mover_change": round(top_mover["fitted_change"] * 100),
        "bottom_mover_name": bottom_mover["artist"],
        "bottom_mover_change": round(bottom_mover["fitted_change"] * 100),
        "has_korean_mover": has_korean_mover,
        "used_solid_fallback": used_solid_fallback,
        "movers_df": movers_df
    }

stats = compute_intro_stats(df, trends_df)

if "intro_done" not in st.session_state:
    st.session_state.intro_done = False
if "intro_stage" not in st.session_state:
    st.session_state.intro_stage = 0

# custom
def render_sparkline(series, highlight_year=None, height=180):
    """
    (Intro) Inline trend line with Y-axis labels and value callouts.
    :param series: data
    :param highlight_year: at this year, add dotted line through
    :param height: height of line
    """
    vals = series.values
    years = series.index.tolist()
    vmin, vmax = vals.min(), vals.max()
    span = max(vmax - vmin, 1e-6)
    width, pad_l, pad_r, pad_t, pad_b = 600, 48, 24, 24, 32
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n = len(vals)
    xs = [pad_l + i * plot_w / (n - 1) for i in range(n)]
    ys = [pad_t + plot_h * (1 - (v - vmin) / span) for v in vals]
    path = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    mid_val = (vmin + vmax) / 2
    grid_svg = ""
    for gval, glabel in [(vmax, f"{vmax:.0f}%"), (mid_val, f"{mid_val:.0f}%"), (vmin, f"{vmin:.0f}%")]:
        gy = pad_t + plot_h * (1 - (gval - vmin) / span)
        grid_svg += (
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - pad_r}" y2="{gy:.1f}" '
            f'stroke="#3d2b4f" stroke-width="0.5" stroke-dasharray="2 4"/>'
            f'<text x="{pad_l - 4}" y="{gy + 4:.1f}" font-size="10" fill="#a98fc4" text-anchor="end">{glabel}</text>'
        )

    highlight_svg = ""
    if highlight_year and highlight_year in years:
        idx = years.index(highlight_year)
        lx = min(xs[idx] + 6, width - pad_r - 28)
        highlight_svg = (
            f'<line x1="{xs[idx]:.1f}" y1="{pad_t}" x2="{xs[idx]:.1f}" y2="{height - pad_b}" '
            f'stroke="#c8a8ff" stroke-width="1" stroke-dasharray="3 3"/>'
            f'<text x="{lx:.1f}" y="{pad_t + 13}" font-size="11" fill="#c8a8ff">{highlight_year}</text>'
        )

    start_label = f"{vals[0]:.0f}%"
    end_label = f"{vals[-1]:.0f}%"
    end_label_x = min(xs[-1] + 8, width - pad_r - 5)

    svg = f'''<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="display:block;">
        {grid_svg}
        <line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" y2="{height - pad_b}" stroke="#5e4575" stroke-width="0.5"/>
        <text x="{pad_l}" y="{height - 6}" font-size="11" fill="#a98fc4">{years[0]}</text>
        <text x="{width - pad_r - 30}" y="{height - 6}" font-size="11" fill="#a98fc4">{years[-1]}</text>
        {highlight_svg}
        <path d="{path}" fill="none" stroke="#ff8fab" stroke-width="2.5"/>
        <circle cx="{xs[0]:.1f}" cy="{ys[0]:.1f}" r="4" fill="#ff8fab"/>
        <text x="{xs[0] + 8:.1f}" y="{ys[0] + -10:.1f}" font-size="14" fill="#ff8fab" font-weight="600">{start_label}</text>
        <circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="5" fill="#ff8fab"/>
        <text x="{end_label_x:.1f}" y="{ys[-1] + 22:.1f}" font-size="14" fill="#ff8fab" font-weight="600">{end_label}</text>
    </svg>'''

    components.html(
        f'<div style="background:transparent;">{svg}</div>',
        height=height + 10,
    )

def render_intro():
    """
    renders 4 intro slides, shows 'part x of 4' caption, and a skip button
    """
    stage = st.session_state.intro_stage
    total_stages = 4

    top_l, top_r = st.columns([0.7, 0.3])
    with top_l:
        st.caption(f"✦ part {stage + 1} of {total_stages} ✦")
    with top_r:
        # skip button
        if st.button("skip to the fun interactive charts →", key="skip_intro", use_container_width=True):
            st.session_state.intro_done = True
            st.rerun()

    dots = "".join(
        "✦" if i == stage else "✧" for i in range(total_stages)
    )
    st.markdown(f"<div style='font-size:4em; letter-spacing:4px;'>{dots}</div>", unsafe_allow_html=True)

    
    with st.container(border=True):
        if stage == 0:
            st.subheader(f"kpop went from {stats['start_overall']}% english to {stats['end_overall']}% english since {stats['first_year']}")
            st.markdown(
                f"if you've been on kpop spaces online you've probably seen the discourse: "
                f"*\"why is there so much english now??\"*, *\"i miss when kpop was actually in korean\"*. "
                f"so i pulled up {len(df):,} songs from {df['artist'].nunique()} artists released between "
                f"{stats['first_year']} and {stats['last_year']} and actually measured it word by word."
            )
            render_sparkline(stats["overall_series"])
            st.caption("avg % english words per song, industry wide, by year. yes i counted every word.")

        elif stage == 1:
            # three separate shifts, not one vague trend
            st.subheader("it's not just 'more english', it's three things happening at once")
            st.markdown(
                f"it's easy to picture this as one vague trend. after looking at the data, it's clear that this is actually three separate shifts: "
                f"fully korean releases are disappearing "
                f"(**{stats['kor_only_start']}%** of releases in {stats['first_year']} → "
                f"**{stats['kor_only_recent']}%** in {stats['last_year']}), "
                f"fully english releases went from basically nonexistent to common "
                f"(**{stats['eng_only_start']}%** → **{stats['eng_only_recent']}%**), "
                f"and even the bilingual songs that stayed bilingual are leaning more english "
                f"(**{stats['mixed_start']}%** → **{stats['mixed_recent']}%** english content "
                f"within the mixed-language ones)."
            )
            mcol1, mcol2, mcol3 = st.columns(3)
            st.markdown("""
                <style>
                [data-testid="stMetricValue"] {
                    font-size: 5rem;
                }
                [data-testid="stMetricLabel"] p {
                    font-size: 1.2rem;
                        font-style: italic;
                }
                [data-testid="stMetricDelta"] {
                    font-size: 1.2rem;
                }
                </style>
            """, unsafe_allow_html=True)

            mcol1.metric("korean-only releases", f"{stats['kor_only_recent']}%", f"{stats['kor_only_change']:+.1f} pts")
            mcol2.metric("english-only releases", f"{stats['eng_only_recent']}%", f"{stats['eng_only_change']:+.1f} pts")
            mcol3.metric("english within mixed songs", f"{stats['mixed_recent']}%", f"{stats['mixed_change']:+.0f} pts")

        elif stage == 2:
            st.subheader(f"everything changed in {stats['inflection_year']}")
            st.markdown(
                f"through most of the 2010s the english % was mostly stable, then it picked a "
                f"direction around **{stats['inflection_year']}** and hasn't stopped growing since. "
                f"but the *median* artist barely moved only "
                f"**{stats['median_change']} points** across their whole career (which is basically nothing). "
                f"the shift is coming from a small group of artists who moved a *lot*."
            )
            render_sparkline(stats["overall_series"], highlight_year=stats["inflection_year"])
            st.caption(f"same chart with a line marking {stats['inflection_year']}")

        elif stage == 3:
            if stats["has_korean_mover"]:
                st.subheader("it's not the whole industry, it's only some eras + groups")
                _second_clause = (
                    f"meanwhile **{stats['bottom_mover_name']}** went "
                    f"**{stats['bottom_mover_change']} points** the other way. "
                )
                _closing = "so yes the trend is real but it depends on the group/era/agency. "
            else:
                st.subheader("everyone's headed the same way")
                _second_clause = (
                    f"not a single artist with a statistically solid change is moving towards korean. even **{stats['bottom_mover_name']}**, the group with the smallest real change in either direction, "
                    f"is still moving toward english, just by **{stats['bottom_mover_change']} points**. "
                )
                _closing = (
                    "so, the trend isn't an overall change towards english amongst groups going in opposite "
                    "directions, it's the same change happening through the whole industry, "
                    "just at different speeds. "
                )

            st.markdown(
                f"among artists with a real, consistent trend (not just noise), "
                f"**{stats['top_mover_name']}** shifted **+{stats['top_mover_change']} points** "
                f"toward english over their career. {_second_clause}"
                f"{_closing}"
            )
            if stats["used_solid_fallback"]:
                st.caption(
                    "heads up: no artist currently has a clear, statistically consistent trend, "
                    "so the two names above are just the biggest raw movers in the full dataset "
                    "including artists whose year-to-year english % is noisy rather than a real, "
                    "sustained pattern. take these two with a bigger grain of salt than usual."
                )
            movers_fig = px.bar(
                stats["movers_df"], x="weighted_slope", y="artist", orientation="h",
                color="direction",
                color_discrete_map={"toward english": "#8ecae6", "toward korean": "#ff8fab"},
                labels={"weighted_slope": "weighted shift score", "direction": ""},
                text=stats["movers_df"]["change_pct_pts"].apply(lambda v: f"{v:+.0f} pp"),
            )
            movers_fig.update_layout(**mobile_layout(
                height=500,
                margin=dict(l=4, r=56, t=48, b=48),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
                font=dict(size=10),
            ))
            movers_fig.update_yaxes(
                automargin=True,
                categoryorder="array",
                categoryarray=stats["movers_df"]["artist"].tolist(),
                tickfont=dict(size=9),
            )
            movers_fig.update_xaxes(tickfont=dict(size=10))
            movers_fig.update_traces(
                textposition="outside",
                textfont=dict(size=9),
                hovertemplate="<b>%{y}</b><br>%{text} trend-line change<br>weighted score: %{x:.2f}<extra></extra>",
            )
            movers_fig.add_vline(x=0, line_width=1, line_color="#555")
            st.plotly_chart(movers_fig, use_container_width=True, key="intro_movers_chart")
            st.caption("ranked by a weighted shift score (trend strength × how much data backs it up).")
    # nav buttons
    nav_l, nav_r = st.columns(2)
    with nav_l:
        if stage > 0:
            if st.button("← back", key="intro_back"):
                st.session_state.intro_stage -= 1
                st.rerun()
    with nav_r:
        if stage < total_stages - 1:
            if st.button("next →", key="intro_next"):
                st.session_state.intro_stage += 1
                st.rerun()
        else:
            if st.button("tell me more →", key="intro_finish", type="secondary"):
                st.session_state.intro_done = True
                st.rerun()


if not st.session_state.intro_done:
    render_intro()
    # dont render main dashboard
    st.stop()


# ── mobile section nav ───────────────────────────────────────────────────
st.markdown("""
<style>
html, body, section.main { scroll-behavior: smooth; }
[id^="nav-"] { scroll-margin-top: 60px; }

/* ── mobile chart / layout fixes ── */
@media (max-width: 768px) {
    /* stack two-column chart rows */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: 100% !important;
        width: 100% !important;
        flex: 1 1 100% !important;
    }
    /* shrink giant metric values so they fit */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
    }
    /* filter bar: stack filter columns */
    [data-testid="stExpander"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: 48% !important;
        flex: 1 1 48% !important;
    }
    /* Plotly charts: make sure they don't overflow */
    .js-plotly-plot, .plotly {
        max-width: 100% !important;
    }
    /* reduce radio button label font on mobile */
    .stRadio label { font-size: 0.82rem !important; }
    /* extra clearance so anchor jumps land below the fixed top nav */
    [id^="nav-"] { scroll-margin-top: 96px; }
}

.kpop-mnav {
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0;
    background: rgba(9, 4, 20, 0.96);
    border-bottom: 1px solid #3a2456;
    z-index: 999999;
    padding: 6px 0;
    padding-top: max(6px, env(safe-area-inset-top));
    justify-content: space-around;
    align-items: center;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
}
@media (max-width: 768px) {
    .kpop-mnav { display: flex !important; }
    .block-container { padding-top: 88px !important; }
}
.kpop-mnav a {
    display: flex; flex-direction: column; align-items: center;
    color: #6a4d8c;
    text-decoration: none !important;
    font-size: 9.5px; font-weight: 500; letter-spacing: 0.2px;
    gap: 2px; padding: 5px 10px; border-radius: 12px;
    transition: color 0.15s, background 0.15s;
    -webkit-tap-highlight-color: transparent;
    user-select: none; min-width: 50px; text-align: center;
}
.kpop-mnav a:active,
.kpop-mnav a:focus { color: #c8a8ff; background: rgba(200,168,255,0.12); outline: none; }
.kpop-mnav .ni { font-size: 19px; line-height: 1.15; }
</style>
<nav class="kpop-mnav" role="navigation" aria-label="section navigation">
    <a href="#nav-top"><span class="ni"></span><span>home</span></a>
    <a href="#nav-lang-comp"><span class="ni"></span><span>composition</span></a>
    <a href="#nav-trends"><span class="ni"></span><span>trends</span></a>
    <a href="#nav-artist"><span class="ni"></span><span>artist</span></a>
    <a href="#nav-change"><span class="ni"></span><span>change</span></a>
    <a href="#nav-heatmap"><span class="ni"></span><span>heatmap</span></a>
    <a href="#nav-info"><span class="ni"></span><span>info</span></a>
</nav>
""", unsafe_allow_html=True)


# =========================
# MAIN DASHBOARD
# =========================

# ── header ───────────────────────────────────────────────────────────────
st.markdown('<a id="nav-top"></a>', unsafe_allow_html=True)
st.title("✦ kpop lyrics explorer ✦")

st.markdown("""
if you've been in kpop spaces lately, or have listened to kpop lately, you know that a lot of kpop songs seem to have more and more english. you've heard people talk about how they miss when it was *\"Kpop not kPOP\"* if you've been here for a while, you've experienced the change firsthand.
but is it actually a trend, or are we just hearing the english-heavy songs more often? is this an industry-wide change, or does it depend on agency? how do specific groups compare? groups vs. soloists? 3rd gen vs 4th gen?
 
i scraped 13,000+ songs from [color coded lyrics](https://colorcodedlyrics.com), tagged every word by language, and built this so you can find out for yourself.
            """)


st.divider()

# ── filter bar ──────────────────────────────────────────────────────────
with st.expander("filters (apply to everything below) ✧"):

    f1, f2, f3, f4, f5 = st.columns([2, 2, 2, 2, 2])
    with f1:
        _min_yr, _max_yr = int(df["year"].min()), int(df["year"].max())
        year_range = st.slider("time period", _min_yr, _max_yr, (_min_yr, _max_yr))
        if _max_yr == pd.Timestamp.now().year:
            st.caption(f"{_max_yr} is still in progress. Only a few months of releases so far, so take that year with a grain of salt.")
    with f2:
        selected_agencies = st.multiselect(
            "label", sorted(df["agency"].unique()), default=[],
            help="sm, yg, jyp, hybe, etc"
        )
    with f3:
        _gen_options = sorted(
            (g for g in df["generation"].unique() if g != "Unknown"),
            key=lambda g: int(re.match(r"\d+", g).group())
        )
        selected_gens = st.multiselect(
            "generation",
            _gen_options,
            default=[],
        )
    with f4:
        selected_artists = st.multiselect(
            "narrow it down to these artists", sorted(df["artist"].unique()), default=[],
        )

    toggle_l, toggle_r = st.columns(2)
    with toggle_l:
        big4_toggle = st.toggle("only the big 4 (SM, JYP, YG, HYBE)?", value=False)
    pct_col = "pct_english_without_voc"

# Apply filters
filtered = df[df["year"].between(str(year_range[0]), str(year_range[1]))]
if selected_artists:
    filtered = filtered[filtered["artist"].isin(selected_artists)]
if selected_agencies:
    filtered = filtered[filtered["agency"].isin(selected_agencies)]
if selected_gens:
    filtered = filtered[filtered["generation"].isin(selected_gens)]
if big4_toggle:
    filtered = filtered[filtered["agency"].apply(_is_big4)]

if filtered.empty:
    st.warning("no songs match this filter combination. try loosening a filter above.")
    st.stop()

total_songs = len(df)
total_artists = df["artist"].nunique()
filtered_songs = len(filtered)
filtered_artists = filtered["artist"].nunique()
eng_only_count = (filtered["language_release"] == "English only").sum()
eng_only_total = (df["language_release"] == "English only").sum()

active_filters = bool(selected_artists or selected_agencies or selected_gens
                      or big4_toggle or year_range != (_min_yr, _max_yr))


mcol1, mcol2, mcol3 = st.columns(3)
if active_filters:
    mcol1.metric("songs", f"{filtered_songs:,}", f"↓ from {total_songs:,} total")
    mcol2.metric("artists", f"{filtered_artists:,}", f"↓ from {total_artists:,} total")
else:
    mcol1.metric("songs analyzed", f"{filtered_songs:,}")
    mcol2.metric("artists", f"{filtered_artists:,}")
mcol3.metric("avg english share", f"{round(filtered[pct_col].mean()*100)}%")

st.caption("[how this data was collected + the fine print](#methodology), full breakdown at the bottom")

st.divider()

# ── two charts side-by-side ──────────────────────────────────────────────
chart_l, chart_r = st.columns(2)

with chart_l:
    # ------ composition -----
    with st.container(border=True):
        st.markdown('<a id="nav-lang-comp"></a>', unsafe_allow_html=True)
        st.subheader("what languages groups are releasing music in")

        # percentages for each language type for intro
        _lc_all = df.groupby(["year", "language_release"]).size().unstack(fill_value=0)
        for _c in LANG_LABELS.values():
            if _c not in _lc_all.columns: _lc_all[_c] = 0
        _lt = _lc_all[list(LANG_LABELS.values())].sum(axis=1)
        _lp = _lc_all.div(_lt, axis=0) * 100
        _first_yr  = _lp.index[0]
        _last_yr   = _lp.index[-1]
        _kor_2010 = _lp.loc[_first_yr, "Korean only"]
        _kor_now  = _lp.loc[_last_yr, "Korean only"]
        _eng_2010 = _lp.loc[_first_yr, "English only"]
        _eng_now  = _lp.loc[_last_yr, "English only"]
        _mix_2010_val = (df[df["language_release"]=="Korean & English mix"].groupby("year")[pct_col].mean()*100)
        _mix_start = round(_mix_2010_val.iloc[0])
        _mix_end   = round(_mix_2010_val.iloc[-1])
        st.markdown(
            f"three things are going on at once. **korean-only releases** have basically "
            f"disappeared. they were **{_kor_2010:.0f}%** of releases in {_first_yr} and now it's just "
            f"**{_kor_now:.0f}%** in {_last_yr}. **english-only releases** went the other way, "
            f"**{_eng_2010:.0f}%** → **{_eng_now:.0f}%**. and even the bilingual songs shifted: "
            f"mixed-language songs averaged **{_mix_start}% english** words early on and now it's "
            f"**{_mix_end}%**. and all of this happened at (about) the same time."
        )
        if int(_last_yr) == pd.Timestamp.now().year:
            st.caption(f"({_last_yr} only has a few months of releases so far, so its numbers will keep moving.)")

        # get dataframe for year and language type specifically to plot
        lang_counts = filtered.groupby(["year", "language_release"]).size().unstack(fill_value=0).reset_index()
        for col in LANG_LABELS.values():
            if col not in lang_counts.columns:
                lang_counts[col] = 0
        lang_counts["total"] = lang_counts[list(LANG_LABELS.values())].sum(axis=1)
        for col in LANG_LABELS.values():
            lang_counts[f"{col}_pct"] = (lang_counts[col] / lang_counts["total"] * 100).round(1)

        compare_by_lang = st.radio(
            "view by:",
            ["percentages", "number of songs"],
            horizontal=True,
            key="lang_radio"
        )

        # option 1: compare by percentages 
        if compare_by_lang == "percentages":
            pct_cols = [f"{c}_pct" for c in LANG_LABELS.values()]
            pct_melted = lang_counts[["year"] + pct_cols].melt(
                id_vars="year", var_name="type", value_name="pct"
            )
            pct_melted["type"] = pct_melted["type"].str.replace("_pct", "")
            pct_melted["label"] = pct_melted["pct"].apply(lambda x: f"{x:.0f}%" if x > 3 else "")
            fig2 = px.bar(pct_melted, x="year", y="pct", color="type",
                            labels={"pct": "% of releases"},
                            color_discrete_map=LANG_COLORS, text="label", height=420)
            fig2.update_traces(textposition="inside", textfont_size=10)
            fig2.update_layout(
                yaxis=dict(range=[0, 100]),
                legend_title="",
                **mobile_layout(height=420),
            )
            st.plotly_chart(fig2, use_container_width=True, key="lang_pct")

        # option 2: compare by song number
        else:
            fig1 = px.bar(lang_counts, x="year", y=list(LANG_LABELS.values()),
                            labels={"value": "songs", "variable": "type"},
                            color_discrete_map=LANG_COLORS, height=420)
            fig1.update_layout(
                legend_title="",
                **mobile_layout(height=420),
            )
            st.plotly_chart(fig1, use_container_width=True, key="lang_count")

with chart_r:
    with st.container(border=True):
        st.markdown('<a id="nav-trends"></a>', unsafe_allow_html=True)
        st.subheader("how much english is actually in these lyrics")
 
        compare_by = st.radio(
            "break it down by:",
            ["overall trend", "agency", "generation", "group type"],
            horizontal=True,
            key="trend_radio"
        )
 
        trend_fig = None
 
        if compare_by == "overall trend":
            
            yearly = filtered.groupby("year").agg(
                pct=(pct_col, "mean"), n=("url", "count")
            ).reset_index().rename(columns={"pct": pct_col})
            _yr = yearly.set_index("year")
            _start_val = round(_yr[pct_col].iloc[0] * 100)
            _end_val   = round(_yr[pct_col].iloc[-1] * 100)
            _start_yr  = _yr.index[0]
            _end_yr    = _yr.index[-1]
            _pct_s = _yr[pct_col] * 100
            _climb_yr = None
            for _i in range(len(_pct_s) - 2):
                if _pct_s.iloc[_i+1] > _pct_s.iloc[_i] and _pct_s.iloc[_i+2] > _pct_s.iloc[_i+1]:
                    _climb_yr = _pct_s.index[_i]
                    break
            # pre-climb range for the "bouncing" description
            if _climb_yr and _climb_yr in _pct_s.index:
                _pre = _pct_s[_pct_s.index < _climb_yr]
            else:
                _pre = _pct_s
            if len(_pre) >= 2:
                _pre_min, _pre_max = round(_pre.min()), round(_pre.max())
                _range_str = f"between **{_pre_min}%** and **{_pre_max}%**"
            else:
                _range_str = f"around **{_start_val}%**"
            # first year where english share crosses 50%
            _majority_yr = next((yr for yr, v in _pct_s.items() if v > 50), None)
            st.markdown(
                f"in the early part of this graph ({_start_yr}–{_climb_yr or _end_yr}) the "
                f"average is {_range_str} english. "
                f"then something changes. since {_climb_yr or _start_yr} the average has climbed, "
                f"from around **{round(_pct_s.loc[_climb_yr] if _climb_yr and _climb_yr in _pct_s.index else _start_val)}%** "
                f"to **{_end_val}%** by {_end_yr}."
                + (f" **{_majority_yr}** is the first year the average "
                   f"song had *more* english words than korean ones." if _majority_yr else "")
            )
            
            trend_fig = px.line(yearly, x="year", y=pct_col, markers=True,
                                custom_data=["n"],
                                labels={pct_col: "% english words per song"})
            
            trend_fig.update_traces(
                line=dict(color="#c8a8ff"),
                marker=dict(color="#ff8fab"),
                hovertemplate="year: %{x}<br>% english: %{y:.1%}<br>n=%{customdata[0]:,} songs<extra></extra>"
            )
 
        elif compare_by == "agency":
            yearly_cmp = filtered.groupby(["year", "agency"]).agg(
                pct=(pct_col, "mean"), n=("url", "count")
            ).reset_index().rename(columns={"pct": pct_col})
            _agency_latest = (
                yearly_cmp.groupby("agency")
                .apply(lambda g: g.sort_values("year").iloc[-1][pct_col])
            ).sort_values(ascending=False)
            _agency_change = (
                filtered.groupby(["agency", "year"])[pct_col].mean()
                .unstack()
                .apply(slope_change, axis=1)
            )
            # "steepest" means biggest magnitude of change, not most positive
            # a big move toward korean is just as steep as a big move toward
            # english, so rank by absolute value, then report the real
            # (signed) number for that pick
            _agency_change_ranked = _agency_change.reindex(
                _agency_change.abs().sort_values(ascending=False).index
            )
            _most_eng_agency   = _agency_latest.index[0]
            _least_eng_agency  = _agency_latest.index[-1]
            _biggest_riser     = _agency_change_ranked.index[0]
            _biggest_riser_chg = round(_agency_change_ranked.iloc[0] * 100, 1)
            st.markdown(
                f"this varies a lot with agency. **{_most_eng_agency}** is currently "
                f"the most english-heavy label in this view, **{_least_eng_agency}** is the most "
                f"korean. the label with the steepest trend over time is **{_biggest_riser}**, "
                f"moving by roughly **{_biggest_riser_chg:+.0f} percentage points** based on its "
                f"year-over-year trend line."
            )
            trend_fig = px.line(yearly_cmp, x="year", y=pct_col, color="agency", markers=True,
                                custom_data=["n"],
                                labels={pct_col: "% english words per song"},
                                color_discrete_sequence=px.colors.qualitative.Pastel)
            trend_fig.update_traces(
                hovertemplate="%{fullData.name}<br>year: %{x}<br>% english: %{y:.1%}<br>n=%{customdata[0]:,} songs<extra></extra>"
            )
 
        elif compare_by == "generation":
            yearly_cmp = filtered.groupby(["year", "generation"]).agg(
                pct=(pct_col, "mean"), n=("url", "count")
            ).reset_index().rename(columns={"pct": pct_col})
            _gen_change = (
                filtered.groupby(["generation", "year"])[pct_col].mean()
                .unstack()
                .apply(slope_change, axis=1)
            )
            # rank by magnitude of change, not raw signed value see agency
            # branch above for why
            _gen_change_ranked = _gen_change.reindex(
                _gen_change.abs().sort_values(ascending=False).index
            )
            _gen_latest = (
                yearly_cmp.groupby("generation")
                .apply(lambda g: g.sort_values("year").iloc[-1][pct_col])
            ).sort_values(ascending=False)
            _most_eng_gen  = _gen_latest.index[0]
            _least_eng_gen = _gen_latest.index[-1]
            _biggest_gen   = _gen_change_ranked.index[0]
            _biggest_gen_chg = round(_gen_change_ranked.iloc[0] * 100, 1)
            st.markdown(
                f"generation matters: **{_most_eng_gen}** artists average the "
                f"most english right now, **{_least_eng_gen}** artists the least. **{_biggest_gen}** had "
                f"the steepest climb (**{_biggest_gen_chg:+.0f} pp**)."
            )
            trend_fig = px.line(yearly_cmp, x="year", y=pct_col, color="generation", markers=True,
                                custom_data=["n"],
                                labels={pct_col: "% english words per song"},
                                color_discrete_sequence=px.colors.qualitative.Pastel)
            trend_fig.update_traces(
                hovertemplate="%{fullData.name}<br>year: %{x}<br>% english: %{y:.1%}<br>n=%{customdata[0]:,} songs<extra></extra>"
            )
 
        elif compare_by == "group type":
            
            yearly_cmp = (
                filtered[filtered["group_type"] != "mixed group"]
                .groupby(["year", "group_type"]).agg(
                    pct=(pct_col, "mean"), n=("url", "count")
                ).reset_index().rename(columns={"pct": pct_col})
            )
            _type_latest = (
                yearly_cmp.groupby("group_type")
                .apply(lambda g: g.sort_values("year").iloc[-1][pct_col])
            ).sort_values(ascending=False)
            _type_change = (
                filtered[filtered["group_type"] != "mixed group"]
                .groupby(["group_type", "year"])[pct_col].mean()
                .unstack()
                .apply(slope_change, axis=1)
            )
            # rank by magnitude of change, not raw signed value see agency
            # branch above for why
            _type_change_ranked = _type_change.reindex(
                _type_change.abs().sort_values(ascending=False).index
            )
            def _phrase_type(label):
                """Make group_type labels read naturally in a sentence."""
                if "solo" in label.lower():
                    return f"{label} artists"
                return f"{label}s"  # "boy group" -> "boy groups", "girl group" -> "girl groups"
 
            _most_eng_type   = _phrase_type(_type_latest.index[0])
            _least_eng_type  = _phrase_type(_type_latest.index[-1])
            _biggest_type    = _phrase_type(_type_change_ranked.index[0])
            _biggest_type_chg = round(_type_change_ranked.iloc[0] * 100, 1)
 
            st.markdown(
                f"**{_most_eng_type}** use the most english on average right now, "
                f"**{_least_eng_type}** lean the most korean. biggest shift over time comes "
                f"from **{_biggest_type}** (**{_biggest_type_chg:+.0f} pp**). solo acts and groups seem to "
                f"follow quite different trends, at least where language composition is concerned."
)
            trend_fig = px.line(yearly_cmp, x="year", y=pct_col, color="group_type", markers=True,
                                custom_data=["n"],
                                labels={pct_col: "% english words per song", "group_type": "group type"},
                                color_discrete_sequence=px.colors.qualitative.Pastel)
            trend_fig.update_traces(
                hovertemplate="%{fullData.name}<br>year: %{x}<br>% english: %{y:.1%}<br>n=%{customdata[0]:,} songs<extra></extra>"
            )
 
        if trend_fig:
            if compare_by == "overall trend":
                data = yearly
            elif compare_by == "agency":
                data = filtered.groupby(["year", "agency"]).agg(
                pct=(pct_col, "mean"), n=("url", "count")
                ).reset_index().rename(columns={"pct": pct_col})
            elif compare_by == "generation":
                data = filtered.groupby(["year", "generation"]).agg(
                pct=(pct_col, "mean"), n=("url", "count")
                ).reset_index().rename(columns={"pct": pct_col})
            elif compare_by == "group type":
                data = (
                        filtered[filtered["group_type"] != "mixed group"]
                        .groupby(["year", "group_type"]).agg(
                            pct=(pct_col, "mean"), n=("url", "count")
                        ).reset_index().rename(columns={"pct": pct_col})
                    )
            trend_fig.update_yaxes(tickformat=".0%", tickfont=dict(size=10), range=[0, 1])
            trend_fig.update_xaxes(tickfont=dict(size=10), tickangle=-45)
            trend_fig.update_layout(**mobile_layout(height=380))
            st.plotly_chart(trend_fig, use_container_width=True, key="trend_chart_main")
 
            with st.expander("see the full spread, not just the averages"):
                st.caption("each box = the spread of individual songs that year. middle line is the median, box is the middle 50%, dots are outliers. this is best viewed fullscreen on a computer")
                if compare_by == "overall trend":
                    fig_box = px.box(filtered, x="year", y=pct_col,
                                    labels={pct_col: "% english words per song"},
                                    color_discrete_sequence=["#c8a8ff"])
                elif compare_by == "agency":
                    fig_box = px.box(filtered, x="year", y=pct_col, color="agency",
                                    labels={pct_col: "% english words per song"},
                                    color_discrete_sequence=px.colors.qualitative.Pastel)
                elif compare_by == "generation":
                    fig_box = px.box(filtered, x="year", y=pct_col, color="generation",
                                    labels={pct_col: "% english words per song"},
                                    color_discrete_sequence=px.colors.qualitative.Pastel)
                elif compare_by == "group type":
                    fig_box = px.box(filtered, x="year", y=pct_col, color="group_type",
                                    labels={pct_col: "% english words per song", "group_type": "group type"},
                                    color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_box.update_yaxes(tickformat=".0%", tickfont=dict(size=10), range=[0, 1])
                fig_box.update_xaxes(tickfont=dict(size=10), tickangle=-45)
                fig_box.update_layout(**mobile_layout(height=360))
                st.plotly_chart(fig_box, use_container_width=True, key="box_chart_main")
 
 


# ── artist lookup + movers ────────────────────────────────────────────────
st.divider()
lookup_col, movers_col = st.columns(2)

with lookup_col:
    # ------- fav group ----------
    with st.container(border=True):
        st.markdown('<a id="nav-artist"></a>', unsafe_allow_html=True)
        st.subheader("look up ur fav group ✧")
        artist_counts_lookup = filtered["artist"].value_counts()
        artists_by_count = artist_counts_lookup.index.tolist()

        if not artists_by_count:
            st.info("no artists match the current filters. loosen a filter above to look someone up.")
            st.stop()

        artist_choice = st.selectbox(
            "pick an artist", artists_by_count,
            index=0, label_visibility="collapsed",
        )

        # NOTE: don't filter on filtered["error"].isna() here. that column
        # conflates two different things: genuine "can't compute english %"
        # cases (japanese/chinese-only releases, where pct is always null)
        # with a benign "no hangul content found" tag on fully-english songs
        # (where pct is always valid, just 100%). filtering on error.isna()
        # was silently dropping every english-only song tagged that way from
        # the trend line, peak-year detection, and the english-only % stat.
        # downstream code already drops rows with a null pct where needed
        # (groupby.mean() skips NaN, valid_songs does an explicit dropna,
        # and plotly skips NaN y-values automatically), so no separate error
        # filter is needed.
        artist_songs = filtered[filtered["artist"] == artist_choice].sort_values("date")

        _a_yearly = artist_songs.groupby("year")[pct_col].mean() * 100
        _a_fit = fit_trend(_a_yearly)
        if _a_fit is not None:
            _a_valid     = _a_yearly.dropna()
            _a_start_yr  = _a_valid.index[0]
            _a_end_yr    = _a_valid.index[-1]
            _a_start_raw = round(_a_valid.iloc[0])
            _a_end_raw   = round(_a_valid.iloc[-1])
            _a_peak_yr   = _a_valid.idxmax()
            _a_peak_val  = round(_a_valid.max())
            _eng_only_n  = (artist_songs["language_release"] == "English only").sum()
            _eng_only_pct = round(_eng_only_n / max(len(artist_songs), 1) * 100)

            # fitted change across all active years. keeps a single
            # unusually high/low year from making an otherwise-flat artist
            # look like a big mover
            _a_change = round(_a_fit["change"])
            _mag      = abs(_a_change)

            # does the raw first->last value move the same direction as the
            # fitted trend line? if not, this artist isn't a simple "steady
            # drift toward X" story. it's a rise-then-fall (or reverse)
            # shape, and picking one direction to narrate would contradict
            # the very numbers shown next to it (e.g. "down to 36%" when 36
            # is bigger than the starting value)
            _raw_diff = _a_end_raw - _a_start_raw
            _shapes_agree = (_raw_diff == 0) or (_a_change == 0) or ((_raw_diff > 0) == (_a_change > 0))

            # a layman reading "-8pp" right after "46% to 20%" (a -26pp raw
            # gap) will reasonably wonder why those don't match. flag it in
            # plain language whenever the trend-line number is noticeably
            # smaller than the raw endpoint gap, rather than assuming the
            # reader knows what a "fitted trend" means
            _endpoint_note = ""
            if _shapes_agree and abs(_raw_diff) >= 10 and _mag < abs(_raw_diff) * 0.7:
                _endpoint_note = (
                    f" \n\n(just comparing {_a_start_yr} to {_a_end_yr} directly would suggest a bigger "
                    f"change (**{_raw_diff:+d} pp**). the **{_a_change:+d} pp** figure instead comes from "
                    f"fitting a line through *every* year in between, so one unusually high or low year "
                    f"at the very start or end doesn't get to speak for the artist's whole career.)"
                )

            if _mag < 5:
                _trend_desc = f"their english share has been consistent: **{_a_start_raw}%** in {_a_start_yr}, **{_a_end_raw}%** in {_a_end_yr}."
            elif _shapes_agree:
                _direction_word = "toward english" if _a_change > 0 else "toward korean"
                _sign = "+" if _a_change > 0 else ""
                _trend_desc = (
                    f"they've drifted noticeably {_direction_word}: **{_a_start_raw}%** english in {_a_start_yr} "
                    f"to **{_a_end_raw}%** english in {_a_end_yr}, a trend of roughly **{_sign}{_a_change} pp** over that span."
                    f"{_endpoint_note}"
                )
            else:
                # peak/valley shape: raw endpoints go one way, the fitted
                # trend line goes the other. narrate both instead of
                # picking one and contradicting it with the other
                _trend_line_word = "toward english" if _a_change > 0 else "toward korean"
                _trend_desc = (
                    f"it's not a straight line: their raw numbers went from **{_a_start_raw}%** in {_a_start_yr} "
                    f"to **{_a_end_raw}%** in {_a_end_yr} (with a peak of **{_a_peak_val}%** in {_a_peak_yr} "
                    f"along the way), but the overall trend line across the whole span actually points "
                    f"**{_trend_line_word}** (**{_a_change:+d} pp**). the mid-career swing outweighs where "
                    f"they started and ended."
                )

            if not _a_fit["is_solid"]:
                _trend_desc += (
                    " \n\n(**this trend isn't statistically solid**. their year-to-year english % "
                    "bounces around too much to call this a confident, consistent pattern rather than noise.)\n\n"
                )

            _peak_note = (
                f" the year they had the most english in their songs was **{_a_peak_yr}** ({_a_peak_val}%)."
                if _a_peak_yr != _a_end_yr and _shapes_agree else ""
            )
            _eng_only_note = f" **{_eng_only_pct}% of their songs** are fully english releases." if _eng_only_pct > 5 else ""
            st.markdown(f"{_trend_desc}{_peak_note}{_eng_only_note}")

        fig_scatter = px.scatter(
            artist_songs, x="date", y=pct_col, hover_data=["title"],
            title=f"{artist_choice} english share per song",
            labels={pct_col: "% english (per song)", "date": ""},
        )
        fig_scatter.update_yaxes(
            tickformat=".0%", tickfont=dict(size=10), range=[0, 1],
        )
        _a_dates = artist_songs["date"]
        if _a_dates.nunique() <= 1:
            _pad = pd.Timedelta(days=30)
            _xrange = [_a_dates.min() - _pad, _a_dates.max() + _pad]
        else:
            _pad = (_a_dates.max() - _a_dates.min()) * 0.05
            _xrange = [_a_dates.min() - _pad, _a_dates.max() + _pad]
        fig_scatter.update_xaxes(
            tickfont=dict(size=10), tickangle=-45,
            range=_xrange, tickformat="%b %Y",
        )
        fig_scatter.update_traces(
            marker=dict(size=8, color="#ff8fab"),
            hovertemplate="<b>%{customdata[0]}</b><br>%{x}<br>%{y:.0%} english<extra></extra>"
        )
        fig_scatter.update_layout(**mobile_layout(height=340))
        st.plotly_chart(fig_scatter, use_container_width=True)

        mc1, mc2 = st.columns(2)
        valid_songs = artist_songs.dropna(subset=[pct_col])
        if valid_songs.empty:
            st.info("no english/korean lyric data for this artist yet")
        else:
            most_eng = valid_songs.loc[valid_songs[pct_col].idxmax()]
            most_kor = valid_songs.loc[valid_songs[pct_col].idxmin()]
            mc1.markdown(f"**🤍 song with the most english**\n\n{most_eng['title']} ({round(most_eng[pct_col]*100)}% english)")
            mc2.markdown(f"**🩷 song with the most korean**\n\n{most_kor['title']} ({round(most_kor[pct_col]*100)}% english)")

        artist_data = (
            filtered[filtered["artist"] == artist_choice]
            [["title", "date", "language_release", pct_col]]
            .copy()
            .sort_values(pct_col, ascending=False)
            .rename(columns={
                "title": "song",
                "date": "date",
                "language_release": "language",
                pct_col: "% english",
            })
        )
        artist_data["% english"] = artist_data["% english"].apply(
            lambda x: f"{round(x*100)}%" if pd.notna(x) else "—"
        )

        with st.expander(f"{artist_choice}: {len(artist_data)} songs"):
            search = st.text_input(
                "search songs",
                key=f"lookup_search_{artist_choice}",
                placeholder="type to filter..."
            )
            if search:
                artist_data = artist_data[
                    artist_data["song"].str.contains(search, case=False, na=False, regex=False)
                ]
            st.dataframe(artist_data, use_container_width=True, hide_index=True)

        with st.expander("compare with another artist"):
            artist_options_compare = [f"{a} ({c} songs)" for a, c in artist_counts_lookup.items()]
            compare_artists_labeled = st.multiselect(
                "pick artists to compare",
                artist_options_compare,
                key="compare_artists_multiselect"
            )
            compare_artists = [a.rsplit(" (", 1)[0] for a in compare_artists_labeled]

            if compare_artists:
                yearly_cmp = (
                    filtered[filtered["artist"].isin(compare_artists)]
                    .groupby(["year", "artist"]).agg(
                        pct=(pct_col, "mean"), n=("url", "count")
                    ).reset_index().rename(columns={"pct": pct_col})
                )
                fig_cmp = px.line(yearly_cmp, x="year", y=pct_col, color="artist", markers=True,
                                  custom_data=["n"],
                                  labels={pct_col: "% english words per song"},
                                  color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_cmp.update_traces(
                    hovertemplate="%{fullData.name}<br>year: %{x}<br>% english: %{y:.1%}<br>n=%{customdata[0]:,} songs<extra></extra>"
                )
                fig_cmp.update_yaxes(
                    tickformat=".0%", tickfont=dict(size=10), range=[0, 1],
                )
                fig_cmp.update_xaxes(tickfont=dict(size=10), tickangle=-45)
                fig_cmp.update_layout(**mobile_layout(height=360))
                st.plotly_chart(fig_cmp, use_container_width=True, key="trend_chart_compare")

                with st.expander("see the full spread, not just averages"):
                    st.caption("each box = the spread of songs that year. median line, middle 50% box, outlier dots.")
                    artist_filtered = filtered[filtered["artist"].isin(compare_artists)]
                    fig_box = px.box(artist_filtered, x="year", y=pct_col, color="artist",
                                    labels={pct_col: "% english words per song"},
                                    color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_box.update_yaxes(tickformat=".0%", tickfont=dict(size=10), range=[0, 1])
                    fig_box.update_xaxes(tickfont=dict(size=10), tickangle=-45)
                    fig_box.update_layout(**mobile_layout(height=340))
                    st.plotly_chart(fig_box, use_container_width=True, key="box_chart_artists")

                st.markdown("#### song lists")
                for artist in compare_artists:
                    adf = (
                        filtered[filtered["artist"] == artist]
                        [["title", "date", "language_release", pct_col]]
                        .copy()
                        .sort_values("date", ascending=False)
                        .rename(columns={
                            "title": "song", "date": "date",
                            "language_release": "language", pct_col: "% english",
                        })
                    )
                    adf["% english"] = adf["% english"].apply(
                        lambda x: f"{round(x*100)}%" if pd.notna(x) else "—"
                    )
                    with st.expander(f"{artist}: {len(adf)} songs"):
                        search2 = st.text_input("search songs", key=f"compare_search_{artist}",
                                                placeholder="type to filter...")
                        if search2:
                            adf = adf[adf["song"].str.contains(search2, case=False, na=False)]
                        st.dataframe(adf, use_container_width=True, hide_index=True)
            else:
                st.info("pick at least one artist to compare")


with movers_col:
    # ------ biggest change ----------
    with st.container(border=True):
        st.markdown('<a id="nav-change"></a>', unsafe_allow_html=True)
        st.subheader("who's changed the most ✧")
        trends_filtered = trends_df[trends_df["artist"].isin(filtered["artist"].unique())].copy()

        # require artists to still be releasing music: a few songs early on
        # followed by years of silence can look like a huge "shift" even though
        # it's really just one stale data point dragging the trend line
        # so only consider artists with more than 5 songs in the last few
        # years of data actually present in this view. uses a 3-calendar-year
        # window (not 2) since the most recent year is usually still in
        # progress and only has a few months of releases in it
        _recent_cutoff = str(int(filtered["year"].max()) - 2)
        _recent_song_counts = (
            filtered[filtered["year"] >= _recent_cutoff]
            .groupby("artist")["url"].count()
        )
        _active_artists = _recent_song_counts[_recent_song_counts > 5].index
        trends_filtered = trends_filtered[trends_filtered["artist"].isin(_active_artists)].copy()

        # "total_change" in the underlying data is just end-year-avg minus
        # start-year-avg and an artist's first or last *year* can be a
        # single song (or a single same-day album drop), so that endpoint
        # can swing 0%->100% on its own. use the fitted regression slope's
        # total movement across the whole career instead, since it's pulled
        # from every year in between, not just the two edges
        trends_filtered["change_pct_pts"] = (
            trends_filtered["slope_per_year"] * (trends_filtered["n_years_active"] - 1) * 100
        )
        trends_filtered["weighted_slope"] = (
            trends_filtered["slope_per_year"] * np.log1p(trends_filtered["n_songs"])
        )

        # only trust trends that are actually consistent year-to-year, not
        # just noisy bouncing that happens to average out to a slope.
        _is_solid_trend = (trends_filtered["p_value"] < SOLID_TREND_P_MAX) & (trends_filtered["r_squared"] > SOLID_TREND_R2_MIN)
        trends_solid = trends_filtered[_is_solid_trend].copy()

        if trends_filtered.empty:
            st.info(
                "no artists in this view have more than 5 songs in the last few "
                "years. try widening your filters."
            )
        elif trends_solid.empty:
            st.info(
                "no artist in this view has a clear, consistent pattern over time right now. "
                "everyone's year-to-year english % is bouncing around too much to call it a real trend. "
                "try widening your filters to see more artists."
            )
        else:
            _median_chg  = round(trends_solid["change_pct_pts"].median(), 1)
            _unchanged_n = (trends_solid["change_pct_pts"].abs() < 5).sum()

            top_movers = trends_solid[trends_solid["slope_per_year"] > 0].sort_values("weighted_slope", ascending=False).head(10)
            top_korean = trends_solid[trends_solid["slope_per_year"] < 0].sort_values("weighted_slope").head(10)

            _top1 = top_movers.iloc[0]
            _top_name, _top_chg = _top1["artist"], round(_top1["change_pct_pts"])

            if top_korean.empty:
                # the actual headline finding: among artists with a real,
                # consistent trend, none of them are trending
                # toward korean right now
                st.markdown(
                    f"among artists with a clear, consistent pattern over time "
                    f"(not just noisy bouncing), **not a single one in this view is trending toward korean** "
                    f"right now. every artist whose english % moves in a real, sustained direction is moving "
                    f"toward **more english**. **{_top_name}** leads this change (**+{_top_chg} pp** over their "
                    f"career, based on their trend line). artists who *have* drifted toward korean over time "
                    f"tend to be ones who've gone quiet or disbanded. once you only look at who's still "
                    f"actively releasing music, the korean-leaning trend disappears."
                )
            else:
                _bot1 = top_korean.iloc[0]
                _bot_name, _bot_chg = _bot1["artist"], round(_bot1["change_pct_pts"])
                st.markdown(
                    f"among artists with a clear, consistent pattern over time (not just noisy bouncing), "
                    f"the median shift is **{_median_chg} percentage points** over their whole career, and "
                    f"**{_unchanged_n} of them** changed by less than 5 points total. **{_top_name}** leads the "
                    f"english shift (**+{_top_chg} pp**, based on their trend line), **{_bot_name}** went furthest "
                    f"the other way (**{_bot_chg} pp**). disclaimer: a big number here doesn't always mean a "
                    f"deliberate choice."
                )

            combined = pd.concat([top_movers, top_korean]).copy()
            combined["direction"] = combined["weighted_slope"].apply(
                lambda s: "toward english" if s > 0 else "toward korean"
            )
            combined = combined.sort_values("weighted_slope", ascending=True)

            fig_combined = px.bar(
                combined, x="weighted_slope", y="artist", orientation="h",
                color="direction",
                color_discrete_map={"toward english": "#8ecae6", "toward korean": "#ff8fab"},
                labels={"weighted_slope": "weighted shift score", "direction": ""},
                text=combined["change_pct_pts"].apply(lambda v: f"{v:+.0f} pp"),
            )
            fig_combined.update_layout(
                height=640,
                margin=dict(l=4, r=56, t=48, b=48),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
                font=dict(size=10),
            )
            fig_combined.update_yaxes(
                automargin=True,
                categoryorder="array",
                categoryarray=combined["artist"].tolist(),
                tickfont=dict(size=9),
            )
            fig_combined.update_xaxes(tickfont=dict(size=10))
            fig_combined.update_traces(
                textposition="outside",
                textfont=dict(size=9),
                hovertemplate="<b>%{y}</b><br>%{text} trend-line change<br>weighted score: %{x:.2f}<extra></extra>"
            )
            fig_combined.add_vline(x=0, line_width=1, line_color="#555")
            st.plotly_chart(fig_combined, use_container_width=True)

            if top_korean.empty:
                st.caption(
                    "every bar here points the same way. no artist with a real trend is currently moving toward korean."
                )

            st.caption(
                "ranked by a weighted shift score (trend strength × how much data backs it up), "
                "so an artist with a few songs and a big swing doesn't outrank one with a lot of songs and a "
                "consistent change. pp labels show the change implied by each artist's trend line, not just "
                "their first vs. most recent song so a single viral cover or one-off english single won't throw "
                "off the whole picture. "
                "only includes artists with more than 5 songs in the last few years (so groups that have gone "
                "quiet don't show up as 'movers') AND a clear, consistent pattern over time (so normal year-to-"
                "year noise doesn't get mistaken for a real trend)."
            )

st.divider()

with st.container(border=True):
    st.markdown('<a id="nav-heatmap"></a>', unsafe_allow_html=True)
    st.subheader("english word share across every artist + year")
    st.markdown(
        "each **row** = an artist. each **column** = a year. "
        "color = that artist's avg share of english words. "
        "**light blue** is mostly korean, **pink/red** is mostly english "
        "blank cells = no data that year."
    )
    _above50 = (
        filtered.groupby(["artist", "year"])[pct_col].mean()
        .unstack()
        .apply(lambda r: (r > 0.5).any(), axis=1)
        .sum()
    )
    _total_artists_hm = filtered["artist"].nunique()
    _always_below20 = (
        filtered.groupby(["artist", "year"])[pct_col].mean()
        .unstack()
        .apply(lambda r: (r.dropna() < 0.2).all() if r.dropna().shape[0] > 0 else False, axis=1)
        .sum()
    )
    st.markdown(
        f"scroll through and you'll see the pink colors creep rightward over time, but unevenly. "
        f"about **{round(_above50/_total_artists_hm*100)}% of artists** "
        f"({_above50} of {_total_artists_hm}) had at least one year where more than half their "
        f"lyrics were english. meanwhile **{_always_below20} artists** stayed under 20% english "
        f"every single year they released music. "
    )

    sort_by = st.selectbox(
        "sort artists by:",
        ["most english overall", "alphabetical", "biggest change over time"],
        help="sorting only changes row order. the color scale stays the same."
    )

    pivot = (
        filtered.groupby(["artist", "year"])[pct_col]
        .mean()
        .unstack(fill_value=None)
    )

    if sort_by == "most english overall":
        order = filtered.groupby("artist")[pct_col].mean().sort_values(ascending=False).index
        pivot = pivot.reindex(order)
    elif sort_by == "biggest change over time":
        _hm_trends = trends_df[trends_df["artist"].isin(filtered["artist"].unique())].copy()

        # same "still active" + "solid trend" logic as the movers section,
        # so the two rankings agree but nobody gets dropped from the
        # heatmap itself, they just fall back to a simpler sort key
        _hm_recent_cutoff = str(int(filtered["year"].max()) - 2)
        _hm_recent_counts = (
            filtered[filtered["year"] >= _hm_recent_cutoff]
            .groupby("artist")["url"].count()
        )
        _hm_active = set(_hm_recent_counts[_hm_recent_counts > 5].index)

        _hm_trends["weighted_slope"] = (
            _hm_trends["slope_per_year"] * np.log1p(_hm_trends["n_songs"])
        )
        _hm_solid_mask = (
            (_hm_trends["p_value"] < SOLID_TREND_P_MAX)
            & (_hm_trends["r_squared"] > SOLID_TREND_R2_MIN)
            & (_hm_trends["artist"].isin(_hm_active))
        )

        # rank 0: qualifies for a "real" trend -> sort by magnitude of weighted slope
        # rank 1: everyone else -> sort by raw slope, so they still land somewhere
        #         sensible instead of being dropped
        _hm_trends["_rank_tier"] = (~_hm_solid_mask).astype(int)
        _hm_trends["_sort_key"] = np.where(
            _hm_solid_mask,
            _hm_trends["weighted_slope"].abs(),
            _hm_trends["slope_per_year"].abs(),
        )
        order = (
            _hm_trends.sort_values(["_rank_tier", "_sort_key"], ascending=[True, False])
            ["artist"]
        )
        pivot = pivot.reindex([a for a in order if a in pivot.index] +
                              [a for a in pivot.index if a not in set(order)])

    def _short_label(name):
        """Strip ' (한국어)' parenthetical to keep only the romanized name."""
        return re.sub(r"\s*\(.*\)$", "", name).strip()

    pivot_display = pivot.copy()
    pivot_display.index = [_short_label(a) for a in pivot_display.index]

    # build a mapping so hover still shows the full name
    label_to_full = {_short_label(a): a for a in pivot.index}

    height = max(600, len(pivot_display) * 20)
    fig_heat = px.imshow(
        pivot_display,
        color_continuous_scale=[[0, "#b5e8ff"], [1, "#f92257"]],
        zmin=0, zmax=1,
        labels={"color": "% english"},
        height=height
    )
    fig_heat.update_layout(
        yaxis={"tickfont": {"size": 10}, "automargin": True},
        xaxis={"tickfont": {"size": 11}, "side": "bottom"},
        margin=dict(l=4, r=16, t=24, b=32),
        coloraxis_colorbar=dict(
            thickness=12, len=0.7,
            tickfont=dict(size=10),
            title=dict(text="% english", font=dict(size=11)),
        ),
    )
    fig_heat.update_traces(
        hovertemplate="<b>%{y}</b><br>year: %{x}<br>% english: %{z:.0%}<extra></extra>"
    )
    st.plotly_chart(fig_heat, use_container_width=True, key="heatmap")
st.divider()
st.markdown('<a id="nav-info"></a>', unsafe_allow_html=True)
st.markdown('<a name="methodology"></a>', unsafe_allow_html=True)
with st.expander("how this works + where the data's from"):
    st.markdown("""
    **data source:** lyrics and release dates are from [color coded lyrics](https://colorcodedlyrics.com) (ccl), 
    a fan-run lyrics database covering kpop releases from 2010 to now.
                
    **future directions:** this project is ongoing. some ideas i may implement in the future include analyzing line distributions, common words in each language, quantifying repetition (e.g., it counts for more if 10 words of english are all different vs. the same word repeated multiple times.), exploring usage in chorus vs. verses, song length

    **how language is measured:** every word in the hangul column of a song's lyrics page is tagged as 
    korean or english based on its characters. words that are only hangul = korean, words that are only 
    latin characters = english. mixed tokens and vocalizations (uh, oh, yeah...) are tracked separately 
    and excluded from every % shown in this app and are not counted as either language.

    **stuff to keep in mind, no data is perfect:**
    - the dates for each song are taken from the date the CCL lyrics were published, so some may be off. this will be noticeable when analyzing individual groups. however, these inconsistencies are small enough and the dataset so large, they do not affect the overall industry trend data.
    - for ARMYs: i unfortunately do not have data from ARIRANG because it is not on CCL.
    - scraping isn't flawless, so some songs might be missing, duplicated, or messed up if the source structure was neglected in the site structure detection of the scraping. ccl is pretty inconsistent, but i tried.
    - ccl's coverage isn't across every artist/year, older releases are underrepresented.
    - loanwords written in korean script (e.g. 버스 for "bus") get counted as korean.
    - english % isn't calculated for japanese-only or chinese-only releases.
    - agency + generation tags were pulled from wikipedia and fan wikis so they might be outdated/off.

    **disclaimer:** this is just a passion project for fun/learning purposes, not affiliated with 
    color coded lyrics, any artist, or any label. all lyrics belong to their rightful owners obviously.
    """)