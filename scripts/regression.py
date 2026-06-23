import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("csv/kpop_lyrics_clean.csv")
df["year"] = df["date"].str[:4].astype(int)

def compute_artist_trends(df, pct_col="pct_english_without_voc", min_years=3, min_songs=10):
    """
    For each artist, fit a linear trend of pct_col over year.
    Returns a dataframe with slope (change per year), r-squared, and sample size.
    """
    results = []

    for artist, group in df.groupby("artist"):
        group = group.dropna(subset=[pct_col])
        yearly = group.groupby("year")[pct_col].mean().reset_index()

        if len(yearly) < min_years or len(group) < min_songs:
            continue

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            yearly["year"], yearly[pct_col]
        )

        results.append({
            "artist": artist,
            "slope_per_year": slope,        # change in % English per year
            "r_squared": r_value ** 2,
            "p_value": p_value,
            "n_songs": len(group),
            "n_years_active": len(yearly),
            "first_year": yearly["year"].min(),
            "last_year": yearly["year"].max(),
            "start_pct": yearly[pct_col].iloc[0],
            "end_pct": yearly[pct_col].iloc[-1],
            "total_change": yearly[pct_col].iloc[-1] - yearly[pct_col].iloc[0],
        })

    return pd.DataFrame(results)

trends_with_voc = compute_artist_trends(df, "pct_english_with_voc")
trends_without_voc = compute_artist_trends(df, "pct_english_without_voc")

trends_with_voc.to_csv("csv/artist_trends_with_voc.csv", index=False)
trends_without_voc.to_csv("csv/artist_trends_without_voc.csv", index=False)

print(trends_without_voc.sort_values("slope_per_year", ascending=False).head(10))
print(trends_without_voc.sort_values("slope_per_year").head(10))