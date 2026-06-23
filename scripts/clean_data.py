import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==== CLEAN THE DATA =====
lyrics = pd.read_csv("csv/kpop_lyrics_combined.csv")

print(f"Before dropping errors: {len(lyrics)} rows")
KEEP_ERRORS = {"japanese release", "chinese release"}

lyrics_clean = lyrics[
    lyrics["error"].isna() |
    lyrics["error"].isin(KEEP_ERRORS) |
    (lyrics["english_only"] == True)
].copy()
print(f"After dropping errors: {len(lyrics_clean)} rows")

lyrics_clean = lyrics_clean.drop_duplicates(subset=["url"]).copy()
print(f"After deduplicating by URL: {len(lyrics_clean)} rows")

lyrics_clean["year"] = lyrics_clean["date"].str[:4]

conditions = [
    lyrics_clean["error"].str.contains("japanese", case=False, na=False),
    lyrics_clean["error"].str.contains("chinese", case=False, na=False),
    lyrics_clean["english_only"] == True,
    (lyrics_clean["pct_english_without_voc"] == 0) & (lyrics_clean["error"].isna()),
]
choices = ["jpn", "ch", "en", "kor"]
lyrics_clean["language_release"] = np.select(conditions, choices, default="koreng")

lyrics_clean = lyrics_clean[
    (lyrics_clean["year"] >= "2010") & (lyrics_clean["year"] <= "2026")
]

lyrics_clean.to_csv("csv/kpop_lyrics_clean.csv", index=False)