import json
import re
import pandas as pd
from collections import Counter

with open("big4_hardcoded.json") as f:
    big4_map = json.load(f)

with open("big4_names.json") as f:
    sm_map = json.load(f)

big4_map.update(sm_map)

def normalize_for_match(s):
    s = re.sub(r'\(.*?\)', '', s)
    s = re.sub(r'[^\w]', '', s).lower()
    return s

wiki_lookup = {normalize_for_match(name): agency for name, agency in big4_map.items() if name}

df = pd.read_csv("csv/kpop_lyrics_clean.csv")
ccl_artists = sorted(df["artist"].unique())

agency_map = {}
other_artists = []

for artist in ccl_artists:
    norm = normalize_for_match(artist)
    matched_agency = None
    for wiki_norm, agency in wiki_lookup.items():
        if wiki_norm and len(wiki_norm) > 2 and (wiki_norm in norm or norm in wiki_norm):
            matched_agency = agency
            break

    if matched_agency:
        agency_map[artist] = matched_agency
    else:
        agency_map[artist] = "Other"
        other_artists.append(artist)

with open("agency_map.json", "w", encoding="utf-8") as f:
    json.dump(agency_map, f, indent=2, ensure_ascii=False)

# separate file listing just the "Other" artists, for review/manual correction
with open("other_artists.json", "w", encoding="utf-8") as f:
    json.dump(sorted(other_artists), f, indent=2, ensure_ascii=False)

print(Counter(agency_map.values()))
print(f"\n{len(other_artists)} artists set to 'Other' — saved to other_artists.json")