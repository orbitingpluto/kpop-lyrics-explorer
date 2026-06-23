import requests
from bs4 import BeautifulSoup
import json
import re

HEADERS = {"User-Agent": "Mozilla/5.0"}

PAGES = {
    "girl group": "https://en.wikipedia.org/wiki/List_of_South_Korean_girl_groups",
    "boy group":  "https://en.wikipedia.org/wiki/List_of_South_Korean_boy_bands",
}

def extract_names(url):
    r = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")
    names = set()

    for table in soup.find_all("table", class_="wikitable"):
        for row in table.find_all("tr")[1:]:
            first_cell = row.find(["th", "td"])
            if not first_cell:
                continue
            link = first_cell.find("a")
            text = link.get_text(strip=True) if link else first_cell.get_text(strip=True)
            # strip footnotes like [1]
            text = re.sub(r'\[.*?\]', '', text).strip()
            if text and len(text) < 60:
                names.add(text)

    return names

group_type_map = {}
for group_type, url in PAGES.items():
    print(f"Fetching {group_type}...")
    names = extract_names(url)
    print(f"  Found {len(names)} names")
    for name in names:
        group_type_map[name] = group_type

with open("group_type_map.json", "w", encoding="utf-8") as f:
    json.dump(group_type_map, f, indent=2, ensure_ascii=False)

print(f"\nTotal: {len(group_type_map)} entries")

import pandas as pd
import json
import re

with open("group_type_map.json") as f:
    group_type_map = json.load(f)

def normalize(s):
    if not s: return ""
    s = re.sub(r'\(.*?\)', '', s)
    s = re.sub(r'[^\w]', '', s).lower()
    return s.strip()

wiki_lookup = {normalize(name): gtype for name, gtype in group_type_map.items() if name}

df = pd.read_csv("csv/kpop_lyrics_clean.csv")
ccl_artists = sorted(df["artist"].unique())

result = {}
unmatched = []

for artist in ccl_artists:
    norm = normalize(artist)
    matched = None
    for wiki_norm, gtype in wiki_lookup.items():
        if wiki_norm and len(wiki_norm) > 2 and (wiki_norm in norm or norm in wiki_norm):
            matched = gtype
            break
    result[artist] = matched if matched else "unknown"
    if not matched:
        unmatched.append(artist)

with open("group_type_map_ccl.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"Girl groups: {sum(1 for v in result.values() if v == 'girl group')}")
print(f"Boy groups: {sum(1 for v in result.values() if v == 'boy group')}")
print(f"Unknown: {len(unmatched)}")
print("\nUnmatched (likely soloists or mixed groups):")
for a in unmatched[:20]:
    print(f"  {a}")