import pandas as pd
import json

df = pd.read_csv("csv/kpop_lyrics_clean.csv")

# find earliest release date per artist
earliest = df.groupby("artist")["date"].min().reset_index()
earliest["year"] = earliest["date"].str[:4].astype(int)

def classify_generation(debut_year):
    if debut_year < 2004:
        return "1st gen"
    elif debut_year < 2012:
        return "2nd gen"
    elif debut_year < 2018:
        return "3rd gen"
    elif debut_year < 2023:
        return "4th gen"
    else:
        return "5th gen"

earliest["generation"] = earliest["year"].apply(classify_generation)

gen_map = dict(zip(earliest["artist"], earliest["generation"]))

with open("gen_map.json", "w", encoding="utf-8") as f:
    json.dump(gen_map, f, indent=2, ensure_ascii=False)

print(f"Mapped {len(gen_map)} artists")
print(earliest.groupby("generation").size())
print("\nSaved to gen_map.json")