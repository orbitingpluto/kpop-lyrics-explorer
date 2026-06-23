# ✦ kpop lyrics explorer

how much english is actually in kpop lyrics, and has it changed over time?

i scraped 13,000+ songs from [color coded lyrics](https://colorcodedlyrics.com), tagged every word by language, and built this dashboard to find out.

## live demo

👉 [kpop lyrics explorer](https://kpop-lyrics-explorer.streamlit.app/)

## running locally

```bash
git clone https://github.com/orbitingpluto/kpop-lyrics-explorer
cd kpop-lyrics-explorer
pip install -r requirements.txt
streamlit run app.py
```

## project structure

```
data/processed/   cleaned CSVs + lookup JSONs used by the app
data/raw/         original scraped data
scripts/          one-off pipeline scripts (scraping, cleaning, regression)
src/utils.py      shared helpers and constants
```

## data notes

- lyrics and dates from [color coded lyrics](https://colorcodedlyrics.com)
- only artists with 10+ songs included
- english % excludes vocalizations (uh, oh, yeah) by default
- agency/generation tags from wikipedia and fan wikis — may be outdated

## built with

streamlit · plotly · pandas · scipy
