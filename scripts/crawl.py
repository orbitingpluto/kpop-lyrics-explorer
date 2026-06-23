import requests
from bs4 import BeautifulSoup
import re
import time
import random
import csv 
import pandas as pd
from datetime import date


# ── constants ────────────────────────────────────────────────────────────────

MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]

HEADERS = {"User-Agent": "Mozilla/5.0"}

# ── karchives (2023–present) ─────────────────────────────────────────────────

def scrape_karchives_month(year, month):
    """Scrape one month's Melon top 50 from karchives."""
    month_name = MONTH_NAMES[month - 1]
    url = f"https://karchives.com/melon-monthly-chart-{month_name}-{year}/"

    r = requests.get(url, headers=HEADERS)
    if r.status_code == 404:
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if not table:
        return None

    songs = []
    for row in table.find_all("tr")[1:]:  # skip header
        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        try:
            rank = int(cells[0].get_text(strip=True))
        except ValueError:
            continue

        # 3rd cell: "Song\nArtist"
        song_cell = cells[2].get_text(separator="\n", strip=True)
        parts = [p.strip() for p in song_cell.split("\n") if p.strip()]

        if len(parts) >= 2:
            title, artist = parts[0], parts[1]
        else:
            split = re.split(r'\s{2,}', song_cell)
            title = split[0].strip()
            artist = split[1].strip() if len(split) > 1 else ""

        songs.append({
            "rank":       rank,
            "title":      title,
            "artist":     artist,
            "year":       year,
            "month":      month,
            "month_name": MONTH_NAMES[month - 1].capitalize(),
            "source":     "karchives_melon",
        })

    return songs


def scrape_karchives(start_year=2023):
    all_songs = []
    today = date.today()

    for year in range(start_year, today.year + 1):
        for month in range(1, 13):
            if year == today.year and month >= today.month:
                break  # don't request current or future months

            print(f"  karchives: {MONTH_NAMES[month-1].capitalize()} {year}")
            songs = scrape_karchives_month(year, month)

            if songs is None:
                print(f"    -> not found, skipping")
            else:
                print(f"    -> {len(songs)} songs")
                all_songs.extend(songs)

            time.sleep(1)

    return all_songs


# ── wikipedia gaon (2010–2022) ───────────────────────────────────────────────

def scrape_gaon_year(year):
    """Scrape weekly Gaon #1s from Wikipedia, return deduplicated song list."""
    url = f"https://en.wikipedia.org/wiki/List_of_Gaon_Digital_Chart_number_ones_of_{year}"
    r = requests.get(url, headers=HEADERS)

    if r.status_code != 200:
        print(f"    -> not found")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    songs = []
    last_date = ""  # track last seen date for continuation rows

    for table in soup.find_all("table", class_="wikitable"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        # weekly chart table has a date/week column
        if not any("week" in h or "date" in h for h in headers):
            continue

        for row in table.find_all("tr")[1:]:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue

            texts = [c.get_text(strip=True) for c in cells]

            # continuation rows (same song, next week) have only a date cell
            if len(texts) == 1:
                last_date = texts[0]
                continue

            # rows where first cell is empty inherit last_date
            date_text = texts[0] if texts[0] else last_date
            if texts[0]:
                last_date = texts[0]

            # need at least song + artist
            if len(texts) < 3:
                continue

            # skip rows with no linked song (section headers etc.)
            if not any(c.find("a") for c in cells[1:]):
                continue

            song   = texts[1].strip('"').strip('\u2019').strip("'")
            artist = texts[2]

            # extract month from "January 3" or "3 January"
            month_match = re.search(
                r'(January|February|March|April|May|June|July|August|'
                r'September|October|November|December)',
                date_text, re.IGNORECASE
            )
            month_name = month_match.group(1).capitalize() if month_match else None
            month_num  = (MONTH_NAMES.index(month_name.lower()) + 1
                          if month_name else None)

            songs.append({
                "rank":       1,  # all are #1 on Gaon
                "title":      song,
                "artist":     artist,
                "year":       year,
                "month":      month_num,
                "month_name": month_name,
                "source":     "gaon_wikipedia",
            })

        break  # only need first matching table

    # deduplicate: same song can hold #1 for multiple weeks
    seen = set()
    unique = []
    for s in songs:
        key = (s["title"].lower(), s["year"])
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique


def scrape_gaon(start_year=2010, end_year=2022):
    all_songs = []
    for year in range(start_year, end_year + 1):
        print(f"  Gaon Wikipedia: {year}")
        songs = scrape_gaon_year(year)
        print(f"    -> {len(songs)} unique songs")
        all_songs.extend(songs)
        time.sleep(1)
    return all_songs


# ── combined ─────────────────────────────────────────────────────────────────

def build_chart_dataset(output_file="charts_combined.csv"):
    print("Scraping Gaon chart (Wikipedia, 2010–2022)...")
    gaon_songs = scrape_gaon(start_year=2010, end_year=2022)

    print("\nScraping Melon monthly chart (karchives, 2023–present)...")
    karchives_songs = scrape_karchives(start_year=2023)

    df = pd.DataFrame(gaon_songs + karchives_songs)

    # deduplicate across sources — same song shouldn't appear twice in same year
    df = df.drop_duplicates(subset=["title", "year"]).reset_index(drop=True)

    df.to_csv(output_file, index=False)
    print(f"\nDone. {len(df)} unique charted songs saved to {output_file}")
    print(df.groupby(["year", "source"]).size().to_string())
    return df


VERSION_KEYWORDS = ["version", "ver.", "ver)", "remix", "inst.", "instrumental", "acoustic", "repack"]

def get_artist_links_from_index():
    """Scrape the main index page and return {artist_name: category_or_index_url}."""
    r = requests.get("https://colorcodedlyrics.com/index/", headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.text, "html.parser")

    # Artist links are in the entry-content div
    content = soup.find("div", class_="entry-content")
    artists = {}
    for a in content.find_all("a", href=True):
        name = a.get_text(strip=True)
        href = a["href"]
        # Only keep Korean artist links (skip nav links, etc.)
        if "colorcodedlyrics.com" in href and name:
            artists[name] = href
    return artists


def get_song_urls_from_category(category_url, artist_name):
    """
    Given a /category/krn/{artist}/ URL, paginate through all pages
    and return a list of song dicts with url, title, artist, date.
    """
    songs = []
    page = 1

    while True:
        url = category_url if page == 1 else f"{category_url.rstrip('/')}/page/{page}/"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})

        # 404 means no more pages
        if r.status_code == 404:
            break

        soup = BeautifulSoup(r.text, "html.parser")
        articles = soup.find_all("article", class_="archive-post-card")

        if not articles:
            break

        for article in articles:
            title_tag = article.find("h3", class_="entry-title")
            if not title_tag:
                continue

            a_tag = title_tag.find("a")
            if not a_tag:
                continue

            song_url = a_tag["href"]
            song_title = a_tag.get_text(strip=True)

            # exclude japanese releases
            if any(x in song_title.lower() for x in ["japanese ver", "japanese version"]):
                continue
            
            # exclude versions/instrumentals
            if any(kw in song_title.lower() for kw in VERSION_KEYWORDS):
                continue

            # Skip index posts (profile/lyrics index pages)
            if "index" in song_title.lower() or "profile" in song_title.lower():
                continue

            # Get post date
            time_tag = article.find("time", class_="entry-date published")
            date = time_tag["datetime"][:10] if time_tag else None  # YYYY-MM-DD

            # Get language category tags — useful to flag ENGLISH-only posts
            cat_links = [a.get_text(strip=True) for a in article.find_all("a", rel="category tag")]

            if "JAPANESE" in cat_links:
                continue  # skip before appending

            songs.append({
                "artist": artist_name,
                "title": song_title,
                "url": song_url,
                "date": date,
                "categories": cat_links,
            })

        page += 1
        time.sleep(random.randint(0, 3))  # be polite

    return songs


def get_song_urls_from_index_post(index_post_url, artist_name):
    """
    For artists with a dedicated index post instead of a category page,
    scrape all song links from the post body.
    """
    r = requests.get(index_post_url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.text, "html.parser")
    content = soup.find("div", class_="entry-content")
    if not content:
        return []

    songs = []
    for a in content.find_all("a", href=True):
        href = a["href"]
        title = a.get_text(strip=True)

        if href.startswith("//"):
            href = "https:" + href
        # exclude japanese releases
        if any(x in title.lower() for x in ["japanese ver", "japanese version"]):
            continue
        
        # exclude versions/instrumentals
        if any(kw in title.lower() for kw in VERSION_KEYWORDS):
            continue
        # Filter to actual song posts (dated URLs like /2019/03/...)
        if re.search(r'/\d{4}/\d{2}/', href) and "index" not in href.lower():

            if any(p in href.lower() for p in ["/jpn/", "japanese-ver", "japanese-version", "-japan-"]):
                continue
            
            songs.append({
                "artist": artist_name,
                "title": title,
                "url": href,
                "date": None,  # no date available here; will pull from song page
                "categories": [],
            })
    return songs

def normalize_artist(s):
    if not s: return ""
    # strip featuring credits
    s = re.sub(r'\s*(feat\.|featuring|ft\.|with|and)\s+.*', '', s, flags=re.IGNORECASE)
    # strip everything non-alphanumeric
    s = re.sub(r'[^\w]', '', s).lower()
    return s.strip()

def collect_all_songs(target_artists=None):
    all_songs = []
    artist_links = get_artist_links_from_index()

    if target_artists:
        norm_targets = [normalize_artist(t) for t in target_artists]
        norm_targets = [t for t in norm_targets if t]

    for artist_name, artist_url in artist_links.items():
        if target_artists:
            norm_name = normalize_artist(artist_name)
            if not any(t in norm_name or norm_name in t for t in norm_targets):
                continue

        if "/category/" in artist_url:
            songs = get_song_urls_from_category(artist_url, artist_name)
        else:
            songs = get_song_urls_from_index_post(artist_url, artist_name)

        all_songs.extend(songs)
        print(f"Collecting: {artist_name} ({len(songs)} songs, {len(all_songs)} total)")
        time.sleep(1)

    return all_songs
# --- stripping ---

VOCALIZATION_WORDS = {
    "uh", "huh", "woo", "woah", "whoa", "ya", "yah", "yeah", "ye",
    "oh", "ooh", "ah", "aha", "eh", "hey", "ha", "na", "la", "da",
    "mm", "mmm", "hmm", "ugh", "yo", "Na", "boom", "hey", "e", "blakah", "la-la"
}

def strip_structural_noise(text):
    """Remove stage directions and parentheticals before word-splitting."""
    text = re.sub(r'\[.*?\]', '', text)   # [윈/닝], [Win/Ning]
    text = re.sub(r'\(.*?\)', '', text)   # (More LEMONADE), (Yeah yeah)
    return text

def classify_word(word):
    cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", word)
    if not cleaned:
        return "other"
    has_hangul = any('\uAC00' <= c <= '\uD7A3' or '\u1100' <= c <= '\u11FF' for c in cleaned)
    has_latin  = any(c.isascii() and c.isalpha() for c in cleaned)
    if has_hangul and not has_latin:
        return "korean"
    elif has_latin and not has_hangul:
        return "english"
    elif has_hangul and has_latin:
        return "mixed"
    else:
        return "other"
    
def get_hangul_text(soup):
    # --- Modern layout ---
    for col in soup.select("div.wp-block-column"):
        header = col.find("strong")
        if header and "Hangul" in header.get_text():
            paragraphs = col.find_all("p")
            return " ".join(p.get_text(separator=" ") for p in paragraphs)

    # --- Old table layout: find whichever table has a Korean column ---
    for table in soup.find_all("table"):
        headers = table.find_all("th")
        korean_col_index = None
        for i, th in enumerate(headers):
            if "Korean" in th.get_text():
                korean_col_index = i
                break

        if korean_col_index is None:
            continue  # this table isn't the lyrics table, try the next one

        rows = table.find_all("tr")
        texts = []
        for row in rows:
            cells = row.find_all("td")
            if len(cells) > korean_col_index:
                texts.append(cells[korean_col_index].get_text(separator=" "))
        return " ".join(texts)

    return None
def parse_ccl_page(url):
    if not url.startswith("http"):
        return {"url": url, "date": None, "error": "invalid url"}
    
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    
    soup = BeautifulSoup(r.text, "html.parser")

    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    except requests.exceptions.RequestException as e:
        return {"url": url, "date": None, "error": f"request failed: {e}"}


    if r.status_code == 404:
        return {"url": url, "date": None, "error": "404 not found"}

    # Always extract date first
    time_tag = soup.find("time", class_="entry-date published")
    date = time_tag["datetime"][:10] if time_tag else None

    # Now safe to use date in any early return
    cat_tags = [a.get_text(strip=True) for a in soup.find_all("a", rel="category tag")]
    if "JAPANESE" in cat_tags:
        return {"url": url, "date": date, "error": "japanese release"}

    raw_text = get_hangul_text(soup)
    if not raw_text:
        return {"url": url, "date": date, "error": "no hangul content found"}

    clean_text = strip_structural_noise(raw_text)
    words = clean_text.split()

    counts = {"korean": 0, "english": 0, "vocalization": 0, "mixed": 0, "other": 0}
    for w in words:
        label = classify_word(w)
        if label == "english" and w.lower() in VOCALIZATION_WORDS:
            counts["vocalization"] += 1
        else:
            counts[label] += 1

    total_with_voc = counts["korean"] + counts["english"] + counts["vocalization"]
    total_without  = counts["korean"] + counts["english"]

    return {
        "url": url,
        "date": date,
        "korean_words": counts["korean"],
        "english_words": counts["english"],
        "vocalization_words": counts["vocalization"],
        "mixed_words": counts["mixed"],
        "other_words": counts["other"],
        "pct_english_with_voc":    round((counts["english"] + counts["vocalization"]) / total_with_voc, 3) if total_with_voc else None,
        "pct_english_without_voc": round(counts["english"] / total_without, 3) if total_without else None,
        "pct_korean_with_voc":     round(counts["korean"] / total_with_voc, 3) if total_with_voc else None,
        "pct_korean_without_voc":  round(counts["korean"] / total_without, 3) if total_without else None,
    }

def normalize_title(s):
    if not s: return ""
    s = re.sub(r'\(.*?\)', '', s)
    s = re.sub(r'[^\w\s]', '', s.lower())
    return s.strip()

def is_charted(song, chart_keys):
    norm_title  = normalize_title(song["title"])
    norm_artist = normalize_artist(song["artist"])
    
    
    # check if any chart entry matches on title + artist substring
    for chart_title, chart_artist in chart_keys:
        if norm_title == chart_title:
            # artist match: either contains the other (handles "BTS" vs "BTS (방탄소년단)")
            if chart_artist in norm_artist or norm_artist in chart_artist:
                return True
    return False

not_found_on_ccl = []

def scrape_all(target_artists, output_file="kpop_lyrics.csv", chart_file="charts_combined.csv"):
    try:
        chart_df = pd.read_csv(chart_file)
        print(f"Loaded existing chart data: {len(chart_df)} songs")
    except FileNotFoundError:
        print("Chart file not found, scraping now...")
        chart_df = build_chart_dataset(output_file=chart_file)

    # make a set of (normalized_title, normalized_artist) tuples
    chart_keys = set(
        zip(
            chart_df["title"].apply(normalize_title),
            chart_df["artist"].apply(normalize_artist)
        )
    )

    chart_titles = set(chart_df["title"].apply(normalize_title))
    artist_links = get_artist_links_from_index()

    if target_artists:
        norm_targets = [normalize_artist(t) for t in target_artists]
        norm_targets = [t for t in norm_targets if t]

    fieldnames = [
        "artist", "title", "url", "date",
        "korean_words", "english_words", "vocalization_words", "mixed_words", "other_words",
        "pct_english_with_voc", "pct_english_without_voc",
        "pct_korean_with_voc", "pct_korean_without_voc",
        "error"
    ]

    total_written = 0

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        for artist_name, artist_url in artist_links.items():
            if target_artists:
                norm_name = normalize_artist(artist_name)
                if not any(t in norm_name or norm_name in t for t in norm_targets):
                    continue

            if artist_name.strip().upper() <= "Morning":
                continue

            print(f"\nCollecting: {artist_name}")

            if "/category/" in artist_url:
                songs = get_song_urls_from_category(artist_url, artist_name)
            else:
                songs = get_song_urls_from_index_post(artist_url, artist_name)

            # # filter to charted songs immediately
            # charted = [s for s in songs if is_charted(s, chart_keys)]
            # ccl_titles = set(normalize_title(s["title"]) for s in songs)

            # print(f"  {len(songs)} songs found, {len(charted)} on chart")

            # # check which charted songs for this artist weren't in CCL at all
            # for chart_title, chart_artist in chart_keys:
            #     if normalize_artist(artist_name) and (
            #         chart_artist in normalize_artist(artist_name) or 
            #         normalize_artist(artist_name) in chart_artist
            #     ):
            #         if chart_title not in ccl_titles:
            #             not_found_on_ccl.append({
            #                 "chart_artist": chart_artist,
            #                 "chart_title": chart_title,
            #             })

            for song in songs:
                if song["categories"] and all(c == "ENGLISH" for c in song["categories"] if c in ("KOREAN", "ENGLISH")):
                    print(f"  Skipping English-only: {song['title']}")
                    continue

                print(f"  Scraping: {song['title']}")
                result = parse_ccl_page(song["url"])

                row = {
                    "artist": song["artist"],
                    "title":  song["title"],
                    "url":    song["url"],
                    "date":   result.get("date") or song["date"],
                    **{k: result.get(k) for k in fieldnames if k not in ("artist", "title", "url", "date")},
                }
                writer.writerow(row)
                f.flush()
                total_written += 1
                time.sleep(random.randint(1, 3))

            time.sleep(1)

    print(f"\nDone. {total_written} songs written to {output_file}")

chart_df = pd.read_csv("charts_combined.csv")
print(chart_df["artist"].unique())

def extract_main_artist(artist_str):
    """Strip featuring credits and take the first-billed artist."""
    if not isinstance(artist_str, str):
        return None
    # remove featuring
    artist_str = re.sub(r'\s+(feat\.|featuring|ft\.)\s+.*', '', artist_str, flags=re.IGNORECASE)
    # if multiple artists separated by comma or &, take first
    artist_str = re.split(r'\s*[,&]\s*', artist_str)[0]
    return artist_str.strip()

artists = (
    chart_df["artist"]
    .apply(extract_main_artist)
    .dropna()
    .unique()
    .tolist()
)

print(f"{len(artists)} unique artists")
for a in sorted(artists):
    print(a)

TARGET_GROUPS = None
scrape_all(TARGET_GROUPS, output_file="kpop_lyrics_all4.csv")

print(f"\n{len(not_found_on_ccl)} charted songs not found on CCL:")
for s in sorted(not_found_on_ccl, key=lambda x: x["chart_artist"]):
    print(f"  {s['chart_artist']} – {s['chart_title']}")

pd.DataFrame(not_found_on_ccl).drop_duplicates().to_csv("not_found_on_ccl.csv", index=False)