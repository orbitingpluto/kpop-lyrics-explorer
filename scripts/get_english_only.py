import requests
from bs4 import BeautifulSoup
import re
import time
import random
import csv 
import pandas as pd
from datetime import date

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

def get_column_header(col):
    """Extract header text from either a <strong> or centered <p> tag."""
    strong = col.find("strong")
    if strong:
        return strong.get_text()
    # fallback: first centered paragraph is the header on some posts
    centered = col.find("p", class_=lambda c: c and "has-text-align-center" in c)
    if centered:
        return centered.get_text()
    return ""

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

def get_lyric_text(soup):
    # --- Modern layout ---
    for col in soup.select("div.wp-block-column"):
        header_text = get_column_header(col)
        if not header_text:
            continue
        if "Hangul" in header_text:
            # exclude the header paragraph itself when getting text
            paragraphs = [p for p in col.find_all("p")
                          if "has-text-align-center" not in (p.get("class") or [])]
            text = " ".join(p.get_text(separator=" ") for p in paragraphs)
            if text.strip():
                return text, "korean"
        elif "Japanese" in header_text or "Romaji" in header_text:
            return None, "japanese"
        elif "English" in header_text or "Lyrics" in header_text:
            paragraphs = [p for p in col.find_all("p")
                          if "has-text-align-center" not in (p.get("class") or [])]
            text = " ".join(p.get_text(separator=" ") for p in paragraphs)
            if text.strip() and text.strip() != "Coming Soon!":
                return text, "english_only"

    # --- Old table layout (unchanged) ---
    for table in soup.find_all("table"):
        header_cells = table.find_all("th")
        if not header_cells:
            first_row = table.find("tr")
            if first_row:
                header_cells = first_row.find_all(["th", "td"])

        col_index = None
        col_lang = None
        for i, th in enumerate(header_cells):
            th_text = th.get_text()
            if "Korean" in th_text:
                col_index = i
                col_lang = "korean"
                break
            elif "Japanese" in th_text or "Romaji" in th_text:
                return None, "japanese"
            elif "English" in th_text and col_index is None:
                col_index = i
                col_lang = "english_only"

        if col_index is None:
            continue

        rows = table.find_all("tr")[1:]
        texts = []
        for row in rows:
            cells = row.find_all("td")
            if len(cells) > col_index:
                texts.append(cells[col_index].get_text(separator=" "))
        text = " ".join(texts)
        if text.strip():
            return text, col_lang

    return None, None

def parse_ccl_page(url):
    if not url.startswith("http"):
        return {"url": url, "date": None, "error": "invalid url"}

    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    except requests.exceptions.RequestException as e:
        return {"url": url, "date": None, "error": f"request failed: {e}"}

    if r.status_code == 404:
        return {"url": url, "date": None, "error": "404 not found"}

    try:
        soup = BeautifulSoup(r.text, "lxml")
    except Exception as e:
        return {"url": url, "date": None, "error": f"parse failed: {e}"}

    time_tag = soup.find("time", class_="entry-date published")
    date = time_tag["datetime"][:10] if time_tag else None

    # category-level japanese check
    cat_tags = [a.get_text(strip=True) for a in soup.find_all("a", rel="category tag")]
    
    if "JAPANESE" in cat_tags:
        return {"url": url, "date": date, "error": "japanese release"}
    if "CHINESE" in cat_tags:
            return {"url": url, "date": date, "error": "chinese release"}

    raw_text, lang = get_lyric_text(soup)

    # column-level japanese check
    if lang == "japanese":
        return {"url": url, "date": date, "error": "japanese release"}

    if lang == "chinese":
        return {"url": url, "date": date, "error": "chinese release"}

    if not raw_text:
        return {"url": url, "date": date, "error": "no lyric content found"}

    if lang == "english_only":
        words = strip_structural_noise(raw_text).split()
        word_count = sum(1 for w in words if re.sub(r"^[^\w]+|[^\w]+$", "", w))
        return {
            "url": url,
            "date": date,
            "korean_words": 0,
            "english_words": word_count,
            "vocalization_words": 0,
            "mixed_words": 0,
            "other_words": 0,
            "pct_english_with_voc": 1.0,
            "pct_english_without_voc": 1.0,
            "pct_korean_with_voc": 0.0,
            "pct_korean_without_voc": 0.0,
            "english_only": True,
        }

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
        "english_only": False,
    }


def retry_errors(input_file="csv/kpop_lyrics_combined.csv", output_file="kpop_lyrics_retried.csv"):
    df = pd.read_csv(input_file)

    # retry both error types — hangul not found and the old "no hangul content found" label
    retry_mask = df["error"].isin(["no hangul content found", "no lyric content found"])
    error_rows = df[retry_mask].copy()
    print(f"Retrying {len(error_rows)} rows")

    fieldnames = list(df.columns)
    if "english_only" not in fieldnames:
        fieldnames = fieldnames + ["english_only"]
    if "english_only" not in df.columns:
        df["english_only"] = False

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # pass through all non-retried rows unchanged
        for _, row in df[~retry_mask].iterrows():
            r = row.to_dict()
            r.setdefault("english_only", False)
            writer.writerow(r)

        # retry error rows
        for i, (_, row) in enumerate(error_rows.iterrows()):
            print(f"[{i+1}/{len(error_rows)}] Retrying: {row['artist']} – {row['title']}")
            result = parse_ccl_page(row["url"])

            updated = row.to_dict()
            updated.update({k: result.get(k) for k in result if k != "url"})
            updated.setdefault("english_only", result.get("english_only", False))
            writer.writerow(updated)
            f.flush()
            time.sleep(random.randint(1, 2))

    print(f"\nDone. Saved to {output_file}")
    retried = pd.read_csv(output_file)
    print(f"English-only songs found: {retried['english_only'].sum()}")
    print(f"Remaining errors: {retried['error'].value_counts(dropna=False).to_string()}")


retry_errors()