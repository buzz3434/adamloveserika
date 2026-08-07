#!/usr/bin/env python3
"""
Daily site generator for adamloveserika.com

What this does, every time it runs (once a day, via GitHub Actions):
1. Looks in a public Google Drive folder for image files.
2. Deterministically picks "today's" photo (cycles through the folder in
   order, then loops back to the start -- so it always works even if you
   never add another photo).
3. Downloads that photo into the site's /photo/ folder.
4. Scans the photo for its "quietest" area -- the largest patch with the
   least visual clutter (sky, a wall, out-of-focus background, a shirt,
   etc.) -- and places the text there instead of dead center. The text
   size is derived from how much quiet space is available: a big open
   area gets bigger text, a small pocket of space gets smaller text.
5. Picks a text color and font style that suit that specific patch of the
   photo (not just the photo as a whole).
6. Writes index.html with the photo filling the screen and the text
   positioned, sized, and styled to fit.

You should never need to touch this file. If you want to change the
message or the rotation start date, see the CONFIG section right below.
"""

import os
import sys
import hashlib
import datetime
import requests
import numpy as np
from PIL import Image

# ----------------------------- CONFIG ------------------------------------

MESSAGE = "Adam loves Erika"

# The very first day of rotation. Day 0 = first photo in the folder.
ROTATION_START_DATE = datetime.date(2025, 1, 1)

DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

DOWNLOADED_PHOTO_PATH = "photo/current.jpg"
OUTPUT_HTML_PATH = "index.html"

FONT_GROUPS = {
    "warm": [
        ("'Dancing Script', cursive", 700),
        ("'Playfair Display', serif", 700),
        ("'Great Vibes', cursive", 400),
        ("'Parisienne', cursive", 400),
        ("'Cormorant Garamond', serif", 700),
    ],
    "cool": [
        ("'Poppins', sans-serif", 600),
        ("'Montserrat', sans-serif", 600),
        ("'Cormorant Garamond', serif", 600),
        ("'Playfair Display', serif", 600),
        ("'Marcellus', serif", 400),
    ],
    "neutral": [
        ("'Cormorant Garamond', serif", 600),
        ("'Playfair Display', serif", 700),
        ("'Poppins', sans-serif", 500),
        ("'Dancing Script', cursive", 700),
        ("'Marcellus', serif", 400),
    ],
}

GOOGLE_FONTS_IMPORT_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Dancing+Script:wght@700&"
    "family=Great+Vibes&"
    "family=Parisienne&"
    "family=Marcellus&"
    "family=Playfair+Display:wght@600;700&"
    "family=Poppins:wght@500;600&"
    "family=Montserrat:wght@600&"
    "family=Cormorant+Garamond:wght@600;700&display=swap"
)

# A rotating set of text colors, each with a light and dark version. The
# light version is used on dark backgrounds and the dark version on light
# backgrounds, so contrast/legibility is always preserved -- but which
# *hue* gets used cycles through this list, so the text isn't always just
# plain black or white.
COLOR_PALETTE = [
    ("#fdf6f0", "#241417"),  # cream / near-black
    ("#ffd9e6", "#5c1830"),  # blush pink / deep wine
    ("#ffe9b0", "#5c4108"),  # gold / deep amber
    ("#e3d9ff", "#2f1a5c"),  # lavender / deep violet
    ("#d7f0ff", "#0b3a56"),  # sky blue / deep navy
    ("#e3f0df", "#1f3d17"),  # sage / deep forest
]

# How far in from each edge (as a fraction of the image) we require the
# text box to stay. Browsers crop photos differently depending on screen
# shape (background-size: cover), so keeping the chosen spot away from the
# outer edges makes it very unlikely to get cropped off on any screen.
EDGE_MARGIN = 0.08

# Candidate text-box sizes to test, as fractions of the image's width/height.
# Sorted largest-area-first so we prefer the biggest quiet area available.
WIDTH_FRACTIONS = [0.70, 0.60, 0.50, 0.42, 0.34, 0.28]
HEIGHT_FRACTIONS = [0.24, 0.20, 0.16, 0.13, 0.10]

# A region is considered "quiet enough" if its brightness standard
# deviation (0-255 scale) is at or below this. Lower = stricter/flatter.
FLAT_THRESHOLD = 16.0

# Step size (as a fraction of image dimensions) used while scanning for
# the quietest spot. Smaller = more thorough but slower.
SCAN_STEP = 0.04

# ---------------------------------------------------------------------------


def day_index_today() -> int:
    """
    How many days since rotation start. Used to pick photo + font + color.

    Normally this is purely based on today's real date, so the look stays
    stable all day even if the workflow happens to run more than once.
    For testing, you can override it by setting the DAY_OFFSET environment
    variable (the "Run workflow" button lets you type a number in) to
    preview a different day's combination on demand without waiting.
    """
    today = datetime.date.today()
    base = (today - ROTATION_START_DATE).days
    override = os.environ.get("DAY_OFFSET", "").strip()
    if override:
        try:
            return base + int(override)
        except ValueError:
            pass
    return base


def list_drive_images():
    if not DRIVE_FOLDER_ID or not GOOGLE_API_KEY:
        sys.exit(
            "ERROR: DRIVE_FOLDER_ID and GOOGLE_API_KEY environment variables "
            "must be set. See SETUP.md."
        )

    url = "https://www.googleapis.com/drive/v3/files"
    params = {
        "q": f"'{DRIVE_FOLDER_ID}' in parents and (mimeType contains 'image/') and trashed = false",
        "key": GOOGLE_API_KEY,
        "fields": "files(id,name,mimeType)",
        "pageSize": 1000,
        "orderBy": "name",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    files = resp.json().get("files", [])
    if not files:
        sys.exit(
            "ERROR: No images found in the Drive folder. Make sure the "
            "folder is shared as 'Anyone with the link - Viewer' and "
            "contains at least one image."
        )
    return files


def download_drive_file(file_id: str, dest_path: str) -> str:
    """Downloads the file and returns a short content hash, used to
    cache-bust the image URL so browsers/CDNs can't keep showing a stale
    cached copy of photo/current.jpg after the content changes."""
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
    params = {"alt": "media", "key": GOOGLE_API_KEY}
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(resp.content)
    return hashlib.md5(resp.content).hexdigest()[:10]


def pick_todays_photo(files):
    idx = day_index_today() % len(files)
    return files[idx]


def rgb_to_hue(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    delta = mx - mn
    if delta == 0:
        return 0.0
    if mx == r:
        h = 60 * (((g - b) / delta) % 6)
    elif mx == g:
        h = 60 * (((b - r) / delta) + 2)
    else:
        h = 60 * (((r - g) / delta) + 4)
    return h % 360


def classify_mood(r, g, b):
    mx, mn = max(r, g, b), min(r, g, b)
    sat = 0 if mx == 0 else (mx - mn) / mx
    if sat < 0.15:
        return "neutral"
    hue = rgb_to_hue(r, g, b)
    if hue <= 90 or hue >= 300:
        return "warm"
    return "cool"


def perceived_brightness(r, g, b) -> float:
    return 0.299 * r + 0.587 * g + 0.114 * b


def pick_font(mood: str):
    day = day_index_today()
    group = FONT_GROUPS.get(mood, FONT_GROUPS["neutral"])
    return group[day % len(group)]


def pick_text_color(brightness: float):
    """Cycle through COLOR_PALETTE by day, choosing the light or dark
    version of that day's color depending on what will stay legible
    against the specific patch of photo behind it."""
    day = day_index_today()
    light_hex, dark_hex = COLOR_PALETTE[(day + 3) % len(COLOR_PALETTE)]
    return dark_hex if brightness > 150 else light_hex


# --------------------------- quiet-area detection ---------------------------


def _build_integral_images(gray: np.ndarray):
    """Summed-area tables for O(1) mean/variance lookups over any rectangle."""
    arr = gray.astype(np.float64)
    integral = np.pad(arr, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    integral_sq = np.pad(arr ** 2, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    return integral, integral_sq


def _region_std(integral, integral_sq, x0, y0, x1, y1):
    n = (x1 - x0) * (y1 - y0)
    total = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    total_sq = integral_sq[y1, x1] - integral_sq[y0, x1] - integral_sq[y1, x0] + integral_sq[y0, x0]
    mean = total / n
    var = max(total_sq / n - mean ** 2, 0.0)
    return var ** 0.5


def find_quiet_region(gray: np.ndarray):
    """
    Scan the image for the largest, plainest patch of background.

    Returns (left_frac, top_frac, width_frac, height_frac, std) where the
    fractions are all relative to the full image (0-1), and std is the
    "clutter score" of the chosen patch (lower = flatter/plainer).
    """
    h, w = gray.shape
    integral, integral_sq = _build_integral_images(gray)

    candidates = sorted(
        ((wf, hf) for wf in WIDTH_FRACTIONS for hf in HEIGHT_FRACTIONS),
        key=lambda c: -(c[0] * c[1]),  # largest area first
    )

    best_fallback = None  # (std, left, top, wf, hf)

    for wf, hf in candidates:
        win_w = max(2, int(round(wf * w)))
        win_h = max(2, int(round(hf * h)))

        left_min, left_max = EDGE_MARGIN, 1 - EDGE_MARGIN - wf
        top_min, top_max = EDGE_MARGIN, 1 - EDGE_MARGIN - hf
        if left_max < left_min or top_max < top_min:
            continue

        best_here = None
        left = left_min
        while left <= left_max + 1e-9:
            top = top_min
            x0 = int(round(left * w))
            x1 = min(w, x0 + win_w)
            while top <= top_max + 1e-9:
                y0 = int(round(top * h))
                y1 = min(h, y0 + win_h)
                if x1 - x0 >= 2 and y1 - y0 >= 2:
                    std = _region_std(integral, integral_sq, x0, y0, x1, y1)
                    if best_here is None or std < best_here[0]:
                        best_here = (std, left, top)
                    if best_fallback is None or std < best_fallback[0]:
                        best_fallback = (std, left, top, wf, hf)
                top += SCAN_STEP
            left += SCAN_STEP

        if best_here and best_here[0] <= FLAT_THRESHOLD:
            std, left, top = best_here
            return left, top, wf, hf, std

    # No patch was "flat enough" -- fall back to the plainest one found.
    std, left, top, wf, hf = best_fallback
    return left, top, wf, hf, std


# ------------------------------- styling -------------------------------


def analyze_photo(path: str):
    """
    Inspect the photo and decide where the text goes, how big it is, and
    what color/font suit that specific spot.
    """
    img = Image.open(path).convert("RGB")
    img_w, img_h = img.size

    # Analyze at (near) full resolution. Downsampling for speed sounds
    # appealing, but any real amount of it smooths away exactly the fine
    # texture/detail (faces, foliage, patterned fabric) that makes a region
    # "busy" -- which would cause busy areas to be misread as calm. The
    # integral-image technique below is fast enough that we don't need to
    # downsample except for very large source photos.
    analysis = img.copy()
    if max(analysis.size) > 2400:
        analysis.thumbnail((2400, 2400), Image.LANCZOS)
    gray = np.array(analysis.convert("L"))

    left_frac, top_frac, width_frac, height_frac, std = find_quiet_region(gray)

    # Sample the actual chosen patch (at full resolution) for its true
    # average color, so the text color/font are matched to that specific
    # spot rather than the photo as a whole.
    x0 = int(left_frac * img_w)
    y0 = int(top_frac * img_h)
    x1 = int((left_frac + width_frac) * img_w)
    y1 = int((top_frac + height_frac) * img_h)
    patch = img.crop((x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)))
    patch_small = patch.resize((20, 20))
    pixels = np.array(patch_small).reshape(-1, 3).mean(axis=0)
    r, g, b = pixels.tolist()

    brightness = perceived_brightness(r, g, b)
    text_color = pick_text_color(brightness)

    mood = classify_mood(r, g, b)
    font_family, font_weight = pick_font(mood)

    # Font size: derived from how much quiet space is available, expressed
    # in vmin so it scales sensibly across both portrait phones and wide
    # desktop screens. Bigger quiet patch -> bigger text; smaller -> smaller.
    text_len = max(len(MESSAGE), 1)
    avg_char_width_factor = 0.50  # rough average glyph width for the fonts above
    width_based_vmin = (width_frac * 100 * 0.90) / (text_len * avg_char_width_factor)
    height_based_vmin = (height_frac * 100 * 0.80)
    font_size_vmin = max(2.6, min(width_based_vmin, height_based_vmin, 12.0))

    return {
        "left_pct": round(left_frac * 100, 2),
        "top_pct": round(top_frac * 100, 2),
        "width_pct": round(width_frac * 100, 2),
        "height_pct": round(height_frac * 100, 2),
        "text_color": text_color,
        "font_family": font_family,
        "font_weight": font_weight,
        "font_size_vmin": round(font_size_vmin, 2),
        "clutter_score": round(std, 1),
    }


def render_html(photo_path: str, style: dict) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Adam &amp; Erika</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="{GOOGLE_FONTS_IMPORT_URL}">
<style>
  * {{
    box-sizing: border-box;
  }}
  html, body {{
    margin: 0;
    padding: 0;
    height: 100%;
    width: 100%;
    overflow: hidden;   /* guarantees no scrollbars, ever */
  }}
  body {{
    position: relative;
    background-image: url("{photo_path}");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    height: 100vh;
    width: 100vw;
    /* 100dvh accounts for mobile browser address bars so the photo
       still fills the true visible screen with no gap or scroll */
    height: 100dvh;
  }}
  .caption-box {{
    position: absolute;
    left: {style['left_pct']}%;
    top: {style['top_pct']}%;
    width: {style['width_pct']}%;
    height: {style['height_pct']}%;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2%;
    text-align: center;
  }}
  .caption-box h1 {{
    font-family: {style['font_family']};
    font-weight: {style['font_weight']};
    color: {style['text_color']};
    font-size: clamp(1.6rem, {style['font_size_vmin']}vmin, 9rem);
    line-height: 1.15;
    margin: 0;
    text-shadow: 0 2px 14px rgba(0,0,0,0.20);
    letter-spacing: 0.02em;
  }}
</style>
</head>
<body>
  <div class="caption-box">
    <h1>{MESSAGE}</h1>
  </div>
</body>
</html>
"""


def main():
    files = list_drive_images()
    chosen = pick_todays_photo(files)
    print(f"Today's photo ({day_index_today()} days since rotation start): {chosen['name']}")

    content_hash = download_drive_file(chosen["id"], DOWNLOADED_PHOTO_PATH)

    style = analyze_photo(DOWNLOADED_PHOTO_PATH)
    print(
        f"Placed text at ({style['left_pct']}%, {style['top_pct']}%) "
        f"size {style['width_pct']}%x{style['height_pct']}% | "
        f"font-size {style['font_size_vmin']}vmin | "
        f"font {style['font_family']} | clutter {style['clutter_score']}"
    )

    photo_url = f"/{DOWNLOADED_PHOTO_PATH}?v={content_hash}"
    html = render_html(photo_url, style)
    with open(OUTPUT_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Done. index.html and photo/current.jpg updated (cache key: {content_hash}).")


if __name__ == "__main__":
    main()
