import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io

# =========================================================
# CONFIG
# =========================================================

# Colors
GREEN = (0, 196, 79)               # % number
OFFWHITE = (235, 232, 221)         # "OF BETS ON ..." text

# Fonts (macOS Arial Bold paths – change if you have a brand font)
PCT_FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
TEXT_FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

BASE_PCT_FONT_SIZE = 80
BASE_TEXT_FONT_SIZE = 80
MIN_PCT_FONT_SIZE = 55
MIN_TEXT_FONT_SIZE = 55

LINE_SPACING_MULT = 0.9      # spacing between line 1 and line 2 inside a block

# Vertical "safe area" for bets on the template (no header, no logo/footer)
# These are tuned off the template you gave: top of free space and top of footer/logo.
SAFE_TOP_Y = 520
SAFE_BOTTOM_Y = 1120  # slightly conservative so we never touch logo/CTA

# Horizontal layout
LEFT_PCT_X = 120
TEXT_BLOCK_X = 420

TEMPLATE_PATH = "public_betting.png"  # your template


# =========================================================
# SAMPLE DATA (TOP 5 MOST BET NBA SPREADS)
# =========================================================

sample_rows = [
    {"pct": 71, "team": "MIL BUCKS",   "line": "+3.5"},
    {"pct": 61, "team": "ORL MAGIC",   "line": "-3.5"},
    {"pct": 61, "team": "PHO SUNS",    "line": "+9.5"},
    {"pct": 58, "team": "PHI 76ERS",   "line": "+1.5"},
    {"pct": 57, "team": "OKC THUNDER", "line": "-7.5"},
]


# =========================================================
# HELPERS
# =========================================================

def line_height(font, text="Ay"):
    bbox = font.getbbox(text)
    return bbox[3] - bbox[1]


# =========================================================
# CORE RENDER: draw picks ON TOP of template
# =========================================================
def render_public_betting_template(rows):
    # Load template exactly (no resizing)
    base = Image.open(TEMPLATE_PATH).convert("RGBA")
    width, height = base.size

    draw = ImageDraw.Draw(base)

    n = len(rows)
    if n == 0:
        return base

    safe_top = SAFE_TOP_Y
    safe_bottom = SAFE_BOTTOM_Y
    available_height = safe_bottom - safe_top

    # Start with base font sizes, shrink until everything fits the vertical band
    pct_size = BASE_PCT_FONT_SIZE
    text_size = BASE_TEXT_FONT_SIZE
    MIN_SPACING = 10

    while True:
        pct_font = ImageFont.truetype(PCT_FONT_PATH, pct_size)
        text_font = ImageFont.truetype(TEXT_FONT_PATH, text_size)

        lh = line_height(text_font)

        # Height of one bet block (two lines of text):
        # line1 at y, line2 at y + lh * LINE_SPACING_MULT, then lh for line2 body
        block_height = lh * (1 + LINE_SPACING_MULT)

        total_blocks_height = n * block_height
        # minimal spacing layout requirement
        min_needed = total_blocks_height + (n + 1) * MIN_SPACING

        if min_needed <= available_height or (pct_size <= MIN_PCT_FONT_SIZE and text_size <= MIN_TEXT_FONT_SIZE):
            break  # either fits, or we've hit minimum font size and have to live with it

        # shrink fonts and try again
        pct_size -= 3
        text_size -= 3
        if pct_size < MIN_PCT_FONT_SIZE:
            pct_size = MIN_PCT_FONT_SIZE
        if text_size < MIN_TEXT_FONT_SIZE:
            text_size = MIN_TEXT_FONT_SIZE

    # Final fonts after sizing loop
    pct_font = ImageFont.truetype(PCT_FONT_PATH, pct_size)
    text_font = ImageFont.truetype(TEXT_FONT_PATH, text_size)
    lh = line_height(text_font)
    block_height = lh * (1 + LINE_SPACING_MULT)
    total_blocks_height = n * block_height

    if total_blocks_height >= available_height:
        spacing = MIN_SPACING
        first_y = safe_top
    else:
        spacing = max(MIN_SPACING, (available_height - total_blocks_height) / (n + 1))
        first_y = safe_top + spacing

    y = first_y

    for idx, row in enumerate(rows):
        pct_text = f"{row['pct']}%"
        line1 = "OF BETS ON"
        line2 = f"{row['team']} {row['line']}"

        # % in green
        draw.text((LEFT_PCT_X, y), pct_text, font=pct_font, fill=GREEN)

        # "OF BETS ON"
        draw.text((TEXT_BLOCK_X, y), line1, font=text_font, fill=OFFWHITE)

        # "TEAM +LINE"
        line2_y = y + lh * LINE_SPACING_MULT
        draw.text((TEXT_BLOCK_X, line2_y), line2, font=text_font, fill=OFFWHITE)

        y += block_height + spacing

    return base


# =========================================================
# STREAMLIT APP
# =========================================================

st.set_page_config(
    page_title="FoxEdge NBA Public Betting Template Filler",
    page_icon="⚠",
    layout="centered"
)

st.title("FoxEdge PUBLIC BETTING WARNING – NBA Auto Filler")

st.write(
    "Uses the FoxEdge template (public_betting.png) and drops in today's "
    "most-bet NBA spreads. Header, triangle, logo, footer bar all come from the template."
)

edited_rows = []
st.sidebar.header("NBA Public Betting Rows")

for i, row in enumerate(sample_rows):
    with st.sidebar.expander(f"Play {i+1}", expanded=(i == 0)):
        pct_val = st.number_input(
            f"% of Bets {i+1}",
            value=row["pct"],
            min_value=0,
            max_value=100,
            step=1
        )
        team_val = st.text_input(
            f"Team {i+1}",
            value=row["team"]
        )
        line_val = st.text_input(
            f"Line/Spread {i+1}",
            value=row["line"]
        )
        edited_rows.append({
            "pct": pct_val,
            "team": team_val.strip().upper(),
            "line": line_val.strip()
        })

img = render_public_betting_template(edited_rows)

buf = io.BytesIO()
img.save(buf, format="PNG")

st.image(buf.getvalue(), use_container_width=True)

st.download_button(
    label="Download Image",
    data=buf.getvalue(),
    file_name="foxedge_nba_public_betting_post.png",
    mime="image/png"
)

st.caption(
    "Font size and spacing now adapt to the number of rows and the available band "
    "between WARNING and the logo. No more clipping, no more hand-tuning constants."
)
