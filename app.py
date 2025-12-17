import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import requests
import os, math, textwrap, re, time, json
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

# --- CONFIG ---
DB_FILE = "tv_shows.db"

# ==========================================
# 🛠️ FONT FIXER (Downloads Fonts Automatically)
# ==========================================
def install_fonts():
    fonts = {
        "Roboto-Regular.ttf": "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Regular.ttf",
        "Roboto-Bold.ttf": "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Bold.ttf"
    }
    for name, url in fonts.items():
        if not os.path.exists(name):
            try:
                r = requests.get(url)
                with open(name, "wb") as f:
                    f.write(r.content)
            except: pass

install_fonts()

# ==========================================
# 🎨 CUSTOM CSS
# ==========================================
st.set_page_config(layout="wide", page_title="TV Heatmap", page_icon="📺")

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 95% !important;
    }
    header {visibility: hidden;}
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .rec-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 5px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 🎨 VISUALIZATION ENGINE (HD TUNED)
# ==========================================

def strip_html(s):
    if not s: return ""
    return re.sub(r'<[^>]*>', '', s).strip()

def draw_star(draw, center, size, color):
    cx, cy = center
    pts = []
    inner = size * 0.45
    outer = size
    ang = -math.pi / 2
    for i in range(10):
        r = outer if i % 2 == 0 else inner
        pts.append((cx + math.cos(ang) * r, cy + math.sin(ang) * r))
        ang += math.pi / 5
    draw.polygon(pts, fill=color)

def color_for_score(score):
    if score is None or (isinstance(score, float) and np.isnan(score)): return (54, 54, 54)
    s = float(score)
    if s <= 5.5: return (128, 0, 128)
    if s <= 6.9: return (220, 0, 0)
    if s <= 7.9: return (212, 175, 55)
    if s <= 8.5: return (144, 238, 144)
    return (0, 100, 0)

def text_color_for_bg(rgb):
    r, g, b = rgb
    return (0, 0, 0) if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else (255, 255, 255)

def cover_crop(img, W, H):
    if img is None: return None
    sw, sh = img.size
    scale = max(W / sw, H / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    img2 = img.resize((nw, nh), Image.LANCZOS)
    left = max(0, (nw - W) // 2)
    top = max(0, (nh - H) // 2)
    return img2.crop((left, top, left + W, top + H))

def draw_text_rich(draw, pos, text, font, fill, shadow=(0, 0, 0, 200), shadow_offset=(2, 2), anchor=None):
    x, y = pos
    sx, sy = shadow_offset
    if shadow:
        draw.text((x + sx, y + sy), text, font=font, fill=shadow, anchor=anchor)
    draw.text((x, y), text, font=font, fill=fill, anchor=anchor)

def load_font(name_list, size):
    priorities = ["Roboto-Regular.ttf", "Roboto-Bold.ttf"] + name_list
    for name in priorities:
        try:
            return ImageFont.truetype(name, size)
        except:
            continue
    return ImageFont.load_default()

def wrap_text_pixel(draw, text, font, max_w):
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        test_line = ' '.join(current_line + [word])
        w = draw.textlength(test_line, font=font)
        if w <= max_w:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                lines.append(word)
                current_line = []
    if current_line:
        lines.append(' '.join(current_line))
    return lines

def render_page(grid_df, poster_img, title, year_range, summary, main_rating):
    # --- UPSCALED DIMENSIONS FOR READABILITY ---
    left_col_w = 800  # Wider title area
    HEADER_Y = 120    # Lower header
    FIXED_BOX_W = 160 # Wider boxes
    FIXED_BOX_H = 70  # Taller boxes for bigger text
    GAP_BETWEEN_COLS = 100
    
    num_seasons = len(grid_df.columns)
    grid_width = num_seasons * (FIXED_BOX_W + 16)
    required_width = 80 + left_col_w + GAP_BETWEEN_COLS + grid_width + 100
    canvas_w = max(2200, required_width) 

    n_eps = grid_df.shape[0]
    base_top = 450
    spacing = FIXED_BOX_H + 16
    min_required_h = base_top + n_eps * spacing + 400
    canvas_h = max(1200, min_required_h)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # --- UPSCALED FONTS ---
    # We double the font sizes so they look crisp when the image is shrunk
    f_reg = load_font(["Roboto-Regular.ttf", "arial.ttf"], 36)
    title_font = load_font(["Roboto-Bold.ttf", "arialbd.ttf"], 110)
    font_year = load_font(["Roboto-Regular.ttf", "arial.ttf"], 48)
    font_rating = load_font(["Roboto-Bold.ttf", "arialbd.ttf"], 90)
    box_font = load_font(["Roboto-Bold.ttf", "arialbd.ttf"], 40) # Big numbers inside boxes

    if poster_img:
        bg = cover_crop(poster_img, canvas_w, canvas_h)
        if bg:
            canvas.paste(bg, (0, 0))
            overlay = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 160)) # Slightly darker for contrast
            canvas.paste(overlay, (0, 0), overlay)

    x_left = 80
    y = 60
    
    draw_text_rich(draw, (x_left, y), "TV Series", f_reg, (245, 245, 245))
    y += 45

    title_lines = wrap_text_pixel(draw, title, title_font, max_w=750)
    for line in title_lines:
        draw_text_rich(draw, (x_left, y), line, title_font, (245, 245, 245))
        y += 115 

    y += 15
    draw_text_rich(draw, (x_left, y), f"({year_range})", font_year, (245, 245, 245))
    y += 90
    
    draw_star(draw, (x_left + 40, y + 35), 40, (255, 200, 0))
    draw_text_rich(draw, (x_left + 100, y), f"{main_rating}/10", font_rating, (245, 245, 245))
    y += 110

    wrapper = textwrap.TextWrapper(width=45) # Adjusted wrap
    lines = wrapper.wrap(summary)[:8]
    for line in lines:
        draw_text_rich(draw, (x_left, y), line, f_reg, (210, 210, 210))
        y += 45

    grid_start_x = x_left + left_col_w + GAP_BETWEEN_COLS
    seasons = list(grid_df.columns)

    legend = [("Awesome", (0, 100, 0)), ("Great", (144, 238, 144)), ("Good", (212, 175, 55)), ("Bad", (220, 0, 0)), ("Garbage", (128, 0, 128))]
    lx = grid_start_x
    ly = HEADER_Y - 60
    
    for name, col in legend:
        draw.ellipse((lx, ly, lx+24, ly+24), fill=col)
        draw_text_rich(draw, (lx+35, ly+12), name, f_reg, (245,245,245), anchor="lm")
        text_w = draw.textlength(name, font=f_reg)
        lx += (35 + text_w + 50)

    for j, s in enumerate(seasons):
        sx = grid_start_x + j * (FIXED_BOX_W + 16)
        box = [sx, HEADER_Y - 25, sx + FIXED_BOX_W, HEADER_Y + 25]
        draw.rounded_rectangle(box, radius=15, fill=(40, 40, 40))
        draw_text_rich(draw, (sx + FIXED_BOX_W / 2, HEADER_Y), f"S{s}", f_reg, (245, 245, 245), anchor="mm")

    row_top = HEADER_Y + 60
    
    for i, ep in enumerate(grid_df.index):
        ry = row_top + i * (FIXED_BOX_H + 16)
        ebox = [grid_start_x - 100, ry, grid_start_x - 100 + 80, ry + FIXED_BOX_H]
        draw.rounded_rectangle(ebox, 10, (40, 40, 40))
        draw_text_rich(draw, (ebox[0] + 40, ebox[1] + FIXED_BOX_H / 2), f"E{ep}", f_reg, (245, 245, 245), anchor="mm")

        for j, s in enumerate(seasons):
            sx = grid_start_x + j * (FIXED_BOX_W + 16)
            val = grid_df.loc[ep, s]
            box = [sx, ry, sx + FIXED_BOX_W, ry + FIXED_BOX_H]
            fill = color_for_score(val)
            draw.rounded_rectangle(box, radius=12, fill=fill, outline=(12, 12, 12), width=3)
            txt = f"{val:.1f}" if pd.notna(val) else "-"
            tcol = text_color_for_bg(fill)
            draw_text_rich(draw, (sx + FIXED_BOX_W / 2, ry + FIXED_BOX_H / 2), txt, box_font, tcol, anchor="mm", shadow=None)

    sep = row_top + n_eps * (FIXED_BOX_H + 16) + 12
    draw.line([(grid_start_x - 100, sep), (grid_start_x + len(seasons) * (FIXED_BOX_W + 16), sep)], fill=(60, 60, 60), width=3)
    avg_y = sep + 20
    a = [grid_start_x - 100, avg_y, grid_start_x - 100 + 80, avg_y + FIXED_BOX_H]
    draw.rounded_rectangle(a, 12, (40, 40, 40))
    draw_text_rich(draw, (a[0] + 40, avg_y + FIXED_BOX_H / 2), "Avg", f_reg, (245, 245, 245), anchor="mm")

    for j, s in enumerate(seasons):
        sx = grid_start_x + j * (FIXED_BOX_W + 16)
        vals = pd.to_numeric(grid_df[s], errors='coerce').dropna()
        avg = round(vals.mean(), 1) if len(vals) > 0 else None
        b = [sx, avg_y, sx + FIXED_BOX_W, avg_y + FIXED_BOX_H]
        fill = color_for_score(avg)
        draw.rounded_rectangle(b, FIXED_BOX_H // 2, fill)
        txt = f"{avg:.1f}" if avg is not None else "—"
        tcol = text_color_for_bg(fill)
        draw_text_rich(draw, (sx + FIXED_BOX_W / 2, avg_y + FIXED_BOX_H / 2), txt, box_font, tcol, anchor="mm", shadow=None)

    return canvas

# ==========================================
# 🧠 BACKEND LOGIC (Unchanged)
# ==========================================

def get_recommendations(current_tconst, genres):
    if not genres or genres == "Unknown": return pd.DataFrame()
    current_genres = genres.split(',')
    main_genre = current_genres[0]
    conn = sqlite3.connect(DB_FILE)
    score_query = []
    for g in current_genres:
        score_query.append(f"(CASE WHEN s.genres LIKE '%{g}%' THEN 1 ELSE 0 END)")
    total_score_sql = " + ".join(score_query)
    sql = f"""
    SELECT s.tconst, s.primaryTitle, s.startYear, s.numVotes, r.averageRating,
           ({total_score_sql}) as match_score
    FROM shows s
    JOIN ratings r ON s.tconst = r.tconst
    WHERE s.genres LIKE ? AND s.tconst != ?
    ORDER BY match_score DESC, s.numVotes DESC
    LIMIT 4
    """
    params = [f"%{main_genre}%", current_tconst]
    try:
        df = pd.read_sql_query(sql, conn, params=params)
    except:
        df = pd.DataFrame()
    conn.close()
    return df

def search_shows(query):
    conn = sqlite3.connect(DB_FILE)
    parts = query.split()
    sql = "SELECT tconst, primaryTitle, startYear, numVotes, genres FROM shows WHERE "
    conditions = []
    params = []
    for part in parts:
        conditions.append("primaryTitle LIKE ?")
        params.append(f"%{part}%")
    sql += " AND ".join(conditions)
    sql += " ORDER BY numVotes DESC LIMIT 15"
    try:
        df = pd.read_sql_query(sql, conn, params=params)
    except:
        df = pd.DataFrame()
    conn.close()
    return df

def scrape_live_ratings(imdb_id):
    data = []
    ua = UserAgent()
    headers = {"User-Agent": ua.chrome}
    session = requests.Session()
    for s in range(1, 40):
        if len(data) > 0 and s > data[-1]['seasonNumber'] + 2: break
        try:
            url = f"https://www.imdb.com/title/{imdb_id}/episodes?season={s}"
            r = session.get(url, headers=headers, timeout=8)
            if r.status_code != 200: continue
            soup = BeautifulSoup(r.content, 'html.parser')
            found = False
            json_tag = soup.find('script', id='__NEXT_DATA__')
            if json_tag:
                try:
                    js = json.loads(json_tag.string)
                    ep_items = js['props']['pageProps']['contentData']['section']['episodes']['items']
                    for item in ep_items:
                        ep = int(item['episode'])
                        val = float(item['rating']['aggregateRating'])
                        if ep > 0 and val > 0:
                            data.append({'seasonNumber': s, 'episodeNumber': ep, 'averageRating': val})
                            found = True
                except: pass
            if not found:
                stars = soup.select('.ipc-rating-star--rating')
                for i, star in enumerate(stars):
                    try:
                        val = float(star.text.strip())
                        if val > 0:
                            data.append({'seasonNumber': s, 'episodeNumber': i+1, 'averageRating': val})
                            found = True
                    except: pass
        except: pass
    return pd.DataFrame(data).drop_duplicates(subset=['seasonNumber', 'episodeNumber'])

def get_live_overall_rating(tconst):
    try:
        ua = UserAgent()
        headers = {"User-Agent": ua.chrome}
        url = f"https://www.imdb.com/title/{tconst}/"
        r = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(r.content, 'html.parser')
        json_tag = soup.find('script', type='application/ld+json')
        if json_tag:
            data = json.loads(json_tag.string)
            return float(data['aggregateRating']['ratingValue'])
    except: return None
    return None

def get_show_data(tconst, force_live=False):
    conn = sqlite3.connect(DB_FILE)
    source_msg = "Database"
    q_main = "SELECT averageRating FROM ratings WHERE tconst = ?"
    main_rating_df = pd.read_sql_query(q_main, conn, params=(tconst,))
    if not main_rating_df.empty and pd.notna(main_rating_df.iloc[0]['averageRating']):
        main_rating = round(main_rating_df.iloc[0]['averageRating'], 1)
    else: main_rating = 0.0
    df = pd.DataFrame()
    if force_live:
        try:
            df = scrape_live_ratings(tconst)
            if not df.empty: source_msg = "Live IMDb (Fresh)"
            live_main = get_live_overall_rating(tconst)
            if live_main: main_rating = live_main
        except: pass
    if df.empty:
        q_eps = "SELECT e.seasonNumber, e.episodeNumber, r.averageRating FROM episodes e JOIN ratings r ON e.tconst = r.tconst WHERE e.parentTconst = ? ORDER BY e.seasonNumber, e.episodeNumber"
        df = pd.read_sql_query(q_eps, conn, params=(tconst,))
        if force_live: source_msg = "Live Failed (Using DB)"
    conn.close()
    if df.empty: return None, "No episode data found.", source_msg
    df = df[df['seasonNumber'] > 0]
    df = df.drop_duplicates(subset=['seasonNumber', 'episodeNumber'], keep='last')
    grid = df.pivot(index="episodeNumber", columns="seasonNumber", values="averageRating")
    if main_rating == 0.0 and not df.empty: main_rating = round(df['averageRating'].mean(), 1)
    return grid, main_rating, source_msg

def get_metadata(imdb_id, quality="medium"):
    poster_url = None
    summary = ""
    try:
        url = f"https://api.tvmaze.com/lookup/shows?imdb={imdb_id}"
        r = requests.get(url, timeout=2) 
        if r.status_code == 200:
            d = r.json()
            poster_url = d.get("image", {}).get(quality) 
            summary = strip_html(d.get("summary", ""))
    except: pass
    return poster_url, summary

# ==========================================
# 🚀 PRO INTERFACE
# ==========================================

st.markdown('<p class="main-title">🔥 TV HEATMAP</p>', unsafe_allow_html=True)
if not os.path.exists(DB_FILE):
    st.error("⚠️ Database missing! Please run 'build_db.py' first.")
    st.stop()

query = st.text_input("", placeholder="🔍 Search for a show (e.g. Arcane, Breaking Bad)...")

if query:
    results = search_shows(query)
    if results.empty:
        st.warning(f"No shows found for '{query}'.")
    else:
        st.markdown("### Select Show:")
        cols = st.columns(3)
        for i, (idx, row) in enumerate(results.iterrows()):
            with cols[i % 3]:
                label = f"{row['primaryTitle']} ({row['startYear']})"
                if st.button(label, key=f"btn_{row['tconst']}", use_container_width=True):
                    st.session_state['selected_show'] = row.to_dict()

if 'selected_show' in st.session_state:
    row = st.session_state['selected_show']
    target_id = row['tconst']
    
    st.divider()
    
    poster_url, summary = get_metadata(target_id, quality="original")
    
    hero_col1, hero_col2 = st.columns([1, 2])
    
    with hero_col1:
        if poster_url:
            st.image(poster_url, use_container_width=True)
        else:
            st.markdown("📺 No Poster")
            
    with hero_col2:
        st.markdown(f"# {row['primaryTitle']}")
        st.markdown(f"#### {row['startYear']} • {row['genres']}")
        
        c1, c2 = st.columns(2)
        do_db = c1.button("⚡ Fast (DB)", key="act_db", use_container_width=True)
        do_live = c2.button("🌍 Live (Web)", key="act_live", use_container_width=True)
        
        use_live = False
        if do_live: use_live = True
        
        if summary:
            st.markdown(f"_{summary}_")

    if do_db or do_live or 'chart_generated' not in st.session_state:
        with st.spinner("Generating Heatmap..."):
            grid, rating, src_msg = get_show_data(target_id, force_live=use_live)
            if grid is not None:
                if "Live" in src_msg: st.success(f"✅ Data Source: {src_msg}")
                else: st.caption(f"ℹ️ Data Source: {src_msg}")
                
                poster_img = None
                if poster_url:
                    try:
                        resp = requests.get(poster_url)
                        poster_img = Image.open(BytesIO(resp.content)).convert("RGB")
                    except: pass
                
                final_img = render_page(grid, poster_img, row['primaryTitle'], row['startYear'], summary, rating)
                st.image(final_img, use_container_width=True)
                
                buf = BytesIO()
                final_img.save(buf, format="PNG")
                st.download_button("⬇️ Download High-Res Image", data=buf.getvalue(), file_name=f"{row['primaryTitle']}.png", mime="image/png", use_container_width=True)
                
                st.divider()
                st.subheader("You might also like:")
                rec_df = get_recommendations(target_id, row['genres'])
                
                if not rec_df.empty:
                    rcols = st.columns(4)
                    for idx, rec_row in rec_df.iterrows():
                        with rcols[idx]:
                            rec_poster, _ = get_metadata(rec_row['tconst'], quality="medium")
                            if rec_poster: st.image(rec_poster, use_container_width=True)
                            st.markdown(f"<div class='rec-title'>{rec_row['primaryTitle']}</div>", unsafe_allow_html=True)
                            st.caption(f"⭐ {rec_row['averageRating']}")
