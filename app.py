import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import requests
import os, math, textwrap, re, json
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

# --- CONFIG ---
DB_FILE = "tv_shows.db"

# ==========================================
# 🛠️ FONT LOADER (Safe Version)
# ==========================================
def install_fonts():
    # Downloads Roboto to ensure consistent look
    fonts = {
        "Roboto-Regular.ttf": "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Regular.ttf",
        "Roboto-Bold.ttf": "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Bold.ttf"
    }
    for name, url in fonts.items():
        if not os.path.exists(name):
            try:
                r = requests.get(url, timeout=5)
                with open(name, "wb") as f:
                    f.write(r.content)
            except: pass

install_fonts()

# ==========================================
# 🎨 VISUALIZATION ENGINE (Auto-Sized)
# ==========================================

def load_font(name_list, size):
    priorities = ["Roboto-Regular.ttf", "Roboto-Bold.ttf"] + name_list
    for name in priorities:
        try:
            return ImageFont.truetype(name, size)
        except: continue
    return ImageFont.load_default()

def draw_text_rich(draw, pos, text, font, fill):
    # Simple draw without shadow to prevent complex PIL errors
    draw.text(pos, text, font=font, fill=fill)

def color_for_score(score):
    if score is None or (isinstance(score, float) and np.isnan(score)): return (54, 54, 54)
    s = float(score)
    if s <= 5.5: return (128, 0, 128)
    if s <= 6.9: return (220, 0, 0)
    if s <= 7.9: return (212, 175, 55)
    if s <= 8.5: return (144, 238, 144)
    return (0, 100, 0)

def render_page(grid_df, poster_img, title, year_range, summary, main_rating):
    # --- DYNAMIC DIMENSIONS ---
    # We calculate the EXACT size needed, no more "max(2200, ...)"
    
    # Fonts
    f_title = load_font(["arialbd.ttf"], 50)
    f_reg = load_font(["arial.ttf"], 24)
    f_num = load_font(["arialbd.ttf"], 20)
    
    # Layout Config
    BOX_SIZE = 60
    PADDING = 10
    LEFT_WIDTH = 500
    
    num_seasons = len(grid_df.columns)
    num_episodes = len(grid_df.index)
    
    # Calculate Grid Size
    grid_w = num_seasons * (BOX_SIZE + PADDING)
    grid_h = num_episodes * (BOX_SIZE + PADDING)
    
    # Final Canvas Size (Exact fit)
    canvas_w = LEFT_WIDTH + grid_w + 100
    canvas_h = max(700, grid_h + 200) # Minimum height 700 to fit sidebar
    
    # Create Canvas
    canvas = Image.new("RGB", (int(canvas_w), int(canvas_h)), (15, 15, 15))
    draw = ImageDraw.Draw(canvas)
    
    # --- DRAW POSTER BACKGROUND ---
    if poster_img:
        try:
            bg = poster_img.resize((int(canvas_w), int(canvas_h)), Image.LANCZOS)
            overlay = Image.new("RGBA", bg.size, (0, 0, 0, 210)) # Darker overlay for readability
            bg.paste(overlay, (0, 0), overlay)
            canvas.paste(bg, (0, 0))
        except: pass # If poster fails, keep black background

    # --- LEFT SIDE CONTENT ---
    x_margin = 40
    curr_y = 50
    
    # Title (Wrapped)
    wrapper = textwrap.TextWrapper(width=20)
    lines = wrapper.wrap(title)
    for line in lines:
        draw_text_rich(draw, (x_margin, curr_y), line, f_title, (255, 255, 255))
        curr_y += 60
        
    draw_text_rich(draw, (x_margin, curr_y), f"({year_range})", f_reg, (200, 200, 200))
    curr_y += 60
    
    # Rating
    draw_text_rich(draw, (x_margin, curr_y), f"⭐ {main_rating}/10", f_title, (255, 215, 0))
    curr_y += 80
    
    # Summary
    sum_wrap = textwrap.TextWrapper(width=45)
    sum_lines = sum_wrap.wrap(summary)[:10]
    for line in sum_lines:
        draw_text_rich(draw, (x_margin, curr_y), line, f_reg, (180, 180, 180))
        curr_y += 30

    # --- RIGHT SIDE GRID ---
    start_x = LEFT_WIDTH + 40
    start_y = 100
    
    seasons = list(grid_df.columns)
    
    # Headers
    for col_idx, s in enumerate(seasons):
        x = start_x + col_idx * (BOX_SIZE + PADDING)
        # Centering text roughly
        draw_text_rich(draw, (x + 15, start_y - 30), f"S{s}", f_reg, (255,255,255))

    # Grid
    for row_idx, ep in enumerate(grid_df.index):
        y = start_y + row_idx * (BOX_SIZE + PADDING)
        
        # Row Label
        draw_text_rich(draw, (start_x - 40, y + 15), f"E{ep}", f_num, (200,200,200))
        
        for col_idx, s in enumerate(seasons):
            x = start_x + col_idx * (BOX_SIZE + PADDING)
            val = grid_df.loc[ep, s]
            
            # Box
            color = color_for_score(val)
            draw.rounded_rectangle([x, y, x + BOX_SIZE, y + BOX_SIZE], radius=6, fill=color)
            
            # Score
            if pd.notna(val):
                txt = f"{val:.1f}"
                # Rough centering
                offset_x = 10 if len(txt) > 1 else 20
                draw_text_rich(draw, (x + offset_x, y + 18), txt, f_num, (0,0,0) if val > 6 else (255,255,255))

    return canvas

# ==========================================
# 🧠 BACKEND LOGIC
# ==========================================

def get_metadata(imdb_id):
    poster_url = None
    summary = ""
    try:
        url = f"https://api.tvmaze.com/lookup/shows?imdb={imdb_id}"
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            d = r.json()
            poster_url = d.get("image", {}).get("original")
            summary = re.sub(r'<[^>]*>', '', d.get("summary", "")).strip()
    except: pass
    return poster_url, summary

def get_show_data(tconst):
    conn = sqlite3.connect(DB_FILE)
    
    # Main Rating
    try:
        q = "SELECT averageRating FROM ratings WHERE tconst = ?"
        res = pd.read_sql_query(q, conn, params=(tconst,))
        main_rating = round(res.iloc[0]['averageRating'], 1) if not res.empty else 0.0
    except: main_rating = 0.0

    # Episodes
    q_eps = """
        SELECT e.seasonNumber, e.episodeNumber, r.averageRating 
        FROM episodes e JOIN ratings r ON e.tconst = r.tconst 
        WHERE e.parentTconst = ? AND e.seasonNumber > 0
        ORDER BY e.seasonNumber, e.episodeNumber
    """
    df = pd.read_sql_query(q_eps, conn, params=(tconst,))
    conn.close()
    
    if df.empty: return None, 0.0
    
    df = df.drop_duplicates(subset=['seasonNumber', 'episodeNumber'], keep='last')
    grid = df.pivot(index="episodeNumber", columns="seasonNumber", values="averageRating")
    
    return grid, main_rating

def search_shows(query):
    conn = sqlite3.connect(DB_FILE)
    sql = "SELECT tconst, primaryTitle, startYear FROM shows WHERE primaryTitle LIKE ? LIMIT 10"
    df = pd.read_sql_query(sql, conn, params=(f"%{query}%",))
    conn.close()
    return df

# ==========================================
# 🚀 APP UI
# ==========================================
st.set_page_config(layout="wide", page_title="TV Heatmap")
st.markdown("""<style>header {visibility: hidden;} .block-container {padding-top: 2rem;}</style>""", unsafe_allow_html=True)
st.title("🔥 TV Show Heatmap")

if not os.path.exists(DB_FILE):
    st.error("⚠️ Database missing.")
    st.stop()

query = st.text_input("Search Show", placeholder="e.g. Dark, Breaking Bad")

if query:
    results = search_shows(query)
    if not results.empty:
        st.write("### Select Show:")
        cols = st.columns(5)
        for i, (idx, row) in enumerate(results.iterrows()):
            with cols[i % 5]:
                label = f"{row['primaryTitle']} ({row['startYear']})"
                if st.button(label, key=row['tconst'], use_container_width=True):
                    st.session_state['selected'] = row['tconst']
                    st.session_state['title'] = row['primaryTitle']
                    st.session_state['year'] = row['startYear']

if 'selected' in st.session_state:
    st.divider()
    
    with st.spinner("Generating Chart..."):
        grid, rating = get_show_data(st.session_state['selected'])
        
        if grid is not None:
            # Metadata
            poster_url, summary = get_metadata(st.session_state['selected'])
            
            poster_img = None
            if poster_url:
                try:
                    resp = requests.get(poster_url, timeout=5)
                    poster_img = Image.open(BytesIO(resp.content)).convert("RGB")
                except: pass
            
            # Render
            final_img = render_page(grid, poster_img, st.session_state['title'], st.session_state['year'], summary, rating)
            
            # Display
            st.image(final_img, use_container_width=True)
            
            # Download
            buf = BytesIO()
            final_img.save(buf, format="PNG")
            st.download_button("⬇️ Download Image", data=buf.getvalue(), file_name="heatmap.png", mime="image/png")
