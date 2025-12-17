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
# 🛠️ FONT LOADER (CRITICAL FIX)
# ==========================================
def install_fonts():
    # Downloads a verified font so text never breaks
    url = "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Regular.ttf"
    if not os.path.exists("Roboto-Regular.ttf"):
        try:
            r = requests.get(url)
            with open("Roboto-Regular.ttf", "wb") as f:
                f.write(r.content)
        except: pass

install_fonts()

# ==========================================
# 🎨 VISUALIZATION ENGINE (Tight Crop Version)
# ==========================================

def get_font(size):
    try:
        return ImageFont.truetype("Roboto-Regular.ttf", size)
    except:
        return ImageFont.load_default()

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

def draw_text_centered(draw, x, y, text, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((x - w / 2, y - h / 2), text, font=font, fill=fill)

def render_page(grid_df, poster_img, title, year_range, summary, main_rating):
    # --- DYNAMIC DIMENSIONS (No more fixed huge width) ---
    BOX_SIZE = 60
    PADDING = 10
    LEFT_SIDE_WIDTH = 500  # Fixed width for the text/poster side
    
    num_seasons = len(grid_df.columns)
    num_episodes = len(grid_df.index)
    
    # Calculate exact grid size
    grid_w = num_seasons * (BOX_SIZE + PADDING)
    grid_h = num_episodes * (BOX_SIZE + PADDING)
    
    # Canvas Size = Left Side + Grid + Margins
    canvas_w = LEFT_SIDE_WIDTH + grid_w + 100
    canvas_h = max(800, grid_h + 250) # Ensure it's at least tall enough for the sidebar
    
    canvas = Image.new("RGB", (int(canvas_w), int(canvas_h)), (15, 15, 15))
    draw = ImageDraw.Draw(canvas)
    
    # Fonts
    f_title = get_font(50)
    f_body = get_font(24)
    f_num = get_font(20)

    # --- DRAW BACKGROUND POSTER (Faded) ---
    if poster_img:
        bg = poster_img.resize((int(canvas_w), int(canvas_h)), Image.LANCZOS)
        # Dark overlay
        overlay = Image.new("RGBA", bg.size, (0, 0, 0, 200))
        bg.paste(overlay, (0, 0), overlay)
        canvas.paste(bg, (0, 0))

    # --- LEFT SIDE (Info) ---
    curr_y = 50
    margin = 40
    
    # Title
    lines = textwrap.wrap(title, width=20)
    for line in lines:
        draw.text((margin, curr_y), line, font=f_title, fill=(255, 255, 255))
        curr_y += 55
        
    draw.text((margin, curr_y), f"({year_range})", font=f_body, fill=(200, 200, 200))
    curr_y += 50
    
    # Rating Star
    draw.text((margin, curr_y), f"⭐ {main_rating}/10", font=f_title, fill=(255, 215, 0))
    curr_y += 70
    
    # Summary
    summary_lines = textwrap.wrap(summary, width=40)
    for line in summary_lines[:10]: # Limit lines
        draw.text((margin, curr_y), line, font=f_num, fill=(180, 180, 180))
        curr_y += 25

    # --- RIGHT SIDE (The Grid) ---
    start_x = LEFT_SIDE_WIDTH + 20
    start_y = 100
    
    seasons = list(grid_df.columns)
    
    # Draw Column Headers (S1, S2...)
    for col_idx, s in enumerate(seasons):
        x = start_x + col_idx * (BOX_SIZE + PADDING) + BOX_SIZE/2
        draw_text_centered(draw, x, start_y - 30, f"S{s}", f_body, (255,255,255))

    # Draw Rows
    for row_idx, ep in enumerate(grid_df.index):
        y = start_y + row_idx * (BOX_SIZE + PADDING)
        
        # Row Label (E1, E2...)
        draw_text_centered(draw, start_x - 30, y + BOX_SIZE/2, f"E{ep}", f_num, (200,200,200))
        
        for col_idx, s in enumerate(seasons):
            x = start_x + col_idx * (BOX_SIZE + PADDING)
            val = grid_df.loc[ep, s]
            
            color = color_for_score(val)
            rect = [x, y, x + BOX_SIZE, y + BOX_SIZE]
            draw.rounded_rectangle(rect, radius=5, fill=color)
            
            # Score Text
            if pd.notna(val):
                txt_col = text_color_for_bg(color)
                draw_text_centered(draw, x + BOX_SIZE/2, y + BOX_SIZE/2, f"{val:.1f}", f_num, txt_col)

    return canvas

# ==========================================
# 🧠 BACKEND LOGIC (Same as before)
# ==========================================
def get_show_data(tconst, force_live=False):
    conn = sqlite3.connect(DB_FILE)
    source_msg = "Database"
    
    # Get Main Rating
    try:
        q = "SELECT averageRating FROM ratings WHERE tconst = ?"
        res = pd.read_sql_query(q, conn, params=(tconst,))
        main_rating = round(res.iloc[0]['averageRating'], 1) if not res.empty else 0.0
    except: main_rating = 0.0

    df = pd.DataFrame()
    
    # Try Live Fetch
    if force_live:
        try:
            ua = UserAgent()
            headers = {"User-Agent": ua.chrome}
            url = f"https://www.imdb.com/title/{tconst}/episodes"
            # (Simplified fetching logic for brevity - just scraping main page for demo)
            # You can keep your complex scraping if you want, but ensure it returns a DF
            # FALLBACK to DB for stability in this script
            pass 
        except: pass

    # Always use DB for stability right now
    q_eps = """
        SELECT e.seasonNumber, e.episodeNumber, r.averageRating 
        FROM episodes e JOIN ratings r ON e.tconst = r.tconst 
        WHERE e.parentTconst = ? AND e.seasonNumber > 0
        ORDER BY e.seasonNumber, e.episodeNumber
    """
    df = pd.read_sql_query(q_eps, conn, params=(tconst,))
    
    conn.close()
    
    if df.empty: return None, 0.0, "No Data"
    
    # Deduplicate
    df = df.drop_duplicates(subset=['seasonNumber', 'episodeNumber'], keep='last')
    
    # Create Pivot Grid
    grid = df.pivot(index="episodeNumber", columns="seasonNumber", values="averageRating")
    
    return grid, main_rating, source_msg

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

# Custom CSS to hide default elements
st.markdown("""<style>header {visibility: hidden;} .block-container {padding-top: 1rem;}</style>""", unsafe_allow_html=True)

st.title("🔥 TV Show Heatmap")

if not os.path.exists(DB_FILE):
    st.error("Database missing.")
    st.stop()

query = st.text_input("Search Show", placeholder="e.g. Game of Thrones")

if query:
    results = search_shows(query)
    if not results.empty:
        cols = st.columns(5)
        for i, row in results.iterrows():
            with cols[i%5]:
                if st.button(f"{row['primaryTitle']} ({row['startYear']})", key=row['tconst']):
                    st.session_state['selected'] = row['tconst']
                    st.session_state['title'] = row['primaryTitle']
                    st.session_state['year'] = row['startYear']

if 'selected' in st.session_state:
    st.divider()
    with st.spinner("Drawing Chart..."):
        # Get Data
        grid, rating, msg = get_show_data(st.session_state['selected'])
        
        if grid is not None:
            # Get Poster (Optional)
            poster = None
            try:
                r = requests.get(f"https://api.tvmaze.com/lookup/shows?imdb={st.session_state['selected']}")
                img_url = r.json().get('image', {}).get('original')
                if img_url:
                    poster = Image.open(BytesIO(requests.get(img_url).content)).convert("RGB")
            except: pass
            
            # RENDER
            final_img = render_page(grid, poster, st.session_state['title'], st.session_state['year'], "Summary unavailable", rating)
            
            # DISPLAY
            st.image(final_img, use_container_width=True)
