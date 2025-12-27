# 📺 IMDb TV Heatmap Generator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://imdb-heatmap-generator.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated data visualization tool that maps the critical reception of TV series over their entire lifespan. This application generates high-fidelity, color-coded heatmaps of episode ratings, allowing users to visualize a show's "Golden Age" or "Seasonal Rot" at a glance.

**[👉 Live Demo](https://imdb-heatmap-generator.streamlit.app/)**

---

## 📖 Overview

The **TV Heatmap Generator** is a Streamlit-based application designed for data enthusiasts and TV buffs. Unlike standard charting tools that rely on basic plotting libraries, this project utilizes a **custom-built PIL (Python Imaging Library) rendering engine** to generate broadcast-quality static images.

It operates on a **Hybrid Data Architecture**:
1.  **Fast (DB):** Instant retrieval from a local optimized SQLite database.
2.  **Live (Web):** Real-time scraping of IMDb for the absolute latest episode ratings.

## ✨ Key Features

* **Custom Visualization Engine:** Does not use Matplotlib or Plotly. Generates pixel-perfect, high-resolution downloadable images with poster art integration.
* **Hybrid Data Fetching:** Seamlessly toggles between a local SQLite cache for speed and live web scraping for fresh data.
* **Smart Color Grading:** Dynamic color scales ranging from "Garbage" (Purple/Red) to "Awesome" (Green) based on IMDb user scores.
* **Content Discovery:** Recommendation engine powered by genre-matching algorithms.
* **"Pro" Interface:** Heavily customized CSS for a cinematic, dark-mode UI that overrides default Streamlit styling.

## 📸 Screenshots

| **Search & Discovery** | **Generated Heatmap** |
|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/8eb8ccf1-b334-4ea8-a7d8-e0a4f126034f" width="100%"> | <img src="https://github.com/user-attachments/assets/be0a3d91-4399-45f3-974b-90e3d45313d4" width="100%"> |
## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/) (with custom CSS injection)
* **Data Processing:** Pandas, NumPy, SQLite3
* **Web Scraping:** BeautifulSoup4, Requests, Fake-UserAgent
* **Image Rendering:** Pillow (PIL) - *Core engine for drawing the heatmaps*
* **API:** TVMaze API (used for high-res poster metadata)

## 🚀 Installation & Local Setup

To run this application locally, follow these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/imdb-heatmap-generator.git](https://github.com/yourusername/imdb-heatmap-generator.git)
cd imdb-heatmap-generator
```
### 2. Install Dependencies
Ensure you have the required libraries installed:
```bash
pip install streamlit pandas numpy requests beautifulsoup4 fake-useragent Pillow
```

### 3. Build the Database
The app relies on a local SQLite database for fast searching. Run the build script to initialize `tv_shows.db`.
```bash
python build_db.py
```
*(Note: Ensure you have the raw IMDb datasets or the scraper configured in `build_db.py` before running this step.)*

### 4. Run the App
```bash
streamlit run app.py
```

## 📂 Project Structure

```text
├── app.py              # Main application logic & UI rendering
├── build_db.py         # Script to initialize/update the SQLite database
├── tv_shows.db         # Local SQLite database (generated)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## 🧠 How It Works

1.  **User Input:** The user searches for a show (e.g., "Breaking Bad").
2.  **Data Retrieval:**
    * The app checks `tv_shows.db` for the show ID (`tconst`).
    * If "Live (Web)" is selected, it scrapes IMDb using a session-based request with header rotation to bypass basic bot detection.
3.  **Data Processing:** Ratings are normalized, and missing data points are handled gracefully.
4.  **Rendering:** The `render_page()` function creates a blank canvas and draws the heatmap pixel-by-pixel, calculating layout dynamically based on the number of seasons and episodes.
5.  **Output:** A high-quality PNG is displayed and offered for download.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements (e.g., adding Rotten Tomatoes support, interactive tooltips, etc.).

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👏 Acknowledgments

* **IMDb:** For the comprehensive rating data.
* **TVMaze:** For providing excellent API access to show posters and summaries.
