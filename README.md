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
| *Clean, dark-mode search interface with instant results.* | *High-res generated chart showing episode quality trends.* |
| ![Search UI](https://via.placeholder.com/400x200?text=Search+Interface) | ![Heatmap Example](https://via.placeholder.com/400x200?text=Generated+Heatmap) |

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
