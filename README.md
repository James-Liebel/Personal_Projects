# Personal Data Science & Engineering Portfolio

Monorepo layout: the **interactive portfolio** lives at **`projects/web/index.html`** (scripts, vendor, D3, Power BI assets are alongside it). Repo-root **`index.html`** redirects there so `/` on GitHub Pages still lands on the portfolio. Everything else stays under **`projects/`** (ML, games, tools, etc.).

Welcome to my personal portfolio repository. This collection houses a diverse range of projects spanning machine learning, game development, web visualization, and utility tools.

## 📂 Project Structure

This repository is organized as follows:

- **`projects/`**
  - **`machine_learning/`**: A comprehensive collection of ML models, including regressions, clustering algorithms, neural networks, and deep learning implementations.
  - **`finance/`**: Quantitative finance and mathematics notebooks.
  - **`games/`**: Python-based games (e.g., Pygame projects).
  - **`web/`**: Web visualizations and HTML-based projects (including Resume).
  - **`tools/`**: Automation scripts and utility tools.
  - **`sandbox/`**: Experimental code and rough drafts.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Recommended: `pip`, `virtualenv`

### Installation

1. Clone the repository (if not already local).
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** TensorFlow might need specific version handling depending on your Python version.

## 🛠 Featured Projects & Usage

### 🔐 Password Automizer
A secure CLI manager for storing encrypted passwords within a Google Doc.
- **Location:** `projects/tools/password_automizer.py`
- **Setup:**
   1. Obtain `credentials.json` from Google Cloud Console.
   2. Place `credentials.json` in the **root directory** of this project.
- **Usage:**
   ```bash
   python projects/tools/password_automizer.py --doc-id YOUR_DOC_ID
   ```

### 🧠 Machine Learning
Various models including Image Classification, Fraud Detection, and Housing Price prediction.
- **Location:** `projects/machine_learning/`
    - `image_classifier.ipynb`
    - `fraud_detection_setup.ipynb`
    - `housing_price_predictor.ipynb` (in `regression/`)
- **Notebooks:** Run Jupyter Notebook from the root or subfolders:
   ```bash
   jupyter notebook
   ```

### Deployment (full monorepo)
- **Portfolio URL:** **`/projects/web/`** (or **`/projects/web/index.html`**). Root **`/`** redirects there via repo-root `index.html`.
- **Local:** `npm install` then `npm start` → **http://localhost:3000** serves the repo; **`/`** is mapped to **`projects/web/index.html`** so it matches the canonical location.
- **GitHub Pages:** **Settings → Pages → GitHub Actions**. Workflow deploys the full tree (excludes `.git`, `node_modules`, `.venv`, `__pycache__`, `.cursor`).

### 🎮 Games
- **Traffic Simulator:** located at `projects/games/traffic_simulator.ipynb`.

## ⚠️ Notes
- `master.key` and `token.pickle` are generated locally for security tools. **Do not commit these.**
- Some scripts rely on specific Google API permissions.

## 📫 Contact
Created and maintained by James Liebel.
