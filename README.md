# Personal Data Science & Engineering Portfolio

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

### 🌐 Web & Resume
- **Resume:** Located at `projects/web/resume.html`.

### 🎮 Games
- **Traffic Simulator:** located at `projects/games/traffic_simulator.ipynb`.

## ⚠️ Notes
- `master.key` and `token.pickle` are generated locally for security tools. **Do not commit these.**
- Some scripts rely on specific Google API permissions.

## 📫 Contact
Created and maintained by James Liebel.
