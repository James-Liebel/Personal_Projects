# Personal Projects Portfolio

Welcome to my personal projects repository! This collection showcases a variety of Python scripts, machine learning models, and automation tools.

## 🚀 Quick Start

### 1. Environment Setup
Ensure you have Python installed, then install the required dependencies for all projects:

```bash
pip install -r requirements.txt
```

> **Note:** The `tensorflow` library (used for Image Classifier) is commented out in `requirements.txt` as it may have compatibility issues with newer Python versions (3.14+). Please install it manually if you intend to run the Machine Learning notebooks:
> `pip install tensorflow`

### 2. Google Automation Setup
Several scripts in this repository (`Password_Automizer.py`, `Self_Typing_Demo.py`, and `click typer`) interact with Google Docs/Drive via the official API.

**Prerequisites:**
*   You must obtain a `credentials.json` file from the [Google Cloud Console](https://console.cloud.google.com/apis/credentials).
*   Place `credentials.json` in the **root directory** of this project.
*   **Note:** On the first run of any of these scripts, a browser window will open asking you to authenticate. A `token.pickle` (or similar) file will be created to store your session.

## 📂 Project Directory

### 🛠️ Automation & Utilities
*   **`Password_Automizer.py`**: A secure CLI manager for storing encrypted passwords within a Google Doc.
*   **`Self_Typing_Demo.py`**: A demonstration script that mimics realistic human typing patterns (typos, pauses, backtracks) directly into a Google Doc.
*   **`click typer/`**: A dedicated directory for a click-based typing automation tool.
*   **`resume.html`**: The HTML source for my resume.

### 🧠 Data Science & Machine Learning
*   **`Machine Learning/`**: Contains various models including:
    *   Image Classification (TensorFlow)
    *   Neural Networks for numerical data
    *   Sentiment Analysis on news text
*   **`Regression/`**: Projects focused on predictive modeling:
    *   Housing Price Predictor
    *   Healthcare cost regression
*   **`Visualizations/`**: Jupyter notebooks demonstrating data visualization techniques (Heatmaps, Scatter plots).
*   **`HTML visualizations/`**: Standalone HTML files rendering interactive charts.

### 📊 Math & Finance
*   **`Math and Finance/`**: Tools for financial analysis and mathematical simulations:
    *   Financial Calculator
    *   Graphing Calculator
    *   Statistics Simulation

### 🎮 Games
*   **`Games/`**: Fun interactive notebooks, including a Traffic Simulator.

---
*Created and maintained by James Liebel.*
