# Integrated-Fund-Analysis
Integrated Fund Analysis is a Python-based tool that helps investors evaluate investment funds through data downloads, factor regressions and performance analysis. It is designed for evidence-based investing research and produces both graphical outputs and neatly formatted summaries.

The repository also includes a merging utility that combines the Fama–French 2×3 factor dataset with the Momentum data files (both downloadable from Kenneth French’s Data Library). This merged version allows for six-factor regression analysis within the same time window.

You can also use the Emerging Markets (EM) versions of the Fama-French factors. To do this, download the Emerging Markets 5 Factors (2x3) from Ken French’s Data Library and extract the monthly CSV. Be careful when labelling the regression analysis so you know whether you have regressed a fund against the developed or emerging Fama-French factors.

# Important! 
You will need to download Fama/French 5 Factors (2x3) and (optionally) the Momentum Factor (Mom) from Ken French's website and (optionally) merge the two files.
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

If you only wish to regress against the five factors, please download and use version 1.0.1. If you wish to use the six factors, use version 1.0.2. I would advise using the six-factor version to gain more insight.

# Features

📈 Fund Data Download — automatically fetches historical data from Yahoo Finance.

🔍 Factor Regression — runs OLS regressions against Fama–French and other common factors.

📊 Visualisation — generates bar and line charts, scatter plots, and contact-sheet comparisons.

🗂️ Performance Summaries — outputs cumulative returns, risk/return metrics, and other stats as PNG images.

🖼️ Contact Sheet Tool — group charts side-by-side for easy comparison.

🧰 Interactive Workflow — combines console prompts with a Tkinter GUI for chart management.

📝 Export Options — save analysis outputs as images for use in presentations or reports.

# Who it’s for

This tool is aimed at:

Investors curious about how their funds behave relative to academic factors.

Analysts who want reproducible charts and stats without building everything from scratch.

Students of finance who want a hands-on way to explore performance and factor models.

# Download the latest Windows executable from the **Releases** tab.

## Run without Python
1. Go to **Releases** (right side of the repo page).
2. Download the `Integrated Fund Analysis.exe`. file.
3. Double-click `Integrated Fund Analysis.exe`.

## Run with Python instead (developers)
```bash
pip install -r Libraries Required.txt
python "Integrated Fund Analysis.py"
