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

# Merging

This repo includes a small utility, Merge.FF.CSVs.py, that combines the monthly Fama–French 2×3 factor file with the Momentum file from Kenneth French’s Data Library so you can run six-factor analysis on aligned dates.

Get the two input files (monthly)

Download ‘F-F_Research_Data_5_Factors_2x3’ (Monthly) — CSV version.

Download ‘F-F_Momentum_Factor’ (Monthly) — CSV version.

# Command-line usage (PowerShell)
**1. Open PowerShell**

Press Start → type PowerShell → hit Enter.

You should see a black/blue PowerShell window.

**2. Go to the folder where the script is saved**

If you saved it into your Documents\Python Programs folder, type:

cd "C:\Users\Kiera\Documents\Python Programs"

(Replace the path with wherever Merge.FF.CSVs.py actually is for you.)

**3. Run the script with your two CSV files**

Example command:

python "C:\Users\Kiera\Documents\Python Programs\Fund Analysis Tool\Merge FF CSVs.py" "C:\Users\Kiera\Documents\Python Programs\Fund Analysis Tool\F-F_Research_Data_5_Factors_2x3.csv" "C:\Users\Kiera\Documents\Python Programs\Fund Analysis Tool\F-F_Momentum_Factor.csv" -o "C:\Users\Kiera\Documents\Python Programs\Fund Analysis Tool\Merged_FF5_MOM.csv"

Explanation:

Merge.FF.CSVs.py → runs the script

"F-F_Research_Data_5_Factors_2x3.csv" → your 5-factor file

"F-F_Momentum_Factor.csv" → the momentum factor file

-o "Merged_FF5_MOM.csv" → output filename (you can rename it if you want)

**4. If you get an Excel pop-up when inspecting the new merged file**

That popup is just Excel being Excel 😅 — it’s warning you that when it opens a CSV, it might automatically show large numbers (like dates written as 202307) in scientific notation (e.g., 2.02307E+05).

For your merged factor file, you should click Don’t Convert.

**Here’s why:**

The script already cleaned your dates into a proper YYYY-MM-DD format, so there shouldn’t be any big integer values left.

If you click Convert, Excel will still try to 'help' by auto-formatting columns, which can sometimes mess things up.

If you click Don’t Convert, Excel will show the raw values exactly as written in the CSV (no unwanted conversions).

**And importantly:**

This popup only affects how Excel displays the CSV, not how Python or the GUI reads it.

So even if you accidentally click Convert, no problem. The GUI will still read the file fine — it doesn’t use Excel.

**What the script does**

Parses and cleans French’s headers/footers.

Harmonises dates to monthly and takes the intersection of available months.

Normalises column names and keeps returns in decimal units.

**Outputs columns similar to:**

Date, Mkt-RF, SMB, HML, RMW, CMA, Mom, RF

(Date formatted as YYYY-MM.)

# Download the latest Windows executable from the **Releases** tab.

## Run without Python
1. Go to **Releases** (right side of the repo page).
2. Download the `Integrated Fund Analysis.exe`. file.
3. Double-click `Integrated Fund Analysis.exe`.

## Run with Python instead (developers)
```bash
pip install -r Libraries Required.txt
python "Integrated Fund Analysis.py"
