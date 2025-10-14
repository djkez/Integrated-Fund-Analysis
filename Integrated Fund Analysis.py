#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Fund Analysis — GUI Edition
--------------------------------------
Changes in this build (Oct 2025):
• Everything now runs from a modern tkinter GUI (no console prompts).
• New checkbox: “Display coefficients in percent” — toggles bar chart scale/labels only.
• Seamless handling of either FF 5-Factor (2x3) or merged 6-factor CSVs (momentum optional).
• Normalised factor scaling: CSVs in % are auto-converted to decimals; merged decimals are left as-is.
• Removed hard-coded 0.01 scaling of regression params — coefficients are unitless; alpha is monthly decimal.
• Keeps the existing Contact Sheet tool (launchable from the GUI).

Tip: If you previously built an EXE with PyInstaller, this file will work the same way.
"""

import os, math, sys, shutil, re, io
from pathlib import Path
from datetime import date, datetime

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt


def _fallback_parse_french(raw: str) -> pd.DataFrame:
    """
    Secondary robust parser for Ken French CSVs.
    It finds the factor header line (",Mkt-RF,SMB,HML,RMW,CMA,RF"), then
    gathers subsequent YYYYMM rows, and constructs a proper CSV with a Date column.
    """
    import io, re
    lines = raw.splitlines()
    hdr_idx = None
    for i, l in enumerate(lines):
        s = l.strip()
        if s.startswith(",") and "Mkt-RF" in s and ",RF" in s:
            hdr_idx = i
            break
    if hdr_idx is None:
        raise ValueError("Header not found in French CSV")
    # Collect contiguous YYYYMM rows
    rows = []
    for l in lines[hdr_idx+1:]:
        if re.match(r'^\s*\d{6}\s*,', l):
            rows.append(l.strip())
        else:
            break
    # Build minimal CSV
    cols = lines[hdr_idx].strip().lstrip(",").replace("\r"," ").replace("\n"," ")
    mini = "Date," + cols + "\n" + "\n".join(rows)
    df = pd.read_csv(io.StringIO(mini))
    if "Date" in df.columns:
        df = df.set_index("Date")
    return df


# =========================================
# --------- FACTOR CSV LOADING  -----------
# =========================================

def load_ff_csv_generic(csv_path: str) -> pd.DataFrame:
    """
    Robust loader for Fama–French monthly factors.
    Works with:
      1) Plain merged CSVs (header on first row, e.g. Date,Mkt-RF,SMB,...,RF,UMD/Mom/WML).
      2) Ken French library CSVs with a prose preamble and header later in the file.
    Output columns (decimals): MKT, SMB, HML, RMW, CMA, RF, [MOM].
    """
    import re
    from pathlib import Path
    p = Path(csv_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Could not find CSV at: {p}")

    raw = p.read_text(encoding="latin-1")

    def _finish(df):
        # Clean column headers (strip BOM/newlines/whitespace) and normalise common FF names
        df.columns = [str(c).replace('\ufeff','').replace('\r','').replace('\n','').strip() for c in df.columns]
        # Robust normalisation: map variants to canonical
        _canon = {}
        import re as _re
        for c in list(df.columns):
            key = _re.sub(r'[^A-Za-z0-9]', '', str(c)).upper()
            if key in ('MKTRF','MKT_RF','MKTMINUSRF','MKTR'):
                _canon[c] = 'MKT'
            elif key == 'SMB':
                _canon[c] = 'SMB'
            elif key == 'HML':
                _canon[c] = 'HML'
            elif key == 'RMW':
                _canon[c] = 'RMW'
            elif key == 'CMA':
                _canon[c] = 'CMA'
            elif key in ('RF','RFRATE','RISKFREE'):
                _canon[c] = 'RF'
            elif key in ('MOM','UMD','WML'):
                _canon[c] = 'MOM'
        if _canon:
            df = df.rename(columns=_canon)

        # Canonical names
        rename_map = {"Mkt-RF":"MKT","Mkt_RF":"MKT","MKT-RF":"MKT","MKTRF":"MKT",
                      "SMB":"SMB","HML":"HML","RMW":"RMW","CMA":"CMA","RF":"RF",
                      "Mom":"MOM","MOM":"MOM","UMD":"MOM","WML":"MOM"}
        df = df.rename(columns=rename_map)

        needed = ["MKT","SMB","HML","RMW","CMA","RF"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected factor columns: {missing}. Columns present: {list(df.columns)}")
        cols = needed + (["MOM"] if "MOM" in df.columns else [])

        # Scale %→decimals if needed
        def _maybe_scale(s):
            s = pd.to_numeric(s, errors="coerce")
            a = s.dropna().abs()
            if a.empty:
                return s
            q95 = a.quantile(0.95)
            # If values look like percentages (typical 0.3 to 5.0), divide by 100
            return s/100.0 if q95 is not None and q95 > 0.25 else s
        for c in cols:
            df[c] = _maybe_scale(df[c])

        # Date index to month-end DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            # Expect YYYYMM ints/strings
            idx = pd.to_datetime(df.index.astype(str), format="%Y%m", errors="coerce")
            if idx.isna().all():
                # Try day-first dates already on the index
                idx = pd.to_datetime(df.index.astype(str), dayfirst=True, errors="coerce")
            if idx.isna().all():
                # Try common 'Date' column moved to index
                for cand in ("Date","DATE","date","Month","YYYYMM","yyyymm","YearMonth","year_month"):
                    if cand in df.columns:
                        idx = pd.to_datetime(df[cand].astype(str), format="%Y%m", errors="coerce")
                        if idx.isna().all():
                            try:
                                idx = pd.to_datetime(df[cand], errors="coerce")
                            except Exception:
                                pass
                        if not idx.isna().all():
                            df = df.drop(columns=[cand])
                            break
            if idx.isna().all():
                raise ValueError("Could not parse dates in factor CSV (expect YYYYMM or a parseable date column).")
            df.index = idx
        df.index = df.index.to_period("M").to_timestamp("M")
        df = df.sort_index()
        return df[cols]

    # Try plain CSV first
    try:
        df_plain = pd.read_csv(io.StringIO(raw))
        # If the first column looks like a date without a header name, set it as index
        if df_plain.columns[0] not in df_plain.columns[1:] and df_plain.columns[0].lower() in ("date","month","yyyymm","yearmonth","unnamed: 0"):
            df_plain = df_plain.set_index(df_plain.columns[0])
        elif re.fullmatch(r"\\d{6}", str(df_plain.iloc[0,0])):
            # first column is YYYYMM but named something else
            df_plain = df_plain.set_index(df_plain.columns[0])
        return _finish(df_plain)
    except Exception:
        pass

    # Fallback: French preamble style -> find header line that contains Mkt-RF and RF
    hdr_idx = None
    for i, l in enumerate(raw.splitlines()):
        s = l.strip()
        if s.startswith(",") and "Mkt-RF" in s and ",RF" in s:
            hdr_idx = i
            break
    if hdr_idx is None:
        raise ValueError("Could not find a valid header for factors in the CSV.")
    lines = raw.splitlines()
    start = hdr_idx + 1
    block = []
    for l in lines[start:]:
        if re.match(r'^\\s*\\d{6}\\s*,', l):
            block.append(l.strip())
        else:
            break
    header = lines[hdr_idx].lstrip(",").strip()
    mini = header + "\\n" + "\\n".join(block)
    df_preamble = pd.read_csv(io.StringIO(mini), index_col=0)
    try:
        return _finish(df_preamble)
    except Exception:
        # Use secondary parser to build a proper Date column
        df2 = _fallback_parse_french(raw)
        return _finish(df2)


# =========================================
# -------- SHARED CONFIG & HELPERS --------
# =========================================

FIG_SIZE = (12, 7)
FIG_DPI = 150

def _save_fig(fig, out_dir: str | None, fname: str):
    if not out_dir:
        return
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  • saved: {path}")
    except Exception as e:
        print(f"  ! could not save figure '{fname}': {e}")

# =========================================
# -------- FUND PERFORMANCE (unchanged) ---
# =========================================

def _annualised_volatility(daily_returns: pd.Series, trading_days: int = 252) -> float:
    return daily_returns.std(skipna=True) * math.sqrt(trading_days)

def _compute_drawdown(cum_returns: pd.Series):
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1.0
    trough_idx = drawdown.idxmin()
    max_dd = float(drawdown.loc[trough_idx])
    peak_idx = cum_returns.loc[:trough_idx].idxmax()
    recovery_idx = None
    post = cum_returns.loc[trough_idx:]
    recovered = post[post >= cum_returns.loc[peak_idx]]
    if not recovered.empty:
        recovery_idx = recovered.index[0]
    return max_dd, peak_idx, trough_idx, recovery_idx

def _years_between(start_dt: datetime, end_dt: datetime) -> float:
    return (end_dt - start_dt).days / 365.25

def fp_fetch_full_history(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    df = t.history(period="max", auto_adjust=True, actions=False)
    if df is None or df.empty:
        df = yf.download(ticker, period="max", auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for '{ticker}'. Use a Yahoo Finance ticker.")
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(key=ticker, axis=1, level=0, drop_level=False)
        except Exception:
            df.columns = ["_".join([str(lvl) for lvl in tup if lvl]) for tup in df.columns]
    close_col = None
    for cand in ["Close", "Adj Close", "AdjClose", "adj_close", "close"]:
        if cand in df.columns:
            close_col = cand
            break
    if close_col is None:
        raise RuntimeError(f"Close column not found. Columns: {list(df.columns)}")
    out = (
        df[[close_col]]
        .rename(columns={close_col: "AdjClose"})
        .sort_index()
        .dropna(subset=["AdjClose"])
    )
    out.index = pd.to_datetime(out.index)
    return out

def fp_slice_to_range(df: pd.DataFrame, start_date: date, end_date: date):
    earliest = df.index.min().date()
    latest   = df.index.max().date()
    req_start, req_end = start_date, end_date
    if start_date < earliest:
        start_date = earliest
    if end_date > latest:
        end_date = latest
    if start_date > end_date:
        raise ValueError(f"Start date {req_start} is after end date {req_end}.")
    window = df.loc[str(start_date):str(end_date)].copy()
    if window.empty:
        raise RuntimeError("No data in the requested date window.")
    return window

def fp_compute_metrics(df_window: pd.DataFrame):
    prices = df_window["AdjClose"]
    start_price = float(prices.iloc[0])
    end_price   = float(prices.iloc[-1])
    total_return = (end_price / start_price) - 1.0
    rets = prices.pct_change().dropna()
    start_dt = df_window.index[0].to_pydatetime()
    end_dt   = df_window.index[-1].to_pydatetime()
    yrs = _years_between(start_dt, end_dt)
    cagr = float("nan") if yrs <= 0 else (1.0 + total_return) ** (1.0 / yrs) - 1.0
    vol  = float(_annualised_volatility(rets))
    cum  = (1.0 + rets).cumprod()
    max_dd, peak_dt, trough_dt, rec_dt = _compute_drawdown(cum)
    return {
        "start_date": start_dt.date(),
        "end_date": end_dt.date(),
        "years": yrs,
        "start_price": start_price,
        "end_price": end_price,
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": vol,
        "max_drawdown": max_dd,
        "dd_peak": peak_dt.date() if peak_dt is not None else None,
        "dd_trough": trough_dt.date() if trough_dt is not None else None,
        "dd_recovery": rec_dt.date() if rec_dt is not None else None,
        "cum_returns": cum,
        "daily_returns": rets,
        "prices": prices,
    }

def _fmt_or_na(x, pct=False, digits=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    if pct: return f"{float(x)*100:.{digits}f}%"
    return f"{float(x):.{digits}f}"

def fp_create_performance_summary_figure(symbol: str, display_name: str, metrics: dict):
    import matplotlib.gridspec as gridspec
    name = display_name if display_name else symbol
    page    = (242/255.0, 242/255.0, 242/255.0)
    panel   = (236/255.0, 236/255.0, 236/255.0)
    fs_title = 14; fs_lbl = 12; fs_val = 12; fs_head = 13; fs_foot = 9
    def _p(x, d=2): return "n/a" if (x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))) else f"{x*100:.{d}f}%"
    def _n(x, d=4): return "n/a" if (x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))) else f"{x:.{d}f}"
    def _s(x): return "n/a" if x is None else str(x)
    left = [("Period", f"{metrics['start_date']} to {metrics['end_date']}"),
            ("Length", f"{metrics['years']:.2f} years"),
            ("Start Price", _n(metrics['start_price'])),
            ("End Price", _n(metrics['end_price']))]
    right = [("Total Return", _p(metrics["total_return"])),
             ("CAGR", "n/a" if math.isnan(metrics["cagr"]) else _p(metrics["cagr"])),
             ("Volatility (ann.)", _p(metrics["ann_vol"])),
             ("Max Drawdown", _p(metrics["max_drawdown"])),
             ("DD Peak", _s(metrics["dd_peak"])),
             ("DD Trough", _s(metrics["dd_trough"])),
             ("Recovery Date", _s(metrics["dd_recovery"]))]
    fig = plt.figure(figsize=(12,7), facecolor=page, constrained_layout=True)
    gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[0.18, 0.82])
    ax_title = fig.add_subplot(gs[0]); ax_title.set_facecolor(panel)
    ax_perf  = fig.add_subplot(gs[1]); ax_perf.set_facecolor(panel)
    for ax in (ax_title, ax_perf):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    ax_title.text(0.5, 0.55, f"{name} — Performance Summary",
                  ha="center", va="center", fontsize=fs_title, fontweight="bold")
    # Left/Right columns
    import matplotlib.gridspec as gridspec2
    gsp = gridspec2.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.04)
    ax_l = fig.add_subplot(gsp[0]); ax_r = fig.add_subplot(gsp[1])
    for a in (ax_l, ax_r):
        a.set_facecolor(panel); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_visible(False)
    def draw_rows(ax, rows, x_lab=0.06, x_val=0.94, top=0.90):
        n = max(len(rows), 1); step = (top - 0.08) / n; y = top
        for lab, val in rows:
            ax.text(x_lab, y, lab + ":", ha="left", va="center", fontsize=12, fontweight="bold")
            ax.text(x_val, y, val, ha="right", va="center", fontsize=12)
            y -= step
    draw_rows(ax_l, left); draw_rows(ax_r, right)
    fig.text(0.5, 0.02, "Data: Yahoo Finance (auto-adjusted). Volatility annualised. Returns exclude fees/taxes.", ha="center", va="center", fontsize=fs_foot, transform=fig.transFigure)
    return fig

def fp_plot_series(metrics: dict, symbol: str, display_name: str, save_dir: str | None = None):
    cum = metrics["cum_returns"]
    name = display_name if display_name else symbol
    fig_cum = plt.figure(figsize=(12,7))
    plt.plot(cum.index, cum.values, label=f"{name} Cumulative Return (normalised)")
    plt.title(f"{name} — Cumulative Return"); plt.xlabel("Date"); plt.ylabel("Growth of £1"); plt.legend(); plt.tight_layout()
    _save_fig(fig_cum, save_dir, f"{name} — Cumulative Return.png")
    fig_perf = fp_create_performance_summary_figure(symbol, name, metrics)
    _save_fig(fig_perf, save_dir, f"{name} — Performance Summary.png")
    if save_dir:
        plt.close(fig_cum); plt.close(fig_perf)

def run_fund_performance_once(symbol: str, display_name: str, start_date: date, end_date: date, save_dir: str | None = None):
    full = fp_fetch_full_history(symbol)
    window = fp_slice_to_range(full, start_date, end_date)
    metrics = fp_compute_metrics(window)
    fp_plot_series(metrics, symbol, display_name, save_dir=save_dir)

# =========================================
# --------- FACTOR REGRESSION -------------
# =========================================

def run_regression_once(ff_csv_path: str, start_month: pd.Period, end_month: pd.Period,
                        ticker: str, fund_name: str | None = None, save_dir: str | None = None,
                        coeff_in_percent: bool = False):
    """
    coeff_in_percent: if True, multiply plotted coefficients by 100 and label axis accordingly.
                      This is a DISPLAY choice only; the regression is always done in decimals.
    """
    if not fund_name: fund_name = ticker
    factors = load_ff_csv_generic(ff_csv_path).copy()
    ff_trim = factors.copy()
    ff_trim.index = ff_trim.index.to_period("M")
    ff_trim = ff_trim.loc[(ff_trim.index >= start_month) & (ff_trim.index <= end_month)]

    # Pull monthly fund returns
    yf_start = start_month.to_timestamp(how='start')
    yf_end   = end_month.to_timestamp(how='end') + pd.offsets.Day(1)
    fund_data = yf.download(
        ticker, start=yf_start.strftime("%Y-%m-%d"), end=yf_end.strftime("%Y-%m-%d"),
        interval="1mo", auto_adjust=True, progress=False
    )
    if fund_data.empty:
        print(f"❌ No data found for {ticker} in the requested window."); return
    price_col = "Adj Close" if "Adj Close" in fund_data.columns else ("Close" if "Close" in fund_data.columns else None)
    if price_col is None:
        print(f"❌ Could not find a price column (Adj Close/Close) for {ticker}. Columns: {list(fund_data.columns)}"); return
    fund_returns = fund_data[[price_col]].pct_change().dropna()
    fund_returns.columns = ["Fund"]
    fund_returns.index = fund_returns.index.to_period("M")
    fund_returns = fund_returns.loc[(fund_returns.index >= start_month) & (fund_returns.index <= end_month)]
    if fund_returns.empty:
        print("❌ No fund returns left after applying the date window."); return

    # Overlap & model
    overlap = fund_returns.index.intersection(ff_trim.index)
    if len(overlap) == 0:
        print("❌ No overlapping months between fund and factor data. Check CSV and dates."); return
    combined = pd.concat([fund_returns.loc[overlap], ff_trim.loc[overlap]], axis=1)
    optional_mom = ["MOM"] if "MOM" in ff_trim.columns else []
    combined = combined.dropna(subset=["Fund", "RF", "MKT", "SMB", "HML", "RMW", "CMA"] + optional_mom)
    if combined.empty:
        print("❌ Overlap contained NaNs in required columns. Check CSV contents."); return

    combined["Excess Fund"] = combined["Fund"] - combined["RF"]
    factor_cols = ["MKT", "SMB", "HML", "RMW", "CMA"] + optional_mom
    X = sm.add_constant(combined[factor_cols])
    y = combined["Excess Fund"]
    model = sm.OLS(y, X).fit()
    print(f"✔ Regression complete for {ticker} ({len(y)} months). Factors used: {factor_cols}")

    # IMPORTANT: params are already in decimal space; do NOT scale by 0.01 here.
    params = model.params.rename({'const': 'Alpha'})
    plot_params = params.copy()
    y_label = "Coefficient (decimal)"
    if coeff_in_percent:
        plot_params = plot_params * 100.0
        y_label = "Coefficient (percent)"

    alpha_color = 'green' if params['Alpha'] > 0 else 'red'
    colors = [alpha_color] + ['steelblue'] * (len(params) - 1)

    # Bar chart
    fig_bar, ax = plt.subplots(figsize=FIG_SIZE)
    title_suffix = " (incl. Momentum)" if ("MOM" in factor_cols) else ""
    plot_params.plot(kind="bar", color=colors, ax=ax,
                     title=f"{fund_name} — Alpha & Factor Exposures{title_suffix}")
    ax.axhline(0, linewidth=0.8)
    ax.set_ylabel(y_label)
    ax.grid(False)
    fig_bar.tight_layout()
    _save_fig(fig_bar, save_dir, f"{fund_name} — Alpha & Factor Exposures.png")

    # Text summary as image
    fig = plt.figure(figsize=FIG_SIZE)
    ax2 = fig.add_axes([0, 0, 1, 1]); ax2.set_axis_off()
    fig.text(0.5, 0.97, f"{fund_name} — Regression Summary", fontsize=15, weight='bold',
             ha='center', va='top')
    summary_text = model.summary().as_text()
    fig.text(0.5, 0.5, summary_text, ha='center', va='center',
             family='DejaVu Sans Mono', fontsize=10, transform=fig.transFigure)
    _save_fig(fig, save_dir, f"{fund_name} — Regression Summary.png")

    if save_dir:
        plt.close(fig_bar); plt.close(fig)

# =========================================
# -------- CONTACT SHEET (as before) ------
# =========================================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageOps, ImageDraw, ImageFont

TRY_OCR = True
try:
    import pytesseract
except Exception:
    TRY_OCR = False

CONTACT_SHEET_BG = (248, 248, 248)
SECTION_HEADER_BG = (230, 230, 230)
SECTION_HEADER_TXT = (20, 20, 20)
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

GROUP_TITLES = {
    "bar":     "Factor Exposure Charts (Bar)",
    "line":    "Time-Series / Cumulative Charts (Line)",
    "summary": "OLS Regression Summaries",
    "stats":   "Performance Statistics Panels",
    "other":   "Other",
}
GROUP_ORDER = ["bar", "line", "summary", "stats", "other"]

def _load_image(path):
    im = Image.open(path)
    if getattr(im, "n_frames", 1) > 1:
        im.seek(0)
    return im.convert("RGB")

def _resize_to_thumb(path, thumb_width=900):
    im = _load_image(path)
    w, h = im.size
    scale = thumb_width / float(w)
    return im.resize((thumb_width, int(h * scale)), Image.LANCZOS)

def _get_font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()

def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font); return (r - l), (b - t)
    elif hasattr(draw, "textsize"):
        return draw.textsize(text, font=font)
    else:
        return (len(text) * 8, int(getattr(font, "size", 18)))

def _ocr_text(img: Image.Image) -> str:
    if not TRY_OCR: return ""
    gray = ImageOps.grayscale(img); gray = ImageOps.autocontrast(gray, cutoff=2)
    try: return pytesseract.image_to_string(gray)
    except Exception: return ""

def _classify_text(t: str, filename_hint: str = "") -> str:
    s = (t + " " + filename_hint).lower()
    if ("ols regression results" in s) or ("dep. variable" in s and "coef" in s): return "summary"
    stats_tokens = ("performance summary","summary for","period:","cagr","max drawdown","volatility","drawdown peak",
                    "drawdown trough","recovery date")
    if any(tok in s for tok in stats_tokens): return "stats"
    if "alpha & factor exposures" in s or "factor exposures" in s or "exposures" in s: return "bar"
    if any(tok in s for tok in (" mkt","smb","hml","rmw","cma"," mom "," umd "," wml ")) and "ols" not in s: return "bar"
    if ("cumulative return" in s or "cumulative" in s or "growth of" in s or "return" in s): return "line"
    if any(tag in s for tag in ("bar","exposure","factor")): return "bar"
    if any(tag in s for tag in ("line","cum","timeseries","return","perf","equity")): return "line"
    if any(tag in s for tag in ("summary","ols","regression")): return "summary"
    if any(tag in s for tag in ("stats","statistics","metrics")): return "stats"
    return "other"

def _classify_by_filename(name: str) -> str:
    s = name.lower()
    if "alpha" in s and "factor" in s and "exposure" in s: return "bar"
    if "regression summary" in s or "ols" in s: return "summary"
    if "performance summary" in s or "stats" in s: return "stats"
    if "cumulative" in s or "growth of" in s or "return" in s: return "line"
    return "other"

def _make_grouped_contact_sheet(groups: dict, cols=2, thumb_width=900, pad=16, bg=(248,248,248)):
    thumbs_by_group = {}; max_col_w = 0
    for g, paths in groups.items():
        thumbs = []
        for p in paths:
            try:
                im = _resize_to_thumb(p, thumb_width); thumbs.append(im); max_col_w = max(max_col_w, im.size[0])
            except Exception: pass
        thumbs_by_group[g] = thumbs

    header_h = 44; total_h = 0; width = cols * max_col_w + (cols + 1) * pad
    for g in GROUP_ORDER:
        thumbs = thumbs_by_group.get(g, [])
        if not thumbs: continue
        rows = max(1, (len(thumbs) + cols - 1) // cols)
        row_h = max((im.size[1] for im in thumbs), default=0)
        total_h += header_h + rows * row_h + (rows + 1) * pad
    if total_h == 0: raise RuntimeError("No images to process.")
    sheet = Image.new("RGB", (width, total_h), color=bg)

    from PIL import ImageDraw as _ImageDraw
    draw = _ImageDraw.Draw(sheet)
    y = 0
    def _draw_section_header(y, title: str):
        header_h = 44
        draw.rectangle([0, y, width, y + header_h], fill=SECTION_HEADER_BG)
        font = _get_font(18)
        tw, th = _measure_text(draw, title, font)
        draw.text((16, y + (header_h - th) // 2), title, fill=SECTION_HEADER_TXT, font=font)
        return y + header_h

    for g in GROUP_ORDER:
        thumbs = thumbs_by_group.get(g, [])
        if not thumbs: continue
        title = {"bar":"Factor Exposure Charts (Bar)","line":"Time-Series / Cumulative Charts (Line)",
                 "summary":"OLS Regression Summaries","stats":"Performance Statistics Panels","other":"Other"}[g]
        y = _draw_section_header(y, title)
        rows = max(1, (len(thumbs) + cols - 1) // cols); row_h = max((im.size[1] for im in thumbs), default=0)
        y += 16; x = 16; c = 0
        for im in thumbs:
            sheet.paste(im, (x, y)); c += 1
            if c % cols == 0: x = 16; y += row_h + 16
            else: x += max_col_w + 16
        if c % cols != 0: y += row_h + 16
    return sheet

class ContactSheetApp(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master=master)
        self.title("Chart Contact Sheet — Grouped (with Manual Override)")
        self.geometry("900x660"); self.minsize(780, 540)
        self.folder = tk.StringVar(); self.per_row = tk.IntVar(value=2)
        self.thumb_width = tk.IntVar(value=900); self.group_similar = tk.BooleanVar(value=True)
        self.items = []
        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self, padding=10); top.pack(fill="x")
        ttk.Label(top, text="1) Choose folder with your chart screenshots").pack(anchor="w")
        row = ttk.Frame(top); row.pack(fill="x", pady=(5,10))
        ttk.Entry(row, textvariable=self.folder).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Choose Folder", command=self.choose_folder).pack(side="left", padx=6)
        ttk.Button(row, text="Reload", command=self.load_images).pack(side="left")
        ttk.Label(top, text="2) Tick images and (optionally) correct group").pack(anchor="w")
        self.canvas = tk.Canvas(self, borderwidth=0)
        self.scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.listframe = ttk.Frame(self.canvas)
        self.listframe.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0,0), window=self.listframe, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.canvas.pack(side="left", fill="both", expand=True, padx=(10,0), pady=(0,10))
        self.scroll.pack(side="right", fill="y", padx=(0,10), pady=(0,10))
        bottom = ttk.Frame(self, padding=10); bottom.pack(fill="x")
        ttk.Label(bottom, text="3) Charts per row:").pack(side="left")
        ttk.Spinbox(bottom, from_=1, to=8, textvariable=self.per_row, width=4).pack(side="left", padx=(6,12))
        ttk.Label(bottom, text="Thumb width:").pack(side="left")
        ttk.Spinbox(bottom, from_=400, to=1600, increment=50, textvariable=self.thumb_width, width=6).pack(side="left", padx=(6,12))
        ttk.Checkbutton(bottom, text="Auto-group similar (you can override per file)", variable=self.group_similar).pack(side="left", padx=(0,12))
        ttk.Button(bottom, text="Select All", command=lambda: self._set_all(True)).pack(side="left")
        ttk.Button(bottom, text="Select None", command=lambda: self._set_all(False)).pack(side="left")
        ttk.Button(bottom, text="Create Contact Sheet", command=self.create_sheet).pack(side="right")

    def choose_folder(self):
        path = filedialog.askdirectory(title="Select folder with chart screenshots")
        if path:
            self.folder.set(path); self.load_images()

    def load_images(self):
        for w in self.listframe.winfo_children(): w.destroy()
        self.items.clear()
        folder = self.folder.get().strip()
        if not folder or not os.path.isdir(folder): return
        files = [f for f in sorted(os.listdir(folder)) if os.path.splitext(f)[1].lower() in VALID_EXTS]
        if not files:
            ttk.Label(self.listframe, text="No images found in this folder.", foreground="red").pack(anchor="w", padx=5, pady=5)
            return
        header = ttk.Frame(self.listframe); header.pack(fill="x", pady=(0,4))
        ttk.Label(header, text="Include", width=8).grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(header, text="File").grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(header, text="Group").grid(row=0, column=2, sticky="w", padx=5)
        for f in files:
            full = os.path.join(folder, f)
            checked = tk.BooleanVar(value=True)
            guess = _classify_by_filename(os.path.basename(full))
            if guess == "other":
                try:
                    img = _load_image(full)
                    txt = _ocr_text(img)
                    guess = _classify_text(txt, filename_hint=os.path.basename(full))
                except Exception:
                    guess = "other"
            grp = tk.StringVar(value=guess)
            row = ttk.Frame(self.listframe); row.pack(fill="x", pady=2)
            ttk.Checkbutton(row, variable=checked).grid(row=0, column=0, padx=5, sticky="w")
            ttk.Label(row, text=f, width=60).grid(row=0, column=1, sticky="w", padx=5)
            rb = ttk.Frame(row); rb.grid(row=0, column=2, sticky="w")
            for key, lbl in (("bar","Bar"),("line","Line"),("summary","OLS"),("stats","Stats"),("other","Other")):
                ttk.Radiobutton(rb, text=lbl, value=key, variable=grp).pack(side="left", padx=(0,6))
            self.items.append({"path": full, "checked": checked, "group": grp})

    def _set_all(self, state: bool):
        for it in self.items: it["checked"].set(state)

    def create_sheet(self):
        chosen = [it for it in self.items if it["checked"].get()]
        if len(chosen) < 2:
            messagebox.showwarning("Need more images", "Please select at least two images.")
            return
        cols = max(1, int(self.per_row.get() or 2))
        tw   = max(300, int(self.thumb_width.get() or 900))
        groups = {k: [] for k in GROUP_ORDER}
        if self.group_similar.get():
            for it in chosen:
                key = _classify_by_filename(os.path.basename(it["path"]))
                if key not in groups: key = "other"
                groups[key].append(it["path"])
        else:
            for it in chosen:
                key = it["group"].get().strip().lower()
                if key not in groups: key = "other"
                groups[key].append(it["path"])
        groups = {k: v for k, v in groups.items() if v}
        try:
            sheet = _make_grouped_contact_sheet(groups, cols=cols, thumb_width=tw, pad=16, bg=CONTACT_SHEET_BG)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create contact sheet:\\n{e}")
            return
        save_path = filedialog.asksaveasfilename(
            title="Save Contact Sheet",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")]
        )
        if not save_path: return
        try:
            sheet.save(save_path, "PNG")
            messagebox.showinfo("Done", f"Saved contact sheet:\\n{os.path.abspath(save_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save file:\\n{e}")

# =========================================
# ----------------- MAIN GUI --------------
# =========================================

class IntegratedFundAnalysisGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Integrated Fund Analysis")
        self.geometry("860x620")
        self.minsize(820, 580)

        # Vars
        self.tickers = tk.StringVar()
        self.names = tk.StringVar()
        self.start_date = tk.StringVar()
        self.end_date = tk.StringVar()
        self.ff_csv_path = tk.StringVar()
        self.save_dir = tk.StringVar()
        self.coeff_percent = tk.BooleanVar(value=False)
        self.run_performance = tk.BooleanVar(value=True)
        self.run_regression = tk.BooleanVar(value=True)

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 8}

        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)

        # Row 1: Tickers & Names
        r1 = ttk.Frame(frm); r1.pack(fill="x", **pad)
        ttk.Label(r1, text="Tickers (comma-separated):").grid(row=0, column=0, sticky="w")
        ttk.Entry(r1, textvariable=self.tickers).grid(row=0, column=1, sticky="we")
        ttk.Label(r1, text="Display names (optional, comma-separated):").grid(row=1, column=0, sticky="w")
        ttk.Entry(r1, textvariable=self.names).grid(row=1, column=1, sticky="we")
        r1.columnconfigure(1, weight=1)

        # Row 2: Dates
        r2 = ttk.Frame(frm); r2.pack(fill="x", **pad)
        ttk.Label(r2, text="Start date (YYYY-MM-DD):").grid(row=0, column=0, sticky="w")
        ttk.Entry(r2, textvariable=self.start_date, width=16).grid(row=0, column=1, sticky="w")
        ttk.Label(r2, text="End date (YYYY-MM-DD):").grid(row=0, column=2, sticky="w", padx=(16,0))
        ttk.Entry(r2, textvariable=self.end_date, width=16).grid(row=0, column=3, sticky="w")

        # Row 3: FF CSV and Save Dir
        r3 = ttk.Frame(frm); r3.pack(fill="x", **pad)
        ttk.Label(r3, text="Fama–French CSV (5 or 5+Momentum):").grid(row=0, column=0, sticky="w")
        ttk.Entry(r3, textvariable=self.ff_csv_path).grid(row=0, column=1, sticky="we")
        ttk.Button(r3, text="Browse…", command=self._pick_csv).grid(row=0, column=2, sticky="w", padx=(6,0))
        ttk.Label(r3, text="Autosave folder (optional):").grid(row=1, column=0, sticky="w")
        ttk.Entry(r3, textvariable=self.save_dir).grid(row=1, column=1, sticky="we")
        ttk.Button(r3, text="Choose…", command=self._pick_folder).grid(row=1, column=2, sticky="w", padx=(6,0))
        r3.columnconfigure(1, weight=1)

        # Row 4: Options
        r4 = ttk.Frame(frm); r4.pack(fill="x", **pad)
        ttk.Checkbutton(r4, text="Run Fund Performance", variable=self.run_performance).pack(side="left")
        ttk.Checkbutton(r4, text="Run Factor Regression", variable=self.run_regression).pack(side="left", padx=(12,0))
        ttk.Checkbutton(r4, text="Display coefficients in percent", variable=self.coeff_percent).pack(side="left", padx=(24,0))

        # Row 5: Buttons
        r5 = ttk.Frame(frm); r5.pack(fill="x", **pad)
        ttk.Button(r5, text="Run", command=self._run).pack(side="left")
        ttk.Button(r5, text="Open Contact Sheet Tool", command=self._open_contact_sheet).pack(side="left", padx=(8,0))

        # Footer
        r6 = ttk.Frame(frm); r6.pack(fill="x", **pad)
        ttk.Label(r6, text="Notes: The CSV loader auto-detects percent vs decimal factors. "
                           "The percent checkbox affects chart labels/scale only.").pack(anchor="w")

    def _pick_csv(self):
        path = filedialog.askopenfilename(title="Select Fama–French Factors CSV",
                                          filetypes=[("CSV files","*.csv"), ("All files","*.*")])
        if path: self.ff_csv_path.set(path)

    def _pick_folder(self):
        path = filedialog.askdirectory(title="Select folder to autosave charts (optional)")
        if path: self.save_dir.set(path)

    def _open_contact_sheet(self):
        ContactSheetApp(self)

    def _parse_dates(self):
        from dateutil import parser as dateparser
        try:
            s = dateparser.parse(self.start_date.get().strip()).date()
            e = dateparser.parse(self.end_date.get().strip()).date()
            if e < s:
                raise ValueError("End date is before start date.")
            return s, e
        except Exception as ex:
            messagebox.showerror("Invalid Dates", f"Please check your dates. Error: {ex}")
            return None, None

    def _available_date_range_for_ticker(self, ticker: str):
        try:
            t = yf.Ticker(ticker)
            df = t.history(period="max", auto_adjust=True, actions=False)
            if df is None or df.empty:
                df = yf.download(ticker, period="max", auto_adjust=True, progress=False)
            if df is None or df.empty:
                return None, None
            idx = pd.to_datetime(df.index)
            return idx.min().date(), idx.max().date()
        except Exception:
            return None, None

    def _run(self):
        tickers_str = self.tickers.get().strip()
        if not tickers_str:
            messagebox.showerror("Missing Tickers", "Please enter at least one Yahoo ticker (e.g., VWRL.L).")
            return
        tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]

        names_str = self.names.get().strip()
        names = [n.strip() for n in names_str.split(",")] if names_str else []
        if names and len(names) != len(tickers):
            names = []  # fallback to tickers

        ff_csv = self.ff_csv_path.get().strip()
        if not ff_csv:
            messagebox.showerror("Missing CSV", "Please select your Fama–French CSV (5 or 5+Momentum).")
            return

        s, e = self._parse_dates()
        if not s or not e:
            return

        # Align to common available window across tickers
        avail_starts, avail_ends = [], []
        for tkr in tickers:
            s0, e0 = self._available_date_range_for_ticker(tkr)
            if s0 and e0:
                avail_starts.append(s0); avail_ends.append(e0)
        if avail_starts and avail_ends:
            common_start = max(s, max(avail_starts))
            common_end   = min(e,   min(avail_ends))
            if common_end < common_start:
                messagebox.showerror("Date Window", "No overlapping date window across the selected tickers.")
                return
            s, e = common_start, common_end

        start_month = pd.Period(pd.Timestamp(s).to_period("M"), freq="M")
        end_month   = pd.Period(pd.Timestamp(e).to_period("M"),   freq="M")

        save_dir = self.save_dir.get().strip() or None
        percent_flag = bool(self.coeff_percent.get())

        self.config(cursor="watch"); self.update_idletasks()
        try:
            for i, tkr in enumerate(tickers):
                name = names[i] if (i < len(names) and names) else tkr
                if self.run_performance.get():
                    try:
                        run_fund_performance_once(tkr, name, s, e, save_dir=save_dir)
                    except Exception as ex:
                        print(f"[Performance] Error for {tkr}: {ex}")
                if self.run_regression.get():
                    try:
                        run_regression_once(ff_csv, start_month, end_month, tkr, name, save_dir=save_dir,
                                            coeff_in_percent=percent_flag)
                    except Exception as ex:
                        print(f"[Regression] Error for {tkr}: {ex}")
            if not save_dir:
                print("Opening figures window…")
                plt.show()
            else:
                messagebox.showinfo("Done", "Finished. Charts were auto-saved to your chosen folder.")
        finally:
            self.config(cursor="")

def main():
    app = IntegratedFundAnalysisGUI()
    app.mainloop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
