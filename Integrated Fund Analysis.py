#!/usr/bin/env python3

import os, math, sys, time, shutil, re
from pathlib import Path
from datetime import date, datetime

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt

# =============================
# Shared config & helpers
# =============================

FIG_SIZE = (12, 7)
FIG_DPI = 150

def ask_yes_no(prompt: str) -> bool:
    while True:
        s = input(prompt).strip().lower()
        if s in ("y", "yes"): return True
        if s in ("n", "no"):  return False
        print("Please enter Y/Yes or N/No.")

def nice_percent(x: float, decimals: int = 2) -> str:
    return f"{x*100:.{decimals}f}%"

def parse_required_date(prompt: str) -> date:
    from dateutil import parser as dateparser
    while True:
        s = input(prompt).strip()
        try:
            return dateparser.parse(s).date()
        except Exception:
            print("Invalid date. Please use a format like 2020-01-31.")

def banner(title: str):
    terminal_width = shutil.get_terminal_size().columns
    print("=" * terminal_width)
    print(title.center(terminal_width))
    print("=" * terminal_width)

def _sanitize_name(s: str) -> str:
    bad = '<>:"/\\|?*'
    for ch in bad:
        s = s.replace(ch, "-")
    return re.sub(r"\s+", " ", s).strip()

def _save_fig(fig, out_dir: str | None, fname: str):
    if not out_dir:
        return
    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, _sanitize_name(fname))
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  • saved: {path}")
    except Exception as e:
        print(f"  ! could not save figure '{fname}': {e}")

# =============================
# -------- FUND PERFORMANCE ---
# =============================

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
    for cand in ["Close", "Adj Close", "close", "adj_close", "AdjClose"]:
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

def fp_fetch_valuations(symbol: str) -> dict:
    t = yf.Ticker(symbol)
    info = {}
    try:
        info = t.get_info()
    except Exception:
        try:
            info = t.info
        except Exception:
            info = {}
    def _clean(x):
        try:
            x = float(x)
            if 0 < x < 1e4: return x
        except Exception:
            pass
        return None
    def _norm_y(y):
        try:
            y = float(y)
        except Exception:
            return None
        if y < 0: return None
        if y > 1.0: return y / 100.0
        return y
    pe = info.get("trailingPE") or info.get("forwardPE")
    pb = info.get("priceToBook") or info.get("priceToBookRatio") or info.get("pbRatio")
    ps = (info.get("priceToSalesTrailing12Months") or info.get("priceToSales")
          or info.get("priceToSalesTTM") or info.get("psRatio"))
    provider_yield = _norm_y(info.get("dividendYield"))
    ttm_yield = None
    try:
        divs = t.dividends
        if divs is not None and not divs.empty:
            cutoff = pd.Timestamp.today(tz=divs.index.tz) - pd.Timedelta(days=365)
            ttm_cash = divs[divs.index >= cutoff].sum()
            px = t.history(period="1mo", auto_adjust=False, actions=False)
            if px is not None and not px.empty:
                last_price = float(px["Close"].dropna().iloc[-1])
                if last_price > 0 and ttm_cash > 0:
                    ttm_yield = float(ttm_cash / last_price)
    except Exception:
        pass
    if ttm_yield is not None and ttm_yield > 0:
        chosen_yield, yield_source = ttm_yield, "TTM"
    elif provider_yield is not None and provider_yield > 0:
        chosen_yield, yield_source = provider_yield, "provider"
    else:
        chosen_yield, yield_source = None, None
    return {
        "pe": _clean(pe),
        "pb": _clean(pb),
        "ps": _clean(ps),
        "div_yield": chosen_yield,
        "div_yield_source": yield_source,
        "as_of": pd.Timestamp.today().date()
    }

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

def fp_create_performance_summary_figure(symbol: str, display_name: str, metrics: dict, valuations: dict | None = None):
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
    vals = []
    if valuations is not None:
        if valuations.get("as_of"):
            vals.append(("As of", _s(valuations["as_of"])))
        vals += [("Current P/E", _s(_fmt_or_na(valuations.get("pe")))),
                 ("Current P/B", _s(_fmt_or_na(valuations.get("pb")))),
                 ("Current P/S", _s(_fmt_or_na(valuations.get("ps"))))]
        dy = valuations.get("div_yield"); src = valuations.get("div_yield_source")
        dy_txt = _p(dy) if isinstance(dy, (int, float)) else _s(dy)
        if src: dy_txt += f" ({src})"
        vals.append(("Dividend Yield", dy_txt))
    fig = plt.figure(figsize=FIG_SIZE, facecolor=page, constrained_layout=True)
    gs  = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[0.14, 0.62, 0.24])
    ax_title = fig.add_subplot(gs[0]); ax_title.set_facecolor(panel)
    ax_perf  = fig.add_subplot(gs[1]); ax_perf.set_facecolor(panel)
    ax_vals  = fig.add_subplot(gs[2]); ax_vals.set_facecolor(panel)
    for ax in (ax_title, ax_perf, ax_vals):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    ax_title.text(0.5, 0.55, f"{name} – Performance Summary",
                  ha="center", va="center", fontsize=fs_title, fontweight="bold")
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
    if vals:
        ax_vals.text(0.5, 0.90, "Valuations", ha="center", va="center", fontsize=fs_head, fontweight="bold")
        n = max(len(vals), 1); top = 0.78; step = (top - 0.16) / n; y = top
        for lab, val in vals:
            ax_vals.text(0.06, y, lab + ":", ha="left", va="center", fontsize=12, fontweight="bold")
            ax_vals.text(0.94, y, val, ha="right", va="center", fontsize=12)
            y -= step
    ax_vals.text(0.5, 0.07, "Data: Yahoo Finance (auto-adjusted). Volatility annualised. Returns exclude fees/taxes.",
                 ha="center", va="center", fontsize=9)
    return fig

def fp_plot_series(metrics: dict, symbol: str, display_name: str, valuations: dict | None = None, save_dir: str | None = None):
    cum = metrics["cum_returns"]
    name = display_name if display_name else symbol
    fig_cum = plt.figure(figsize=FIG_SIZE)
    plt.plot(cum.index, cum.values, label=f"{name} Cumulative Return (normalised)")
    plt.title(f"{name} – Cumulative Return"); plt.xlabel("Date"); plt.ylabel("Growth of £1"); plt.legend(); plt.tight_layout()
    _save_fig(fig_cum, save_dir, f"{name} — Cumulative Return.png")
    fig_perf = fp_create_performance_summary_figure(symbol, name, metrics, valuations)
    _save_fig(fig_perf, save_dir, f"{name} — Performance Summary.png")
    if save_dir:
        plt.close(fig_cum); plt.close(fig_perf)

def run_fund_performance_once(symbol: str, display_name: str, start_date: date, end_date: date, save_dir: str | None = None):
    full = fp_fetch_full_history(symbol)
    window = fp_slice_to_range(full, start_date, end_date)
    metrics    = fp_compute_metrics(window)
    valuations = fp_fetch_valuations(symbol)
    fp_plot_series(metrics, symbol, display_name, valuations, save_dir=save_dir)

# =============================
# -------- FACTOR REGRESSION --
# =============================

def load_ff5_csv(csv_path: str) -> pd.DataFrame:
    csv_path = Path(csv_path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV at: {csv_path}")
    df = pd.read_csv(csv_path, skiprows=3)
    first_col = df.columns[0]
    df = df[df[first_col].astype(str).str.len() == 6].copy()
    df.columns = ["Date", "MKT", "SMB", "HML", "RMW", "CMA", "RF"]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m") + pd.offsets.MonthEnd(0)
    df.set_index("Date", inplace=True)
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0
    return df

def pick_ff_csv_gui() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    try:
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True); root.lift(); root.focus_force()
        path = filedialog.askopenfilename(
            parent=root, title="Select Fama–French 5 Factors CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        root.destroy()
        return path if path else None
    except Exception:
        try: root.destroy()
        except Exception: pass
        return None

def run_regression_once(ff_csv_path: str, start_month: pd.Period, end_month: pd.Period,
                        ticker: str, fund_name: str | None = None, save_dir: str | None = None):
    if not fund_name: fund_name = ticker
    factors = load_ff5_csv(ff_csv_path).copy()
    ff_trim = factors.copy()
    ff_trim.index = ff_trim.index.to_period("M")
    ff_trim = ff_trim.loc[(ff_trim.index >= start_month) & (ff_trim.index <= end_month)]
    yf_start = start_month.to_timestamp(how='start')
    yf_end   = end_month.to_timestamp(how='end') + pd.offsets.Day(1)
    fund_data = yf.download(
        ticker, start=yf_start.strftime("%Y-%m-%d"), end=yf_end.strftime("%Y-%m-%d"),
        interval="1mo", auto_adjust=True, progress=False
    )
    if fund_data.empty:
        print(f"❌ No data found for {ticker} in the requested window."); return
    fund_returns = fund_data[["Close"]].pct_change().dropna()
    fund_returns.columns = ["Fund"]
    fund_returns.index = fund_returns.index.to_period("M")
    fund_returns = fund_returns.loc[(fund_returns.index >= start_month) & (fund_returns.index <= end_month)]
    if fund_returns.empty:
        print("❌ No fund returns left after applying the date window."); return
    overlap = fund_returns.index.intersection(ff_trim.index)
    if len(overlap) == 0:
        print("❌ No overlapping months between fund and factor data. Check CSV and dates."); return
    combined = pd.concat([fund_returns.loc[overlap], ff_trim.loc[overlap]], axis=1)
    combined = combined.dropna(subset=["Fund", "RF", "MKT", "SMB", "HML", "RMW", "CMA"])
    if combined.empty:
        print("❌ Overlap contained NaNs in required columns. Check CSV contents."); return
    combined["Excess Fund"] = combined["Fund"] - combined["RF"]
    X = sm.add_constant(combined[["MKT", "SMB", "HML", "RMW", "CMA"]])
    y = combined["Excess Fund"]
    model = sm.OLS(y, X).fit()
    params = model.params.rename({'const': 'Alpha'})
    alpha_color = 'green' if params['Alpha'] > 0 else 'red'
    colors = [alpha_color] + ['steelblue'] * (len(params) - 1)
    fig_bar, ax = plt.subplots(figsize=FIG_SIZE)
    params.plot(kind="bar", color=colors, ax=ax, title=f"{fund_name} — Alpha & Factor Exposures")
    ax.axhline(0, linewidth=0.8); ax.set_ylabel("Coefficient"); ax.grid(False)
    fig_bar.tight_layout()
    _save_fig(fig_bar, save_dir, f"{fund_name} — Alpha & Factor Exposures.png")
    fig = plt.figure(figsize=FIG_SIZE)
    ax2 = fig.add_axes([0, 0, 1, 1]); ax2.set_axis_off()
    fig.text(0.5, 0.97, f"{fund_name} — Regression Summary", fontsize=15, weight='bold', ha='center', va='top')
    summary_text = model.summary().as_text()
    fig.text(0.5, 0.5, summary_text, ha='center', va='center', family='DejaVu Sans Mono', fontsize=10, transform=fig.transFigure)
    _save_fig(fig, save_dir, f"{fund_name} — Regression Summary.png")
    if save_dir:
        plt.close(fig_bar); plt.close(fig)

# =============================
# -------- CONTACT SHEET GUI --
# =============================

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
                    "drawdown trough","recovery date","current p/e","current p/b","dividend yield")
    if any(tok in s for tok in stats_tokens): return "stats"
    if "alpha & factor exposures" in s or "factor exposures" in s or "exposures" in s: return "bar"
    if any(tok in s for tok in (" mkt","smb","hml","rmw","cma")) and "ols" not in s: return "bar"
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

def _classify_image(path: str) -> str:
    guess = _classify_by_filename(os.path.basename(path))
    if guess != "other": return guess
    try:
        img = _load_image(path); txt = _ocr_text(img); return _classify_text(txt, filename_hint=os.path.basename(path))
    except Exception:
        return "other"

def _draw_section_header(canvas: Image.Image, y: int, width: int, pad: int, title: str) -> int:
    header_h = 44; draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, y, width, y + header_h], fill=SECTION_HEADER_BG)
    font = _get_font(18); tw, th = _measure_text(draw, title, font)
    draw.text((pad, y + (header_h - th) // 2), title, fill=SECTION_HEADER_TXT, font=font)
    return y + header_h

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
    y = 0
    for g in GROUP_ORDER:
        thumbs = thumbs_by_group.get(g, [])
        if not thumbs: continue
        title = GROUP_TITLES[g]; y = _draw_section_header(sheet, y, width, pad, title)
        rows = max(1, (len(thumbs) + cols - 1) // cols); row_h = max((im.size[1] for im in thumbs), default=0)
        y += pad; x = pad; c = 0
        for im in thumbs:
            sheet.paste(im, (x, y)); c += 1
            if c % cols == 0: x = pad; y += row_h + pad
            else: x += max_col_w + pad
        if c % cols != 0: y += row_h + pad
    return sheet

def _make_contact_sheet(image_paths, cols=2, thumb_width=900, pad=16, bg=(248,248,248)):
    thumbs = [_resize_to_thumb(p, thumb_width) for p in image_paths]
    if not thumbs: raise RuntimeError("No images to process.")
    rows = (len(thumbs) + cols - 1) // cols
    col_w = max(im.size[0] for im in thumbs); row_h = max(im.size[1] for im in thumbs)
    sheet_w = cols * col_w + (cols + 1) * pad; sheet_h = rows * row_h + (rows + 1) * pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), color=bg)
    x = y = pad; c = 0
    for im in thumbs:
        sheet.paste(im, (x, y)); c += 1
        if c % cols == 0: x = pad; y += row_h + pad
        else: x += col_w + pad
    return sheet

class ContactSheetApp(tk.Tk):
    def __init__(self):
        super().__init__()
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
                guess = _classify_image(full)
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
            if list(groups.keys()) == ["other"]:
                sheet = _make_contact_sheet(groups["other"], cols=cols, thumb_width=tw, pad=16, bg=CONTACT_SHEET_BG)
            else:
                sheet = _make_grouped_contact_sheet(groups, cols=cols, thumb_width=tw, pad=16, bg=CONTACT_SHEET_BG)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create contact sheet:\n{e}")
            return
        save_path = filedialog.asksaveasfilename(
            title="Save Contact Sheet",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")]
        )
        if not save_path: return
        try:
            sheet.save(save_path, "PNG")
            messagebox.showinfo("Done", f"Saved contact sheet:\n{os.path.abspath(save_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save file:\n{e}")

# --- end: GUI embed ---

def launch_contact_sheet_gui():
    print(
        "\n========================================================\n"
        "Chart Contact Sheet Tool — Grouped with Manual Override\n"
        "========================================================\n"
        "Now with categories: Bar / Line / OLS / Performance Stats / Other.\n"
        "1) Choose a folder and tick images\n"
        "2) Correct the group with the radio buttons if needed\n"
        "3) Set per-row and thumbnail width\n"
        "4) Save As… one PNG; grouped sections are rendered side-by-side\n"
        "(Launching the GUI now...)\n"
        "--------------------------------------------------------\n"
    )
    ContactSheetApp().mainloop()

# =============================
# ---------- MAIN RUNNER ------
# =============================

def _available_date_range_for_ticker(ticker: str):
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

def main():
    banner("INTEGRATED FUND ANALYSIS TOOL")
    print("This tool runs Fund Performance, Factor Regression, and the Contact Sheet GUI.")
    print("You will only need to enter tickers, names, and dates once.\n")

    print("Enter one or more Yahoo tickers separated by commas (e.g., VWRL.L, IVV, VEVE.L).")
    tickers_str = input("Tickers: ").strip()
    if not tickers_str:
        print("No tickers entered. Exiting."); return
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]

    print("\nOptionally enter display names (same order), or press Enter to use tickers:")
    names_str = input("Names (comma-separated): ").strip()
    names = [n.strip() for n in names_str.split(",")] if names_str else []
    if names and len(names) != len(tickers):
        print("Name count doesn't match tickers — using tickers as names."); names = []

    start_date = parse_required_date("\nStart date (YYYY-MM-DD): ")
    end_date   = parse_required_date("End date   (YYYY-MM-DD): ")
    if end_date < start_date:
        print("End date is before start date. Exiting."); return

    # Align to common available window across tickers
    avail_starts, avail_ends = [], []
    for tkr in tickers:
        s, e = _available_date_range_for_ticker(tkr)
        if s and e: avail_starts.append(s); avail_ends.append(e)
    if avail_starts and avail_ends:
        common_start = max(start_date, max(avail_starts))
        common_end   = min(end_date,   min(avail_ends))
        if common_end < common_start:
            print("No overlapping date window across the selected tickers. Exiting."); return
        if (common_start != start_date) or (common_end != end_date):
            print(f"\nAligning dates across funds: using {common_start} to {common_end} to match data availability.")
        start_date, end_date = common_start, common_end

    # Derive monthly window for regression (based on aligned dates)
    start_month = pd.Period(pd.Timestamp(start_date).to_period("M"), freq="M")
    end_month   = pd.Period(pd.Timestamp(end_date).to_period("M"),   freq="M")

    # Optional: choose an output folder to auto-save all figures
    save_dir = None
    try:
        import tkinter as _tk
        from tkinter import filedialog as _fd
        root = _tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        if ask_yes_no("Would you like to choose a folder to auto-save all charts? [Y/N]: "):
            save_dir = _fd.askdirectory(parent=root, title="Select destination folder for charts") or None
        root.destroy()
    except Exception:
        save_dir = None

    print("\nOpening file dialog to select your Fama–French 5 Factors CSV…")
    ff_csv_path = pick_ff_csv_gui()
    if not ff_csv_path:
        ff_csv_path = input("(No file selected) Enter the full path to your Fama–French CSV: ").strip()
        if not ff_csv_path:
            print("No factor CSV provided. Exiting."); return

    # Run both analyses for each ticker
    for i, ticker in enumerate(tickers):
        name = names[i] if (i < len(names) and names) else ticker
        try:
            run_fund_performance_once(ticker, name, start_date, end_date, save_dir=save_dir)
        except Exception as e:
            print(f"[Performance] Error for {ticker}: {e}")
        try:
            run_regression_once(ff_csv_path, start_month, end_month, ticker, name, save_dir=save_dir)
        except Exception as e:
            print(f"[Regression] Error for {ticker}: {e}")

    if not save_dir:
        print("\nDisplaying all charts. Close the windows to continue...")
        plt.show()
    else:
        print("\nAutosave enabled — charts were saved and figure windows were suppressed.")

    # Launch the contact sheet GUI unchanged
    launch_contact_sheet_gui()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!"); sys.exit(0)
