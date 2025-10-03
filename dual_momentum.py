#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

SYMBOLS = ['VOO', 'SCZ', 'TLT']

YEARS_BACK = 1
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# 每檔之間隨機睡秒數
TICKER_DELAY_MIN = 0.2
TICKER_DELAY_MAX = 0.8

MAX_RETRIES = 5
BASE_SLEEP = 2.0

DEBUG = True

def send_telegram_message(message: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("[WARN] Telegram 未設定，訊息如下：\n", message)
        return None
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    try:
        resp = requests.post(url, data=data, timeout=15)
        return resp.json()
    except Exception as e:
        print("[WARN] Telegram 通知失敗：", e)
        return None

def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; MomentumBot/1.3; +https://example.com)"
    })
    return s

def fetch_single_symbol(symbol: str, start_date: str, end_date: str,
                        session, max_retries: int = MAX_RETRIES, base_sleep: float = BASE_SLEEP) -> pd.DataFrame:
    def dl_with_period():
        return yf.download(symbol, period="450d", interval="1d",
                           auto_adjust=False, progress=False, threads=False, session=session)

    def dl_with_start_end():
        return yf.download(symbol, start=start_date, end=end_date, interval="1d",
                           auto_adjust=False, progress=False, threads=False, session=session)

    def dl_via_chart_api():
        # 直連 v8 chart，優先 query1，失敗換 query2
        hosts = ["https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com"]
        for host in hosts:
            url = f"{host}/v8/finance/chart/{symbol}?range=450d&interval=1d"
            try:
                r = session.get(url, timeout=12)
                if r.status_code != 200:
                    continue
                js = r.json()
                res = js.get("chart", {}).get("result", [])
                if not res:
                    continue
                r0 = res[0]
                ts = r0.get("timestamp", [])
                indicators = r0.get("indicators", {}).get("quote", [{}])[0]
                adj = r0.get("indicators", {}).get("adjclose", [{}])[0]
                if not ts or not indicators:
                    continue
                idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York")
                df = pd.DataFrame({
                    "Open": indicators.get("open"),
                    "High": indicators.get("high"),
                    "Low":  indicators.get("low"),
                    "Close":indicators.get("close"),
                    "Volume":indicators.get("volume"),
                }, index=idx)
                if adj and adj.get("adjclose"):
                    df["Adj Close"] = adj.get("adjclose")
                df = df.dropna(how="all")
                return df
            except Exception:
                continue
        return pd.DataFrame()

    for attempt in range(1, max_retries + 1):
        try:
            # 1) period 方式
            df = dl_with_period()
            if df is not None and not df.empty:
                return df

            # 2) start/end 備援
            df = dl_with_start_end()
            if df is not None and not df.empty:
                return df

            # 3) 直連 v8 chart
            df = dl_via_chart_api()
            if df is not None and not df.empty:
                return df

            raise RuntimeError("Empty DataFrame after period/start-end/api fallbacks")
        except Exception as e:
            sleep_s = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.75)
            print(f"[WARN] {symbol} 第{attempt}次下載失敗：{e}，{sleep_s:.1f}s 後重試")
            time.sleep(sleep_s)

    return pd.DataFrame()

def cache_path(symbol: str) -> str:
    return os.path.join(CACHE_DIR, f"{symbol}.csv")

def load_cache(symbol: str) -> pd.DataFrame:
    p = cache_path(symbol)
    if os.path.exists(p):
        try:
            df = pd.read_csv(p, parse_dates=["Date"], index_col="Date")
            return df.sort_index()
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def save_cache(symbol: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
    p = cache_path(symbol)
    df = df.sort_index()
    df.to_csv(p, index=True)

def fetch_with_cache(symbols, start_date, end_date, session) -> pd.DataFrame:
    frames = []
    failed = []

    for sym in symbols:
        cached = load_cache(sym)
        need_end = pd.to_datetime(end_date)

        if not cached.empty:
            last = cached.index.max()
            if last >= need_end - pd.Timedelta(days=1):
                df = cached
            else:
                df_new = fetch_single_symbol(sym, start_date, end_date, session=session)
                if df_new is not None and not df_new.empty:
                    df = pd.concat([cached, df_new]).sort_index().drop_duplicates()
                    save_cache(sym, df)
                else:
                    df = cached
        else:
            df = fetch_single_symbol(sym, start_date, end_date, session=session)
            if df is not None and not df.empty:
                save_cache(sym, df)

        if df is not None and not df.empty and 'Close' in df.columns:
            frames.append(df[['Close']].rename(columns={'Close': sym}))
        else:
            failed.append(sym)

        # 每檔之間短暫隨機延遲，降低限流機率
        time.sleep(random.uniform(TICKER_DELAY_MIN, TICKER_DELAY_MAX))

    if failed:
        send_telegram_message("⚠️ yfinance 抓取部分失敗（已降級處理）：\n" + "、".join(failed))

    return pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()

def clip_to_completed_months(daily_df: pd.DataFrame, monthly_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty or monthly_df.empty:
        return monthly_df

    last_daily = daily_df.index.max()
    if getattr(last_daily, "tzinfo", None) is not None:
        last_daily_naive = last_daily.tz_convert(None)
    else:
        last_daily_naive = last_daily

    current_month_end = last_daily_naive + pd.offsets.MonthEnd(0)
    prev_month_end = last_daily_naive + pd.offsets.MonthEnd(-1)

    m_idx = monthly_df.index
    if getattr(m_idx, "tz", None) is not None:
        m_idx_naive = m_idx.tz_convert(None)
    else:
        m_idx_naive = m_idx

    cutoff = prev_month_end if last_daily_naive < current_month_end else current_month_end
    mask = m_idx_naive <= cutoff
    return monthly_df[mask]

def main():
    session = make_session()

    today = pd.Timestamp.today().normalize()
    start_date = (today - pd.DateOffset(years=YEARS_BACK)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')

    data = fetch_with_cache(SYMBOLS, start_date, end_date, session=session)

    if data.empty:
        send_telegram_message("❌ 動能策略：今日資料全數抓取失敗（可能為限流或網路問題），已跳過運行。")
        print("[ERROR] 全部 ticker 取得失敗，略過本次策略")
        return

    available_syms = [c for c in SYMBOLS if c in data.columns]
    data = data[available_syms].copy()

    monthly_raw = data.resample('ME').last()
    monthly = clip_to_completed_months(data, monthly_raw)

    if monthly.empty:
        print("[WARN] 月度資料在裁切後為空，可能因為原始資料只涵蓋到本月初。")
        return

    returns = pd.DataFrame(index=monthly.index)
    for sym in available_syms:
        returns[f'{sym}_1m'] = monthly[sym].pct_change(1)
        returns[f'{sym}_3m'] = monthly[sym].pct_change(3)
        returns[f'{sym}_6m'] = monthly[sym].pct_change(6)

    # 風險資產動能（VOO, SCZ）
    if 'VOO' in available_syms:
        cols = [c for c in ['VOO_1m', 'VOO_3m', 'VOO_6m'] if c in returns.columns]
        if cols:
            returns['VOO_momentum'] = returns[cols].mean(axis=1)
    if 'SCZ' in available_syms:
        cols = [c for c in ['SCZ_1m', 'SCZ_3m', 'SCZ_6m'] if c in returns.columns]
        if cols:
            returns['SCZ_momentum'] = returns[cols].mean(axis=1)

    positions = []
    for idx, row in returns.iterrows():
        voo_mom = row.get('VOO_momentum', np.nan)
        scz_mom = row.get('SCZ_momentum', np.nan)
        tlt_ret = row.get('TLT_1m', np.nan)

        if DEBUG:
            def p(x):
                return f"{(x*100):.2f}%" if pd.notna(x) and np.isfinite(x) else "N/A"
            idx_print = idx.tz_convert(None) if hasattr(idx, "tzinfo") and idx.tzinfo else idx
            print(f"Date: {idx_print}, VOO Momentum: {p(voo_mom)}, SCZ Momentum: {p(scz_mom)}, TLT Return: {p(tlt_ret)}")

        candidates = {}
        if pd.notna(voo_mom):
            candidates['VOO'] = float(voo_mom)
        if pd.notna(scz_mom):
            candidates['SCZ'] = float(scz_mom)

        best_asset, best_mom = None, -np.inf
        for k, v in candidates.items():
            if pd.notna(v) and v > best_mom:
                best_asset, best_mom = k, v

        if best_asset is not None and np.isfinite(best_mom) and best_mom > 0:
            positions.append(best_asset)
        else:
            if ('TLT' in available_syms) and ('TLT_1m' in returns.columns):
                tlt_val = float(tlt_ret) if pd.notna(tlt_ret) else np.nan
                positions.append('TLT' if np.isfinite(tlt_val) and tlt_val > 0 else 'CASH')
            else:
                positions.append('CASH')

    returns['Position'] = positions

    # 通知：只有部位變更才推送
    if len(returns) >= 2:
        last_two = returns['Position'].iloc[-2:].tolist()
        if last_two[0] != last_two[1]:
            msg = (
                "🔔 雙動能策略通知\n\n"
                f"上次選擇: {last_two[0]}\n本次選擇: {last_two[1]}"
            )
            send_telegram_message(msg)

    print(returns[['Position']].tail(10))

if __name__ == "__main__":
    main()
