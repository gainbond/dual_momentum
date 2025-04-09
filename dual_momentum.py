import yfinance as yf
import pandas as pd
import numpy as np
import os
import requests
from dotenv import load_dotenv

# 讀取環境變數
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    response = requests.post(url, data=data)
    return response.json()

# 設定 ETF 代號
symbols = ['VOO', 'SCZ', 'TLT']

# 抓取歷史資料（往回推 2 年以獲取足夠數據）
start_date = (pd.Timestamp.today() - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

def fetch_data(symbols, start_date, end_date):
    data_frames = []
    for symbol in symbols:
        df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)
        if not df.empty:
            df = df[['Close']].rename(columns={'Close': symbol})
            data_frames.append(df)
    
    if data_frames:
        return pd.concat(data_frames, axis=1)
    else:
        return pd.DataFrame()

# 下載數據
data = fetch_data(symbols, start_date, end_date)

# 確保 DataFrame 不是空的
if data.empty:
    raise ValueError("下載的數據為空，請檢查 yfinance 或網路連線。")

# 轉換為月度數據（取每月最後一個交易日）
data = data.resample('ME').last()

# 計算報酬率（1個月、3個月、6個月）
returns = data.pct_change()
returns['VOO_1m'] = returns['VOO']
returns['VOO_3m'] = data['VOO'].pct_change(3)
returns['VOO_6m'] = data['VOO'].pct_change(6)
returns['SCZ_1m'] = returns['SCZ']
returns['SCZ_3m'] = data['SCZ'].pct_change(3)
returns['SCZ_6m'] = data['SCZ'].pct_change(6)
returns['TLT_1m'] = returns['TLT']
returns.drop(columns=['VOO', 'SCZ', 'TLT'], inplace=True, errors='ignore')
returns.dropna(inplace=True)

# 計算平均動能
returns['VOO_momentum'] = returns[['VOO_1m', 'VOO_3m', 'VOO_6m']].mean(axis=1)
returns['SCZ_momentum'] = returns[['SCZ_1m', 'SCZ_3m', 'SCZ_6m']].mean(axis=1)

# 設定投資策略
def dual_momentum_strategy(returns):
    positions = []
    for index, row in returns.iterrows():
        voo_mom = row['VOO_momentum']
        scz_mom = row['SCZ_momentum']
        tlt_return = row['TLT_1m']
        
        # 確保數據為標量
        voo_mom = voo_mom.item()
        scz_mom = scz_mom.item()
        tlt_return = tlt_return.item()
        
        print(f"Date: {index}, VOO Momentum: {voo_mom:.2%}, SCZ Momentum: {scz_mom:.2%}, TLT Return: {tlt_return:.2%}")

        # 相對動能選擇
        if voo_mom > scz_mom:
            best_asset = 'VOO'
            best_momentum = voo_mom
        else:
            best_asset = 'SCZ'
            best_momentum = scz_mom
        
        # 絕對動能判斷
        if best_momentum > 0:
            positions.append(best_asset)
        elif tlt_return > 0:
            positions.append('TLT')
        else:
            positions.append('CASH')
    
    returns['Position'] = positions
    return returns

# 執行策略
returns = dual_momentum_strategy(returns)

# 檢查是否有變更，並發送 Telegram 通知
if len(returns) > 1:
    last_two_positions = returns['Position'].iloc[-2:].tolist()
    if last_two_positions[0] != last_two_positions[1]:
        message = f"🔔 雙動能策略通知\n\n上次選擇: {last_two_positions[0]}\n本次選擇: {last_two_positions[1]}"
        send_telegram_message(message)

# 顯示結果
print(returns[['Position']].tail(10))

