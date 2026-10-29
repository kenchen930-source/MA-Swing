import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title='波段均線選股器', layout='wide')
st.title('波段均線選股器 V1.1')
st.caption('獨立於 60KD｜上市＋上櫃｜日線收盤資料｜免費 Yahoo 行情，僅供候選股研究')
TZ = ZoneInfo('Asia/Taipei')

# TWSE/TPEx official industry classification (code or Chinese industry label).
INDUSTRY_CODES = {
    '01':'水泥工業', '02':'食品工業', '03':'塑膠工業', '04':'紡織纖維',
    '05':'電機機械', '06':'電器電纜', '08':'玻璃陶瓷', '09':'造紙工業',
    '10':'鋼鐵工業', '11':'橡膠工業', '12':'汽車工業', '14':'建材營造',
    '15':'航運業', '16':'觀光餐旅', '17':'金融保險', '18':'貿易百貨',
    '20':'其他業', '21':'化學工業', '22':'生技醫療', '23':'油電燃氣',
    '24':'半導體', '25':'電腦及週邊設備', '26':'光電', '27':'通信網路',
    '28':'電子零組件', '29':'電子通路', '30':'資訊服務', '31':'其他電子',
    '32':'文化創意', '33':'農業科技', '34':'電子商務', '35':'綠能環保',
    '36':'數位雲端', '37':'運動休閒', '38':'居家生活', '39':'其他',
}

def industry_of(entry):
    keys = ('產業別', '產業別名稱', '產業名稱', '產業類別',
            'Industry', 'IndustryCode', 'IndustryName', 'SecuritiesIndustryCode')
    raw = next((str(entry[k]).strip() for k in keys if entry.get(k) is not None and str(entry[k]).strip()), '')
    if not raw:
        return '未分類'
    if raw.isdigit():
        return INDUSTRY_CODES.get(raw.zfill(2), f'其他分類（{raw}）')
    raw = raw.replace('產業', '').replace('工業', '工業') if raw.endswith('產業') else raw
    return raw

@st.cache_data(ttl=86400, show_spinner=False)
def stock_universe():
    sources = [('上市', 'https://openapi.twse.com.tw/v1/opendata/t187ap03_L', '.TW'),
               ('上櫃', 'https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O', '.TWO')]
    items, errors = [], []
    for market, url, suffix in sources:
        try:
            r = requests.get(url, timeout=25, headers={'User-Agent':'Mozilla/5.0'})
            r.raise_for_status()
            payload = r.json()
            if not isinstance(payload, list):
                raise ValueError('清單格式異常')
            count = 0
            for entry in payload:
                code = str(next((entry[k] for k in ('公司代號', 'SecuritiesCompanyCode', 'CompanyCode', 'Code') if entry.get(k)), '')).strip()
                name = str(next((entry[k] for k in ('公司簡稱', '公司名稱', 'CompanyName', 'SecuritiesCompanyName', 'Name') if entry.get(k)), code)).strip()
                if re.fullmatch(r'\d{4}', code):
                    items.append({'代號':code, '名稱':name, '市場':market, '產業':industry_of(entry), 'ticker':code+suffix})
                    count += 1
            if count < 100:
                raise ValueError(f'僅辨識 {count} 檔')
        except Exception as e:
            errors.append(f'{market}清單失敗：{e}')
    return list({x['ticker']:x for x in items}.values()), errors

@st.cache_data(ttl=1800, show_spinner=False)
def download_batch(tickers):
    try:
        data = yf.download(list(tickers), period='1y', interval='1d', group_by='ticker',
                           auto_adjust=False, progress=False, threads=4, timeout=18)
        return data, ''
    except Exception as e:
        return pd.DataFrame(), str(e)

def get_frame(data, ticker, batch_count):
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(0):
            df = data[ticker].copy()
        elif ticker in data.columns.get_level_values(1):
            df = data.xs(ticker, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    elif batch_count == 1:
        df = data.copy()
    else:
        return pd.DataFrame()
    if 'Close' not in df.columns:
        return pd.DataFrame()
    return df.dropna(subset=['Close']).sort_index()

def completed_daily(df):
    # Daily Yahoo bar can be in progress during the Taiwan trading session.
    now = datetime.now(TZ)
    if not df.empty and now.weekday() < 5 and 9 <= now.hour < 14:
        last = pd.Timestamp(df.index[-1])
        if last.tzinfo is None:
            last = last.tz_localize(TZ)
        else:
            last = last.tz_convert(TZ)
        if last.date() == now.date():
            return df.iloc[:-1]
    return df

def evaluate(df, convergence_limit, divergence_limit, flat_tolerance):
    if len(df) < 95:
        raise ValueError('有效日線少於 95 根')
    close = df['Close'].astype(float)
    ma = pd.DataFrame({n:close.rolling(n, min_periods=n).mean() for n in (5,10,20,60)})
    valid = ma.notna().all(axis=1)
    aligned = valid & (ma[5] > ma[10]) & (ma[10] > ma[20]) & (ma[20] > ma[60])
    if not bool(aligned.iloc[-1]):
        return None
    # First day of the current uninterrupted alignment run must be within 3 trading days.
    age = 0
    for flag in aligned.iloc[::-1]:
        if bool(flag):
            age += 1
        else:
            break
    if age not in (1, 2, 3):
        return None
    # Latest 5/10/20-day averages rise relative to yesterday.
    if any(not (ma[n].iloc[-1] > ma[n].iloc[-2]) for n in (5,10,20)):
        return None
    # 'Flat or up' for MA60: 5-session change >= -tolerance%.
    ma60_change = (ma[60].iloc[-1] / ma[60].iloc[-6] - 1) * 100
    if ma60_change < -flat_tolerance:
        return None
    deviation = (close.iloc[-1] / ma[20].iloc[-1] - 1) * 100
    if not (0 < deviation <= divergence_limit):
        return None
    # Require convergence to PRECEDE the first day of alignment.
    # Within 20 sessions ending the day before the first alignment day.
    formation_pos = len(df) - age
    start = max(0, formation_pos - 20)
    past = ma.iloc[start:formation_pos].dropna()
    if past.empty:
        return None
    concentration = (past.max(axis=1) - past.min(axis=1)) / past.mean(axis=1) * 100
    minimum = float(concentration.min())
    if minimum > convergence_limit:
        return None
    date = pd.Timestamp(df.index[formation_pos]).strftime('%Y-%m-%d')
    return {'排列第幾天':age, '形成日期':date,
            '收盤價':round(float(close.iloc[-1]),2),
            '20MA乖離%':round(float(deviation),2),
            '糾結最小集中度%':round(minimum,2),
            '60MA近5日變化%':round(float(ma60_change),2),
            '5MA':round(float(ma[5].iloc[-1]),2),
            '10MA':round(float(ma[10].iloc[-1]),2),
            '20MA':round(float(ma[20].iloc[-1]),2),
            '60MA':round(float(ma[60].iloc[-1]),2)}

stocks, roster_errors = stock_universe()
for error in roster_errors:
    st.warning(error)
if not stocks:
    st.error('無法取得官方股票清單，停止掃描。')
    st.stop()
st.write(f'股票清單：**{len(stocks):,} 檔**（四位數上市／上櫃公司代號）')
st.markdown('**核心條件**：5MA ＞ 10MA ＞ 20MA ＞ 60MA；5／10／20MA 向上；最近 3 日首次形成並持續；收盤價高於 20MA。')
with st.expander('調整篩選門檻', expanded=True):
    concentration_limit = st.number_input('形成前 20 日內，四線最小集中度上限（%）', 0.5, 15.0, 2.0, 0.5)
    deviation_limit = st.number_input('20MA 最大正乖離（%）', 0.5, 20.0, 3.0, 0.5)
    flat_tolerance = st.number_input('60MA 近 5 日允許微幅下彎（%）', 0.0, 2.0, 0.10, 0.05,
                                      help='例如 0.10 表示 60MA 最近 5 日跌幅不超過 0.10%，視為走平。')
    st.caption('糾結集中度＝(四線最大值－最小值)／四線平均值 ×100%；必須在首次形成排列之前出現。未啟用成交量門檻。')
scan_range = st.selectbox('本次掃描範圍', ['先測試 30 檔', '先測試 100 檔', '全市場'])
st.caption('僅使用已完成的日 K；Yahoo 免費行情可能延遲、缺漏或限流。全市場可能需要數分鐘。')
if st.button('開始掃描', type='primary'):
    selected = stocks[:30] if scan_range == '先測試 30 檔' else stocks[:100] if scan_range == '先測試 100 檔' else stocks
    results, failures = [], []
    progress = st.progress(0)
    status = st.empty()
    size = 15
    for start in range(0, len(selected), size):
        batch = selected[start:start+size]
        data, error = download_batch(tuple(x['ticker'] for x in batch))
        for stock in batch:
            try:
                df = completed_daily(get_frame(data, stock['ticker'], len(batch)))
                if len(df) < 95:
                    raise ValueError('日線資料不足或下載失敗' + (f'：{error}' if error else ''))
                matched = evaluate(df, concentration_limit, deviation_limit, flat_tolerance)
                if matched is not None:
                    results.append({'代號':stock['代號'], '名稱':stock['名稱'], '市場':stock['市場'], '產業':stock['產業'], **matched})
            except Exception as e:
                failures.append({'股票':stock['ticker'], '原因':str(e)[:140]})
        done = min(start + len(batch), len(selected))
        progress.progress(done / len(selected))
        status.caption(f'已處理 {done:,}/{len(selected):,}｜資料成功 {done-len(failures):,}｜失敗 {len(failures):,}')
        if done < len(selected):
            time.sleep(0.4)
    success = len(selected)-len(failures)
    st.success(f'掃描完成｜成功 {success:,}/{len(selected):,}（{success/len(selected):.1%}）｜失敗 {len(failures):,}')
    if failures or roster_errors:
        st.warning('資料未完整覆蓋；失敗股票不能視為不符合條件。')
    st.session_state['ma_swing_last_results'] = results
    st.session_state['ma_swing_last_failures'] = failures
    if failures:
        with st.expander(f'查看失敗股票（{len(failures)} 檔）'):
            st.dataframe(pd.DataFrame(failures), hide_index=True)

# Results are persisted across widget reruns, so changing industry does NOT trigger another scan.
if 'ma_swing_last_results' in st.session_state:
    saved = st.session_state['ma_swing_last_results']
    st.subheader(f'剛形成多頭完美排列：{len(saved)} 檔')
    if saved:
        out = pd.DataFrame(saved).sort_values(['排列第幾天','20MA乖離%'], ascending=[True, True]).reset_index(drop=True)
        sectors = sorted(out['產業'].fillna('未分類').unique().tolist())
        selected_sectors = st.multiselect('依產業查看結果（可複選）', sectors, default=sectors,
                                          help='不會重新掃描。若要看全部，選取所有產業。')
        shown = out[out['產業'].isin(selected_sectors)].copy()
        st.caption(f'目前顯示 {len(shown)} / {len(out)} 檔；產業依證交所／櫃買中心公司基本資料分類。')
        st.dataframe(shown, hide_index=True, use_container_width=True)
        st.download_button('下載目前顯示結果 CSV', shown.to_csv(index=False).encode('utf-8-sig'),
                           'ma_swing_results_v1_1.csv', mime='text/csv')
    else:
        st.info('本次成功取得行情的股票中，沒有符合所有條件的股票。可考慮放寬 2% 糾結或 3% 乖離門檻。')
st.info('本工具僅作為波段候選股雷達，請用券商日線確認均線與成交量、MACD，再自行評估風險。')
