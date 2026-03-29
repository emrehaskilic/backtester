"""Triple Barrier Pattern Backtest"""
import numpy as np, pandas as pd, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_parquet('data/ETHUSDT_3m_vol_11mo.parquet')
df['dt'] = pd.to_datetime(df['open_time'], unit='ms')
oi5 = pd.read_parquet('data/ETHUSDT_OI_5m_11mo.parquet')
df['oi'] = np.nan
oi_map = dict(zip(oi5['open_time'], oi5['open_interest']))
for idx in range(len(df)):
    ot = df['open_time'].iloc[idx]
    if ot in oi_map: df.iloc[idx, df.columns.get_loc('oi')] = oi_map[ot]
df['oi'] = df['oi'].ffill().fillna(0.0)

total = len(df)
closes = df['close'].values.astype(np.float64)
highs = df['high'].values.astype(np.float64)
lows = df['low'].values.astype(np.float64)
lookback = 20

# ATR
tr = np.zeros(total); tr[0] = highs[0]-lows[0]
for i in range(1,total): tr[i] = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
atr = np.full(total, np.nan); atr[14] = np.mean(tr[1:15])
for i in range(15,total): atr[i] = (atr[i-1]*13+tr[i])/14

def lin_slope(data,end,period):
    if end<period: return np.nan
    y=data[end+1-period:end+1]; x=np.arange(period,dtype=np.float64)
    n=period; sx=x.sum();sy=y.sum();sxy=(x*y).sum();sxx=(x*x).sum()
    d=n*sxx-sx*sx
    if abs(d)<1e-12: return 0
    return (n*sxy-sx*sy)/d

price_slopes = np.full(total, np.nan)
for i in range(lookback,total):
    ps = lin_slope(closes,i,lookback)
    price_slopes[i] = ps/atr[i] if not np.isnan(atr[i]) and atr[i]>0 else 0

atr_pctile = np.full(total, np.nan)
for i in range(100,total):
    w=atr[i-99:i+1]; v=w[~np.isnan(w)]
    if len(v)>0: atr_pctile[i]=np.sum(v<atr[i])/len(v)*100

ema20=np.full(total,np.nan); k=2/21; ema20[0]=closes[0]
for i in range(1,total): ema20[i]=closes[i]*k+ema20[i-1]*(1-k)
kc_width=np.full(total,np.nan)
for i in range(total):
    if not np.isnan(ema20[i]) and not np.isnan(atr[i]) and closes[i]>0:
        kc_width[i]=(2*atr[i]*2.0)/closes[i]*100

price_vs_ema = np.full(total, np.nan)
for i in range(total):
    if not np.isnan(ema20[i]) and not np.isnan(atr[i]) and atr[i]>0:
        price_vs_ema[i]=(closes[i]-ema20[i])/atr[i]

train_end = 8*30*480
valid = np.array([not(np.isnan(price_slopes[i]) or np.isnan(atr_pctile[i]) or np.isnan(kc_width[i]) or np.isnan(price_vs_ema[i])) for i in range(total)])

def qth(data,end):
    vals=np.sort(data[:end][valid[:end]&~np.isnan(data[:end])])
    return [vals[int(q/5*len(vals))] for q in range(1,5)]

ps_th=qth(price_slopes,train_end)
atr_th=qth(atr_pctile,train_end)
kc_th=qth(kc_width,train_end)
pve_th=qth(price_vs_ema,train_end)

def get_q(val,th):
    q=0
    for t in th:
        if val>t: q+=1
    return q

# ── TRIPLE BARRIER ──
TP=0.5; SL=0.5; TIMEOUT=160
MARGIN_RATIO=1/40; LEVERAGE=25; TAKER_FEE=0.0005
balance=10000.0; peak=10000.0; max_dd=0.0
total_trades=0; wins=0; total_pnl=0.0; total_fees=0.0
start=200; in_position=False; position_end=0
trades=[]

for i in range(start, total-TIMEOUT):
    if not valid[i]: continue
    if in_position and i < position_end: continue

    ps_q=get_q(price_slopes[i],ps_th)
    atr_q=get_q(atr_pctile[i],atr_th)
    kc_q=get_q(kc_width[i],kc_th)
    pve_q=get_q(price_vs_ema[i],pve_th)

    short_sig = (ps_q==0 and atr_q==4 and kc_q==2)
    long_sig = (pve_q==4 and kc_q==1)
    if not short_sig and not long_sig: continue

    direction = -1.0 if short_sig else 1.0
    entry = closes[i]
    if direction < 0:
        tp_price=entry*(1-TP/100); sl_price=entry*(1+SL/100)
    else:
        tp_price=entry*(1+TP/100); sl_price=entry*(1-SL/100)

    result='TIMEOUT'; exit_price=closes[min(i+TIMEOUT,total-1)]; exit_bar=i+TIMEOUT
    for j in range(i+1, min(i+TIMEOUT+1, total)):
        if direction<0:
            if lows[j]<=tp_price: result='TP'; exit_price=tp_price; exit_bar=j; break
            if highs[j]>=sl_price: result='SL'; exit_price=sl_price; exit_bar=j; break
        else:
            if highs[j]>=tp_price: result='TP'; exit_price=tp_price; exit_bar=j; break
            if lows[j]<=sl_price: result='SL'; exit_price=sl_price; exit_bar=j; break

    if direction<0: ret=(entry-exit_price)/entry*100
    else: ret=(exit_price-entry)/entry*100

    margin=balance*MARGIN_RATIO; notional=margin*LEVERAGE
    pnl=notional*ret/100; fee=notional*TAKER_FEE*2
    balance+=pnl-fee; total_pnl+=pnl; total_fees+=fee; total_trades+=1
    if pnl>0: wins+=1
    if balance>peak: peak=balance
    dd=(peak-balance)/peak*100
    if dd>max_dd: max_dd=dd
    in_position=True; position_end=exit_bar
    trades.append({'bar':i,'date':df['dt'].iloc[i].strftime('%Y-%m-%d'),'dir':'SHORT' if direction<0 else 'LONG','result':result,'ret':ret,'pnl':pnl,'balance':balance,'is_train':i<train_end})

print("TRIPLE BARRIER PATTERN STRATEJISI")
print("Short: price_slope=Q1 AND ATR=Q5 AND KC_width=Q3")
print("Long : price_vs_EMA=Q5 AND KC_width=Q2")
print(f"Exit : TP={TP}% SL={SL}% Timeout=8h | Ayni anda tek pozisyon")
print("="*80)
print()

tp_cnt=sum(1 for t in trades if t['result']=='TP')
sl_cnt=sum(1 for t in trades if t['result']=='SL')
to_cnt=sum(1 for t in trades if t['result']=='TIMEOUT')
short_cnt=sum(1 for t in trades if t['dir']=='SHORT')
long_cnt=sum(1 for t in trades if t['dir']=='LONG')

print(f"Toplam: {total_trades} trade | Short: {short_cnt} | Long: {long_cnt}")
print(f"TP: {tp_cnt} ({tp_cnt/max(total_trades,1)*100:.1f}%) | SL: {sl_cnt} ({sl_cnt/max(total_trades,1)*100:.1f}%) | Timeout: {to_cnt} ({to_cnt/max(total_trades,1)*100:.1f}%)")
print(f"Win: {wins} | WR: {wins/max(total_trades,1)*100:.1f}%")
print(f"Balance: ${balance:,.0f} | Net: {(balance-10000)/10000*100:+.2f}%")
print(f"Max DD: {max_dd:.2f}%")
print(f"PnL: ${total_pnl:+,.0f} | Fee: ${total_fees:,.0f}")
print()

train_t=[t for t in trades if t['is_train']]; test_t=[t for t in trades if not t['is_train']]
tr_w=sum(1 for t in train_t if t['pnl']>0); te_w=sum(1 for t in test_t if t['pnl']>0)
print(f"TRAIN (8 ay): {len(train_t)} trade | WR: {tr_w/max(len(train_t),1)*100:.1f}% | PnL: ${sum(t['pnl'] for t in train_t):+,.0f}")
print(f"TEST  (3 ay): {len(test_t)} trade | WR: {te_w/max(len(test_t),1)*100:.1f}% | PnL: ${sum(t['pnl'] for t in test_t):+,.0f}")
print()

for label,subset in [('Train Short',[t for t in train_t if t['dir']=='SHORT']),
                      ('Train Long',[t for t in train_t if t['dir']=='LONG']),
                      ('Test Short',[t for t in test_t if t['dir']=='SHORT']),
                      ('Test Long',[t for t in test_t if t['dir']=='LONG'])]:
    if subset:
        w=sum(1 for t in subset if t['pnl']>0)
        print(f"  {label:12s}: {len(subset):>3d} trade | WR: {w/len(subset)*100:>5.1f}% | PnL: ${sum(t['pnl'] for t in subset):>+8,.0f}")
    else:
        print(f"  {label:12s}:   0 trade")

print()
print("HAFTALIK:")
print(f"{'Hf':>3s} {'Tarih':>10s} {'Trd':>4s} {'TP':>3s} {'SL':>3s} {'TO':>3s} {'WR%':>5s} {'PnL$':>8s} {'Bal$':>10s} {'Donem':>5s}")
print("-"*65)
bpw=480*7; ws=start; week_num=1
while ws<total:
    we=min(ws+bpw,total)
    wt=[t for t in trades if t['bar']>=ws and t['bar']<we]
    if wt:
        wtp=sum(1 for t in wt if t['result']=='TP')
        wsl=sum(1 for t in wt if t['result']=='SL')
        wto=sum(1 for t in wt if t['result']=='TIMEOUT')
        ww=sum(1 for t in wt if t['pnl']>0)
        wr=ww/len(wt)*100; wpnl=sum(t['pnl'] for t in wt)
        wbal=wt[-1]['balance']; donem='TEST' if not wt[0]['is_train'] else 'TRAIN'
        date=df['dt'].iloc[ws].strftime('%m/%d')
        print(f"{week_num:3d} {date:>10s} {len(wt):>4d} {wtp:>3d} {wsl:>3d} {wto:>3d} {wr:>5.1f} {wpnl:>+8.0f} {wbal:>10,.0f} {donem:>5s}")
    week_num+=1; ws+=bpw
