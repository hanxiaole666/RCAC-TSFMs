# -*- coding: utf-8 -*-
"""18_spx_full.py — S&P 500 全11方法对照 (补齐 Sec 5.11 的方法对称性)
与 v6 主实验完全同构: 嵌套协议(5验证/5测试), 3seeds, 3水平, 11方法, 单侧Kupiec.
统计口径: 每水平 3seeds x 5窗 x 20步 = 300步 池化 -> 3 cells x 11 methods = 33行.
【运行】约 8-12 分钟 -> v6_spx_full.csv + 终端合规计数
"""
import os, sys
PROJECT_ROOT = r"D:\RCAC-TSFMs"
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, "hf_cache")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT, "hf_cache", "matplotlib")
sys.path.insert(0, PROJECT_ROOT)
import numpy as np, pandas as pd
DATA, RESULTS = os.path.join(PROJECT_ROOT, "data"), os.path.join(PROJECT_ROOT, "results")
LOOKBACK, PRED_LEN, N_VAL, N_TEST, SAMPLES = 256, 20, 5, 5, 16
GAMMAS, CAPS, RESID_CAP = [0.02, 0.05, 0.10], [2.0, 2.5, 3.0], 500
os.makedirs(RESULTS, exist_ok=True)

# ---------- 数据 (与15号脚本同源) ----------
def normalize_spx(raw):
    cmap = {"Date":"timestamps","date":"timestamps","日期":"timestamps","timestamps":"timestamps",
            "Open":"open","open":"open","开盘":"open","High":"high","high":"high","最高":"high",
            "Low":"low","low":"low","最低":"low","Close":"close","close":"close","收盘":"close",
            "Volume":"volume","volume":"volume","成交量":"volume"}
    raw = raw.rename(columns={c: cmap.get(c, c) for c in raw.columns})
    raw["timestamps"] = pd.to_datetime(raw["timestamps"])
    for c in ["open","high","low","close"]:
        if c not in raw.columns: raise ValueError(f"缺少列 {c}")
    if "volume" not in raw.columns: raw["volume"] = 0.0
    raw["amount"] = raw.get("amount", raw["volume"] * raw["close"])
    out = raw[["open","high","low","close","volume","amount","timestamps"]].dropna().reset_index(drop=True)
    out.to_csv(os.path.join(DATA, "spx_daily.csv"), index=False)
    return out

cache = os.path.join(DATA, "spx_daily.csv")
assert os.path.exists(cache), "先跑 15_spx_extension.py 生成 spx_daily.csv (或手动放置)"
df = pd.read_csv(cache, parse_dates=["timestamps"])
print(f"数据: {len(df)} bars, {df['timestamps'].iloc[0].date()} ~ {df['timestamps'].iloc[-1].date()}")

from model import Kronos, KronosTokenizer, KronosPredictor
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("加载 Kronos...")
tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
pred = KronosPredictor(model, tok, device=device, max_context=2048)

def realized_vol(close, win=20):
    ret = np.diff(np.log(np.clip(close,1e-8,None)))
    return max(float(np.std(ret[-win:])),1e-4) if len(ret)>=win else 1e-4

def conf_engine(pred_mean, truth, preds, state, a, g=None, cap=2.0, mode="rcac", vol_ratio=1.0):
    if mode == "raw":
        return (np.quantile(preds, a/2, axis=0), np.quantile(preds, 1-a/2, axis=0)), state
    if mode == "sc":
        r = state["resid"]
        if len(r)==0: return None, state
        h = np.quantile(np.abs(r), 1-a)
        return (pred_mean-h, pred_mean+h), state
    alpha, resid = state["alpha"], state["resid"]
    lo=np.empty(len(truth)); hi=np.empty(len(truth))
    for t in range(len(truth)):
        h = np.quantile(np.abs(resid), 1-alpha) if len(resid)>=10 else (np.std(resid) if resid else 0.05*abs(pred_mean[t]))
        if   mode=="rcac":  f = min(cap, max(1.0, vol_ratio))
        elif mode=="gated": f = min(cap, max(1.0, vol_ratio)) if alpha > a else 1.0
        elif mode=="sym":   f = min(cap, max(1.0/cap, vol_ratio))
        else:               f = 1.0
        lo[t], hi[t] = pred_mean[t]-h*f, pred_mean[t]+h*f
        miss = 0.0 if pred_mean[t]-h <= truth[t] <= pred_mean[t]+h else 1.0
        gam = g if g else state["gamma0"]
        if state.get("agaci"):
            gam = state["gamma0"]/(1+state["t"]/200)
        state["t"] += 1
        alpha = min(0.5, max(0.02, alpha + gam*(a - miss)))
        resid.append(pred_mean[t]-truth[t]); resid = resid[-RESID_CAP:]
    state["alpha"], state["resid"] = alpha, resid
    return (lo, hi), state

def garch_intervals(x_df, last_price, a):
    try:
        from arch import arch_model
        r = 100*np.diff(np.log(x_df["close"].values))
        m = arch_model(r, vol="Garch", p=1, q=1, dist="t").fit(disp="off")
        fc = m.forecast(horizon=PRED_LEN)
        mu = fc.mean.values[-1]/100; var = fc.variance.values[-1]/10000
    except Exception:
        r = np.diff(np.log(x_df["close"].values)); lam=0.94
        e=np.zeros(len(r)); e[0]=r[0]**2
        for i in range(1,len(r)): e[i]=lam*e[i-1]+(1-lam)*r[i]**2
        mu=np.zeros(PRED_LEN); var=np.full(PRED_LEN, e[-1])
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1-a/2)
    t = np.arange(1, PRED_LEN+1)
    return (last_price*np.exp(np.cumsum(mu)-z*np.sqrt(np.cumsum(var))),
            last_price*np.exp(np.cumsum(mu)+z*np.sqrt(np.cumsum(var))))

def hsim_intervals(x_df, last_price, a):
    r = np.diff(np.log(x_df["close"].values))[-LOOKBACK:]
    q_lo, q_hi = np.quantile(r, a/2), np.quantile(r, 1-a/2)
    t = np.sqrt(np.arange(1, PRED_LEN+1))
    return last_price*np.exp(q_lo*t), last_price*np.exp(q_hi*t)

def ml_quantile_band(x_df, last_price, a, mode):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    c = df["close"].values if False else x_df["close"].values
    ret = np.diff(np.log(np.clip(c,1e-8,None))); ret = np.concatenate([[0], ret])
    rng_hl = (x_df["high"].values - x_df["low"].values)/np.clip(c,1e-8,None)
    vol20 = pd.Series(ret).rolling(20).std().bfill().values
    amt = x_df["amount"].values if x_df["amount"].sum()>0 else x_df["volume"].values
    amtz = (amt-np.mean(amt))/(np.std(amt)+1e-9)
    X = np.column_stack([np.roll(ret,k) for k in range(1,6)]+[vol20, pd.Series(rng_hl).rolling(5).mean().bfill().values, amtz])
    X, y = X[30:], ret[30:]
    qs = [a/2, 1-a/2]
    if mode == "cqr":
        models = [HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=100).fit(X,y) for q in qs]
        resid = np.maximum(models[0].predict(X)-y, y-models[1].predict(X))
        off = np.quantile(np.maximum(resid,0), 1-a) if len(resid) else 0.0
        xm = X.mean(axis=0, keepdims=True)
        q_lo = models[0].predict(xm)[0]-off; q_hi = models[1].predict(xm)[0]+off
    else:
        rng2 = np.random.default_rng(0); resids=[]
        for b in range(20):
            idx = rng2.integers(0, len(X), len(X))
            m = Ridge().fit(X[idx], y[idx])
            resids.append(y - m.predict(X))
        R = np.concatenate(resids)
        q_lo, q_hi = np.quantile(R, a/2), np.quantile(R, 1-a/2)
    t = np.sqrt(np.arange(1, PRED_LEN+1))
    return last_price*np.exp(q_lo*t), last_price*np.exp(q_hi*t)

from math import comb
def binom_sf(x, n, p): return sum(comb(n,k)*p**k*(1-p)**(n-k) for k in range(x, n+1))
A = {"80":0.20,"90":0.10,"95":0.05}

base = len(df) - LOOKBACK - PRED_LEN*(N_VAL+N_TEST)
METHODS = ["raw","sc","aci","agaci","garch","hsim","cqr","enbpi","gated","sym","rcac"]

print(f"滚动评测: spx x 3seeds x 10窗 x 3水平 x 11方法 ...")
rows = []
for a_lvl in [0.20, 0.10, 0.05]:
    tag = {0.2:"80",0.1:"90",0.05:"95"}[a_lvl]
    # 验证期选参
    resid_pool, vol_hist = [], []
    states = {(g,c): {"alpha":a_lvl,"resid":[],"gamma0":g,"t":0} for g in GAMMAS for c in CAPS}
    agg = {(g,c): [] for g in GAMMAS for c in CAPS}
    for w in range(N_VAL):
        s0 = base + w*PRED_LEN
        x_df = df.iloc[s0:s0+LOOKBACK][["open","high","low","close","volume","amount"]].reset_index(drop=True)
        x_ts = df["timestamps"].iloc[s0:s0+LOOKBACK].reset_index(drop=True)
        y_ts = df["timestamps"].iloc[s0+LOOKBACK:s0+LOOKBACK+PRED_LEN].reset_index(drop=True)
        truth = df["close"].iloc[s0+LOOKBACK:s0+LOOKBACK+PRED_LEN].values
        torch.manual_seed(42); np.random.seed(42)
        preds = np.array([pred.predict(df=x_df,x_timestamp=x_ts,y_timestamp=y_ts,pred_len=PRED_LEN,
                         T=1.0,top_p=0.9,sample_count=1)["close"].values for _ in range(SAMPLES)])
        pm = preds.mean(axis=0)
        sig = realized_vol(x_df["close"].values); vol_hist.append(sig); vol_hist=vol_hist[-200:]
        vr = sig/max(np.mean(vol_hist[:-1]),1e-4) if len(vol_hist)>1 else 1.0
        for (g,c),st in states.items():
            (lo,hi), st = conf_engine(pm, truth, preds, st, a_lvl, g=g, cap=c, mode="rcac", vol_ratio=vr)
            agg[(g,c)].append(np.mean((truth>=lo)&(truth<=hi))*100)
        resid_pool.extend((pm-truth).tolist()); resid_pool=resid_pool[-RESID_CAP:]
        for k in states: states[k]["resid"]=list(resid_pool)
    best = None
    for (g,c),lst in agg.items():
        picp = np.mean(lst)
        if picp >= 100*(1-a_lvl)-1.0 and (best is None or picp>best[0]): best=(picp,g,c)
    gstar, capstar = (best[1],best[2]) if best else max(((np.mean(v),k) for k,v in agg.items()), key=lambda z:-z[0])[1]
    # 测试期: 例外流按 (方法) 池化 3seeds x 5窗 = 300步
    exc_pool = {m: [] for m in METHODS}
    wink_pool = {m: [] for m in METHODS}
    for seed in [42,43,44]:
        torch.manual_seed(seed); np.random.seed(seed)
        st_aci  = {"alpha":a_lvl,"resid":list(resid_pool),"gamma0":gstar,"t":0}
        st_rcac = {"alpha":a_lvl,"resid":list(resid_pool),"gamma0":gstar,"t":0}
        st_gate = {"alpha":a_lvl,"resid":list(resid_pool),"gamma0":gstar,"t":0}
        st_sym  = {"alpha":a_lvl,"resid":list(resid_pool),"gamma0":gstar,"t":0}
        st_aga  = {"alpha":a_lvl,"resid":list(resid_pool),"gamma0":gstar,"t":0,"agaci":True}
        resid_shared = list(resid_pool)
        vol_hist2 = list(vol_hist)
        for w in range(N_VAL, N_VAL+N_TEST):
            s0 = base + w*PRED_LEN
            x_df = df.iloc[s0:s0+LOOKBACK][["open","high","low","close","volume","amount"]].reset_index(drop=True)
            x_ts = df["timestamps"].iloc[s0:s0+LOOKBACK].reset_index(drop=True)
            y_ts = df["timestamps"].iloc[s0+LOOKBACK:s0+LOOKBACK+PRED_LEN].reset_index(drop=True)
            truth = df["close"].iloc[s0+LOOKBACK:s0+LOOKBACK+PRED_LEN].values
            preds = np.array([pred.predict(df=x_df,x_timestamp=x_ts,y_timestamp=y_ts,pred_len=PRED_LEN,
                             T=1.0,top_p=0.9,sample_count=1)["close"].values for _ in range(SAMPLES)])
            pm = preds.mean(axis=0)
            sig = realized_vol(x_df["close"].values); vol_hist2.append(sig); vol_hist2=vol_hist2[-200:]
            vr = sig/max(np.mean(vol_hist2[:-1]),1e-4) if len(vol_hist2)>1 else 1.0
            lp = df["close"].iloc[s0+LOOKBACK-1]
            intervals = {}
            intervals["raw"],_ = conf_engine(pm, truth, preds, {}, a_lvl, mode="raw")
            r,_ = conf_engine(pm, truth, preds, {"resid":resid_shared}, a_lvl, mode="sc"); intervals["sc"]=r
            intervals["aci"],  st_aci  = conf_engine(pm, truth, preds, st_aci,  a_lvl, g=gstar, mode="aci")
            intervals["rcac"], st_rcac = conf_engine(pm, truth, preds, st_rcac, a_lvl, g=gstar, cap=capstar, mode="rcac", vol_ratio=vr)
            intervals["gated"],st_gate = conf_engine(pm, truth, preds, st_gate, a_lvl, g=gstar, cap=capstar, mode="gated", vol_ratio=vr)
            intervals["sym"],  st_sym  = conf_engine(pm, truth, preds, st_sym,  a_lvl, g=gstar, cap=capstar, mode="sym", vol_ratio=vr)
            intervals["agaci"],st_aga  = conf_engine(pm, truth, preds, st_aga,  a_lvl, g=gstar, mode="aci")
            try:    intervals["garch"] = garch_intervals(x_df, lp, a_lvl)
            except Exception: intervals["garch"] = None
            intervals["hsim"] = hsim_intervals(x_df, lp, a_lvl)
            try:    intervals["cqr"]   = ml_quantile_band(x_df, lp, a_lvl, "cqr")
            except Exception: intervals["cqr"]=None
            try:    intervals["enbpi"] = ml_quantile_band(x_df, lp, a_lvl, "enbpi")
            except Exception: intervals["enbpi"]=None
            for m,(lo,hi) in intervals.items():
                if lo is None: continue
                exc_pool[m].extend(((truth<lo)|(truth>hi)).astype(int).tolist())
                wink_pool[m].append(np.mean((hi-lo)+(2/a_lvl)*(np.clip(lo-truth,0,None)+np.clip(truth-hi,0,None))))
            resid_shared.extend((pm-truth).tolist()); resid_shared = resid_shared[-RESID_CAP:]
            for stt in (st_aci,st_rcac,st_gate,st_sym,st_aga): stt["resid"]=list(resid_shared)
        print(f"  spx {tag}% seed{seed} done")
    for m in METHODS:
        e = np.array(exc_pool[m]); n = len(e); x = int(e.sum())
        p1 = binom_sf(x, n, a_lvl)
        rows.append({"symbol":"spx","level":tag,"method":m,"n":n,"PICP":round(100*(1-e.mean()),1),
                     "exc":x,"p_1sided":round(p1,4),"pass1s":bool(p1>0.05),
                     "Winkler":round(float(np.mean(wink_pool[m])),1)})

r = pd.DataFrame(rows)
r.to_csv(os.path.join(RESULTS,"v6_spx_full.csv"), index=False, encoding="utf-8-sig")
print("\n=== S&P 500 全方法 单侧合规 (3 cells, n=300/格) ===")
cnt = r.groupby("method")["pass1s"].sum().astype(int).sort_values(ascending=False)
for m,c in cnt.items(): print(f"  {m:<8} {c}/3")
print("\n=== 明细 ===")
print(r.pivot_table(index="method", columns="level", values=["PICP","p_1sided","Winkler"]).round(2).to_string())
print("已保存 v6_spx_full.csv")
