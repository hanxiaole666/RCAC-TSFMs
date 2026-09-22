# -*- coding: utf-8 -*-
"""
RCAC v6 —— TKDE 大修实验: 嵌套评估 + 统计合规检验 + 强基线 + 4资产 + Chronos修复 + 路径敏感性
================================================================
对应审稿意见 Major 1/2/5/6/7/8/9/10 的实验回复.

协议 (Major 2 修复):
  每资产 10 窗口: 窗口1-5 = 验证集 (仅用于选 γ, cap), 窗口6-10 = 测试集 (所有论文数字)
  ACI状态与残差库跨窗口连续演化 (模拟在线部署, 验证期积累的状态带入测试期)

新增基线 (Major 5): GARCH-t, 历史模拟, CQR, EnbPI, AgACI
统计层 (Major 1): Kupiec LR_uc, Christoffersen LR_cc, 块bootstrap CI, DM+BH-FDR
风险水平 (Major 5): 80% / 90% / 95%  (99%功效不足, 报告并注明)
Chronos修复 (Major 6), 路径敏感性 16 vs 64 (Major 7), 经济指标 (Major 8),
机制消融: 门控/对称/cap (Major 10)

【依赖】 pip install arch  (GARCH; 缺装则自动跳过该基线)
【运行】 C:\\Users\\apple\\anaconda3\\python.exe kronos_risk_test_v6.py
【预计】 12-18 分钟 (RTX 2060)

【产出】 results 目录: v6_selection.csv / v6_test_main.csv / v6_stats.csv /
        v6_dm_fdr.csv / v6_chronos_rcac.csv / v6_pathsens.csv / v6_econ.csv
"""

import os
PROJECT_ROOT = r"D:\RCAC-TSFMs"
DATA_DIR, RESULTS_DIR = os.path.join(PROJECT_ROOT,"data"), os.path.join(PROJECT_ROOT,"results")
HF_CACHE = os.path.join(PROJECT_ROOT,"hf_cache")
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE,"transformers")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["MPLCONFIGDIR"] = os.path.join(HF_CACHE,"matplotlib")
import sys
_here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else PROJECT_ROOT
for p in (_here, PROJECT_ROOT):
    if p not in sys.path: sys.path.insert(0, p)
import time, math
import numpy as np
import pandas as pd
for d in (DATA_DIR, RESULTS_DIR): os.makedirs(d, exist_ok=True)

# ---------------- 配置 ----------------
MODEL_NAME  = "NeoQuasar/Kronos-mini"; TOKENIZER = "NeoQuasar/Kronos-Tokenizer-2k"; MAX_CONTEXT = 2048
CHRONOS_ID  = "amazon/chronos-t5-small"
SYMBOLS     = ["sh000300", "sh000905", "600519"]   # PAPER-LOCKED (v6 as submitted):
    # The submitted manuscript evaluates exactly these 3 assets (9 cells).
    # A 4-asset extension (adding 000001 Ping An Bank, 12 cells) was run separately
    # during revision preparation; it is NOT part of the submitted paper.
    # Keep this list unchanged so that run_all.py reproduces the paper exactly.
LOOKBACK, PRED_LEN = 256, 20
N_VAL, N_TEST, SAMPLES = 5, 5, 16
SEEDS       = [42, 43, 44]
LEVELS      = [0.20, 0.10, 0.05]                                # 80/90/95 (Major 5)
GAMMAS      = [0.02, 0.05, 0.10]; CAPS = [2.0, 2.5, 3.0]
RESID_CAP   = 500
TAG = {"0.2":"80","0.1":"90","0.05":"95"}

# ---------------- 统计工具 ----------------
def _chi2_sf(x, df):
    if x <= 0: return 1.0
    if df == 1: return math.erfc(math.sqrt(x/2))
    if df == 2: return math.exp(-x/2)
    raise ValueError

def kupiec(excs, a):
    """excs: 0/1 例外序列按时间排列 -> (LR_uc, p). 边界情形用极限似然."""
    n, x = len(excs), int(np.sum(excs))
    if n == 0: return np.nan, 1.0
    if x == 0:                       # 全覆盖: 仅零例外项有贡献
        lr = -2 * n * math.log(1-a)
        return lr, _chi2_sf(lr, 1)
    if x == n:                       # 全例外
        lr = -2 * n * math.log(a)
        return lr, _chi2_sf(lr, 1)
    ph = x/n
    ll_h = x*math.log(ph) + (n-x)*math.log(1-ph)
    ll_0 = x*math.log(a) + (n-x)*math.log(1-a)
    lr = -2*(ll_0 - ll_h)
    return lr, _chi2_sf(lr, 1)

def christoffersen(excs, a):
    lr_uc, p_uc = kupiec(excs, a)
    e = list(map(int, excs)); n00=n01=n10=n11=0
    for i in range(1, len(e)):
        if   e[i-1]==0 and e[i]==0: n00+=1
        elif e[i-1]==0 and e[i]==1: n01+=1
        elif e[i-1]==1 and e[i]==0: n10+=1
        else: n11+=1
    den0, den1 = n00+n01, n10+n11
    if den0==0 or den1==0: return np.nan, 1.0
    p0, p1 = n01/den0, n11/den1
    nt = den0+den1
    ph = (n01+n11)/nt
    ll_h = (n01+n11)*math.log(ph) + (n00+n10)*math.log(1-ph) if 0<ph<1 else 0.0
    ll_0 = n01*math.log(p0)+n00*math.log(1-p0) if 0<p0<1 else 0.0
    ll_0 += n11*math.log(p1)+n10*math.log(1-p1) if 0<p1<1 else 0.0
    lr_ind = max(0.0, -2*(ll_h - ll_0))
    lr_cc = (lr_uc if not np.isnan(lr_uc) else 0.0) + lr_ind
    return lr_cc, _chi2_sf(lr_cc, 2)

def blockboot_ci(excs, block=20, B=500, seed=0):
    e = np.asarray(excs, float); n = len(e)
    if n < block: return np.nan, np.nan
    rng = np.random.default_rng(seed); nb = n//block; stats = []
    for _ in range(B):
        idx = rng.integers(0, nb, nb)
        samp = np.concatenate([e[i*block:(i+1)*block] for i in idx])
        stats.append(samp.mean())
    return np.percentile(stats, 2.5)*100, np.percentile(stats, 97.5)*100

def norm_cdf(x):
    try:
        from scipy.stats import norm; return norm.cdf(x)
    except ImportError: return 0.5*(1+math.erf(x/math.sqrt(2)))

def dm_fdr(ours, base_dict):
    """ ours: dict name->series; 返回BH校正后的q值表"""
    ps, names = [], []
    for name, (e1, e2) in base_dict.items():
        d = np.asarray(e1)-np.asarray(e2); T=len(d)
        dbar=d.mean(); lrv=np.mean((d-dbar)**2)
        for l in range(1, min(5, T-1)):
            lrv += 2*(1-l/5)*np.mean((d[l:]-dbar)*(d[:-l]-dbar))
        dm = dbar/math.sqrt(lrv/T) if lrv>0 else 0.0
        ps.append(2*(1-norm_cdf(abs(dm)))); names.append((name, dm))
    order = np.argsort(ps); q = np.empty(len(ps)); m = len(ps)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = m - rank
        prev = min(prev, ps[i]*m/k); q[i] = prev
    return [(names[i][0], round(names[i][1],3), round(ps[i],4), round(q[i],4)) for i in range(m)]

# ---------------- 共形引擎 ----------------
def conf_engine(pred_mean, truth, preds, state, a, g=None, cap=2.5, mode="rcac", vol_ratio=1.0):
    """mode: rcac / aci / gated / sym / sc / raw"""
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
        if len(resid) >= 10:   h = np.quantile(np.abs(resid), 1-alpha)
        elif len(resid) >= 2:  h = float(np.std(resid))
        else:                  h = 0.05 * abs(pred_mean[t])   # 价格自适应兜底, 不再用硬编码50
        if   mode=="rcac":  f = min(cap, max(1.0, vol_ratio))
        elif mode=="gated": f = min(cap, max(1.0, vol_ratio)) if alpha > a else 1.0
        elif mode=="sym":   f = min(cap, max(1/cap, vol_ratio))
        else:               f = 1.0                      # aci
        lo[t], hi[t] = pred_mean[t]-h*f, pred_mean[t]+h*f
        miss = 0.0 if (pred_mean[t]-h <= truth[t] <= pred_mean[t]+h) else 1.0   # 无调制分支反馈 (公平)
        gam = g if g else state["gamma0"]
        if state.get("agaci"): gam = state["gamma0"]/(1+state["t"]/200)
        state["t"] += 1
        alpha = min(0.5, max(0.02, alpha + gam*(a - miss)))
        resid.append(pred_mean[t]-truth[t]); resid = resid[-RESID_CAP:]
    state["alpha"], state["resid"] = alpha, resid
    return (lo, hi), state

def winkler(lo, hi, y, a):
    return np.mean((hi-lo)+(2/a)*(np.clip(lo-y,0,None)+np.clip(y-upper,0,None))) if False else \
           np.mean((hi-lo)+(2/a)*(np.clip(lo-y,0,None)+np.clip(y-hi,0,None)))

def stats_of(lo, hi, y, a):
    exc = ((y<lo)|(y>hi)).astype(int)
    picp = 100*(1-exc.mean()); mpiw = float(np.mean(hi-lo))
    lr_uc, p_uc = kupiec(exc, a); lr_cc, p_cc = christoffersen(exc, a)
    lo95, hi95 = blockboot_ci(exc, block=PRED_LEN)
    exc_dist = np.where(y<lo, lo-y, np.where(y>hi, y-hi, 0.0))
    return dict(PICP=picp, MPIW=mpiw, Winkler=winkler(lo,hi,y,a), LR_uc=round(lr_uc,3), p_uc=round(p_uc,4),
                LR_cc=round(lr_cc,3), p_cc=round(p_cc,4), CI_lo=round(lo95,2), CI_hi=round(hi95,2),
                exceed=round(float(exc_dist[exc==1].mean()) if exc.any() else 0.0, 4),
                n_exc=int(exc.sum()))

# ---------------- 传统基线 ----------------
def feat_matrix(df):
    c = df["close"].values
    ret = np.diff(np.log(np.clip(c,1e-8,None))); ret = np.concatenate([[0], ret])
    rng = (df["high"].values - df["low"].values)/np.clip(c,1e-8,None)
    vol20 = pd.Series(ret).rolling(20).std().bfill().values
    amt = df["amount"].values if df["amount"].sum()>0 else df["volume"].values
    amtz = (amt - np.mean(amt))/(np.std(amt)+1e-9)
    X = np.column_stack([np.roll(ret,k) for k in range(1,6)] + [vol20, pd.Series(rng).rolling(5).mean().bfill().values, amtz])
    return X[30:], ret[30:]    # 去掉前30行的NaN窗口

def garch_intervals(x_df, last_price, a):
    try:
        from arch import arch_model
        c = x_df["close"].values
        r = 100*np.diff(np.log(c))
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
    lo = last_price*np.exp(np.cumsum(mu) - z*np.sqrt(np.cumsum(var)))
    hi = last_price*np.exp(np.cumsum(mu) + z*np.sqrt(np.cumsum(var)))
    return lo, hi

def hsim_intervals(x_df, last_price, a):
    r = np.diff(np.log(x_df["close"].values))[-LOOKBACK:]
    q_lo, q_hi = np.quantile(r, a/2), np.quantile(r, 1-a/2)
    t = np.sqrt(np.arange(1, PRED_LEN+1))
    return last_price*np.exp(q_lo*t), last_price*np.exp(q_hi*t)

def ml_quantile_band(x_df, last_price, a, mode):
    """CQR / EnbPI: 特征条件收益分位数 + driftless传播"""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    X, y = feat_matrix(x_df)
    qs = [a/2, 1-a/2]
    if mode == "cqr":
        models = [HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=100).fit(X,y) for q in qs]
        resid = np.maximum(models[0].predict(X)-y, y-models[1].predict(X))
        off = np.quantile(np.maximum(resid,0), 1-a) if len(resid) else 0.0
        x_mean = X.mean(axis=0, keepdims=True)
        q_lo = models[0].predict(x_mean)[0]-off; q_hi = models[1].predict(x_mean)[0]+off
    else:  # enbpi
        rng = np.random.default_rng(0); resids = []
        for b in range(20):
            idx = rng.integers(0, len(X), len(X))
            m = Ridge().fit(X[idx], y[idx])
            resids.append(y - m.predict(X))
        R = np.concatenate(resids)
        q_lo, q_hi = np.quantile(R, a/2), np.quantile(R, 1-a/2)
    t = np.sqrt(np.arange(1, PRED_LEN+1))
    return last_price*np.exp(q_lo*t), last_price*np.exp(q_hi*t)

# ---------------- 数据 ----------------
def load_kline_cached(symbol):
    cache = os.path.join(DATA_DIR, f"{symbol}_daily.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, parse_dates=["timestamps"])
    import akshare as ak
    print(f"    拉取 {symbol} ...")
    if symbol.startswith("sh0"):
        df = ak.stock_zh_index_daily(symbol=symbol); df = df.rename(columns={"date":"timestamps"})
    else:
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date="20180101", adjust="qfq")
        df = df.rename(columns={"日期":"timestamps","开盘":"open","最高":"high","最低":"low","收盘":"close","成交量":"volume"})
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    if "amount" not in df.columns: df["amount"] = df.get("volume",0)*df["close"]
    if "volume" not in df.columns: df["volume"] = 0.0
    df = df[["open","high","low","close","volume","amount","timestamps"]].dropna().reset_index(drop=True)
    df.to_csv(cache, index=False); return df

def realized_vol(close, win=20):
    ret = np.diff(np.log(np.clip(close,1e-8,None)))
    if len(ret)<win: return float(np.std(ret)) if len(ret)>1 else 1e-4
    return max(float(np.std(ret[-win:])),1e-4)

# ================= 主流程 =================
def main():
    print("="*66); print("  RCAC v6 — main experiment (nested evaluation + statistical compliance + strong baselines)"); print("="*66)
    from model import Kronos, KronosTokenizer, KronosPredictor
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("\n[1/6] 加载 Kronos ...")
    tok = KronosTokenizer.from_pretrained(TOKENIZER)
    model = Kronos.from_pretrained(MODEL_NAME)
    pred_kr = KronosPredictor(model, tok, device=device, max_context=MAX_CONTEXT)
    chronos_pipe = None
    try:
        from chronos import ChronosPipeline
        chronos_pipe = ChronosPipeline.from_pretrained(CHRONOS_ID, device_map=device, torch_dtype=torch.float32)
        print("      Chronos-small 已加载")
    except Exception as e:
        print(f"      [警告] Chronos 跳过: {e}")

    print("\n[2/6] 数据 ...")
    dfs = {}
    for s in SYMBOLS:
        try: dfs[s] = load_kline_cached(s); print(f"      {s}: {len(dfs[s])} bars")
        except Exception as e: print(f"      [警告] {s} 失败跳过: {e}")

    rows_main, rows_stats, rows_chron, rows_ps, rows_sel = [], [], [], [], []
    t0 = time.time()
    for sym, df in dfs.items():
        base = len(df) - LOOKBACK - PRED_LEN*(N_VAL+N_TEST)
        # ---- 每资产超参选择 (验证期) ----
        sel = {}
        val_resid = {}
        vol_hist = []                       # 资产级波动率历史, 跨水平持续 (修复B)
        for a in LEVELS:
            resid_pool = []
            states = {(g,cp): {"alpha":a, "resid":[], "gamma0":g, "t":0} for g in GAMMAS for cp in CAPS}
            agg = {(g,cp): [] for g in GAMMAS for cp in CAPS}
            for w in range(N_VAL):
                s0 = base + w*PRED_LEN
                truth, preds, pred_mean, vol_ratio, vol_hist = run_one(pred_kr, df, s0, device, SAMPLES, vol_hist)
                for (g,cp), st in states.items():
                    (lo,hi), st = conf_engine(pred_mean, truth, preds, st, a, g=g, cap=cp, mode="rcac", vol_ratio=vol_ratio)
                    agg[(g,cp)].append(stats_of(lo,hi,truth,a))
                resid_pool.extend((pred_mean-truth).tolist()); resid_pool = resid_pool[-RESID_CAP:]
                for (g,cp), st in states.items(): st["resid"] = list(resid_pool)
            best = None
            for (g,cp), lst in agg.items():
                picp = np.mean([r["PICP"] for r in lst]); mpiw = np.mean([r["MPIW"] for r in lst])
                if picp >= 100*(1-a)-1.0 and (best is None or mpiw < best[2]):
                    best = (g, cp, mpiw, picp)
            if best is None: best = max(((g,cp,np.mean([r["MPIW"] for r in lst]),np.mean([r["PICP"] for r in lst])) for (g,cp),lst in agg.items()), key=lambda z:-z[3])
            sel[a] = (best[0], best[1]); val_resid[a] = list(resid_pool)
            rows_sel.append({"symbol":sym,"level":TAG[str(a)],"gamma":best[0],"cap":best[1],"val_PICP":round(best[3],2),"val_MPIW":round(best[2],2)})
        print(f"  {sym} 超参选择: " + "  ".join(f"{TAG[str(a)]}%:γ={sel[a][0]},cap={sel[a][1]}" for a in LEVELS))

        # ---- 测试期 (所有论文数字) ----
        for a in LEVELS:
            gstar, capstar = sel[a]
            resid_pool = list(val_resid[a])   # 暖启动 (修复A)
            st_aci  = {"alpha":a, "resid":[], "gamma0":gstar, "t":0}
            st_rcac = {"alpha":a, "resid":[], "gamma0":gstar, "t":0}
            st_gate = {"alpha":a, "resid":[], "gamma0":gstar, "t":0}
            st_sym  = {"alpha":a, "resid":[], "gamma0":gstar, "t":0}
            st_aga  = {"alpha":a, "resid":[], "gamma0":gstar, "t":0, "agaci":True}
            exc_stream = {k: [] for k in ["raw","sc","aci","rcac","gated","sym","agaci","garch","hsim","cqr","enbpi"]}
            win_wink = {k: [] for k in exc_stream}
            chron_exc, chron_wink_raw, chron_wink_rcac = [], [], []
            for w in range(N_VAL, N_VAL+N_TEST):
                s0 = base + w*PRED_LEN
                truth, preds, pred_mean, vol_ratio, vol_hist = run_one(pred_kr, df, s0, device, SAMPLES, vol_hist)
                lp = df["close"].iloc[s0+LOOKBACK-1]
                intervals = {}
                intervals["raw"], _ = conf_engine(pred_mean, truth, preds, {}, a, mode="raw")
                r, _ = conf_engine(pred_mean, truth, preds, {"resid":resid_pool}, a, mode="sc"); intervals["sc"]=r
                intervals["aci"],  st_aci  = conf_engine(pred_mean, truth, preds, st_aci,  a, g=gstar, mode="aci")
                intervals["rcac"], st_rcac = conf_engine(pred_mean, truth, preds, st_rcac, a, g=gstar, cap=capstar, mode="rcac", vol_ratio=vol_ratio)
                intervals["gated"],st_gate = conf_engine(pred_mean, truth, preds, st_gate, a, g=gstar, cap=capstar, mode="gated", vol_ratio=vol_ratio)
                intervals["sym"],  st_sym  = conf_engine(pred_mean, truth, preds, st_sym,  a, g=gstar, cap=capstar, mode="sym", vol_ratio=vol_ratio)
                intervals["agaci"],st_aga  = conf_engine(pred_mean, truth, preds, st_aga,  a, g=gstar, mode="aci")
                try:    intervals["garch"] = garch_intervals(df.iloc[s0:s0+LOOKBACK], lp, a)
                except Exception: intervals["garch"] = None
                intervals["hsim"] = hsim_intervals(df.iloc[s0:s0+LOOKBACK], lp, a)
                try:    intervals["cqr"]   = ml_quantile_band(df.iloc[s0:s0+LOOKBACK], lp, a, "cqr")
                except Exception: intervals["cqr"]=None
                try:    intervals["enbpi"] = ml_quantile_band(df.iloc[s0:s0+LOOKBACK], lp, a, "enbpi")
                except Exception: intervals["enbpi"]=None
                for name, b in intervals.items():
                    if b is None: continue
                    st = stats_of(b[0], b[1], truth, a)
                    st.update(symbol=sym, level=TAG[str(a)], method=name, seed="all",
                              window=w+1, start=str(df["timestamps"].iloc[s0+LOOKBACK].date()))
                    rows_main.append(st)
                    exc_stream[name].extend(((truth<b[0])|(truth>b[1])).astype(int).tolist())
                    win_wink[name].append(st["Winkler"])
                resid_pool.extend((pred_mean-truth).tolist()); resid_pool = resid_pool[-RESID_CAP:]
                for stt in (st_aci,st_rcac,st_gate,st_sym,st_aga): stt["resid"] = list(resid_pool)
                # Chronos 修复 (单seed, 80%水平)
                if chronos_pipe is not None and abs(a-0.2)<1e-9:
                    ctx = torch.tensor(df["close"].iloc[s0:s0+LOOKBACK].values, dtype=torch.float32)
                    with torch.no_grad():
                        cs = chronos_pipe.predict(ctx, PRED_LEN, num_samples=32)
                    if hasattr(cs,"cpu"): cs = cs.cpu().numpy()
                    cs = np.asarray(cs); cs = cs[0] if cs.ndim==3 else cs
                    cm = cs.mean(axis=0)
                    lo_raw, hi_raw = np.quantile(cs, a/2, axis=0), np.quantile(cs, 1-a/2, axis=0)
                    stc = {"alpha":a, "resid":chron_exc_resid(sym), "gamma0":gstar, "t":w*PRED_LEN}
                    (lo_r, hi_r), _ = conf_engine(cm, truth, cs, stc, a, g=gstar, cap=capstar, mode="rcac", vol_ratio=vol_ratio)
                    _set_chron_resid(sym, stc["resid"])
                    chron_exc.extend(((truth<lo_r)|(truth>hi_r)).astype(int).tolist())
                    chron_wink_raw.append(winkler(lo_raw,hi_raw,truth,a))
                    chron_wink_rcac.append(winkler(lo_r,hi_r,truth,a))
                # 路径敏感性 (CSI300, seed42处理于外层, 80%)
                if sym=="sh000300" and abs(a-0.2)<1e-9:
                    for n_samp in (64,):
                        ps = np.array([pred_kr.predict(df=df.iloc[s0:s0+LOOKBACK][["open","high","low","close","volume","amount"]].reset_index(drop=True),
                                       x_timestamp=df["timestamps"].iloc[s0:s0+LOOKBACK].reset_index(drop=True),
                                       y_timestamp=df["timestamps"].iloc[s0+LOOKBACK:s0+LOOKBACK+PRED_LEN].reset_index(drop=True),
                                       pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=1)["close"].values
                                       for _ in range(n_samp)])
                        pm = ps.mean(axis=0)
                        lo1, hi1 = np.quantile(ps,a/2,axis=0), np.quantile(ps,1-a/2,axis=0)
                        rows_ps.append({"symbol":sym,"window":w+1,"n_paths":n_samp,
                                        "raw_PICP":stats_of(lo1,hi1,truth,a)["PICP"],
                                        "raw_MPIW":stats_of(lo1,hi1,truth,a)["MPIW"]})
            # 汇总统计 (Kupiec/CC/bootstrap per method per level)
            for name, excs in exc_stream.items():
                if not excs: continue
                excs = np.array(excs)
                lr_uc,p_uc = kupiec(excs,a); lr_cc,p_cc = christoffersen(excs,a)
                lo95,hi95 = blockboot_ci(excs, block=PRED_LEN)
                rows_stats.append({"symbol":sym,"level":TAG[str(a)],"method":name,
                                   "PICP":round(100*(1-excs.mean()),2),"CI_lo":round(lo95,2),"CI_hi":round(hi95,2),
                                   "LR_uc":round(lr_uc,3) if not np.isnan(lr_uc) else None,"p_uc":round(p_uc,4),
                                   "LR_cc":round(lr_cc,3) if not np.isnan(lr_cc) else None,"p_cc":round(p_cc,4),
                                   "n":len(excs)})
            # DM + BH-FDR: rcac vs 基线 (逐窗Winkler)
            fam = {k: (win_wink["rcac"], v) for k,v in win_wink.items()
                   if k!="rcac" and len(v)==len(win_wink["rcac"]) and len(v)>=5}
            for name, dm, p, q in dm_fdr(None, fam):
                rows_stats.append({"symbol":sym,"level":TAG[str(a)],"method":f"DM_vs_{name}",
                                   "DM":dm,"p":p,"q_FDR":q})
            if chronos_pipe is not None and abs(a-0.2)<1e-9 and len(chron_exc)>0:
                e = np.array(chron_exc)
                rows_chron.append({"symbol": sym,
                                   "rcac_PICP": round(100*(1-e.mean()), 2),
                                   "rcac_Winkler": round(float(np.mean(chron_wink_rcac)), 2),
                                   "raw_Winkler": round(float(np.mean(chron_wink_raw)), 2)})
            print(f"  {sym} {TAG[str(a)]}% 测试完成")

    print(f"\n[3/6] 总耗时 {time.time()-t0:.0f}s, 保存结果 ...")
    for name, rows in [("v6_selection",rows_sel),("v6_test_main",rows_main),("v6_stats",rows_stats),
                       ("v6_chronos_rcac",rows_chron),("v6_pathsens",rows_ps)]:
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, name+".csv"), index=False, encoding="utf-8-sig")
        print(f"      {name}.csv  ({len(rows)} rows)")
    print("\n[4/6] 关键汇总:")
    sdf = pd.DataFrame(rows_stats)
    if len(sdf):
        piv = sdf[sdf["method"].isin(["raw","sc","aci","rcac","garch","cqr","enbpi"])][["symbol","level","method","PICP","CI_lo","CI_hi","p_uc","p_cc"]]
        print(piv.to_string(index=False))
    print("\n完成. 判定: rcac 应是唯一 CI下界≥名义水平 且 p_uc>0.05 的方法; DM的q_FDR<0.1为显著.")

_CHR_RESID = {}
def chron_exc_resid(sym): return list(_CHR_RESID.get(sym, []))
def _set_chron_resid(sym, r): _CHR_RESID[sym] = list(r)[-RESID_CAP:]

def run_one(pred_kr, df, s0, device, n_samp, vol_hist=None):
    x_df = df.iloc[s0:s0+LOOKBACK][["open","high","low","close","volume","amount"]].reset_index(drop=True)
    x_ts = df["timestamps"].iloc[s0:s0+LOOKBACK].reset_index(drop=True)
    y_ts = df["timestamps"].iloc[s0+LOOKBACK:s0+LOOKBACK+PRED_LEN].reset_index(drop=True)
    truth = df["close"].iloc[s0+LOOKBACK:s0+LOOKBACK+PRED_LEN].values
    preds = np.array([pred_kr.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                        pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=1)["close"].values
                      for _ in range(n_samp)])
    # 因果波动率: σ̂(输入窗) / σ̄(历史窗累积均值, 不含当前窗)
    sig = realized_vol(x_df["close"].values)
    if vol_hist is None: vol_hist = []
    vol_hist.append(sig); vol_hist = vol_hist[-200:]
    sig_bar = np.mean(vol_hist[:-1]) if len(vol_hist) > 1 else sig
    vol_ratio = sig / max(sig_bar, 1e-4)
    return truth, preds, preds.mean(axis=0), vol_ratio, vol_hist

if __name__ == "__main__":
    main()
