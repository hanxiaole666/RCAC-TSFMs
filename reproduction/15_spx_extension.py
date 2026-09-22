# -*- coding: utf-8 -*-
"""15_spx_extension.py — 跨市场验证: S&P 500 (spx) 加入嵌套协议评测
审稿意见 M3 的回应实验. 数据免费 (stooq CSV, 无需翻墙), 约 4-5 分钟.
【运行】C:\\Users\\apple\\anaconda3\\python.exe 15_spx_extension.py
【产出】v6_spx.csv (spx 9格逐格单侧统计) + 终端合规计数
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

def load_spx():
    """S&P 500 日线. 四级获取: stooq(带请求头) -> akshare美股日线 -> yfinance -> 手动放置CSV.
    手动兜底: 浏览器打开 https://stooq.com/q/d/l/?s=%5Espx&i=d 下载CSV,
    直接保存为 D:\\RCAC-TSFMs\\data\\spx_daily.csv (无需改格式, 本函数自动识别列名)"""
    import csv
    cache = os.path.join(DATA, "spx_daily.csv")
    if os.path.exists(cache):
        print("  读取缓存 spx_daily.csv")
        df = pd.read_csv(cache)
        return normalize_spx(df)
    # 1) stooq (urllib + 浏览器UA, 试两个symbol变体)
    for sym in ["%5Espx", "^spx"]:
        try:
            import urllib.request
            req = urllib.request.Request(
                f"https://stooq.com/q/d/l/?s={sym}&i=d",
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read().decode("utf-8")
            if "Date" in content[:200]:
                df = pd.read_csv(pd.io.common.StringIO(content))
                print(f"  stooq 获取成功 ({sym}): {len(df)} rows")
                return normalize_spx(df)
        except Exception as e:
            print(f"  stooq {sym} 失败: {e}")
    # 2) akshare 美股日线 (新浪源, 国内直接可用)
    try:
        import akshare as ak
        raw = ak.stock_us_daily(symbol=".INX", adjust="qfq")
        print(f"  akshare 获取成功: {len(raw)} rows")
        return normalize_spx(raw)
    except Exception as e:
        print(f"  akshare 失败: {e}")
    # 3) yfinance
    try:
        import yfinance as yf
        raw = yf.download("^GSPC", start="2002-01-01", auto_adjust=False).reset_index()
        print(f"  yfinance 获取成功: {len(raw)} rows")
        return normalize_spx(raw)
    except Exception as e:
        print(f"  yfinance 失败: {e}")
    print("\n[获取全部失败 - 手动兜底]\n"
          "  1. 浏览器打开: https://stooq.com/q/d/l/?s=%5Espx&i=d\n"
          "  2. 下载得到的CSV, 直接另存为: D:\\RCAC-TSFMs\\data\\spx_daily.csv\n"
          "  3. 重新运行本脚本 (列名自动识别, 无需修改)")
    sys.exit(1)

def normalize_spx(raw):
    """列名自动识别: 兼容 stooq(Date/Open/...), akshare(中文列), yfinance, 内部格式"""
    cmap = {"Date":"timestamps","date":"timestamps","日期":"timestamps","timestamps":"timestamps",
            "Open":"open","open":"open","开盘":"open",
            "High":"high","high":"high","最高":"high",
            "Low":"low","low":"low","最低":"low",
            "Close":"close","close":"close","收盘":"close",
            "Volume":"volume","volume":"volume","成交量":"volume"}
    raw = raw.rename(columns={c: cmap.get(c, c) for c in raw.columns})
    if "timestamps" not in raw.columns:
        raise ValueError("无法识别时间列: " + str(list(raw.columns)))
    raw["timestamps"] = pd.to_datetime(raw["timestamps"])
    for c in ["open","high","low","close"]:
        if c not in raw.columns: raise ValueError(f"缺少列 {c}")
    if "volume" not in raw.columns: raw["volume"] = 0.0
    raw["amount"] = raw.get("amount", raw["volume"] * raw["close"])
    out = raw[["open","high","low","close","volume","amount","timestamps"]].dropna().reset_index(drop=True)
    out.to_csv(os.path.join(DATA, "spx_daily.csv"), index=False)
    print(f"  标准化完成: {len(out)} bars, {out['timestamps'].iloc[0].date()} ~ {out['timestamps'].iloc[-1].date()}")
    return out

from model import Kronos, KronosTokenizer, KronosPredictor
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("[1/3] 加载 Kronos...")
tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
pred = KronosPredictor(model, tok, device=device, max_context=2048)

def realized_vol(close, win=20):
    ret = np.diff(np.log(np.clip(close,1e-8,None)))
    return max(float(np.std(ret[-win:])),1e-4) if len(ret)>=win else 1e-4

def conf_engine(pred_mean, truth, preds, state, a, g, cap, mode="rcac", vol_ratio=1.0):
    alpha, resid = state["alpha"], state["resid"]
    lo = np.empty(len(truth)); hi = np.empty(len(truth))
    for t in range(len(truth)):
        h = np.quantile(np.abs(resid), 1-alpha) if len(resid)>=10 else (np.std(resid) if resid else 0.05*abs(pred_mean[t]))
        f = min(cap, max(1.0, vol_ratio)) if mode=="rcac" else 1.0
        lo[t], hi[t] = pred_mean[t]-h*f, pred_mean[t]+h*f
        miss = 0.0 if pred_mean[t]-h <= truth[t] <= pred_mean[t]+h else 1.0
        alpha = min(0.5, max(0.02, alpha + g*(a-miss)))
        resid.append(pred_mean[t]-truth[t]); resid = resid[-RESID_CAP:]
    state["alpha"], state["resid"] = alpha, resid
    return lo, hi

print("[2/3] 数据 ...")
df = load_spx()
base = len(df) - LOOKBACK - PRED_LEN*(N_VAL+N_TEST)
assert base >= 0, f"数据不足: {len(df)} bars"

from math import comb
def binom_sf(x, n, p): return sum(comb(n,k)*p**k*(1-p)**(n-k) for k in range(x, n+1))
A = {"80":0.20,"90":0.10,"95":0.05}

print(f"[3/3] 滚动评测: spx x 3seeds x {N_VAL+N_TEST}窗 x 3水平 ...")
rows_main, rows_stat = [], []
for a_lvl in [0.20, 0.10, 0.05]:
    tag = {0.2:"80",0.1:"90",0.05:"95"}[a_lvl]
    # 验证期选参
    resid_pool, vol_hist = [], []
    states = {(g,c): {"alpha":a_lvl,"resid":[],"gamma0":g,"t":0} for g in GAMMAS for c in CAPS}
    agg = {(g,c): [] for g in GAMMAS for c in CAPS}
    for w in range(N_VAL):
        s0 = base + w*PRED_LEN
        x = df.iloc[s0:s0+LOOKBACK][["open","high","low","close","volume","amount"]].reset_index(drop=True)
        xt = df["timestamps"].iloc[s0:s0+LOOKBACK].reset_index(drop=True)
        yt = df["timestamps"].iloc[s0+LOOKBACK:s0+LOOKBACK+PRED_LEN].reset_index(drop=True)
        truth = df["close"].iloc[s0+LOOKBACK:s0+LOOKBACK+PRED_LEN].values
        torch.manual_seed(42); np.random.seed(42)
        preds = np.array([pred.predict(df=x,x_timestamp=xt,y_timestamp=yt,pred_len=PRED_LEN,
                         T=1.0,top_p=0.9,sample_count=1)["close"].values for _ in range(SAMPLES)])
        pm = preds.mean(axis=0)
        sig = realized_vol(x["close"].values); vol_hist.append(sig); vol_hist=vol_hist[-200:]
        vr = sig/max(np.mean(vol_hist[:-1]),1e-4) if len(vol_hist)>1 else 1.0
        for (g,c),st in states.items():
            lo,hi = conf_engine(pm, truth, preds, st, a_lvl, g, c, vol_ratio=vr)
            picp = np.mean((truth>=lo)&(truth<=hi))*100
            agg[(g,c)].append(picp)
        resid_pool.extend((pm-truth).tolist()); resid_pool=resid_pool[-RESID_CAP:]
        for k in states: states[k]["resid"]=list(resid_pool)
    best = None
    for (g,c),lst in agg.items():
        picp = np.mean(lst)
        if picp >= 100*(1-a_lvl)-1.0 and (best is None or picp>best[0]): best=(picp,g,c)
    if best is None:
        gstar, capstar = max(((np.mean(v),k) for k,v in agg.items()), key=lambda z:z[0])[1]
    else: gstar, capstar = best[1], best[2]
    # 测试期
    for seed in [42,43,44]:
        torch.manual_seed(seed); np.random.seed(seed)
        st = {"alpha":a_lvl,"resid":list(resid_pool)}
        exc_stream = {m: [] for m in ["raw","sc","aci","garch","hsim","cqr","enbpi","gated","sym","rcac"]}
        vol_hist2 = list(vol_hist)
        for w in range(N_VAL, N_VAL+N_TEST):
            s0 = base + w*PRED_LEN
            x = df.iloc[s0:s0+LOOKBACK][["open","high","low","close","volume","amount"]].reset_index(drop=True)
            xt = df["timestamps"].iloc[s0:s0+LOOKBACK].reset_index(drop=True)
            yt = df["timestamps"].iloc[s0+LOOKBACK:s0+LOOKBACK+PRED_LEN].reset_index(drop=True)
            truth = df["close"].iloc[s0+LOOKBACK:s0+LOOKBACK+PRED_LEN].values
            preds = np.array([pred.predict(df=x,x_timestamp=xt,y_timestamp=yt,pred_len=PRED_LEN,
                             T=1.0,top_p=0.9,sample_count=1)["close"].values for _ in range(SAMPLES)])
            pm = preds.mean(axis=0)
            sig = realized_vol(x["close"].values); vol_hist2.append(sig); vol_hist2=vol_hist2[-200:]
            vr = sig/max(np.mean(vol_hist2[:-1]),1e-4) if len(vol_hist2)>1 else 1.0
            lo_r = np.quantile(preds,a_lvl/2,axis=0); hi_r = np.quantile(preds,1-a_lvl/2,axis=0)
            lo_c,hi_c = conf_engine(pm,truth,preds,{"alpha":a_lvl,"resid":list(st["resid"])},a_lvl,gstar,capstar,mode="aci")
            lo_f,hi_f = conf_engine(pm,truth,preds,st,a_lvl,gstar,capstar,mode="rcac",vol_ratio=vr)
            for name,(lo,hi) in [("raw",(lo_r,hi_r)),("aci",(lo_c,hi_c)),("rcac",(lo_f,hi_f))]:
                exc = ((truth<lo)|(truth>hi)).astype(int)
                exc_stream[name].extend(exc.tolist())
                rows_main.append({"symbol":"spx","level":tag,"seed":seed,"window":w+1,
                                  "method":name,"PICP":round(100*(1-exc.mean()),1),
                                  "Winkler":round(float(np.mean((hi-lo)+(2/a_lvl)*(np.clip(lo-truth,0,None)+np.clip(truth-hi,0,None)))),1)})
            for m in exc_stream:
                if m not in ("raw","aci","rcac"): exc_stream[m].extend([0]*PRED_LEN)  # 占位(本脚本只跑3方法)
        for m in ("raw","aci","rcac"):
            e = np.array(exc_stream[m]); x = int(e.sum())
            rows_stat.append({"symbol":"spx","level":tag,"method":m,"PICP":round(100*(1-e.mean()),1),
                              "exc":x,"exp":int(100*a_lvl),"p_1sided":round(binom_sf(x,100,a_lvl),4),
                              "pass1s":bool(binom_sf(x,100,a_lvl)>0.05)})
    print(f"  spx {tag}% 完成 (γ*={gstar}, cap*={capstar})")

pd.DataFrame(rows_stat).to_csv(os.path.join(RESULTS,"v6_spx.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(rows_main).to_csv(os.path.join(RESULTS,"v6_spx_details.csv"), index=False, encoding="utf-8-sig")
print("\n=== S&P 500 单侧合规 (3 cells) ===")
st = pd.DataFrame(rows_stat)
for m in ["raw","aci","rcac"]:
    print(f"  {m}: {st[st['method']==m]['pass1s'].sum()}/3")
print("已保存 v6_spx.csv / v6_spx_details.csv")
