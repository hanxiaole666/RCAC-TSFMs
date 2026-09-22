# -*- coding: utf-8 -*-
"""17_metric_comparison.py — 与标准评分规则对照: pinball loss + CRPS (审稿 Minor 回应)
重放 CSI300 测试窗 (80%水平, seed42), 逐步存储 raw/aci/rcac 的区间端点与采样路径:
  - Pinball loss at τ ∈ {0.10, 0.90} (即80%区间的两端点分位数)
  - CRPS (从16条采样路径的经验CDF, 用 fair-CRPS 公式)
  - 与 Winkler 对照 (一致性检查)
【运行】约2-3分钟 -> v6_metric_comparison.csv
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
ALPHA, GAMMA, CAP = 0.20, 0.10, 2.0

from model import Kronos, KronosTokenizer, KronosPredictor
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
pred = KronosPredictor(model, tok, device=device, max_context=2048)

def realized_vol(close, win=20):
    ret = np.diff(np.log(np.clip(close,1e-8,None)))
    return max(float(np.std(ret[-win:])),1e-4) if len(ret)>=win else 1e-4

def pinball(y, q, tau): return np.mean(np.maximum(tau*(y-q), (tau-1)*(y-q)))
def fair_crps(samples, y):
    """samples: [S, T], y: [T] -> 平均 CRPS (无偏方程, S=16)"""
    S = samples.shape[0]
    term1 = np.mean(np.abs(samples - y[None,:]), axis=0)
    diffs = np.abs(samples[:,None,:] - samples[None,:,:])
    term2 = np.sum(diffs, axis=(0,1)) / (2*S*(S-1))
    return np.mean(term1 - term2)

def conf_engine(pred_mean, truth, state, a=ALPHA, g=GAMMA, cap=CAP, vol_ratio=1.0, mode="rcac"):
    alpha, resid = state["alpha"], state["resid"]
    lo=np.empty(len(truth)); hi=np.empty(len(truth))
    for t in range(len(truth)):
        h = np.quantile(np.abs(resid), 1-alpha) if len(resid)>=10 else (np.std(resid) if resid else 0.05*abs(pred_mean[t]))
        f = min(cap, max(1.0, vol_ratio)) if mode=="rcac" else 1.0
        lo[t], hi[t] = pred_mean[t]-h*f, pred_mean[t]+h*f
        miss = 0.0 if pred_mean[t]-h <= truth[t] <= pred_mean[t]+h else 1.0
        alpha = min(0.5, max(0.02, alpha + g*(a-miss)))
        resid.append(pred_mean[t]-truth[t]); resid=resid[-500:]
    state["alpha"], state["resid"] = alpha, resid
    return lo, hi

df = pd.read_csv(os.path.join(DATA, "sh000300_daily.csv"), parse_dates=["timestamps"])
base = len(df) - LOOKBACK - PRED_LEN*(N_VAL+N_TEST)
torch.manual_seed(42); np.random.seed(42)
# 暖启动残差 (验证期最后100步)
st = {"alpha":ALPHA, "resid":[]}
for ww in range(N_VAL):
    ss = base + ww*PRED_LEN
    st["resid"].extend((df["close"].iloc[ss+LOOKBACK:ss+LOOKBACK+PRED_LEN].values
                        - df["close"].iloc[ss+LOOKBACK-1]).tolist())
st["resid"] = st["resid"][-500:]
vol_hist = []

rows = []
for w in range(N_VAL, N_VAL+N_TEST):
    s0 = base + w*PRED_LEN
    x_df = df.iloc[s0:s0+LOOKBACK][["open","high","low","close","volume","amount"]].reset_index(drop=True)
    x_ts = df["timestamps"].iloc[s0:s0+LOOKBACK].reset_index(drop=True)
    y_ts = df["timestamps"].iloc[s0+LOOKBACK:s0+LOOKBACK+PRED_LEN].reset_index(drop=True)
    truth = df["close"].iloc[s0+LOOKBACK:s0+LOOKBACK+PRED_LEN].values
    torch.manual_seed(42)
    preds = np.array([pred.predict(df=x_df,x_timestamp=x_ts,y_timestamp=y_ts,pred_len=PRED_LEN,
                     T=1.0,top_p=0.9,sample_count=1)["close"].values for _ in range(SAMPLES)])
    pm = preds.mean(axis=0)
    sig = realized_vol(x_df["close"].values); vol_hist.append(sig); vol_hist=vol_hist[-200:]
    vr = sig/max(np.mean(vol_hist[:-1]),1e-4) if len(vol_hist)>1 else 1.0
    lo_r = np.quantile(preds, ALPHA/2, axis=0); hi_r = np.quantile(preds, 1-ALPHA/2, axis=0)
    lo_a, hi_a = conf_engine(pm, truth, {"alpha":ALPHA,"resid":list(st["resid"])}, mode="aci")
    lo_f, hi_f = conf_engine(pm, truth, st, vol_ratio=vr, mode="rcac")
    for name,(lo,hi) in [("raw",(lo_r,hi_r)),("aci",(lo_a,hi_a)),("rcac",(lo_f,hi_f))]:
        rows.append({"window":w+1,"method":name,
            "pinball_lo":round(pinball(truth, lo, ALPHA/2),2),
            "pinball_hi":round(pinball(truth, hi, 1-ALPHA/2),2),
            "CRPS":round(fair_crps(preds, truth),2),
            "Winkler":round(float(np.mean((hi-lo)+(2/ALPHA)*(np.clip(lo-truth,0,None)+np.clip(truth-hi,0,None)))),2)})
    print(f"  w{w+1} done")

r = pd.DataFrame(rows)
r.to_csv(os.path.join(RESULTS,"v6_metric_comparison.csv"), index=False, encoding="utf-8-sig")
print("\n=== Pinball/CRPS 对照 (CSI300, 80%水平, seed42, 5窗) ===")
print(r.groupby("method")[["pinball_lo","pinball_hi","CRPS","Winkler"]].mean().round(2).to_string())
print("\n判定: 若 rcac 的 pinball/CRPS 劣于 aci -> 宽度代价在标准评分下同样存在(与5.4一致);")
print("      若相当或更优 -> 非对称调制的代价在尾部评分下被低估。")
