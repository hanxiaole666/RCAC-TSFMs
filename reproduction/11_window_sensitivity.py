# -*- coding: utf-8 -*-
"""window_sensitivity.py — 回看窗(128/256/512) × 步长(10/20) 敏感性 (CSI300, 80%水平)
实质性审稿意见 R2-5 实验补充。约 8-10 分钟。
【运行】C:\\Users\\apple\\anaconda3\\python.exe window_sensitivity.py
"""
import os, sys
PROJECT_ROOT = r"D:\RCAC-TSFMs"
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT,"hf_cache")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT,"hf_cache","matplotlib")
sys.path.insert(0, PROJECT_ROOT)
import numpy as np, pandas as pd
DATA, RESULTS = os.path.join(PROJECT_ROOT,"data"), os.path.join(PROJECT_ROOT,"results")
SAMPLES, ALPHA, GAMMA, CAP = 16, 0.20, 0.10, 2.0

from model import Kronos, KronosTokenizer, KronosPredictor
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
pred = KronosPredictor(model, tok, device=device, max_context=2048)

def realized_vol(close, win=20):
    ret = np.diff(np.log(np.clip(close,1e-8,None)))
    return max(float(np.std(ret[-win:])),1e-4) if len(ret)>=win else 1e-4

df = pd.read_csv(os.path.join(DATA,"sh000300_daily.csv"), parse_dates=["timestamps"])
rows = []
for LOOKBACK, PRED_LEN, NW in [(128,20,10),(256,20,5),(512,20,5),(256,10,10)]:
    base = len(df) - LOOKBACK - PRED_LEN*NW
    torch.manual_seed(42); np.random.seed(42)
    resid, vol_hist = [], []
    st = {"alpha":ALPHA, "resid":[]}
    res_lo, res_hi, res_tr = [], [], []
    for w in range(NW):
        s0 = base + w*PRED_LEN
        x_df = df.iloc[s0:s0+LOOKBACK][["open","high","low","close","volume","amount"]].reset_index(drop=True)
        x_ts = df["timestamps"].iloc[s0:s0+LOOKBACK].reset_index(drop=True)
        y_ts = df["timestamps"].iloc[s0+LOOKBACK:s0+LOOKBACK+PRED_LEN].reset_index(drop=True)
        truth = df["close"].iloc[s0+LOOKBACK:s0+LOOKBACK+PRED_LEN].values
        preds = np.array([pred.predict(df=x_df,x_timestamp=x_ts,y_timestamp=y_ts,pred_len=PRED_LEN,
                         T=1.0,top_p=0.9,sample_count=1)["close"].values for _ in range(SAMPLES)])
        pm = preds.mean(axis=0)
        sig = realized_vol(x_df["close"].values); vol_hist.append(sig); vol_hist=vol_hist[-200:]
        vr = sig/max(np.mean(vol_hist[:-1]),1e-4) if len(vol_hist)>1 else 1.0
        lo_r = np.quantile(preds,ALPHA/2,axis=0); hi_r = np.quantile(preds,1-ALPHA/2,axis=0)
        # RCAC
        alpha, res = st["alpha"], st["resid"]
        lo=np.empty(PRED_LEN); hi=np.empty(PRED_LEN)
        for t in range(PRED_LEN):
            h = np.quantile(np.abs(res),1-alpha) if len(res)>=10 else (np.std(res) if res else 0.05*abs(pm[t]))
            f = min(CAP,max(1.0,vr))
            lo[t],hi[t] = pm[t]-h*f, pm[t]+h*f
            miss = 0.0 if pm[t]-h<=truth[t]<=pm[t]+h else 1.0
            alpha = min(0.5,max(0.02,alpha+GAMMA*(ALPHA-miss)))
            res.append(pm[t]-truth[t]); res=res[-500:]
        st["alpha"],st["resid"] = alpha,res
        res_lo.append(lo); res_hi.append(hi); res_tr.append(truth)
    all_lo=np.concatenate(res_lo); all_hi=np.concatenate(res_hi); all_tr=np.concatenate(res_tr)
    picp = np.mean((all_tr>=all_lo)&(all_tr<=all_hi))*100
    mpiw = float(np.mean(all_hi-all_lo))
    rows.append({"lookback":LOOKBACK,"pred_len":PRED_LEN,"n_windows":NW,
                 "rcac_PICP":round(picp,1),"rcac_MPIW":round(mpiw,1)})
    print(f"  LB={LOOKBACK} PL={PRED_LEN}: RCAC PICP {picp:.1f}% MPIW {mpiw:.1f}")
pd.DataFrame(rows).to_csv(os.path.join(RESULTS,"v6_window_sens.csv"), index=False, encoding="utf-8-sig")
print("\n完成 -> v6_window_sens.csv")
