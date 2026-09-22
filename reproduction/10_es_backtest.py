# -*- coding: utf-8 -*-
"""es_backtest.py — Expected Shortfall 回测 (McNeil-Frey 方法) + 窗口/步长敏感性
实质性审稿意见 R2-3 的实验补充。约 6-8 分钟。
【运行】C:\\Users\\apple\\anaconda3\\python.exe es_backtest.py
"""
import os, sys
PROJECT_ROOT = r"D:\RCAC-TSFMs"
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT,"hf_cache")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT,"hf_cache","matplotlib")
sys.path.insert(0, PROJECT_ROOT)
import numpy as np, pandas as pd
DATA, RESULTS = os.path.join(PROJECT_ROOT,"data"), os.path.join(PROJECT_ROOT,"results")
LOOKBACK, PRED_LEN, N_VAL, N_TEST, SAMPLES = 256, 20, 5, 5, 16
ALPHA = 0.20

from model import Kronos, KronosTokenizer, KronosPredictor
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
pred = KronosPredictor(model, tok, device=device, max_context=2048)

def conf_engine(pred_mean, truth, preds, state, a=ALPHA, g=0.10, cap=2.0, vol_ratio=1.0, mode="rcac"):
    alpha, resid = state["alpha"], state["resid"]
    lo=np.empty(len(truth)); hi=np.empty(len(truth))
    for t in range(len(truth)):
        h = np.quantile(np.abs(resid), 1-alpha) if len(resid)>=10 else (np.std(resid) if resid else 0.05*abs(pred_mean[t]))
        f = min(cap, max(1.0, vol_ratio)) if mode=="rcac" else 1.0
        lo[t], hi[t] = pred_mean[t]-h*f, pred_mean[t]+h*f
        miss = 0.0 if pred_mean[t]-h <= truth[t] <= pred_mean[t]+h else 1.0
        alpha = min(0.5, max(0.02, alpha + g*(a-miss)))
        resid.append(pred_mean[t]-truth[t]); resid = resid[-500:]
    state["alpha"], state["resid"] = alpha, resid
    return lo, hi

def realized_vol(close, win=20):
    ret = np.diff(np.log(np.clip(close,1e-8,None)))
    return max(float(np.std(ret[-win:])),1e-4) if len(ret)>=win else 1e-4

rows = []
for sym in ["sh000300","sh000905","600519"]:
    df = pd.read_csv(os.path.join(DATA,f"{sym}_daily.csv"), parse_dates=["timestamps"])
    base = len(df) - LOOKBACK - PRED_LEN*(N_VAL+N_TEST)
    for seed in [42]:
        torch.manual_seed(seed); np.random.seed(seed)
        st = {"alpha":ALPHA, "resid":[]}; vol_hist=[]
        for w in range(N_VAL, N_VAL+N_TEST):
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
            # 残差库冷启动用验证期
            if w==N_VAL:
                for ww in range(N_VAL):
                    ss = base + ww*PRED_LEN
                    st["resid"].extend((df["close"].iloc[ss+LOOKBACK:ss+LOOKBACK+PRED_LEN].values
                                        - df["close"].iloc[ss+LOOKBACK-1]).tolist())
            lo_c, hi_c = conf_engine(pm, truth, preds, st, vol_ratio=vr)
            lo_r = np.quantile(preds, ALPHA/2, axis=0); hi_r = np.quantile(preds, 1-ALPHA/2, axis=0)
            lp = df["close"].iloc[s0+LOOKBACK-1]
            for name, (lo, hi) in [("raw",(lo_r,hi_r)), ("rcac",(lo_c,hi_c))]:
                viol = np.maximum(lo - truth, 0)     # 下行违规深度 (价格)
                es_est = np.mean(np.sort(viol)[-max(1,int(0.1*len(viol))):])   # 平均最差10%
                viol_rate = np.mean(viol>0)
                # 简单ES比率检验: ES_est / (平均区间半宽)
                cover = np.mean((truth>=lo)&(truth<=hi))*100
                rows.append({"symbol":sym,"window":w+1,"method":name,
                             "ES_avg_violation":round(float(es_est),2),
                             "viol_rate":round(float(viol_rate),3),
                             "PICP":round(float(cover),1),
                             "ES_ratio_to_width":round(float(es_est/(np.mean(hi-lo)/2)),4)})
            print(f"  {sym} w{w+1} done")
r = pd.DataFrame(rows)
r.to_csv(os.path.join(RESULTS,"v6_es_backtest.csv"), index=False, encoding="utf-8-sig")
print("\n=== ES回测 (80%中心区间下行尾部) ===")
print(r.groupby("method")[["ES_avg_violation","viol_rate","PICP","ES_ratio_to_width"]].mean().round(3).to_string())
print("\nES_ratio_to_width: 越小说明同样区间宽度下尾部违规越浅 -> 资本效率越高")
