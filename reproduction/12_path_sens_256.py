# -*- coding: utf-8 -*-
"""path_sens_256.py — 256路径敏感性 (CSI300 测试5窗, 约4分钟)
【运行】C:\\Users\\apple\\anaconda3\\python.exe path_sens_256.py
"""
import os, sys
PROJECT_ROOT = r"D:\finrisk_project"
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT,"hf_cache")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT,"hf_cache","matplotlib")
sys.path.insert(0, PROJECT_ROOT)
import numpy as np, pandas as pd
DATA, RESULTS = os.path.join(PROJECT_ROOT,"data"), os.path.join(PROJECT_ROOT,"results")
LOOKBACK, PRED_LEN, N_VAL, N_TEST, ALPHA = 256, 20, 5, 5, 0.20

from model import Kronos, KronosTokenizer, KronosPredictor
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
pred = KronosPredictor(model, tok, device=device, max_context=2048)

df = pd.read_csv(os.path.join(DATA,"sh000300_daily.csv"), parse_dates=["timestamps"])
base = len(df) - LOOKBACK - PRED_LEN*(N_VAL+N_TEST)
rows = []
for w in range(N_VAL, N_VAL+N_TEST):
    s0 = base + w*PRED_LEN
    x_df = df.iloc[s0:s0+LOOKBACK][["open","high","low","close","volume","amount"]].reset_index(drop=True)
    x_ts = df["timestamps"].iloc[s0:s0+LOOKBACK].reset_index(drop=True)
    y_ts = df["timestamps"].iloc[s0+LOOKBACK:s0+LOOKBACK+PRED_LEN].reset_index(drop=True)
    truth = df["close"].iloc[s0+LOOKBACK:s0+LOOKBACK+PRED_LEN].values
    torch.manual_seed(42); np.random.seed(42)
    preds = np.array([pred.predict(df=x_df,x_timestamp=x_ts,y_timestamp=y_ts,pred_len=PRED_LEN,
                     T=1.0,top_p=0.9,sample_count=1)["close"].values for _ in range(256)])
    lo, hi = np.quantile(preds, ALPHA/2, axis=0), np.quantile(preds, 1-ALPHA/2, axis=0)
    picp = np.mean((truth>=lo)&(truth<=hi))*100
    mpiw = np.mean(hi-lo)
    mc_se = np.std(np.mean((truth[None,:]>=preds)&(truth[None,:]<=np.sort(preds,axis=0)),axis=1))*0  # placeholder
    # 蒙特卡洛误差: 对路径重抽样估计分位数区间的PICP波动
    rng = np.random.default_rng(0); boots = []
    for _ in range(200):
        idx = rng.integers(0, 256, 256)
        lo_b, hi_b = np.quantile(preds[idx], ALPHA/2, axis=0), np.quantile(preds[idx], 1-ALPHA/2, axis=0)
        boots.append(np.mean((truth>=lo_b)&(truth<=hi_b))*100)
    rows.append({"window": w+1, "start": str(y_ts.iloc[0].date()),
                 "raw_PICP_256": round(picp,1), "MPIW": round(mpiw,1),
                 "MC_SD": round(float(np.std(boots)),1)})
    print(f"  w{w+1} ({y_ts.iloc[0].date()}): PICP {picp:.0f}%  MPIW {mpiw:.0f}  MC_SD {np.std(boots):.1f}pp")
r = pd.DataFrame(rows)
r.to_csv(os.path.join(RESULTS,"v6_pathsens_256.csv"), index=False, encoding="utf-8-sig")
print(f"\n256路径均值: {r['raw_PICP_256'].mean():.1f}%  (16路径≈61%, 64路径≈55%)")
print("-> 若256路径均值仍显著低于80%, 欠分散诊断彻底坐实; 若反而回升, 需在稿中修正路径数结论。")
