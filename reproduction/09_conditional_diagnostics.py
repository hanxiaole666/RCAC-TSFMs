# -*- coding: utf-8 -*-
"""conditional_diagnostics.py — 波动率状态分组的条件覆盖诊断 (纯pandas, 无需GPU, 秒级)
【运行】C:\\Users\\apple\\anaconda3\\python.exe conditional_diagnostics.py
【原理】vol_ratio 仅需输入窗价格即可重算 (与v6完全相同的因果定义), 无需重跑TSFM
"""
import pandas as pd, numpy as np, os

DATA, RESULTS = r"D:\finrisk_project\data", r"D:\finrisk_project\results"
LOOKBACK, PRED_LEN, N_VAL, N_TEST = 256, 20, 5, 5

def realized_vol(close, win=20):
    ret = np.diff(np.log(np.clip(close,1e-8,None)))
    return max(float(np.std(ret[-win:])),1e-4) if len(ret)>=win else (float(np.std(ret)) if len(ret)>1 else 1e-4)

main = pd.read_csv(os.path.join(RESULTS,"v6_test_main.csv"), encoding="utf-8-sig")
main80 = main[main["level"]==80].copy()

vols = []
for _, r in main80.iterrows():
    df = pd.read_csv(os.path.join(DATA, f"{r['symbol']}_daily.csv"), parse_dates=["timestamps"])
    base = len(df) - LOOKBACK - PRED_LEN*(N_VAL+N_TEST)
    s0 = base + (int(r["window"])-1)*PRED_LEN
    hist = []
    for w in range(int(r["window"])):
        ss = base + w*PRED_LEN
        hist.append(realized_vol(df["close"].iloc[ss:ss+LOOKBACK].values))
    sig = hist[-1]; sig_bar = np.mean(hist[:-1]) if len(hist)>1 else sig
    vols.append(sig/max(sig_bar,1e-4))
main80["vol_ratio"] = vols
main80["vol_state"] = pd.qcut(main80["vol_ratio"], 3, labels=["calm","mid","turbulent"])

print("=== 条件覆盖诊断 (80%水平, 按波动率状态三分位, PICP均值) ===")
print(f"  {'method':<8} {'calm':>6} {'mid':>6} {'turbulent':>9} 极差")
piv = main80.groupby(["method","vol_state"], observed=True)["PICP"].mean().unstack()
for mth in ["raw","sc","aci","garch","enbpi","gated","rcac"]:
    if mth not in piv.index: continue
    g = piv.loc[mth]
    print(f"  {mth:<8} {g['calm']:6.1f} {g['mid']:6.1f} {g['turbulent']:9.1f} {g.max()-g.min():5.1f}")

main80.to_csv(os.path.join(RESULTS,"v6_conditional.csv"), index=False, encoding="utf-8-sig")
print("\n已保存 v6_conditional.csv (含vol_ratio/vol_state列)")
