# -*- coding: utf-8 -*-
"""fix_and_summarize_v6.py — 修正CI方向 + 汇总全部论文表格 (秒出, 无需重跑实验)
【运行】C:\\Users\\apple\\anaconda3\\python.exe fix_and_summarize_v6.py
"""
import pandas as pd, numpy as np, os

R = r"D:\finrisk_project\results"
NOM = {"80": 80.0, "90": 90.0, "95": 95.0}

st  = pd.read_csv(os.path.join(R, "v6_stats.csv"), encoding="utf-8-sig")
main = pd.read_csv(os.path.join(R, "v6_test_main.csv"), encoding="utf-8-sig")

print("="*80)
print("【修正1】覆盖率CI翻转 + Kupiec合规判定 (Kupiec通过 = p_uc>0.05; 严格合规 = CI下界>=名义)")
print("="*80)
rows = []
for _, r in st[st["method"].str.contains("DM")==False].iterrows():
    if pd.isna(r.get("CI_lo")): continue
    cov_lo = 100 - r["CI_hi"]; cov_hi = 100 - r["CI_lo"]
    nom = NOM[str(r["level"])]
    rows.append({"symbol": r["symbol"], "level": r["level"], "method": r["method"],
                 "PICP": r["PICP"], "CI_lo": round(cov_lo,1), "CI_hi": round(cov_hi,1),
                 "p_uc": r["p_uc"], "p_cc": r["p_cc"],
                 "Kupiec": "PASS" if r["p_uc"]>0.05 else "FAIL",
                 "严格合规": "YES" if cov_lo>=nom else "no"})
cdf = pd.DataFrame(rows)
for lv in ["80","90","95"]:
    print(f"\n--- 名义 {lv}% ---")
    print(cdf[cdf["level"]==lv].to_string(index=False))

print("\n【合规率汇总 (Kupiec PASS 的 资产×水平 格数 / 总格数)】")
summ = cdf.groupby("method")["Kupiec"].apply(lambda s: f"{(s=='PASS').sum()}/{len(s)}")
print(summ.to_string())

print("\n" + "="*80)
print("【修正2】Winkler / MPIW / 超额损失 汇总 (测试期, 按水平合并3资产)")
print("="*80)
agg = main.groupby(["level","method"]).agg(
    Winkler=("Winkler","mean"), MPIW=("MPIW","mean"),
    PICP=("PICP","mean"), exceed=("exceed","mean")).round(2).reset_index()
for lv in ["80","90","95"]:
    print(f"\n--- 名义 {lv}% (按Winkler排序) ---")
    print(agg[agg["level"]==lv].sort_values("Winkler").to_string(index=False))

print("\n" + "="*80)
print("【表】DM + BH-FDR (rcac vs 基线, 来自v6_stats)")
print("="*80)
dm = st[st["method"].str.startswith("DM_vs_")][["symbol","level","method","DM","p","q_FDR"]]
print(dm.to_string(index=False) if len(dm) else "(无DM行)")

print("\n" + "="*80)
print("【表】Chronos 修复 / 路径敏感性 (原始CSV)")
print("="*80)
for f in ["v6_chronos_rcac.csv","v6_pathsens.csv","v6_selection.csv"]:
    p = os.path.join(R, f)
    if os.path.exists(p):
        print(f"\n--- {f} ---"); print(pd.read_csv(p, encoding="utf-8-sig").to_string(index=False))

cdf.to_csv(os.path.join(R,"v6_stats_corrected.csv"), index=False, encoding="utf-8-sig")
agg.to_csv(os.path.join(R,"v6_tables_pooled.csv"), index=False, encoding="utf-8-sig")
print("\n已保存: v6_stats_corrected.csv / v6_tables_pooled.csv")
