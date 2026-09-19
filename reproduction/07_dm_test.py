# -*- coding: utf-8 -*-
"""
DM 显著性检验 —— RCAC vs 基线方法 (基于 v5_details.csv, 零推理耗时, 秒出结果)
================================================================
Diebold-Mariano (1995) 检验: 比较两组预测的损失序列差异是否显著
H0: 两种方法预测能力相同 (E[d_t] = 0)

【运行】
  C:\\Users\\apple\\anaconda3\\python.exe dm_test.py
【输入】
  D:\\finrisk_project\\results\\v5_details.csv   (v5 已生成)
【输出】
  D:\\finrisk_project\\results\\dm_results.csv
"""

import os
import numpy as np
import pandas as pd

RESULTS_DIR = r"D:\finrisk_project\results"
DETAIL = os.path.join(RESULTS_DIR, "v5_details.csv")
OUT    = os.path.join(RESULTS_DIR, "dm_results.csv")

H_LIST = [1, 5]          # DM长滞后: h=1(保守) 与 h=5; 窗口长度20, 更大h会高估
OURS   = "Winkler_aci_g10_asym"          # RCAC
BASELINES = [("Winkler_sc", "Split Conformal"),
             ("Winkler_aci", "ACI γ=0.05"),
             ("Winkler_aci_asym", "ACI γ=0.05+asym"),
             ("Winkler_aci_g10", "ACI γ=0.1"),
             ("Winkler_raw", "Raw Quantile")]

def norm_cdf(x):
    try:
        from scipy.stats import norm
        return norm.cdf(x)
    except ImportError:
        import math
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def dm_test(e1, e2, h=1):
    """e1: 本文方法损失, e2: 基线损失. 返回 (DM统计量, p值, 均值差)."""
    d = np.asarray(e1, dtype=float) - np.asarray(e2, dtype=float)
    T = len(d)
    if T < 10:
        return np.nan, np.nan, np.nan
    dbar = d.mean()
    # Newey-West 长程方差 (Bartlett 权重)
    lrv = np.mean((d - dbar) ** 2)
    for l in range(1, min(h, T - 1)):
        cov = np.mean((d[l:] - dbar) * (d[:-l] - dbar))
        lrv += 2 * (1 - l / h) * cov
    if lrv <= 0:                       # 差异恒定(方差0)
        if dbar == 0:
            return 0.0, 1.0, dbar        # 完全相同: DM=0, p=1
        return (np.inf if dbar < 0 else -np.inf), 0.0, dbar
    dm = dbar / np.sqrt(lrv / T)
    p = 2 * (1 - norm_cdf(abs(dm)))
    return dm, p, dbar

def main():
    df = pd.read_csv(DETAIL, encoding="utf-8-sig")
    print(f"读取 {len(df)} 行明细 (标的: {df['symbol'].unique().tolist()})\n")

    rows = []
    # ---- 总体 (跨标的×seed×窗口) ----
    print("=" * 78)
    print("  DM 检验: RCAC (aci_g10_asym) vs 基线   [损失=Winkler, 越小越好]")
    print("=" * 78)
    for scope, sub in [("总体", df)] + [(s, df[df["symbol"] == s]) for s in df["symbol"].unique()]:
        print(f"\n【{scope}】 n={len(sub)}")
        print(f"  {'vs 基线':<24} {'h=1 DM':>9} {'p':>8} {'h=5 DM':>9} {'p':>8}   判定")
        for col, name in BASELINES:
            pair = sub[[OURS, col]].dropna()
            if len(pair) < 10:
                print(f"  {name:<24} 样本不足({len(pair)})")
                continue
            dm1, p1, d1 = dm_test(pair[OURS], pair[col], h=H_LIST[0])
            dm5, p5, _  = dm_test(pair[OURS], pair[col], h=H_LIST[1])
            sig1 = "***" if p1 < 0.01 else "**" if p1 < 0.05 else "*" if p1 < 0.1 else "ns"
            sig5 = "***" if p5 < 0.01 else "**" if p5 < 0.05 else "*" if p5 < 0.1 else "ns"
            print(f"  {name:<24} {dm1:9.3f} {p1:8.4f} {dm5:9.3f} {p5:8.4f}   {sig1}/{sig5}")
            rows.append({"scope": scope, "baseline": name, "n": len(pair),
                         "DM_h1": round(dm1, 4), "p_h1": round(p1, 4),
                         "DM_h5": round(dm5, 4), "p_h5": round(p5, 4),
                         "mean_diff": round(d1, 4)})

    # 附加: 数据一致性自检 (同一序列与自身比较, 预期 DM=0, p=1)
    print("\n【一致性自检】同一损失序列与自身比较 (预期 DM=0, p=1):")
    self_col = "Winkler_aci"
    dm1, p1, _ = dm_test(df[self_col].dropna(), df[self_col].dropna(), h=1)
    print(f"  {self_col} vs 自身: DM={dm1:.3f}, p={p1:.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存: {OUT}")
    print("显著性标记: *** p<0.01  ** p<0.05  * p<0.1  ns 不显著")
    print("\n论文写法: 报告 h=1 结果为主 (保守), h=5 为稳健性;")
    print("          DM<0 且 p<0.05 => RCAC 显著优于该基线.")

if __name__ == "__main__":
    main()
