# -*- coding: utf-8 -*-
"""make_table5.py — 一键重算 RCAC (γ=0.1) 分资产数字, 更新论文 Table 5
【运行】C:\\Users\\apple\\anaconda3\\python.exe make_table5.py
"""
import pandas as pd
df = pd.read_csv(r"D:\RCAC-TSFMs\results\v5_details.csv", encoding="utf-8-sig")
print("Per-asset means (RCAC = aci_g10_asym):")
for sym, sub in df.groupby("symbol"):
    r = {"PICP": sub["PICP_aci_g10_asym"].mean(),
         "MPIW": sub["MPIW_aci_g10_asym"].mean(),
         "Winkler": sub["Winkler_aci_g10_asym"].mean()}
    print(f"  {sym}: " + "  ".join(f"{k}={v:.2f}" for k, v in r.items()))
print("\n把上面的 Winkler 数字填回 Word 表 Table 5 的 RCAC 列即可。")
