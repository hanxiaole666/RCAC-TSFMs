# -*- coding: utf-8 -*-
"""fix_and_summarize_v6.py — 修正CI方向 + 汇总全部论文表格 (12格适配版)
修复: level列int/str类型兼容 | 空CSV文件防御 | 直接输出单侧权威表
【运行】C:\\Users\\apple\\anaconda3\\python.exe 08_fix_and_summarize_v6.py
"""
import pandas as pd, numpy as np, os, math
from math import comb

R = r"D:\RCAC-TSFMs\results"
NOM = {"80": 80.0, "90": 90.0, "95": 95.0}
A   = {"80":0.20, "90":0.10, "95":0.05}

def binom_sf(x, n, p):
    return sum(comb(n,k)*p**k*(1-p)**(n-k) for k in range(x, n+1))

# ---- 读取 (关键修复1: level统一为字符串) ----
st   = pd.read_csv(os.path.join(R, "v6_stats.csv"), encoding="utf-8-sig")
st["level"] = st["level"].astype(str)
main = pd.read_csv(os.path.join(R, "v6_test_main.csv"), encoding="utf-8-sig")
main["level"] = main["level"].astype(str)

# ---- 单侧 Kupiec 全量重算 + 权威表输出 ----
# 关键修复4: v6_stats.csv 混有 DM 检验行 (方法名 DM_vs_*, PICP 为 NaN), 先过滤
st_cells = st[~st["method"].str.startswith("DM")].copy()
n_cells = st_cells[["symbol","level"]].drop_duplicates().shape[0]
print(f"检测到 {n_cells} 个 (资产 x 水平) 格 | 有效方法数: {st_cells['method'].nunique()}")
rows = []
for _, r in st_cells.iterrows():
    a = A[r["level"]]; x = int(round(100 - r["PICP"]))
    p1 = binom_sf(x, 100, a)
    rows.append({"symbol":r["symbol"],"level":r["level"],"method":r["method"],
                 "PICP":r["PICP"],"exc":x,"exp_exc":int(100*a),
                 "p_1sided":round(p1,4),"pass1s":bool(p1>0.05),
                 "p_cc":round(float(r["p_cc"]),4) if not pd.isna(r.get("p_cc")) else np.nan,
                 "CI_lo":r.get("CI_lo"), "CI_hi":r.get("CI_hi")})
one = pd.DataFrame(rows)
one.to_csv(os.path.join(R, "v6_onesided_kupiec.csv"), index=False, encoding="utf-8-sig")

print("\n=== 单侧 Kupiec 合规计数 (每格n=100市场步, H1=下溢覆盖) ===")
counts = one.groupby("method")["pass1s"].sum().astype(int).sort_values(ascending=False)
for m, c in counts.items():
    print(f"  {m:<10} {c}/{n_cells}")

print("\n=== RCAC 逐格明细 ===")
rc = one[one["method"]=="rcac"].sort_values(["symbol","level"])
for _, r in rc.iterrows():
    print(f"  {r['symbol']} {r['level']}%: PICP {r['PICP']:.0f} exc {r['exc']}/{r['exp_exc']} "
          f"p1s {r['p_1sided']:.3f} {'PASS' if r['pass1s'] else 'FAIL'}")

# ---- 修正1: 覆盖率CI (level已字符串化, 过滤恢复正常) ----
print("\n=== 覆盖率CI + 合规判定 ===")
def cov_ci(row):
    if pd.isna(row.get("CI_lo")): return None
    return (round(100-row["CI_hi"],1), round(100-row["CI_lo"],1))
out_rows = []
for _, r in st_cells.iterrows():
    ci = cov_ci(r)
    o = one[(one["symbol"]==r["symbol"])&(one["level"]==r["level"])&(one["method"]==r["method"])].iloc[0]
    out_rows.append({"symbol":r["symbol"],"level":r["level"],"method":r["method"],
        "PICP":r["PICP"], "CI": "[{},{}]".format(ci[0],ci[1]) if ci else "-",
        "p_1sided": o["p_1sided"],
        "Kupiec": "PASS" if o["pass1s"] else "FAIL"})
cdf = pd.DataFrame(out_rows)
for lv in ["80","90","95"]:
    print(f"\n--- 名义 {lv}% ---")
    print(cdf[cdf["level"]==lv].to_string(index=False))
cdf.to_csv(os.path.join(R, "v6_stats_corrected.csv"), index=False, encoding="utf-8-sig")

# ---- 修正2: pooled 汇总 (防御 level 过滤) ----
print("\n=== 80%水平合并3资产 (按Winkler排序) ===")
agg = main.groupby(["level","method"]).agg(
    Winkler=("Winkler","mean"), MPIW=("MPIW","mean"),
    PICP=("PICP","mean"), exceed=("exceed","mean")).round(2).reset_index()
for lv in ["80","90","95"]:
    print(f"\n--- {lv}% ---")
    print(agg[agg["level"]==lv].sort_values("Winkler").to_string(index=False))
agg.to_csv(os.path.join(R, "v6_tables_pooled.csv"), index=False, encoding="utf-8-sig")

# ---- DM 表 (level已字符串化) ----
dm = st[st["method"].str.startswith("DM")].copy()
dm["baseline"] = dm["method"].str.replace("DM_vs_","")
print("\n=== DM + BH-FDR ===")
print(dm[["symbol","level","baseline","DM","p","q_FDR"]].to_string(index=False))

# ---- Chronos / 路径敏感性 (关键修复2: 空文件防御) ----
print("\n=== Chronos / 路径敏感性 ===")
for f in ["v6_chronos_rcac.csv","v6_pathsens.csv","v6_pathsens_256.csv","v6_selection.csv"]:
    p = os.path.join(R, f)
    if not os.path.exists(p):
        print(f"  [缺失] {f}"); continue
    try:
        df_x = pd.read_csv(p, encoding="utf-8-sig")
        if len(df_x) == 0:
            print(f"  [空文件] {f} (该段未产出, 跳过)"); continue
        print(f"\n--- {f} ---"); print(df_x.to_string(index=False))
    except pd.errors.EmptyDataError:
        print(f"  [空文件] {f} (该段未产出, 跳过)")

print("\n完成. 权威单侧表: v6_onesided_kupiec.csv ({}格 x {}方法)".format(n_cells, one["method"].nunique()))
