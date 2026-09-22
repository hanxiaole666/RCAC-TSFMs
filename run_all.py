#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py — RCAC 论文一键复现主控脚本
================================================================
按顺序执行完整实验流水线, 每个阶段产出对应论文表格/图片。
支持断点续跑: 已存在的 CSV 输出会自动跳过 (加 --force 强制重跑)。

【用法】
  python run_all.py            # 断点续跑全部
  python run_all.py --force    # 强制全部重跑
  python run_all.py --stage 6  # 只跑第6阶段及之后

【阶段对照表】 (论文 Table/图 <- 阶段)
  阶段 脚本                              产出                        预计耗时
  0   00_local_ai_bench.py              硬件环境记录                <1 min
  1   01_kronos_risk_test.py            Fig 2/3 底稿 (合成数据)     ~1 min
  2   02_kronos_risk_test_v2.py         真实数据+分位数区间          ~5 min
  3   03_kronos_risk_test_v3.py         ACI 自适应校准              ~5 min
  4   04_kronos_risk_test_v4.py         2x2 消融 (对称调制负结果)    ~6 min
  5   05_kronos_risk_test_v5.py         非对称调制+Chronos+gamma    ~6 min
  6   06_kronos_risk_test_v6.py         Table 2a/2b, B3 (主实验)    ~15 min
  7   07_dm_test.py                     Table 4 (DM检验)            <10 s
  8   08_fix_and_summarize_v6.py        CI修正+汇总表               <10 s
  9   09_conditional_diagnostics.py     Table B2 (条件覆盖)         <10 s
  10  10_es_backtest.py                 Table B4 (ES诊断)           ~8 min
  11  11_window_sensitivity.py          Table B5 (窗口敏感性)       ~10 min
  12  12_path_sens_256.py               256路径确认                 ~4 min
  13  13_make_figures.py                Fig 1-4                     ~5 min
  14  14_make_table5.py                Table 2a 辅助核对           <10 s

总计约 60-70 分钟 (RTX 2060 6GB 基准)。
"""
import os, sys, subprocess, time, argparse

HERE  = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.join(HERE, "reproduction")
# 产物可能在仓库 results/ 或脚本默认的 D:\\RCAC-TSFMs\\results (各脚本的 PROJECT_ROOT)
RESULTS_CANDIDATES = [os.path.join(HERE, "results"), r"D:\RCAC-TSFMs\results"]

def markers_exist(markers):
    return any(all(os.path.exists(os.path.join(rd, m)) for m in markers)
               for rd in RESULTS_CANDIDATES)

STAGES = [
    (0,  "00_local_ai_bench.py",          []),
    (1,  "01_kronos_risk_test.py",        []),
    (2,  "02_kronos_risk_test_v2.py",     []),
    (3,  "03_kronos_risk_test_v3.py",     []),
    (4,  "04_kronos_risk_test_v4.py",     []),
    (5,  "05_kronos_risk_test_v5.py",     []),
    (6,  "06_kronos_risk_test_v6.py",     ["v6_stats.csv", "v6_test_main.csv", "v6_selection.csv"]),
    (7,  "07_dm_test.py",                 ["dm_results.csv"]),
    (8,  "08_fix_and_summarize_v6.py",    ["v6_stats_corrected.csv", "v6_tables_pooled.csv"]),
    (9,  "09_conditional_diagnostics.py", ["v6_conditional.csv"]),
    (10, "10_es_backtest.py",             ["v6_es_backtest.csv"]),
    (11, "11_window_sensitivity.py",      ["v6_window_sens.csv"]),
    (12, "12_path_sens_256.py",           ["v6_pathsens_256.csv"]),
    (13, "13_make_figures.py",            ["fig1_framework.png", "fig4_cases.png"]),
    (14, "14_make_table5.py",             []),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="强制重跑全部")
    ap.add_argument("--stage", type=int, default=0, help="从该阶段开始")
    args = ap.parse_args()
    t0 = time.time()
    for stage, script, markers in STAGES:
        if stage < args.stage: continue
        path = os.path.join(REPRO, script)
        if not os.path.exists(path):
            print("[中止] 找不到脚本: {}".format(path)); sys.exit(1)
        if not args.force and markers and markers_exist(markers):
            print("[跳过] 阶段{} {} (产物已存在, --force 可重跑)".format(stage, script)); continue
        print("="*70); print("[阶段 {}] {} ...".format(stage, script)); print("="*70)
        r = subprocess.run([sys.executable, path], cwd=HERE)
        if r.returncode != 0:
            print("\n[中止] 阶段{} 失败 (exit {}). 修复后加 --stage {} 续跑。".format(stage, r.returncode, stage))
            sys.exit(1)
    print("\n[完成] 全部阶段耗时 {:.0f} 分钟。产物在 results\\ 目录。".format((time.time()-t0)/60))

if __name__ == "__main__":
    main()
