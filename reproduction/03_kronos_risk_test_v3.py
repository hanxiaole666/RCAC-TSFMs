# -*- coding: utf-8 -*-
"""
Kronos 金融风险预测 v3 —— ACI 自适应校准 + 多种子统计 (论文实验骨架版)
================================================================
相对 v2 的升级:
  1. 三层区间方法对比:  raw quantile PI -> split conformal -> ACI (Gibbs & Candes 2021)
  2. ACI: 按在线覆盖反馈动态调整误差率 α_t, 适应金融残差非平稳性
  3. 多种子 (3 seeds) × 多窗口 (10): 报告 mean±std, 满足顶刊统计要求
  4. Winkler Score (覆盖+宽度联合指标, 区间预测论文标配)
  5. akshare 数据本地缓存, 重复运行不重新下载

【运行】
  C:\\Users\\apple\\anaconda3\\python.exe kronos_risk_test_v3.py

【预计耗时】 10窗口 × 3seeds × 16采样 ≈ 3-5 分钟 (RTX 2060)

【产出 (D盘 results 目录)】
  v3_summary.csv          <- 论文 Table 1: 三种方法 × (PICP/MPIW/Winkler) 的 mean±std
  v3_details.csv          <- 逐窗口×逐seed明细
  v3_last_window.png      <- 三方法区间对比图
"""

import os

PROJECT_ROOT = r"D:\finrisk_project"
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
HF_CACHE_DIR = os.path.join(PROJECT_ROOT, "hf_cache")

os.environ["HF_HOME"]            = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_CACHE_DIR, "transformers")
os.environ["HF_ENDPOINT"]        = "https://hf-mirror.com"
os.environ["MPLCONFIGDIR"]       = os.path.join(HF_CACHE_DIR, "matplotlib")

import sys
for p in [os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "", PROJECT_ROOT]:
    if p and p not in sys.path:
        sys.path.insert(0, p)

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for d in [DATA_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ------------------------------------------------------------------
# 0. 配置区
# ------------------------------------------------------------------
MODEL_NAME  = "NeoQuasar/Kronos-mini"
TOKENIZER   = "NeoQuasar/Kronos-Tokenizer-2k"
MAX_CONTEXT = 2048

SYMBOL      = "sh000300"     # 沪深300; 可换 "600519" 等
LOOKBACK    = 256
PRED_LEN    = 20
N_WINDOWS   = 10
SAMPLE_CNT  = 16
SEEDS       = [42, 43, 44]   # 多种子, 报告 mean±std
ALPHA       = 0.20           # 目标误差率 (80% 区间)
GAMMA       = 0.05           # ACI 学习率 (Gibbs & Candes 建议 0.01~0.1)
RESID_CAP   = 500            # 校准残差滑窗容量

OUT_SUMMARY = os.path.join(RESULTS_DIR, "v3_summary.csv")
OUT_DETAIL  = os.path.join(RESULTS_DIR, "v3_details.csv")
OUT_FIG     = os.path.join(RESULTS_DIR, "v3_last_window.png")
# ------------------------------------------------------------------

def load_kline_cached():
    """优先读本地缓存CSV, 没有再联网拉取"""
    cache = os.path.join(DATA_DIR, f"{SYMBOL}_daily.csv")
    if os.path.exists(cache):
        print(f"  读取本地缓存: {cache}")
        df = pd.read_csv(cache, parse_dates=["timestamps"])
        return df
    import akshare as ak
    print(f"  通过 akshare 拉取 {SYMBOL} ...")
    df = ak.stock_zh_index_daily(symbol=SYMBOL)
    df = df.rename(columns={"date": "timestamps"})
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df["amount"] = df["volume"] * df["close"]
    df = df[["open", "high", "low", "close", "volume", "amount", "timestamps"]].dropna().reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df

# ---------------- 三种区间方法 ----------------
def pi_raw(preds):
    """方法1: 采样分位数区间"""
    return np.quantile(preds, ALPHA/2, axis=0), np.quantile(preds, 1-ALPHA/2, axis=0)

def pi_split_conformal(pred_mean, calib_resid):
    """方法2: 固定α的 split conformal (v2 同款)"""
    if len(calib_resid) == 0:
        return None
    h = np.quantile(np.abs(calib_resid), 1 - ALPHA)
    return pred_mean - h, pred_mean + h

def aci_run(pred_mean, truth, calib_resid_init, alpha0=ALPHA, gamma=GAMMA):
    """方法3: ACI — 在线逐步调整 α_t.
    返回: (lower数组, upper数组, alpha轨迹). 严格因果: t步只用t之前的残差和反馈."""
    alpha = alpha0
    resid = list(calib_resid_init)
    lowers, uppers, alphas = [], [], []
    for t in range(len(truth)):
        h = np.quantile(np.abs(resid), 1 - alpha) if len(resid) >= 10 else np.std(resid) if resid else 50.0
        lo, hi = pred_mean[t] - h, pred_mean[t] + h
        lowers.append(lo); uppers.append(hi); alphas.append(alpha)
        err = 0.0 if lo <= truth[t] <= hi else 1.0        # 覆盖反馈
        alpha = min(0.5, max(0.02, alpha + gamma * (ALPHA - err)))  # α_t 更新
        resid.append(pred_mean[t] - truth[t])              # 残差入库
        resid = resid[-RESID_CAP:]
    return np.array(lowers), np.array(uppers), np.array(alphas)

# ---------------- 指标 ----------------
def winkler(lower, upper, y, alpha=ALPHA):
    """Winkler score: 区间宽度 +  miss 惩罚 (越小越好)"""
    width = upper - lower
    below = np.clip(lower - y, 0, None)
    above = np.clip(y - upper, 0, None)
    return np.mean(width + (2/alpha) * (below + above))

def picp_mpiw(lower, upper, y):
    cover = np.mean((y >= lower) & (y <= upper)) * 100
    width = np.mean(upper - lower)
    return cover, width

def point_metrics(pred_mean, truth):
    err = pred_mean - truth
    mape = np.mean(np.abs(err) / np.clip(truth, 1e-8, None)) * 100
    da = np.mean(np.sign(np.diff(pred_mean)) == np.sign(np.diff(truth))) * 100
    return mape, da

def main():
    print("=" * 64)
    print(f"  Kronos 金融风险预测 v3  ACI校准版   标的: {SYMBOL}")
    print("=" * 64)

    print("\n[1/5] 获取数据...")
    df = load_kline_cached()
    print(f"      {len(df)} 根日线")

    print("\n[2/5] 加载模型...")
    from model import Kronos, KronosTokenizer, KronosPredictor
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER)
    model = Kronos.from_pretrained(MODEL_NAME)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=MAX_CONTEXT)
    print(f"      设备: {device.upper()}")

    base = len(df) - LOOKBACK - PRED_LEN * N_WINDOWS
    if base < 0:
        raise SystemExit(f"数据不足: 需要至少 {LOOKBACK + PRED_LEN*N_WINDOWS} 根, 实际 {len(df)}")

    print(f"\n[3/5] 滚动评测: {N_WINDOWS}窗口 × {len(SEEDS)}seeds × {SAMPLE_CNT}采样 ...")
    all_rows = []
    t_start = time.time()
    last_plot = None

    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        # ACI 的 alpha 和残差滑窗跨窗口连续演化 (模拟真实在线部署)
        aci_resid, aci_alpha = [], ALPHA
        calib_resid = []   # split conformal 的残差库 (同样跨窗口累积)

        for w in range(N_WINDOWS):
            s = base + w * PRED_LEN
            x_df        = df.iloc[s:s+LOOKBACK][["open","high","low","close","volume","amount"]].reset_index(drop=True)
            x_timestamp = df["timestamps"].iloc[s:s+LOOKBACK].reset_index(drop=True)
            y_timestamp = df["timestamps"].iloc[s+LOOKBACK:s+LOOKBACK+PRED_LEN].reset_index(drop=True)
            truth       = df["close"].iloc[s+LOOKBACK:s+LOOKBACK+PRED_LEN].values

            preds = []
            for _ in range(SAMPLE_CNT):
                p = predictor.predict(df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                                      pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=1)
                preds.append(p["close"].values)
            preds = np.array(preds)
            pred_mean = preds.mean(axis=0)

            # ---- 三方法区间 ----
            lo_raw, hi_raw = pi_raw(preds)
            sc = pi_split_conformal(pred_mean, calib_resid)
            lo_aci, hi_aci, alphas = aci_run(pred_mean, truth, aci_resid, alpha0=aci_alpha)

            # ---- 指标 ----
            mape, da = point_metrics(pred_mean, truth)
            c_raw, w_raw = picp_mpiw(lo_raw, hi_raw, truth)
            ws_raw = winkler(lo_raw, hi_raw, truth)
            if sc is not None:
                c_sc, w_sc = picp_mpiw(sc[0], sc[1], truth); ws_sc = winkler(sc[0], sc[1], truth)
            else:
                c_sc = w_sc = ws_sc = np.nan
            c_aci, w_aci = picp_mpiw(lo_aci, hi_aci, truth); ws_aci = winkler(lo_aci, hi_aci, truth)

            all_rows.append({"seed": seed, "window": w+1, "start_date": str(y_timestamp.iloc[0].date()),
                             "MAPE(%)": mape, "DA(%)": da,
                             "PICP_raw": c_raw, "MPIW_raw": w_raw, "Winkler_raw": ws_raw,
                             "PICP_sc": c_sc, "MPIW_sc": w_sc, "Winkler_sc": ws_sc,
                             "PICP_aci": c_aci, "MPIW_aci": w_aci, "Winkler_aci": ws_aci,
                             "alpha_aci_end": alphas[-1]})

            # 更新残差库 (供后续窗口), 保持因果性
            calib_resid.extend((pred_mean - truth).tolist()); calib_resid = calib_resid[-RESID_CAP:]
            aci_resid = list(np.atleast_1d(np.array(calib_resid)))  # ACI 与SC共用残差库, 公平对比
            aci_alpha = alphas[-1]

            last_plot = (np.arange(LOOKBACK), x_df["close"].values, np.arange(LOOKBACK, LOOKBACK+PRED_LEN),
                         truth, pred_mean, lo_raw, hi_raw, sc, lo_aci, hi_aci, alphas)
            print(f"  seed{seed} 窗口{w+1:2d} ({y_timestamp.iloc[0].date()}) | "
                  f"MAPE {mape:.2f}% | PICP raw {c_raw:3.0f}% sc {c_sc if not np.isnan(c_sc) else 0:3.0f}% aci {c_aci:3.0f}% | α_end {alphas[-1]:.2f}")

    print(f"\n      总耗时: {time.time()-t_start:.0f} 秒")

    # ---- 汇总: mean±std ----
    print("\n[4/5] 汇总 (mean±std, 跨 seed×window):")
    rdf = pd.DataFrame(all_rows)
    rdf.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")
    metric_cols = [c for c in rdf.columns if c not in ("seed", "window", "start_date")]
    summary = pd.DataFrame({
        "mean": rdf[metric_cols].mean(),
        "std":  rdf[metric_cols].std(),
    }).round(4)
    summary.to_csv(OUT_SUMMARY, encoding="utf-8-sig")
    print(summary.to_string())

    # ---- 论文核心对比表 (仅三种方法的 PICP/MPIW/Winkler) ----
    print("\n  ===== 论文 Table 1: 区间预测方法对比 =====")
    print(f"  {'方法':<18} {'PICP%':>8} {'MPIW':>8} {'Winkler':>8}")
    for name, pc, wc, wsc in [("Raw Quantile", "PICP_raw", "MPIW_raw", "Winkler_raw"),
                              ("Split Conformal", "PICP_sc", "MPIW_sc", "Winkler_sc"),
                              ("ACI (ours step1)", "PICP_aci", "MPIW_aci", "Winkler_aci")]:
        print(f"  {name:<18} {rdf[pc].mean():8.2f} {rdf[wc].mean():8.2f} {rdf[wsc].mean():8.2f}")

    # ---- 可视化 ----
    print("\n[5/5] 可视化...")
    hx, hclose, fx, truth, pred_mean, lo_r, hi_r, sc, lo_a, hi_a, alphas = last_plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(hx, hclose, "b-", label="History")
    ax.plot(fx, truth, "k-", lw=2, label="Ground Truth")
    ax.plot(fx, pred_mean, "r--", lw=1.5, label="Forecast mean")
    ax.fill_between(fx, lo_r, hi_r, color="orange", alpha=0.25, label="Raw Quantile 80% PI")
    if sc is not None:
        ax.fill_between(fx, sc[0], sc[1], color="purple", alpha=0.15, label="Split Conformal 80% PI")
    ax.fill_between(fx, lo_a, hi_a, color="green", alpha=0.15, label="ACI 80% PI")
    ax.legend(loc="upper left"); ax.set_title(f"{SYMBOL} - Three Interval Methods (last seed/window)")
    ax.grid(alpha=0.3)
    axes[1].plot(fx, alphas, "g-", lw=2)
    axes[1].axhline(ALPHA, color="gray", ls="--", alpha=0.6, label=f"target α={ALPHA}")
    axes[1].set_title("ACI adaptive α_t trajectory"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT_FIG, dpi=150)
    print(f"      已保存: {OUT_FIG}")
    print("\n完成! 若 ACI 的 Winkler < Split Conformal 且 PICP 更接近80%, 故事成立:")
    print("  下一步 = 在 ACI 残差上加波动率加权/分位数回归, 形成你的最终方法。")

if __name__ == "__main__":
    main()
