# -*- coding: utf-8 -*-
"""
Kronos 金融风险预测 升级版 v2
================================================================
新增能力 (相对 v1):
  1. akshare 拉取真实 A 股数据 (默认沪深300指数, 可换个股)
  2. 分位数预测区间 (10%/90% 分位数, 替代粗糙的 mean±1.28σ)
  3. Split Conformal 校准 —— 把 80% 覆盖率从严重欠覆盖拉回 ~80%
  4. 滚动样本外评测 (rolling-origin, 金融时序论文必须这么做)
  5. 对比表: 校准前 vs 校准后 (PICP / 区间宽度) -> 论文消融表雏形

【依赖】
  C:\\Users\\apple\\anaconda3\\python.exe -m pip install akshare einops
  (einops 已装可跳过)

【运行】
  C:\\Users\\apple\\anaconda3\\python.exe kronos_risk_test_v2.py

【产出 (D盘)】
  D:\\finrisk_project\\results\\v2_metrics_summary.csv   <- 总对比表
  D:\\finrisk_project\\results\\v2_window_details.csv    <- 逐窗口明细
  D:\\finrisk_project\\results\\v2_last_window.png       <- 最近窗口可视化
"""

import os

PROJECT_ROOT = r"D:\finrisk_project"
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
HF_CACHE_DIR = os.path.join(PROJECT_ROOT, "hf_cache")

# ---- 缓存重定向 (import transformers 之前) ----
os.environ["HF_HOME"]               = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"]    = os.path.join(HF_CACHE_DIR, "transformers")
os.environ["HF_ENDPOINT"]           = "https://hf-mirror.com"
os.environ["MPLCONFIGDIR"]          = os.path.join(HF_CACHE_DIR, "matplotlib")

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

SYMBOL      = "sh000300"   # 沪深300指数; 个股如 "600519"(贵州茅台) / "000001"(平安银行)
START_DATE  = "20180101"
LOOKBACK    = 256            # 历史窗口 (日线约1年, 兼顾速度与信息)
PRED_LEN    = 20             # 预测未来20个交易日 (约1个月, 风险预测常用尺度)
N_WINDOWS   = 5              # 滚动评测窗口数 (论文建议 >=5, 可加大)
SAMPLE_CNT  = 16             # 采样路径数 (分位数区间需要 >=16 才稳定)
ALPHA       = 0.20           # 目标误差率 -> 80% 区间

OUT_SUMMARY = os.path.join(RESULTS_DIR, "v2_metrics_summary.csv")
OUT_DETAIL  = os.path.join(RESULTS_DIR, "v2_window_details.csv")
OUT_FIG     = os.path.join(RESULTS_DIR, "v2_last_window.png")
# ------------------------------------------------------------------

def load_real_kline():
    """用 akshare 拉取真实行情, 失败则回退到 v1 的合成数据"""
    try:
        import akshare as ak
        print(f"  正在通过 akshare 拉取 {SYMBOL} 日线数据...")
        df = ak.stock_zh_index_daily(symbol=SYMBOL) if SYMBOL.startswith("sh0") \
             else ak.stock_zh_a_hist(symbol=SYMBOL, period="daily", start_date=START_DATE, adjust="qfq")
        # 统一列名
        colmap = {"日期": "timestamps", "date": "timestamps", "开盘": "open", "open": "open",
                  "最高": "high", "high": "high", "最低": "low", "low": "low",
                  "收盘": "close", "close": "close", "成交量": "volume", "volume": "volume"}
        df = df.rename(columns=colmap)
        df["timestamps"] = pd.to_datetime(df["timestamps"])
        if "amount" not in df.columns:
            df["amount"] = df.get("volume", 0) * df["close"]
        if "volume" not in df.columns:
            df["volume"] = 0.0
        df = df[["open", "high", "low", "close", "volume", "amount", "timestamps"]].dropna().reset_index(drop=True)
        df.to_csv(os.path.join(DATA_DIR, f"{SYMBOL}_daily.csv"), index=False)
        print(f"  获取成功: {len(df)} 根日线, {df['timestamps'].iloc[0].date()} ~ {df['timestamps'].iloc[-1].date()}")
        return df
    except Exception as e:
        print(f"  [警告] akshare 获取失败 ({e}), 回退到合成GARCH数据")
        from kronos_risk_test import make_synthetic_kline
        return make_synthetic_kline()

def conformal_factor(calib_residuals, alpha):
    """Split conformal: 用校准集绝对残差的分位数作为区间半宽"""
    if len(calib_residuals) == 0:
        return None
    q = np.quantile(np.abs(calib_residuals), 1 - alpha)
    return float(q)

def evaluate_window(pred_q10, pred_q90, pred_mean, truth_close, conf_halfwidth):
    """单窗口评测: 点预测 + 区间预测 (校准前后)"""
    err = pred_mean - truth_close
    mae, rmse = np.mean(np.abs(err)), np.sqrt(np.mean(err**2))
    mape = np.mean(np.abs(err) / np.clip(truth_close, 1e-8, None)) * 100
    da = np.mean(np.sign(np.diff(pred_mean)) == np.sign(np.diff(truth_close))) * 100

    # 校准前: 分位数区间
    cover_raw = np.mean((truth_close >= pred_q10) & (truth_close <= pred_q90)) * 100
    width_raw = np.mean(pred_q90 - pred_q10)

    # 校准后: conformal 区间 (若该校准因子已计算)
    if conf_halfwidth is not None:
        lo, hi = pred_mean - conf_halfwidth, pred_mean + conf_halfwidth
        cover_conf = np.mean((truth_close >= lo) & (truth_close <= hi)) * 100
        width_conf = np.mean(hi - lo)
    else:
        cover_conf, width_conf = np.nan, np.nan

    return {"MAE": mae, "RMSE": rmse, "MAPE(%)": mape, "DA(%)": da,
            "PICP_raw(%)": cover_raw, "MPIW_raw": width_raw,
            "PICP_conformal(%)": cover_conf, "MPIW_conformal": width_conf}

def main():
    print("=" * 64)
    print(f"  Kronos 金融风险预测 v2   标的: {SYMBOL}   模型: {MODEL_NAME}")
    print("=" * 64)

    # ---- 1. 数据 ----
    print("\n[1/5] 获取数据...")
    df = load_real_kline()
    need = LOOKBACK + PRED_LEN * (N_WINDOWS + 1)   # 额外+1个窗口做首个校准集
    if len(df) < need:
        print(f"  [警告] 数据仅 {len(df)} 根, 不足 {need}, 自动减少窗口数")
        n_win = max(1, (len(df) - LOOKBACK) // PRED_LEN - 1)
    else:
        n_win = N_WINDOWS

    # ---- 2. 模型 ----
    print("\n[2/5] 加载 Kronos...")
    from model import Kronos, KronosTokenizer, KronosPredictor
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER)
    model = Kronos.from_pretrained(MODEL_NAME)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=MAX_CONTEXT)
    print(f"      设备: {device.upper()}")

    # ---- 3. 滚动评测 ----
    print(f"\n[3/5] 滚动样本外评测: {n_win} 个窗口 × 预测{PRED_LEN}天 × 采样{SAMPLE_CNT}条...")
    results, calib_residuals = [], []
    total_start = time.time()
    last_plot = None

    base = len(df) - LOOKBACK - PRED_LEN * n_win
    for w in range(n_win):
        s = base + w * PRED_LEN
        x_df        = df.iloc[s:s+LOOKBACK][["open","high","low","close","volume","amount"]].reset_index(drop=True)
        x_timestamp = df["timestamps"].iloc[s:s+LOOKBACK].reset_index(drop=True)
        y_timestamp = df["timestamps"].iloc[s+LOOKBACK:s+LOOKBACK+PRED_LEN].reset_index(drop=True)
        truth       = df["close"].iloc[s+LOOKBACK:s+LOOKBACK+PRED_LEN].values

        # 前一个窗口的残差 -> 本窗口的 conformal 校准因子 (严格的时序因果, 无泄漏)
        cw = conformal_factor(calib_residuals, ALPHA)

        preds = []
        for i in range(SAMPLE_CNT):
            p = predictor.predict(df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                                  pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=1)
            preds.append(p["close"].values)
        preds = np.array(preds)                      # [SAMPLE_CNT, PRED_LEN]
        pred_mean = preds.mean(axis=0)
        pred_q10  = np.quantile(preds, ALPHA/2, axis=0)
        pred_q90  = np.quantile(preds, 1-ALPHA/2, axis=0)

        # 记录本窗口残差, 供下一个窗口校准
        calib_residuals.extend((pred_mean - truth).tolist())
        calib_residuals = calib_residuals[-500:]     # 滑窗保留最近500个, 适应分布漂移

        m = evaluate_window(pred_q10, pred_q90, pred_mean, truth, cw)
        m["window"] = w + 1
        m["start_date"] = str(y_timestamp.iloc[0].date())
        results.append(m)
        print(f"  窗口 {w+1}/{n_win} ({m['start_date']}) | MAPE {m['MAPE(%)']:.2f}% "
              f"| PICP_raw {m['PICP_raw(%)']:.0f}% -> PICP_conf {m['PICP_conformal(%)'] if not np.isnan(m['PICP_conformal(%)']) else float('nan'):.0f}%")

        last_plot = (np.arange(LOOKBACK), x_df["close"].values,
                     np.arange(LOOKBACK, LOOKBACK+PRED_LEN), truth, pred_mean, pred_q10, pred_q90, cw)

    print(f"      总耗时: {time.time()-total_start:.0f} 秒")

    # ---- 4. 汇总 ----
    print("\n[4/5] 汇总结果:")
    rdf = pd.DataFrame(results)
    numeric_cols = [c for c in rdf.columns if c not in ("window", "start_date")]
    summary = rdf[numeric_cols].mean()
    for k, v in summary.items():
        print(f"      {k}: {v:.4f}")

    # 保存明细 + 汇总
    rdf.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_SUMMARY, encoding="utf-8-sig")
    print(f"\n      明细: {OUT_DETAIL}")
    print(f"      汇总: {OUT_SUMMARY}")

    # ---- 5. 可视化最近窗口 ----
    print("\n[5/5] 可视化...")
    hx, hclose, fx, truth, pred_mean, q10, q90, cw = last_plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hx, hclose, "b-", label="History")
    ax.plot(fx, truth, "k-", lw=2, label="Ground Truth")
    ax.plot(fx, pred_mean, "r--", lw=2, label="Forecast (mean)")
    ax.fill_between(fx, q10, q90, color="orange", alpha=0.25, label="Quantile 80% PI (raw)")
    if cw is not None:
        ax.fill_between(fx, pred_mean-cw, pred_mean+cw, color="green", alpha=0.15,
                        label=f"Conformal 80% PI (±{cw:.2f})")
    ax.legend(); ax.set_title(f"{SYMBOL} last window - Raw vs Conformal Prediction Interval")
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT_FIG, dpi=150)
    print(f"      已保存: {OUT_FIG}")
    print("\n完成! 论文用法: v2_metrics_summary.csv 即 '校准前 vs 校准后' 消融表,")
    print("      你的方法只需在 conformal 基础上再改进 (如自适应宽度/加权校准)。")

if __name__ == "__main__":
    main()
