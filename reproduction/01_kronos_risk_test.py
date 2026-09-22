# -*- coding: utf-8 -*-
"""
Kronos 金融风险预测 跑通 + 评测脚本 (D盘版 · C盘空间保护版)
================================================================
适用环境: Windows 10 + Anaconda Python 3.11 + RTX 2060 6GB, C盘紧张

【为什么需要这个版本】
  默认情况下, 程序会偷偷往 C 盘写大量文件:
    1. HuggingFace 模型缓存   -> C:\\Users\\apple\\.cache\\huggingface   (Kronos模型约50-100MB)
    2. pip 下载的依赖安装包    -> C:\\Users\\apple\\AppData\\Local\\pip\\cache
    3. matplotlib 字体缓存等   -> C:\\Users\\apple\\.matplotlib
  本脚本已通过环境变量把所有缓存和产出重定向到 D盘项目目录。

【推荐目录结构 (先手动建好, 或运行脚本自动创建)】
  D:\\RCAC-TSFMs\\            <- 项目根目录 (PROJECT_ROOT)
    ├─ kronos_risk_test.py         <- 本脚本
    ├─ model\\                    <- Kronos 官方仓库的 model 包 (见下)
    ├─ data\\                      <- K线数据CSV、Qlib数据
    ├─ results\\                   <- 评测输出图、指标CSV
    ├─ checkpoints\\               <- 微调后的模型权重
    └─ hf_cache\\                  <- HuggingFace 模型缓存 (代替C盘)

【首次使用步骤】
1. 克隆官方仓库的 model 代码到项目目录 (在 D 盘操作, CMD 执行):
     D:
     mkdir D:\\RCAC-TSFMs
     cd D:\\RCAC-TSFMs
     git clone https://gh-proxy.com/https://github.com/shiyu-coder/Kronos.git
     然后把 Kronos 仓库里的 model/ 文件夹复制到 D:\\RCAC-TSFMs\\model\\
     (本脚本只需 model 包, 不必保留整个仓库)

2. 安装依赖 (torch/pandas/matplotlib/transformers 已有则跳过):
     pip install transformers matplotlib pandas

3. 运行:
     python D:\\RCAC-TSFMs\\kronos_risk_test.py

【国内下载模型慢?】 脚本已内置镜像, 无需手动设置。
"""

import os

# ==================================================================
# ★★★ D盘项目根目录: 按需修改为你的实际路径 ★★★
# ==================================================================
PROJECT_ROOT = r"D:\RCAC-TSFMs"
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
CKPT_DIR     = os.path.join(PROJECT_ROOT, "checkpoints")
HF_CACHE_DIR = os.path.join(PROJECT_ROOT, "hf_cache")

# ---- 关键: 在 import transformers 之前重定向所有缓存到 D 盘 ----
os.environ["HF_HOME"]              = HF_CACHE_DIR    # HF总缓存 (默认在用户目录C盘!)
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
os.environ["TRANSFORMERS_CACHE"]    = os.path.join(HF_CACHE_DIR, "transformers")
os.environ["HF_DATASETS_CACHE"]     = os.path.join(HF_CACHE_DIR, "datasets")
os.environ["HF_ENDPOINT"]           = "https://hf-mirror.com"   # 国内镜像
os.environ["MPLCONFIGDIR"]          = os.path.join(HF_CACHE_DIR, "matplotlib")  # 字体缓存也挪走

# 把脚本所在目录加入 import 路径 (确保能找到 model 包)
sys_path_added = os.path.join(os.path.dirname(os.path.abspath(__file__))) \
    if "__file__" in dir() else PROJECT_ROOT
import sys
if sys_path_added not in sys.path:
    sys.path.insert(0, sys_path_added)
# 若 model 包在 PROJECT_ROOT 下, 也加入路径
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # 不弹窗, 后台保存图片
import matplotlib.pyplot as plt

# 自动创建目录
for d in [DATA_DIR, RESULTS_DIR, CKPT_DIR, HF_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

# ------------------------------------------------------------------
# 0. 配置区 (按需修改)
# ------------------------------------------------------------------
MODEL_NAME  = "NeoQuasar/Kronos-mini"        # 2060 6GB 推荐 mini
TOKENIZER   = "NeoQuasar/Kronos-Tokenizer-2k" if "mini" in MODEL_NAME else "NeoQuasar/Kronos-Tokenizer-base"
MAX_CONTEXT = 2048 if "mini" in MODEL_NAME else 512
LOOKBACK    = 512                              # 历史窗口长度 (mini 可到 2048)
PRED_LEN    = 60                               # 预测未来 60 根K线
SAMPLE_CNT  = 8                                # 采样路径数 (概率预测, 算置信区间)
USE_SYNTHETIC_DATA = True                      # True=内置模拟数据; False=读 D:\\RCAC-TSFMs\\data\\ 下的CSV
CSV_PATH    = os.path.join(DATA_DIR, "my_kline.csv")
OUT_FIG     = os.path.join(RESULTS_DIR, "kronos_result.png")
OUT_METRICS = os.path.join(RESULTS_DIR, "metrics.csv")
# ------------------------------------------------------------------

def make_synthetic_kline(n=1200, seed=42):
    """生成带波动率聚集(GARCH效应)的合成K线 -> 模拟真实风险场景"""
    rng = np.random.default_rng(seed)
    ret = np.zeros(n); vol = np.zeros(n)
    vol[0] = 0.01
    for t in range(1, n):                       # GARCH(1,1)
        vol[t] = np.sqrt(0.00001 + 0.08*ret[t-1]**2 + 0.90*vol[t-1]**2)
        ret[t] = vol[t] * rng.standard_normal()
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.roll(close, 1); open_[0] = close[0]
    spread = np.abs(rng.standard_normal(n)) * close * vol * 0.5
    high = np.maximum(open_, close) + spread
    low  = np.minimum(open_, close) - spread
    volume = rng.lognormal(10, 0.8, n)
    amount = volume * close
    dates = pd.bdate_range("2021-01-01", periods=n)
    df = pd.DataFrame({"open": open_, "high": high, "low": low,
                       "close": close, "volume": volume, "amount": amount})
    df["timestamps"] = dates
    # 顺便把合成数据存到 D 盘 data 目录, 方便后续离线调试
    df.to_csv(os.path.join(DATA_DIR, "synthetic_kline.csv"), index=False)
    return df

def load_kline_csv(path):
    """读取自己的K线数据. CSV 需包含: open, high, low, close 列,
       可选 volume, amount; 时间列命名为 timestamps 或 date"""
    df = pd.read_csv(path)
    tcol = "timestamps" if "timestamps" in df.columns else "date"
    df["timestamps"] = pd.to_datetime(df[tcol])
    for c in ["volume", "amount"]:
        if c not in df.columns:
            df[c] = 0.0
    return df[["open", "high", "low", "close", "volume", "amount", "timestamps"]]

def evaluate(pred, truth_close):
    """金融风险预测核心指标"""
    err = pred - truth_close
    mae  = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    mape = np.mean(np.abs(err) / np.clip(truth_close, 1e-8, None)) * 100
    pred_dir = np.sign(np.diff(pred)); true_dir = np.sign(np.diff(truth_close))
    dir_acc = np.mean(pred_dir == true_dir) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE(%)": mape, "方向准确率(%)": dir_acc}

def main():
    print("=" * 60)
    print(f"  Kronos 金融风险预测测试 (D盘版)   模型: {MODEL_NAME}")
    print(f"  项目根目录: {PROJECT_ROOT}")
    print(f"  HF缓存目录: {HF_CACHE_DIR}  <- 模型不再占用C盘")
    print("=" * 60)

    # ---- 1. 数据 ----
    if USE_SYNTHETIC_DATA:
        print("\n[1/5] 生成带 GARCH 波动率聚集的合成K线数据...")
        df = make_synthetic_kline()
    else:
        print(f"\n[1/5] 读取数据: {CSV_PATH}")
        df = load_kline_csv(CSV_PATH)
    print(f"      共 {len(df)} 根K线, 区间 {df['timestamps'].iloc[0].date()} ~ {df['timestamps'].iloc[-1].date()}")

    x_df        = df.iloc[:LOOKBACK][["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    x_timestamp = df["timestamps"].iloc[:LOOKBACK].reset_index(drop=True)
    y_timestamp = df["timestamps"].iloc[LOOKBACK:LOOKBACK+PRED_LEN].reset_index(drop=True)
    truth       = df.iloc[LOOKBACK:LOOKBACK+PRED_LEN].reset_index(drop=True)

    # ---- 2. 加载模型 ----
    print("\n[2/5] 加载 Kronos 模型 (首次运行自动下载到 D盘 hf_cache)...")
    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
    except ImportError:
        print("\n[错误] 找不到 model 包! 请先把 Kronos 仓库的 model/ 文件夹")
        print(f"       复制到: {PROJECT_ROOT}\\model\\")
        print("       克隆命令: git clone https://gh-proxy.com/https://github.com/shiyu-coder/Kronos.git")
        sys.exit(1)

    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"      推理设备: {device.upper()}")
    if device == "cuda:0":
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"      显存剩余: {free_gb:.1f} GB")

    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER)
    model = Kronos.from_pretrained(MODEL_NAME)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=MAX_CONTEXT)

    # ---- 3. 多次采样预测 (概率预测 -> 置信区间) ----
    print(f"\n[3/5] 预测未来 {PRED_LEN} 根K线, 采样 {SAMPLE_CNT} 条路径...")
    preds = []
    t0 = time.time()
    for i in range(SAMPLE_CNT):
        p = predictor.predict(df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                              pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=1)
        preds.append(p["close"].values)
        print(f"      采样 {i+1}/{SAMPLE_CNT} 完成")
    print(f"      推理总耗时: {time.time()-t0:.1f} 秒")
    preds = np.array(preds)
    pred_mean = preds.mean(axis=0)
    pred_std  = preds.std(axis=0)

    # ---- 4. 评测 ----
    print("\n[4/5] 评测结果 (收盘价预测):")
    m = evaluate(pred_mean, truth["close"].values)
    for k, v in m.items():
        print(f"      {k}: {v:.4f}")

    realized_vol = (truth["high"] - truth["low"]).values
    vol_mae = np.mean(np.abs(pred_std * 2 - realized_vol))
    print(f"\n      风险专项 - 预测波动带宽度 vs 实际振幅 MAE: {vol_mae:.4f}")

    lower = pred_mean - 1.28 * pred_std; upper = pred_mean + 1.28 * pred_std
    coverage = np.mean((truth["close"].values >= lower) & (truth["close"].values <= upper)) * 100
    print(f"      80%置信区间覆盖率: {coverage:.1f}%  (理想值≈80%)")

    # 指标落盘到 D盘 results 目录
    import csv
    with open(OUT_METRICS, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in m.items():
            w.writerow([k, round(v, 6)])
        w.writerow(["风险带MAE", round(vol_mae, 6)])
        w.writerow(["80%CI覆盖率(%)", round(coverage, 2)])
    print(f"      指标已保存: {OUT_METRICS}")

    # ---- 5. 可视化 ----
    print(f"\n[5/5] 保存可视化结果 -> {OUT_FIG}")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    idx = np.arange(LOOKBACK, LOOKBACK + PRED_LEN)
    axes[0].plot(np.arange(LOOKBACK), x_df["close"], "b-", label="History (input)")
    axes[0].plot(idx, truth["close"], "k-", lw=2, label="Ground Truth")
    axes[0].plot(idx, pred_mean, "r--", lw=2, label="Kronos Forecast (mean)")
    axes[0].fill_between(idx, lower, upper, color="red", alpha=0.2, label="80% CI")
    axes[0].legend(); axes[0].set_title(f"{MODEL_NAME}  Close-Price Forecast"); axes[0].grid(alpha=0.3)
    axes[1].plot(idx, realized_vol, "k-", lw=2, label="Realized Range (H-L)")
    axes[1].plot(idx, pred_std * 2, "r--", lw=2, label="Predicted Risk Band (2*std)")
    axes[1].legend(); axes[1].set_title("Volatility / Risk Forecast"); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT_FIG, dpi=150)
    print("      完成! 本次运行未向C盘写入任何文件 (除非路径已存在旧缓存)。")

if __name__ == "__main__":
    main()
