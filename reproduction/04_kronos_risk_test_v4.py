# -*- coding: utf-8 -*-
"""
Kronos 金融风险预测 v4 —— 波动率调制 ACI + 跨资产 + 2×2 消融 (方法模块实验闭环版)
================================================================
在 v3 (raw / split conformal / ACI) 基础上, 新增:
  1. 波动率调制半宽:  h = Q(残差) × clip(σ̂/σ̄, 0.5, 2.5)
     平静期收窄(压MPIW), 动荡期加宽(保PICP)
  2. 2×2 消融:  {固定α / ACI} × {无调制 / 波动调制}  四种组合
  3. 跨资产批量: 沪深300 + 贵州茅台 (可继续加), 验证结论稳健性

【运行】
  C:\\Users\\apple\\anaconda3\\python.exe kronos_risk_test_v4.py

【预计耗时】 2标的 × 3seeds × 10窗口 × 16采样 ≈ 4-6 分钟 (RTX 2060)

【产出 (D盘 results)】
  v4_ablation.csv       <- 论文消融表: 4方法 × (PICP/MPIW/Winkler) mean±std
  v4_by_symbol.csv      <- 分资产结果 (稳健性)
  v4_details.csv        <- 逐标的×seed×窗口明细
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

for d in [DATA_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ------------------------------------------------------------------
# 0. 配置区
# ------------------------------------------------------------------
MODEL_NAME  = "NeoQuasar/Kronos-mini"
TOKENIZER   = "NeoQuasar/Kronos-Tokenizer-2k"
MAX_CONTEXT = 2048

SYMBOLS     = ["sh000300", "600519"]   # 沪深300指数 + 贵州茅台; 加 "000001" 等可扩展
LOOKBACK    = 256
PRED_LEN    = 20
N_WINDOWS   = 10
SAMPLE_CNT  = 16
SEEDS       = [42, 43, 44]
ALPHA       = 0.20
GAMMA       = 0.05
RESID_CAP   = 500
VOL_CLIP    = (0.5, 2.5)               # 调制因子截断, 防止极端波动把区间打爆

OUT_ABL   = os.path.join(RESULTS_DIR, "v4_ablation.csv")
OUT_SYM   = os.path.join(RESULTS_DIR, "v4_by_symbol.csv")
OUT_DET   = os.path.join(RESULTS_DIR, "v4_details.csv")
# ------------------------------------------------------------------

def load_kline_cached(symbol):
    cache = os.path.join(DATA_DIR, f"{symbol}_daily.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, parse_dates=["timestamps"])
    import akshare as ak
    print(f"    联网拉取 {symbol} ...")
    if symbol.startswith("sh0"):                      # 指数
        df = ak.stock_zh_index_daily(symbol=symbol)
        df = df.rename(columns={"date": "timestamps"})
    else:                                             # 个股 (前复权)
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date="20180101", adjust="qfq")
        df = df.rename(columns={"日期": "timestamps", "开盘": "open", "最高": "high",
                                "最低": "low", "收盘": "close", "成交量": "volume"})
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    if "amount" not in df.columns:
        df["amount"] = df.get("volume", 0) * df["close"]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[["open","high","low","close","volume","amount","timestamps"]].dropna().reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df

def realized_vol(close, win=20):
    """历史窗口内的滚动实现波动率 (因果: 只用预测起点之前的数据)"""
    ret = np.diff(np.log(np.clip(close, 1e-8, None)))
    if len(ret) < win:
        return float(np.std(ret)) if len(ret) > 1 else 1e-4
    rv = np.std(ret[-win:])
    return max(rv, 1e-4)

def winkler(lower, upper, y, alpha=ALPHA):
    width = upper - lower
    return np.mean(width + (2/alpha)*(np.clip(lower-y,0,None) + np.clip(y-upper,0,None)))

def picp_mpiw(lo, hi, y):
    return np.mean((y>=lo)&(y<=hi))*100, np.mean(hi-lo)

# ---------------- 四种区间方法 (2×2) ----------------
def run_methods(pred_mean, truth, preds, calib_resid, alpha_state, vol_ratio):
    """返回 dict: 四种组合的 (lo, hi)
    组合: sc(固定α) / aci(自适应α) × 裸半宽 / 波动调制半宽"""
    q_lo_raw, q_hi_raw = np.quantile(preds, ALPHA/2, axis=0), np.quantile(preds, 1-ALPHA/2, axis=0)
    out = {"raw": (q_lo_raw, q_hi_raw)}

    def modulate(h):
        return h * np.clip(vol_ratio, *VOL_CLIP)

    # --- split conformal 分支 (固定 α) ---
    if len(calib_resid) > 0:
        h_sc = np.quantile(np.abs(calib_resid), 1-ALPHA)
        out["sc"]      = (pred_mean - h_sc,      pred_mean + h_sc)
        out["sc_vol"]  = (pred_mean - modulate(h_sc), pred_mean + modulate(h_sc))
    else:
        out["sc"] = out["sc_vol"] = None

    # --- ACI 分支 (在线 α_t, 残差库严格因果) ---
    alpha, resid, results = alpha_state["alpha"], list(alpha_state["resid"]), {}
    for t in range(len(truth)):
        h = np.quantile(np.abs(resid), 1-alpha) if len(resid) >= 10 else (np.std(resid) if resid else 50.0)
        results.setdefault("aci",      []).append((pred_mean[t]-h,      pred_mean[t]+h))
        hm = modulate(h)
        results.setdefault("aci_vol",  []).append((pred_mean[t]-hm,     pred_mean[t]+hm))
        # 用"无调制"分支的覆盖反馈更新 α (标准 ACI 定义, 保证消融公平)
        err = 0.0 if pred_mean[t]-h <= truth[t] <= pred_mean[t]+h else 1.0
        alpha = min(0.5, max(0.02, alpha + GAMMA*(ALPHA - err)))
        resid.append(pred_mean[t]-truth[t]); resid = resid[-RESID_CAP:]
    alpha_state["alpha"], alpha_state["resid"] = alpha, resid
    out["aci"]     = (np.array([r[0] for r in results["aci"]]),     np.array([r[1] for r in results["aci"]]))
    out["aci_vol"] = (np.array([r[0] for r in results["aci_vol"]]), np.array([r[1] for r in results["aci_vol"]]))
    return out

def main():
    print("=" * 64)
    print("  Kronos 金融风险预测 v4  波动率调制ACI + 跨资产 + 2×2消融")
    print("=" * 64)

    from model import Kronos, KronosTokenizer, KronosPredictor
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("\n[1/4] 加载模型...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER)
    model = Kronos.from_pretrained(MODEL_NAME)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=MAX_CONTEXT)
    print(f"      设备: {device.upper()}")

    print("\n[2/4] 获取数据...")
    dfs = {}
    for sym in SYMBOLS:
        dfs[sym] = load_kline_cached(sym)
        print(f"      {sym}: {len(dfs[sym])} 根日线")

    print(f"\n[3/4] 滚动评测: {len(SYMBOLS)}标的 × {len(SEEDS)}seeds × {N_WINDOWS}窗口 ...")
    all_rows = []
    t0 = time.time()
    for sym in SYMBOLS:
        df = dfs[sym]
        base = len(df) - LOOKBACK - PRED_LEN*N_WINDOWS
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            aci_state  = {"alpha": ALPHA, "resid": []}
            calib      = []                       # SC 残差库
            vol_hist   = []                       # σ̄ 的因果估计 (历史窗口vol的累积均值)
            for w in range(N_WINDOWS):
                s = base + w*PRED_LEN
                x_df   = df.iloc[s:s+LOOKBACK][["open","high","low","close","volume","amount"]].reset_index(drop=True)
                x_ts   = df["timestamps"].iloc[s:s+LOOKBACK].reset_index(drop=True)
                y_ts   = df["timestamps"].iloc[s+LOOKBACK:s+LOOKBACK+PRED_LEN].reset_index(drop=True)
                truth  = df["close"].iloc[s+LOOKBACK:s+LOOKBACK+PRED_LEN].values

                preds = np.array([predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                                     pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=1)["close"].values
                                  for _ in range(SAMPLE_CNT)])
                pred_mean = preds.mean(axis=0)

                # 波动率调制因子 (因果)
                sig = realized_vol(x_df["close"].values)
                vol_hist.append(sig); vol_hist = vol_hist[-200:]
                sig_bar = np.mean(vol_hist[:-1]) if len(vol_hist) > 1 else sig   # 不含本窗的均值
                vol_ratio = sig / max(sig_bar, 1e-4)

                ints = run_methods(pred_mean, truth, preds, calib, aci_state, vol_ratio)
                row = {"symbol": sym, "seed": seed, "window": w+1, "start_date": str(y_ts.iloc[0].date()),
                       "MAPE(%)": np.mean(np.abs(pred_mean-truth)/np.clip(truth,1e-8,None))*100,
                       "vol_ratio": vol_ratio}
                for name, bounds in ints.items():
                    if bounds is None:          # 首个窗口无校准残差, sc 系方法记 NaN
                        row[f"PICP_{name}"] = row[f"MPIW_{name}"] = row[f"Winkler_{name}"] = np.nan
                    else:
                        lo, hi = bounds
                        c, wdt = picp_mpiw(lo, hi, truth)
                        row[f"PICP_{name}"], row[f"MPIW_{name}"], row[f"Winkler_{name}"] = c, wdt, winkler(lo, hi, truth)
                all_rows.append(row)
                calib.extend((pred_mean-truth).tolist()); calib = calib[-RESID_CAP:]
            print(f"  {sym} seed{seed} 完成 | 末窗 vol_ratio={vol_ratio:.2f}")
    print(f"      总耗时: {time.time()-t0:.0f} 秒")

    print("\n[4/4] 生成消融表...")
    rdf = pd.DataFrame(all_rows)
    rdf.to_csv(OUT_DET, index=False, encoding="utf-8-sig")
    methods = ["raw", "sc", "sc_vol", "aci", "aci_vol"]

    abl = []
    for mth in methods:
        pc = rdf[f"PICP_{mth}"].dropna(); wd = rdf[f"MPIW_{mth}"].dropna(); wk = rdf[f"Winkler_{mth}"].dropna()
        if len(pc) == 0:
            continue
        abl.append({"method": mth, "n": len(pc),
                    "PICP_mean": pc.mean(), "PICP_std": pc.std(),
                    "MPIW_mean": wd.mean(), "MPIW_std": wd.std(),
                    "Winkler_mean": wk.mean(), "Winkler_std": wk.std()})
    adf = pd.DataFrame(abl).round(3)
    adf.to_csv(OUT_ABL, index=False, encoding="utf-8-sig")

    sym_rows = []
    for sym in SYMBOLS:
        sub = rdf[rdf["symbol"] == sym]
        for mth in methods:
            pc = sub[f"PICP_{mth}"].dropna(); wk = sub[f"Winkler_{mth}"].dropna()
            if len(pc) == 0: continue
            sym_rows.append({"symbol": sym, "method": mth,
                             "PICP": pc.mean(), "Winkler": wk.mean()})
    sdf = pd.DataFrame(sym_rows).round(3)
    sdf.to_csv(OUT_SYM, index=False, encoding="utf-8-sig")

    print("\n  ===== 论文消融表 (2×2 + 两层基线) =====")
    print(f"  {'方法':<12} {'PICP%':>14} {'MPIW':>14} {'Winkler':>14}   组合含义")
    meaning = {"raw": "基线: 分位数区间", "sc": "固定α, 无调制", "sc_vol": "固定α + 波动调制",
               "aci": "自适应α, 无调制", "aci_vol": "自适应α + 波动调制 ★本文方法"}
    for _, r in adf.iterrows():
        print(f"  {r['method']:<12} {r['PICP_mean']:6.2f}±{r['PICP_std']:<5.2f} "
              f"{r['MPIW_mean']:6.2f}±{r['MPIW_std']:<5.2f} {r['Winkler_mean']:6.2f}±{r['Winkler_std']:<5.2f}   {meaning[r['method']]}")
    print("\n  ===== 分资产稳健性 =====")
    print(sdf.to_string(index=False))
    print(f"\n已保存: {OUT_ABL}\n        {OUT_SYM}\n        {OUT_DET}")
    print("\n完成! 判定: aci_vol 应同时满足 PICP≈80 且 Winkler < aci 且 MPIW < aci。")

if __name__ == "__main__":
    main()
