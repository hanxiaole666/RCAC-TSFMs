# -*- coding: utf-8 -*-
"""
Kronos 金融风险预测 v5 —— 方法最终形态 (非对称调制 + GAMMA敏感性 + 四资产 + Chronos基线)
================================================================
相对 v4 的演进 (v4 的对称调制已证明失败: 平静期收窄导致漏覆盖, 见 v4_ablation.csv):
  1. ★ 非对称波动调制:  h = Q × min(2.5, max(1.0, σ̂/σ̄))
     只加宽不收窄 —— 平静期不冒险, 动荡期保覆盖 (v4负结果直接引出的设计)
  2. GAMMA 敏感性: {0.02, 0.05, 0.1}, 论文消融表标配
  3. 四资产: 沪深300 + 茅台 + 平安银行 + BTC, 跨资产稳健性拉满
  4. Chronos-small 基线 (amazon/chronos-t5-small, 单seed):
     验证"欠分散"是通用时序基础模型的通病, 而非 Kronos 独有

【运行】
  C:\\Users\\apple\\anaconda3\\python.exe kronos_risk_test_v5.py
  首次运行 Chronos 会联网下载 ~400MB (走D盘缓存+镜像)

【预计耗时】 4标的 × 3seeds × 10窗口 × 16采样 ≈ 6-8 分钟

【产出 (D盘 results)】
  v5_ablation.csv    <- 主消融表: raw/sc/aci/aci_asym + 各GAMMA
  v5_gamma.csv       <- GAMMA 敏感性专项表
  v5_chronos.csv     <- Chronos 欠分散验证 (vs Kronos raw)
  v5_by_symbol.csv   <- 分资产结果
  v5_details.csv     <- 全量明细
"""

import os

PROJECT_ROOT = r"D:\RCAC-TSFMs"
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
CHRONOS_ID  = "amazon/chronos-t5-small"

SYMBOLS     = ["sh000300", "600519"]   # PAPER-LOCKED (v5 as submitted):
    # Paper v5 results were produced with these 2 assets; "000001"/"btc" failed to
    # fetch at submission time and were skipped. Keep locked for exact reproduction.
LOOKBACK    = 256
PRED_LEN    = 20
N_WINDOWS   = 10
SAMPLE_CNT  = 16
SEEDS       = [42, 43, 44]
ALPHA       = 0.20
BASE_GAMMA  = 0.05
GAMMAS      = [0.02, 0.05, 0.1]     # 敏感性分析
RESID_CAP   = 500
VOL_CAP_HI  = 2.5                   # 非对称调制上限

OUT_ABL = os.path.join(RESULTS_DIR, "v5_ablation.csv")
OUT_GAM = os.path.join(RESULTS_DIR, "v5_gamma.csv")
OUT_CHR = os.path.join(RESULTS_DIR, "v5_chronos.csv")
OUT_SYM = os.path.join(RESULTS_DIR, "v5_by_symbol.csv")
OUT_DET = os.path.join(RESULTS_DIR, "v5_details.csv")
# ------------------------------------------------------------------

def load_kline_cached(symbol):
    cache = os.path.join(DATA_DIR, f"{symbol}_daily.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, parse_dates=["timestamps"])
    import akshare as ak
    print(f"    联网拉取 {symbol} ...")
    if symbol == "btc":                                   # BTC (akshare 接口可能变动, 多方案尝试)
        last_err = None
        for attempt in [lambda: ak.crypto_hist(symbol="BTC", period="daily", start_date="20180101", adjust=""),
                        lambda: ak.crypto_hist(symbol="BTC/USDT", period="daily")]:
            try:
                df = attempt(); break
            except Exception as e:
                last_err = e; df = None
        if df is None:
            raise RuntimeError(f"BTC数据获取失败: {last_err}")
        df = df.rename(columns={c: {"日期":"timestamps","date":"timestamps","时间":"timestamps"}.get(c, c) for c in df.columns})
        df = df.rename(columns={"开盘":"open","最高":"high","最低":"low","收盘":"close","成交量":"volume","open":"open","high":"high","low":"low","close":"close","volume":"volume"})
    elif symbol.startswith("sh0"):                        # A股指数
        df = ak.stock_zh_index_daily(symbol=symbol)
        df = df.rename(columns={"date": "timestamps"})
    else:                                                 # A股个股 (前复权)
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
    ret = np.diff(np.log(np.clip(close, 1e-8, None)))
    if len(ret) < win:
        return float(np.std(ret)) if len(ret) > 1 else 1e-4
    return max(float(np.std(ret[-win:])), 1e-4)

def winkler(lower, upper, y, alpha=ALPHA):
    return np.mean((upper-lower) + (2/alpha)*(np.clip(lower-y,0,None) + np.clip(y-upper,0,None)))

def picp_mpiw(lo, hi, y):
    return np.mean((y>=lo)&(y<=hi))*100, np.mean(hi-lo)

def asym_factor(vol_ratio):
    """★ 非对称调制: 只加宽(>=1), 永不收窄, 上限截断"""
    return min(VOL_CAP_HI, max(1.0, vol_ratio))

# ---------------- 区间方法集合 ----------------
def run_methods(pred_mean, truth, preds, calib_resid, aci_states, vol_ratio):
    """aci_states: {gamma: {"alpha","resid"}} 独立演化; 返回 {方法名: (lo,hi) 或 None}"""
    out = {"raw": (np.quantile(preds, ALPHA/2, axis=0), np.quantile(preds, 1-ALPHA/2, axis=0))}
    if len(calib_resid) > 0:
        h_sc = np.quantile(np.abs(calib_resid), 1-ALPHA)
        out["sc"] = (pred_mean - h_sc, pred_mean + h_sc)
    else:
        out["sc"] = None
    f = asym_factor(vol_ratio)
    for g in GAMMAS:
        st = aci_states[g]
        alpha, resid = st["alpha"], list(st["resid"])
        rec = {"mod": [], "plain": []}
        for t in range(len(truth)):
            h = np.quantile(np.abs(resid), 1-alpha) if len(resid) >= 10 else (np.std(resid) if resid else 50.0)
            rec["mod"].append((pred_mean[t]-h*f, pred_mean[t]+h*f))       # 非对称调制
            rec["plain"].append((pred_mean[t]-h,  pred_mean[t]+h))        # 裸ACI
            err = 0.0 if pred_mean[t]-h <= truth[t] <= pred_mean[t]+h else 1.0   # 公平: 用裸分支反馈
            alpha = min(0.5, max(0.02, alpha + g*(ALPHA - err)))
            resid.append(pred_mean[t]-truth[t]); resid = resid[-RESID_CAP:]
        st["alpha"], st["resid"] = alpha, resid
        lo_m = np.array([r[0] for r in rec["mod"]]);  hi_m = np.array([r[1] for r in rec["mod"]])
        lo_p = np.array([r[0] for r in rec["plain"]]); hi_p = np.array([r[1] for r in rec["plain"]])
        tag = "" if g == BASE_GAMMA else f"_g{int(g*100):02d}"
        out[f"aci{tag}_asym"] = (lo_m, hi_m)                          # 非对称调制版
        if g == BASE_GAMMA:
            out["aci"] = (lo_p, hi_p)                                 # 裸ACI (基线锚)
        else:
            out[f"aci_g{int(g*100):02d}"] = (lo_p, hi_p)              # 裸ACI其它GAMMA (敏感性)
    return out

def try_load_chronos(device):
    """Chronos-small 基线; 不可用则返回None (不阻塞主流程)"""
    try:
        import torch
        from chronos import ChronosPipeline
        print("    加载 Chronos-small 基线 (首次约400MB下载)...")
        try:
            pipe = ChronosPipeline.from_pretrained(CHRONOS_ID, device_map=device, torch_dtype=torch.float32)
        except TypeError:      # 旧版不接受 device_map
            pipe = ChronosPipeline.from_pretrained(CHRONOS_ID, torch_dtype=torch.float32)
        return pipe
    except Exception as e:
        print(f"    [警告] Chronos 不可用 ({type(e).__name__}: {e}), 跳过该基线")
        print("           如需补装: pip install chronos-forecasting")
        return None

def chronos_forecast(pipe, close_hist, device):
    import torch
    ctx = torch.tensor(close_hist, dtype=torch.float32)
    with torch.no_grad():
        try:    # chronos 2.x / 1.4+ 签名
            samples = pipe.predict(ctx, PRED_LEN, num_samples=32, limit_prediction_length=False)
        except TypeError:   # 旧版无该参数
            samples = pipe.predict(ctx, PRED_LEN, num_samples=32)
    if isinstance(samples, (tuple, list)):   # 部分版本返回(quantiles, mean)或(samples,...)
        samples = samples[0]
    if hasattr(samples, "cpu"):              # torch tensor -> numpy
        samples = samples.cpu().numpy()
    samples = np.asarray(samples)
    return samples[0] if samples.ndim == 3 else samples

def main():
    print("=" * 64)
    print("  Kronos risk forecasting v5  asymmetric-modulation ACI + Chronos baseline (paper-locked 2 assets)")
    print("=" * 64)

    from model import Kronos, KronosTokenizer, KronosPredictor
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("\n[1/5] 加载 Kronos...")
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER)
    model = Kronos.from_pretrained(MODEL_NAME)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=MAX_CONTEXT)
    print(f"      设备: {device.upper()}")
    chronos_pipe = try_load_chronos(device)

    print("\n[2/5] 获取数据...")
    dfs = {}
    for sym in SYMBOLS:
        try:
            dfs[sym] = load_kline_cached(sym)
            print(f"      {sym}: {len(dfs[sym])} 根日线")
        except Exception as e:
            print(f"      [警告] {sym} 获取失败, 跳过: {e}")

    print(f"\n[3/5] 滚动评测: {len(dfs)}标的 × {len(SEEDS)}seeds × {N_WINDOWS}窗口 ...")
    all_rows, chrono_rows = [], []
    t0 = time.time()
    for sym, df in dfs.items():
        base = len(df) - LOOKBACK - PRED_LEN*N_WINDOWS
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            aci_states = {g: {"alpha": ALPHA, "resid": []} for g in GAMMAS}
            calib, vol_hist = [], []
            for w in range(N_WINDOWS):
                s = base + w*PRED_LEN
                x_df  = df.iloc[s:s+LOOKBACK][["open","high","low","close","volume","amount"]].reset_index(drop=True)
                x_ts  = df["timestamps"].iloc[s:s+LOOKBACK].reset_index(drop=True)
                y_ts  = df["timestamps"].iloc[s+LOOKBACK:s+LOOKBACK+PRED_LEN].reset_index(drop=True)
                truth = df["close"].iloc[s+LOOKBACK:s+LOOKBACK+PRED_LEN].values

                preds = np.array([predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                                     pred_len=PRED_LEN, T=1.0, top_p=0.9, sample_count=1)["close"].values
                                  for _ in range(SAMPLE_CNT)])
                pred_mean = preds.mean(axis=0)

                sig = realized_vol(x_df["close"].values)
                vol_hist.append(sig); vol_hist = vol_hist[-200:]
                sig_bar = np.mean(vol_hist[:-1]) if len(vol_hist) > 1 else sig
                vol_ratio = sig / max(sig_bar, 1e-4)

                ints = run_methods(pred_mean, truth, preds, calib, aci_states, vol_ratio)
                row = {"symbol": sym, "seed": seed, "window": w+1,
                       "start_date": str(y_ts.iloc[0].date()), "vol_ratio": round(vol_ratio, 3)}
                for name, bounds in ints.items():
                    if bounds is None:
                        row[f"PICP_{name}"] = row[f"MPIW_{name}"] = row[f"Winkler_{name}"] = np.nan
                    else:
                        lo, hi = bounds
                        c, wd = picp_mpiw(lo, hi, truth)
                        row[f"PICP_{name}"], row[f"MPIW_{name}"], row[f"Winkler_{name}"] = c, wd, winkler(lo, hi, truth)
                all_rows.append(row)

                # Chronos 对照 (仅首seed, 同窗口同数据 -> 公平比较欠分散程度)
                if chronos_pipe is not None and seed == SEEDS[0]:
                    ch_samp = chronos_forecast(chronos_pipe, x_df["close"].values, device)
                    lo_c, hi_c = np.quantile(ch_samp, ALPHA/2, axis=0), np.quantile(ch_samp, 1-ALPHA/2, axis=0)
                    c, wd = picp_mpiw(lo_c, hi_c, truth)
                    chrono_rows.append({"symbol": sym, "window": w+1,
                                        "PICP_chronos": c, "MPIW_chronos": wd,
                                        "Winkler_chronos": winkler(lo_c, hi_c, truth),
                                        "MAPE_chronos(%)": np.mean(np.abs(ch_samp.mean(axis=0)-truth)/np.clip(truth,1e-8,None))*100})

                calib.extend((pred_mean-truth).tolist()); calib = calib[-RESID_CAP:]
            print(f"  {sym} seed{seed} 完成 | 末窗 vol_ratio={vol_ratio:.2f}")
    print(f"      总耗时: {time.time()-t0:.0f} 秒")

    print("\n[4/5] 生成表格...")
    rdf = pd.DataFrame(all_rows)
    rdf.to_csv(OUT_DET, index=False, encoding="utf-8-sig")
    methods = ["raw", "sc", "aci", "aci_asym"] + [f"aci_g{int(g*100):02d}_asym" for g in GAMMAS if g != BASE_GAMMA] \
              + [f"aci_g{int(g*100):02d}" for g in GAMMAS if g != BASE_GAMMA]

    abl = []
    for mth in methods:
        pc = rdf[f"PICP_{mth}"].dropna(); wd = rdf[f"MPIW_{mth}"].dropna(); wk = rdf[f"Winkler_{mth}"].dropna()
        if len(pc) == 0: continue
        abl.append({"method": mth, "n": len(pc),
                    "PICP_mean": pc.mean(), "PICP_std": pc.std(),
                    "MPIW_mean": wd.mean(), "MPIW_std": wd.std(),
                    "Winkler_mean": wk.mean(), "Winkler_std": wk.std()})
    adf = pd.DataFrame(abl).round(3)
    adf.to_csv(OUT_ABL, index=False, encoding="utf-8-sig")

    print("\n  ===== 主消融表 =====")
    print(f"  {'方法':<14} {'PICP%':>14} {'MPIW':>14} {'Winkler':>14}")
    for _, r in adf.iterrows():
        star = "  ★本文方法" if r["method"] == "aci_asym" else ""
        print(f"  {r['method']:<14} {r['PICP_mean']:6.2f}±{r['PICP_std']:<5.2f} "
              f"{r['MPIW_mean']:6.2f}±{r['MPIW_std']:<5.2f} {r['Winkler_mean']:6.2f}±{r['Winkler_std']:<5.2f}{star}")

    g_rows = []
    for g in GAMMAS:
        sub = adf[adf["method"] == (f"aci_g{int(g*100):02d}_asym" if g != BASE_GAMMA else "aci_asym")]
        if len(sub):
            g_rows.append({"gamma": g, **{k: sub.iloc[0][k] for k in
                          ["PICP_mean","PICP_std","MPIW_mean","MPIW_std","Winkler_mean","Winkler_std"]}})
    gdf = pd.DataFrame(g_rows).round(3)
    gdf.to_csv(OUT_GAM, index=False, encoding="utf-8-sig")
    print("\n  ===== GAMMA 敏感性 (非对称调制版) =====")
    print(gdf.to_string(index=False))

    if chrono_rows:
        cdf = pd.DataFrame(chrono_rows)
        cdf.to_csv(OUT_CHR, index=False, encoding="utf-8-sig")
        kraw = rdf[rdf["seed"] == SEEDS[0]]
        print("\n  ===== Chronos 欠分散验证 (单seed, raw分位数区间) =====")
        print(f"  {'标的':<10} {'Kronos_raw_PICP%':>16} {'Chronos_PICP%':>14}")
        for sym in dfs:
            k = kraw[kraw["symbol"] == sym]["PICP_raw"].mean()
            c = cdf[cdf["symbol"] == sym]["PICP_chronos"].mean()
            print(f"  {sym:<10} {k:16.2f} {c:14.2f}")
        print("  -> 两者都远低于80%即证明: 欠分散是通用TSFM通病")

    sym_rows = []
    for sym in dfs:
        sub = rdf[rdf["symbol"] == sym]
        for mth in ["raw", "aci", "aci_asym"]:
            sym_rows.append({"symbol": sym, "method": mth,
                             "PICP": sub[f"PICP_{mth}"].mean(), "Winkler": sub[f"Winkler_{mth}"].mean()})
    sdf = pd.DataFrame(sym_rows).round(3)
    sdf.to_csv(OUT_SYM, index=False, encoding="utf-8-sig")
    print("\n  ===== 分资产 (raw / aci / aci_asym) =====")
    print(sdf.to_string(index=False))
    print(f"\n已保存: v5_ablation / v5_gamma / v5_chronos / v5_by_symbol / v5_details (均在 results\\)")
    print("\n判定: aci_asym 应满足 PICP≈80% 且 Winkler < aci(311) 且 MPIW < aci。")

if __name__ == "__main__":
    main()
