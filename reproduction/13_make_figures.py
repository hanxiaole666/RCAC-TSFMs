# -*- coding: utf-8 -*-
"""make_figures.py — 一键生成论文全部4张图 (约4-6分钟, 需要GPU加载Kronos)
Fig 1: RCAC框架示意图 (纯绘制, 无数据依赖)
Fig 2: 三方法区间对比 (CSI300 测试窗8: raw/sc/RCAC 80%区间 vs 真实价格)
Fig 3: ACI α_t 在线轨迹 + vol_ratio (同一窗口)
Fig 4: 案例研究双联图 (CSI300-w8 拯救链 / 茅台-w6 崩盘窗)
【运行】C:\\Users\\apple\\anaconda3\\python.exe make_figures.py
【产出】D:\\finrisk_project\\results\\fig1..fig4.png
"""
import os, sys
PROJECT_ROOT = r"D:\finrisk_project"
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT,"hf_cache")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["MPLCONFIGDIR"] = os.path.join(PROJECT_ROOT,"hf_cache","matplotlib")
sys.path.insert(0, PROJECT_ROOT)
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
DATA, RESULTS = os.path.join(PROJECT_ROOT,"data"), os.path.join(PROJECT_ROOT,"results")
LOOKBACK, PRED_LEN, N_VAL, N_TEST, SAMPLES = 256, 20, 5, 5, 16
ALPHA, GAMMA, CAP = 0.20, 0.10, 2.0

# ---------- Fig 1: 框架示意 ----------
fig, ax = plt.subplots(figsize=(11,3.2)); ax.axis("off")
boxes = [("TSFM\n(Kronos/Chronos)","#dbeafe"),("Sample paths\n{S=16}","#dbeafe"),
         ("ACI\nonline α_t (Eq.3)","#fef3c7"),("Asymmetric\nmodulation (Eq.4)","#fef3c7"),
         ("Deployed interval\nπ̃_t ⊇ π_t","#dcfce7"),("Unmodulated miss\n→ feedback (Prop.1)","#fee2e2")]
xs = np.linspace(0.06, 0.86, 6)
for (txt,c),x in zip(boxes,xs):
    ax.add_patch(plt.Rectangle((x,0.32),0.12,0.44,fc=c,ec="k",lw=1.2))
    ax.text(x+0.06,0.54,txt,ha="center",va="center",fontsize=9)
for i in range(5):
    ax.annotate("",xy=(xs[i+1]-0.005,0.54),xytext=(xs[i]+0.125,0.54),
                arrowprops=dict(arrowstyle="->",lw=1.4))
ax.annotate("",xy=(0.86+0.06,0.30),xytext=(0.92,0.20),arrowprops=dict(arrowstyle="->",lw=1.2))
ax.text(0.55,0.08,"feedback uses UNMODULATED misses  ⇒  coverage guarantee preserved (Prop. 1)",
        ha="center",fontsize=9,style="italic")
plt.tight_layout(); plt.savefig(os.path.join(RESULTS,"fig1_framework.png"),dpi=200); plt.close()
print("fig1 完成")

# ---------- 加载模型, 重放窗口生成 Fig2-4 数据 ----------
from model import Kronos, KronosTokenizer, KronosPredictor
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
pred = KronosPredictor(model, tok, device=device, max_context=2048)

def run_window(df, s0):
    x_df = df.iloc[s0:s0+LOOKBACK][["open","high","low","close","volume","amount"]].reset_index(drop=True)
    x_ts = df["timestamps"].iloc[s0:s0+LOOKBACK].reset_index(drop=True)
    y_ts = df["timestamps"].iloc[s0+LOOKBACK:s0+LOOKBACK+PRED_LEN].reset_index(drop=True)
    truth = df["close"].iloc[s0+LOOKBACK:s0+LOOKBACK+PRED_LEN].values
    torch.manual_seed(42); np.random.seed(42)
    preds = np.array([pred.predict(df=x_df,x_timestamp=x_ts,y_timestamp=y_ts,pred_len=PRED_LEN,
                     T=1.0,top_p=0.9,sample_count=1)["close"].values for _ in range(SAMPLES)])
    ret = np.diff(np.log(x_df["close"].values))
    sig = max(float(np.std(ret[-20:])),1e-4)
    return x_df, x_ts, y_ts, truth, preds, sig

def rcac_trace(truth, pred_mean, preds, vol_ratio, a=ALPHA, g=GAMMA, cap=CAP):
    """返回 lo,hi,alpha轨迹 (暖启动残差=本窗内前步, 演示用途)"""
    alpha, resid, los, his, alps = a, [], [], [], []
    h0 = np.quantile(np.abs(pred_mean-truth), 0.8)   # 演示暖启动
    resid.append(h0)
    for t in range(len(truth)):
        h = np.quantile(np.abs(resid), 1-alpha)
        f = min(cap, max(1.0, vol_ratio))
        los.append(pred_mean[t]-h*f); his.append(pred_mean[t]+h*f); alps.append(alpha)
        miss = 0.0 if pred_mean[t]-h <= truth[t] <= pred_mean[t]+h else 1.0
        alpha = min(0.5, max(0.02, alpha + g*(a-miss)))
        resid.append(pred_mean[t]-truth[t]); resid = resid[-500:]
    return np.array(los), np.array(his), np.array(alps)

base_plot = lambda df: len(df)-LOOKBACK-PRED_LEN*(N_VAL+N_TEST)
plots = {}
for sym, w in [("sh000300",8),("sh000300",6),("600519",6)]:
    df = pd.read_csv(os.path.join(DATA,f"{sym}_daily.csv"), parse_dates=["timestamps"])
    s0 = base_plot(df) + (w-1)*PRED_LEN
    x_df,x_ts,y_ts,truth,preds,sig = run_window(df,s0)
    ret = np.diff(np.log(df["close"].iloc[:s0+LOOKBACK].values))
    sigbar = max(float(np.std(ret[-60:])),1e-4)
    vr = sig/sigbar
    pm = preds.mean(axis=0)
    lo_r,hig = np.quantile(preds,ALPHA/2,axis=0), np.quantile(preds,1-ALPHA/2,axis=0)
    lo_c,hi_c,alps = rcac_trace(truth, pm, preds, vr)
    plots[(sym,w)] = dict(x=x_df,x_ts=x_ts,y_ts=y_ts,truth=truth,pm=pm,lo_r=lo_r,hig=hig,lo_c=lo_c,hi_c=hi_c,alps=alps,vr=vr)

# ---------- Fig 2: 区间对比 (CSI300 w8) ----------
d = plots[("sh000300",8)]
fig, ax = plt.subplots(figsize=(11,5))
hx = np.arange(LOOKBACK); fx = np.arange(LOOKBACK,LOOKBACK+PRED_LEN)
ax.plot(hx, d["x"]["close"], "b-", lw=1.2, label="History")
ax.plot(fx, d["truth"], "k-", lw=2, label="Realized")
ax.plot(fx, d["pm"], "r--", lw=1.5, label="TSFM mean forecast")
ax.fill_between(fx, d["lo_r"], d["hig"], color="orange", alpha=0.3, label="Raw 80% interval")
ax.fill_between(fx, d["lo_c"], d["hi_c"], color="green", alpha=0.18, label="RCAC 80% interval")
ax.legend(loc="upper left"); ax.grid(alpha=0.3)
ax.set_title("Fig 2 — CSI 300 test window 8 (2026-06): raw vs RCAC intervals")
plt.tight_layout(); plt.savefig(os.path.join(RESULTS,"fig2_intervals.png"),dpi=200); plt.close()
print("fig2 完成")

# ---------- Fig 3: α_t 轨迹 ----------
fig, axes = plt.subplots(2,1,figsize=(11,6),sharex=True)
axes[0].plot(fx, d["alps"], "g-", lw=2)
axes[0].axhline(ALPHA,color="gray",ls="--",alpha=0.6); axes[0].set_ylabel("α_t")
axes[0].set_title("Fig 3 — Online miscoverage level α_t (ACI, unmodulated feedback)")
axes[1].plot(fx, np.full(PRED_LEN, d["vr"]), "r-", lw=2, label=f"vol_ratio = {d['vr']:.2f}")
axes[1].axhline(1.0,color="gray",ls="--",alpha=0.6); axes[1].legend(); axes[1].set_ylabel("σ̂/σ̄")
plt.tight_layout(); plt.savefig(os.path.join(RESULTS,"fig3_alpha.png"),dpi=200); plt.close()
print("fig3 完成")

# ---------- Fig 4: 案例研究双联 ----------
fig, axes = plt.subplots(1,2,figsize=(13,5))
for ax,(sym,w),ttl in zip(axes,[("sh000300",8),("600519",6)],
                          ["CSI 300 w8: turbulent rescue","Moutai w6: crash onset"]):
    dd = plots[(sym,w)]
    ax.plot(hx, dd["x"]["close"], "b-", lw=1, label="History")
    ax.plot(fx, dd["truth"], "k-", lw=2, label="Realized")
    ax.plot(fx, dd["pm"], "r--", lw=1.2)
    ax.fill_between(fx, dd["lo_r"], dd["hig"], color="orange", alpha=0.3, label="Raw 80%")
    ax.fill_between(fx, dd["lo_c"], dd["hi_c"], color="green", alpha=0.18, label="RCAC 80%")
    ax.set_title(ttl); ax.grid(alpha=0.3); ax.legend(fontsize=8)
plt.suptitle("Fig 4 — Case studies")
plt.tight_layout(); plt.savefig(os.path.join(RESULTS,"fig4_cases.png"),dpi=200); plt.close()
print("fig4 完成 | 全部图片在 results\\")
