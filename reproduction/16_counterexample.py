# -*- coding: utf-8 -*-
"""16_counterexample.py — Counterexample 1: shadow coupling voids the ACI guarantee
论文 Sec 4.4 Counterexample 1 的可复现脚本. 纯numpy, 秒级, 无需GPU/网络.
三种反馈模式对照:
  coupled_equal : 反馈区间 = 部署区间 (f·h)          -> 自修正, 覆盖->1-α (保证成立)
  shadow        : 反馈区间 = f·h (影子宽带), 部署 = h (生产窄带) -> 覆盖->1-f·α=60% (失效)
  decoupled     : 反馈区间 = h, 部署 = f·h (RCAC)     -> Observation 1, >= 1-α
【运行】python 16_counterexample.py
"""
import numpy as np

def aci_sim(mode, f, T=50000, alpha=0.20, gamma=0.10, seed=0):
    rng = np.random.default_rng(seed)
    eps = 1.0/rng.uniform(1e-9, 1.0, T)          # |eps| ~ Pareto(1): P(|eps|>x)=1/x
    a, pool, dep_miss = alpha, [], 0
    for t in range(T):
        h = np.quantile(pool, 1-a) if len(pool)>=10 else (np.std(pool) if pool else 10.0)
        if   mode=="coupled_equal": fb_band, dep_band = f*h, f*h
        elif mode=="shadow":        fb_band, dep_band = f*h, h
        else:                       fb_band, dep_band = h,   f*h
        miss_fb  = 1.0 if eps[t] > fb_band  else 0.0
        dep_miss += 1.0 if eps[t] > dep_band else 0.0
        a = min(0.5, max(0.02, a + gamma*(alpha - miss_fb)))
        pool.append(eps[t]); pool = pool[-2000:]
    return 100*(1-dep_miss/T)

if __name__ == "__main__":
    print("Counterexample 1 — Pareto(1) errors, alpha=0.2, f=2.0, T=50000")
    print(f"{'seed':>5} {'coupled_equal':>14} {'shadow(fails)':>14} {'decoupled(RCAC)':>16}")
    for s in [42, 1, 7, 99]:
        row = [aci_sim(m, 2.0, seed=s) for m in ["coupled_equal","shadow","decoupled"]]
        print(f"{s:>5} {row[0]:>13.1f}% {row[1]:>13.1f}% {row[2]:>15.1f}%")
    print("\n理论: shadow 收敛于 1-f*alpha = 60%; decoupled >= 80% (Observation 1)")
