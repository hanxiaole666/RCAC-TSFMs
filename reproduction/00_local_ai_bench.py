#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本机 AI 算力测试程序 (Local AI Capability Benchmark)
测试项目:
  1. 硬件信息: CPU / 内存 / 磁盘
  2. CPU 算力: 大矩阵乘法 FLOPS 基准
  3. 内存性能: 连续读写速度
  4. GPU 能力: 显存、矩阵计算 (需要 PyTorch + CUDA)
  5. 模拟推理: 简单神经网络前向传播耗时
"""

import time, os, sys, platform, math, statistics

# ---------- 工具 ----------
def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def ok(msg):   print(f"[通过] {msg}")
def warn(msg): print(f"[提示] {msg}")
def fail(msg): print(f"[未测] {msg}")

def fmt_bytes(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"

def fmt_flops(n):
    for unit in ["FLOP/s", "KFLOP/s", "MFLOP/s", "GFLOP/s", "TFLOP/s"]:
        if abs(n) < 1000:
            return f"{n:.2f} {unit}"
        n /= 1000
    return f"{n:.2f} PFLOP/s"

def score_bar(value, max_value, width=30):
    filled = int(min(1.0, value / max_value) * width)
    return "█" * filled + "░" * (width - filled)

# ---------- 1. 硬件信息 ----------
def hardware_info():
    section("1. 硬件信息")
    print(f"系统: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python: {sys.version.split()[0]}")
    print(f"CPU: {platform.processor() or os.cpu_count()}")
    print(f"CPU 逻辑核心数: {os.cpu_count()}")

    try:
        import psutil
        print(f"内存总量: {fmt_bytes(psutil.virtual_memory().total)}")
        print(f"当前内存占用: {psutil.virtual_memory().percent}%")
    except ImportError:
        print("内存信息: 未安装 psutil (pip install psutil)")

# ---------- 2. CPU 算力测试 ----------
def cpu_benchmark(n=1024, rounds=3):
    section("2. CPU 浮点算力测试 (矩阵乘法)")
    print(f"使用 {n}x{n} 浮点矩阵相乘, 重复 {rounds} 次取平均值...")

    try:
        import numpy as np          # 优先用 numpy (快得多)
        random = np.random.default_rng(42)
        A, B = random.random((n, n)), random.random((n, n))
        use_np = True
    except ImportError:
        random = __import__("random")
        random.seed(42)
        A = [[random.random() for _ in range(n)] for _ in range(n)]
        B = [[random.random() for _ in range(n)] for _ in range(n)]
        use_np = False
    if use_np:
        print("(使用 NumPy 加速)")
    else:
        print("(未安装 NumPy, 用纯 Python, 较慢; 建议 pip install numpy)")

    flops_per_op = 2 * n ** 3          # 乘法+加法
    times = []
    for i in range(rounds):
        t0 = time.perf_counter()
        if use_np:
            C = A @ B
        else:  # 转置 B 提升缓存命中率
            Bt = list(zip(*B))
            C = [[sum(ar * bc for ar, bc in zip(row, col)) for col in Bt] for row in A]
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  第 {i+1} 轮: {dt:.2f} 秒")

    avg = statistics.mean(times)
    gflops = flops_per_op / avg
    print(f"平均耗时: {avg:.2f} 秒")
    print(f"CPU 算力: {fmt_flops(gflops)}")
    print(f"评分: {score_bar(gflops, 10_000)}")
    return gflops

# ---------- 3. 内存性能 ----------
def memory_benchmark(size_mb=256):
    section("3. 内存读写性能")
    n = size_mb * 1024 * 1024 // 8   # float64 元素个数
    try:
        data = [0.0] * n
    except MemoryError:
        fail("内存不足, 跳过内存测试")
        return

    t0 = time.perf_counter()
    for i in range(n):
        data[i] = i * 1.0
    write_dt = time.perf_counter() - t0

    t0 = time.perf_counter()
    total = 0.0
    for i in range(0, n, 4):   # 步进采样加速
        total += data[i]
    read_dt = time.perf_counter() - t0

    write_bw = size_mb / 1024 / write_dt
    read_bw  = (size_mb / 1024) / (read_dt * 4)
    print(f"写入带宽 (估算): {write_bw:.2f} GB/s")
    print(f"读取带宽 (估算): {read_bw:.2f} GB/s")
    print(f"评分: {score_bar(read_bw, 50)}")
    del data

# ---------- 4. GPU 测试 ----------
def gpu_benchmark():
    section("4. GPU 能力测试 (PyTorch)")
    try:
        import torch
    except ImportError:
        fail("未安装 PyTorch (pip install torch), 跳过 GPU 测试")
        return None

    print(f"PyTorch 版本: {torch.__version__}")
    if not torch.cuda.is_available():
        fail("无可用 CUDA GPU, 使用 CPU 模式测试推理")
        device = "cpu"
    else:
        device = "cuda"
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"显存: {props.total_memory / 1024**3:.1f} GB")
        print(f"CUDA 核心: {props.multi_processor_count} SM")

    n = 4096
    print(f"\n运行 {n}x{n} 矩阵乘法基准...")
    A = torch.randn(n, n, device=device)
    B = torch.randn(n, n, device=device)
    for _ in range(3):      # 预热
        C = A @ B
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(5):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        C = A @ B
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = statistics.mean(times)
    tflops = 2 * n ** 3 / avg / 1e12
    print(f"平均耗时: {avg*1000:.1f} ms")
    print(f"GPU 算力: {tflops:.2f} TFLOP/s")
    print(f"评分: {score_bar(tflops, 100)}")
    return tflops

# ---------- 5. 模拟推理 ----------
def inference_benchmark():
    section("5. 模拟模型推理 (前向传播)")
    try:
        import torch
        import torch.nn as nn
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"推理设备: {device.upper()}")

        # 小型 CNN, 类似轻量级视觉模型
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 1000),
        ).to(device).eval()

        x = torch.randn(32, 3, 224, 224, device=device)   # batch=32 图像
        with torch.no_grad():
            for _ in range(5):
                _ = model(x)          # 预热
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 20

        imgs_per_sec = 32 / dt
        print(f"单次前向 (batch=32): {dt*1000:.1f} ms")
        print(f"吞吐率: {imgs_per_sec:.1f} 张/秒")
        print(f"评分: {score_bar(imgs_per_sec, 1000)}")
    except ImportError:
        fail("未安装 PyTorch, 跳过推理测试")

# ---------- 主流程 ----------
def main():
    print("╔══════════════════════════════════════════╗")
    print("║      本机 AI 算力测试  Local AI Bench    ║")
    print("╚══════════════════════════════════════════╝")
    hardware_info()
    cpu_benchmark()
    memory_benchmark()
    gpu_benchmark()
    inference_benchmark()
    section("测试完成")
    print("建议: 若 GPU 算力 > 5 TFLOP/s, 可流畅运行 7B 以下本地大模型推理")

if __name__ == "__main__":
    main()
