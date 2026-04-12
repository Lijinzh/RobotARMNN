"""
机械臂可视化统一入口

将 visualize_ik.py（IK 误差评估）和 visualize_mujoco.py（MuJoCo 交互仿真）
合并为一个脚本，通过子命令控制执行模式。

子命令：
    ik     — IK 误差对比可视化（解析 vs MLP vs ESN，静态离线评估）
    mujoco — MuJoCo 交互可视化（实时键鼠控制目标点追踪）

用法：
    # IK 误差评估（图片自动保存到 export/）
    uv run src/visualize.py ik --arm 2dof
    uv run src/visualize.py ik --arm 3dof
    uv run src/visualize.py ik --arm all
    uv run src/visualize.py ik --arm all --no-save    # 仅显示窗口
    uv run src/visualize.py ik --arm 2dof --n_test 20000

    # MuJoCo 交互可视化
    uv run src/visualize.py mujoco --arm 2dof
    uv run src/visualize.py mujoco --arm 3dof
    uv run src/visualize.py mujoco --arm 3dof --use_nn
    uv run src/visualize.py mujoco --arm 3dof --use_nn --model_type esn

MuJoCo 键盘操作：
    ↑ / ↓       — 目标 +Y / -Y（前后）
    ← / →       — 目标 -X / +X（左右）
    KP8 / KP2   — 目标 +Z / -Z（上下，仅 3-DOF）
    R           — 复位目标点
    SPACE       — 启停自动轨迹演示
    P           — 打印当前状态
"""

import argparse
import os
import sys
import time

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── 中文字体配置 ────────────────────────────────────────────────────
for _font in ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei"]:
    if _font in [f.name for f in matplotlib.font_manager.fontManager.ttflist]:
        matplotlib.rcParams["font.sans-serif"] = [_font] + matplotlib.rcParams["font.sans-serif"]
        break
matplotlib.rcParams["axes.unicode_minus"] = False
# ────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.kinematics_2dof import (
    PlanarArmConfig,
    forward_kinematics_planar,
    inverse_kinematics_planar,
)
from src.kinematics_3dof import (
    Arm3DOFConfig,
    forward_kinematics_3dof,
    inverse_kinematics_3dof,
)
from src.models import IKNN_ESN, IKNN_MLP

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
EXPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "export")

# =====================================================================
# ====================== IK 功能区 ====================================
# =====================================================================


def sample_targets_2dof(cfg: PlanarArmConfig, n: int, seed: int = 42) -> np.ndarray:
    """在 2-DOF 平面臂工作空间内均匀采样目标点（环形 XY）。"""
    rng = np.random.default_rng(seed)
    r_min = cfg.r_min + 0.01
    r_max = cfg.r_max - 0.01
    targets = []
    while len(targets) < n:
        batch = rng.uniform(-r_max, r_max, (n * 3, 2))
        r = np.linalg.norm(batch, axis=1)
        valid = (r >= r_min) & (r <= r_max)
        targets.append(batch[valid])
    return np.concatenate(targets, axis=0)[:n]


def sample_targets_3dof(cfg: Arm3DOFConfig, n: int, seed: int = 42) -> np.ndarray:
    """在 3-DOF 空间臂工作空间内均匀采样目标点（球壳内）。"""
    rng = np.random.default_rng(seed)
    r_max = cfg.L1 + cfg.L2 - 0.01
    r_min = 0.05
    targets = []
    while len(targets) < n:
        batch = rng.uniform(-r_max, r_max, (n * 3, 3))
        batch[:, 2] += cfg.L0
        r3d = np.linalg.norm(batch - np.array([0, 0, cfg.L0]), axis=1)
        z_ok = (batch[:, 2] >= 0.05) & (batch[:, 2] <= cfg.L0 + cfg.L1 + cfg.L2 - 0.01)
        valid = (r3d >= r_min) & (r3d <= r_max) & z_ok
        targets.append(batch[valid])
    return np.concatenate(targets, axis=0)[:n]


def solve_2dof_all(
    targets: np.ndarray,
    cfg: PlanarArmConfig,
    mlp_model: IKNN_MLP | None,
    esn_model: IKNN_ESN | None,
) -> dict:
    """对所有目标批量求解 2-DOF IK（解析 + MLP + ESN）。"""
    N = len(targets)
    results = {}

    # ---- 解析解 ----
    thetas_ana = np.full((N, 2), np.nan)
    for i, t in enumerate(targets):
        sol = inverse_kinematics_planar(t, cfg, elbow_up=True)
        if sol is not None:
            thetas_ana[i] = sol
    ee_ana = np.array([
        forward_kinematics_planar(thetas_ana[i, 0], thetas_ana[i, 1], cfg)
        if not np.isnan(thetas_ana[i, 0])
        else [np.nan, np.nan]
        for i in range(N)
    ])
    valid_ana = ~np.isnan(ee_ana[:, 0])
    err_ana = np.where(valid_ana, np.linalg.norm(ee_ana - targets, axis=1) * 1000, np.nan)
    results["analytical"] = {"ee": ee_ana, "err_mm": err_ana, "valid": valid_ana}

    # ---- MLP ----
    if mlp_model is not None:
        mlp_model.eval()
        x_t = torch.from_numpy(targets.astype(np.float32))
        with torch.no_grad():
            sincos = mlp_model(x_t).numpy()
        theta1 = np.arctan2(sincos[:, 0], sincos[:, 1])
        theta2 = np.arctan2(sincos[:, 2], sincos[:, 3])
        ee_mlp = np.column_stack([
            cfg.L1 * np.cos(theta1) + cfg.L2 * np.cos(theta1 + theta2),
            cfg.L1 * np.sin(theta1) + cfg.L2 * np.sin(theta1 + theta2),
        ])
        results["mlp"] = {
            "ee": ee_mlp,
            "err_mm": np.linalg.norm(ee_mlp - targets, axis=1) * 1000,
            "valid": np.ones(N, bool),
        }

    # ---- ESN ----
    if esn_model is not None:
        esn_model.eval()
        x_t = torch.from_numpy(targets.astype(np.float32))
        with torch.no_grad():
            sincos = esn_model(x_t).numpy()
        theta1 = np.arctan2(sincos[:, 0], sincos[:, 1])
        theta2 = np.arctan2(sincos[:, 2], sincos[:, 3])
        ee_esn = np.column_stack([
            cfg.L1 * np.cos(theta1) + cfg.L2 * np.cos(theta1 + theta2),
            cfg.L1 * np.sin(theta1) + cfg.L2 * np.sin(theta1 + theta2),
        ])
        results["esn"] = {
            "ee": ee_esn,
            "err_mm": np.linalg.norm(ee_esn - targets, axis=1) * 1000,
            "valid": np.ones(N, bool),
        }

    return results


def solve_3dof_all(
    targets: np.ndarray,
    cfg: Arm3DOFConfig,
    mlp_model: IKNN_MLP | None,
    esn_model: IKNN_ESN | None,
) -> dict:
    """对所有目标批量求解 3-DOF IK（解析 + MLP + ESN，几何解耦）。"""
    N = len(targets)
    results = {}

    # ---- 解析解 ----
    thetas_ana = np.full((N, 3), np.nan)
    for i, t in enumerate(targets):
        sol = inverse_kinematics_3dof(t, cfg, elbow_up=True)
        if sol is not None:
            thetas_ana[i] = sol
    ee_ana = np.array([
        forward_kinematics_3dof(thetas_ana[i, 0], thetas_ana[i, 1], thetas_ana[i, 2], cfg)
        if not np.isnan(thetas_ana[i, 0])
        else [np.nan, np.nan, np.nan]
        for i in range(N)
    ])
    valid_ana = ~np.isnan(ee_ana[:, 0])
    err_ana = np.where(valid_ana, np.linalg.norm(ee_ana - targets, axis=1) * 1000, np.nan)
    results["analytical"] = {"ee": ee_ana, "err_mm": err_ana, "valid": valid_ana}

    # 几何解耦特征（θ1 精确解析）
    r_xy = np.sqrt(targets[:, 0] ** 2 + targets[:, 1] ** 2)
    z_prime = targets[:, 2] - cfg.L0
    theta1_ana = np.arctan2(targets[:, 1], targets[:, 0])

    def decode_3dof(sincos: np.ndarray) -> np.ndarray:
        t2 = np.arctan2(sincos[:, 0], sincos[:, 1])
        t3 = np.arctan2(sincos[:, 2], sincos[:, 3])
        r = cfg.L1 * np.cos(t2) + cfg.L2 * np.cos(t2 + t3)
        z = cfg.L0 + cfg.L1 * np.sin(t2) + cfg.L2 * np.sin(t2 + t3)
        x = r * np.cos(theta1_ana)
        y = r * np.sin(theta1_ana)
        return np.column_stack([x, y, z])

    feat = np.column_stack([r_xy, z_prime]).astype(np.float32)

    # ---- MLP ----
    if mlp_model is not None:
        mlp_model.eval()
        with torch.no_grad():
            sincos = mlp_model(torch.from_numpy(feat)).numpy()
        ee_mlp = decode_3dof(sincos)
        results["mlp"] = {
            "ee": ee_mlp,
            "err_mm": np.linalg.norm(ee_mlp - targets, axis=1) * 1000,
            "valid": np.ones(N, bool),
            "targets_2d": np.column_stack([r_xy, z_prime]),
        }

    # ---- ESN ----
    if esn_model is not None:
        esn_model.eval()
        with torch.no_grad():
            sincos = esn_model(torch.from_numpy(feat)).numpy()
        ee_esn = decode_3dof(sincos)
        results["esn"] = {
            "ee": ee_esn,
            "err_mm": np.linalg.norm(ee_esn - targets, axis=1) * 1000,
            "valid": np.ones(N, bool),
        }

    return results


def plot_ik_comparison(
    arm_label: str,
    results: dict,
    targets: np.ndarray,
    save_path: str | None = None,
) -> None:
    """绘制四子图 IK 误差对比图并自动保存（可通过 save_path=None 禁用）。"""
    SCHEMES = [
        ("analytical", "#888888", "解析 IK（基准）"),
        ("mlp", "#2196F3", "MLP IK"),
        ("esn", "#F44336", "ESN IK"),
    ]
    is_3dof = arm_label == "3DOF"

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"{arm_label} 逆运动学误差对比：解析解 / MLP / ESN",
        fontsize=26,
        fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.44, wspace=0.35)

    # ---- (a) 误差 CDF ----
    ax0 = fig.add_subplot(gs[0, 0])
    for key, color, label in SCHEMES:
        if key not in results:
            continue
        errs_v = results[key]["err_mm"]
        errs_v = errs_v[~np.isnan(errs_v)]
        if len(errs_v) == 0:
            continue
        sorted_e = np.sort(np.clip(errs_v, 0, 30))
        cdf = np.linspace(0, 100, len(sorted_e))
        ax0.plot(sorted_e, cdf, color=color, linewidth=2.0, label=label)
    ax0.set_xlabel("末端位置误差 (mm)", fontsize=15)
    ax0.set_ylabel("累积百分比 (%)", fontsize=15)
    ax0.set_title("(a) 末端误差累积分布（CDF）", fontsize=18)
    ax0.axvline(1.0, color="k", linestyle="--", alpha=0.4, linewidth=1.2)
    ax0.text(1.05, 8, "1 mm", fontsize=13, color="k", alpha=0.6)
    ax0.legend(fontsize=15)
    ax0.grid(True, alpha=0.3)
    ax0.set_xlim(0, 30)
    ax0.set_ylim(0, 102)

    # ---- (b) 柱状图 ----
    ax1 = fig.add_subplot(gs[0, 1])
    bar_keys = [k for k, _, _ in SCHEMES if k in results]
    bar_colors = [c for k, c, _ in SCHEMES if k in results]
    bar_labels = [lb for k, _, lb in SCHEMES if k in results]
    means, p95s = [], []
    for key in bar_keys:
        errs_v = results[key]["err_mm"]
        errs_v = errs_v[~np.isnan(errs_v)]
        means.append(float(np.mean(errs_v)) if len(errs_v) > 0 else 0.0)
        p95s.append(float(np.percentile(errs_v, 95)) if len(errs_v) > 0 else 0.0)
    x_pos = np.arange(len(bar_keys))
    bars = ax1.bar(x_pos, means, color=bar_colors, alpha=0.85, width=0.5, label="均值")
    ax1.bar(
        x_pos,
        [p - m for p, m in zip(p95s, means)],
        bottom=means,
        color=bar_colors,
        alpha=0.35,
        width=0.5,
        label="均值→95%分位",
    )
    for bar, m, p in zip(bars, means, p95s):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            p + 0.2,
            f"均={m:.2f}\np95={p:.1f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(bar_labels, fontsize=14)
    ax1.set_ylabel("末端误差 (mm)", fontsize=15)
    ax1.set_title("(b) 平均误差 + 95% 分位数", fontsize=18)
    ax1.legend(fontsize=14)
    ax1.grid(True, alpha=0.3, axis="y")

    # ---- (c) MLP 误差空间分布 ----
    ax2 = fig.add_subplot(gs[1, 0])
    if "mlp" in results:
        errs_mlp = results["mlp"]["err_mm"]
        valid = ~np.isnan(errs_mlp)
        clamp_errs = np.clip(errs_mlp[valid], 0, 15)
        if is_3dof:
            feat_2d = results["mlp"]["targets_2d"]
            sc = ax2.scatter(
                feat_2d[valid, 0],
                feat_2d[valid, 1],
                c=clamp_errs,
                cmap="RdYlGn_r",
                s=4,
                alpha=0.6,
                vmin=0,
                vmax=15,
            )
            ax2.set_xlabel("水平半径 r_xy (m)", fontsize=15)
            ax2.set_ylabel("相对高度 z' (m)", fontsize=15)
            ax2.set_title("(c) MLP 误差 — 竖直平面分布（3DOF）", fontsize=18)
        else:
            sc = ax2.scatter(
                targets[valid, 0],
                targets[valid, 1],
                c=clamp_errs,
                cmap="RdYlGn_r",
                s=4,
                alpha=0.6,
                vmin=0,
                vmax=15,
            )
            ax2.set_xlabel("X (m)", fontsize=15)
            ax2.set_ylabel("Y (m)", fontsize=15)
            ax2.set_title("(c) MLP 误差 — 工作空间分布（2DOF）", fontsize=18)
            ax2.set_aspect("equal")
        plt.colorbar(sc, ax=ax2, label="误差 (mm, 截断至 15mm)")
    else:
        ax2.text(
            0.5,
            0.5,
            "无 MLP 检查点",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=18,
        )
        ax2.set_title("(c) MLP 工作空间误差分布", fontsize=18)

    # ---- (d) 误差直方图 ----
    ax3 = fig.add_subplot(gs[1, 1])
    bins = np.logspace(-2, 2, 60)
    for key, color, label in SCHEMES:
        if key not in results:
            continue
        errs_v = results[key]["err_mm"]
        errs_v = errs_v[~np.isnan(errs_v)]
        if len(errs_v) == 0:
            continue
        if errs_v.max() < 1e-6:
            ax3.axvline(1e-3, color=color, linewidth=2, linestyle="--", label=f"{label}（误差≈0）")
            continue
        ax3.hist(errs_v, bins=bins, color=color, alpha=0.5, label=label, density=True)
    ax3.set_xscale("log")
    ax3.set_xlabel("末端位置误差 (mm, 对数轴)", fontsize=15)
    ax3.set_ylabel("概率密度", fontsize=15)
    ax3.set_title("(d) 误差分布直方图（对数 x 轴）", fontsize=18)
    ax3.axvline(1.0, color="k", linestyle="--", alpha=0.4, linewidth=1.2)
    ax3.legend(fontsize=15)
    ax3.grid(True, alpha=0.3)

    # 自动保存
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize] 已保存图片: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def print_summary_ik(arm_label: str, results: dict) -> None:
    """打印 IK 误差结果表。"""
    print(f"\n{'=' * 65}")
    print(f"  {arm_label} IK 误差对比（末端位置误差 mm；低→好）")
    print("=" * 65)
    fmt = "  {:<22}  均值={:7.3f}mm  中位={:7.3f}mm  p95={:7.2f}mm  可达率={:.1f}%"
    for key, label in [("analytical", "解析 IK（基准）"), ("mlp", "MLP IK"), ("esn", "ESN IK")]:
        if key not in results:
            continue
        errs = results[key]["err_mm"]
        valid = results[key]["valid"]
        errs_v = errs[~np.isnan(errs)]
        reach = valid.mean() * 100
        if len(errs_v) == 0:
            print(f"  {label:<22}  （无数据）")
            continue
        print(
            fmt.format(label, np.mean(errs_v), np.median(errs_v), np.percentile(errs_v, 95), reach)
        )
    print("=" * 65)


def _load_ik_checkpoint(ckpt_path: str, model_class, default_kwargs: dict):
    """通用 IK 检查点加载辅助（支持新/旧格式）。"""
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "config" in ckpt:
        model = model_class(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = model_class(**default_kwargs)
        model.load_state_dict(ckpt)
    model.eval()
    return model


def run_ik(args) -> None:
    """IK 误差评估入口（对应 'ik' 子命令）。"""
    run_2dof = args.arm in ("2dof", "all")
    run_3dof = args.arm in ("3dof", "all")

    # ---- 2-DOF ----
    if run_2dof:
        print("\n[vis_ik] === 2-DOF 平面臂 ===")
        cfg2 = PlanarArmConfig()

        mlp2 = _load_ik_checkpoint(
            os.path.join(CKPT_DIR, "ik_2dof_mlp_best.pt"),
            IKNN_MLP,
            {"input_dim": 2, "hidden_dim": 256, "output_dim": 4},
        )
        esn2 = _load_ik_checkpoint(
            os.path.join(CKPT_DIR, "ik_2dof_esn_best.pt"),
            IKNN_ESN,
            {"input_dim": 2, "output_dim": 4},
        )
        for name, model in [("MLP", mlp2), ("ESN", esn2)]:
            if model:
                print(f"[vis_ik] 已加载 2DOF {name}")
            else:
                print(f"[vis_ik] 未找到 2DOF {name} 检查点，跳过")

        print(f"[vis_ik] 采样 {args.n_test} 个 2DOF 目标点...")
        targets_2d = sample_targets_2dof(cfg2, args.n_test, seed=args.seed)
        results_2d = solve_2dof_all(targets_2d, cfg2, mlp2, esn2)
        print_summary_ik("2DOF", results_2d)

        save_path = None if args.no_save else os.path.join(EXPORT_DIR, "ik_comparison_2dof.png")
        plot_ik_comparison("2DOF", results_2d, targets_2d, save_path=save_path)

    # ---- 3-DOF ----
    if run_3dof:
        print("\n[vis_ik] === 3-DOF 空间臂 ===")
        cfg3 = Arm3DOFConfig()

        mlp3 = _load_ik_checkpoint(
            os.path.join(CKPT_DIR, "ik_3dof_mlp_best.pt"),
            IKNN_MLP,
            {"input_dim": 2, "hidden_dim": 384, "output_dim": 4, "n_layers": 5},
        )
        esn3 = _load_ik_checkpoint(
            os.path.join(CKPT_DIR, "ik_3dof_esn_best.pt"),
            IKNN_ESN,
            {"input_dim": 2, "output_dim": 4},
        )
        for name, model in [("MLP", mlp3), ("ESN", esn3)]:
            if model:
                print(f"[vis_ik] 已加载 3DOF {name}")
            else:
                print(f"[vis_ik] 未找到 3DOF {name} 检查点，跳过")

        print(f"[vis_ik] 采样 {args.n_test} 个 3DOF 目标点...")
        targets_3d = sample_targets_3dof(cfg3, args.n_test, seed=args.seed)
        results_3d = solve_3dof_all(targets_3d, cfg3, mlp3, esn3)
        print_summary_ik("3DOF", results_3d)

        save_path = None if args.no_save else os.path.join(EXPORT_DIR, "ik_comparison_3dof.png")
        plot_ik_comparison("3DOF", results_3d, targets_3d, save_path=save_path)


# =====================================================================
# ================= MuJoCo 交互功能区 =================================
# =====================================================================

# GLFW 按键码
_GLFW_RIGHT = 262
_GLFW_LEFT = 263
_GLFW_DOWN = 264
_GLFW_UP = 265
_GLFW_KP8 = 328
_GLFW_KP2 = 322
_HELD_TIMEOUT = 0.12  # 长按超时（秒）
_STEP = 0.01  # 目标点移动步长（米）

# 全局控制状态
_target = np.zeros(3, dtype=np.float64)
_demo_running = False
_demo_t = 0.0
_print_info = False
_held_keys: dict[int, float] = {}


def _make_key_callback(demo_config: dict):
    global _target, _demo_running, _print_info, _held_keys

    def key_callback(keycode: int) -> None:
        global _target, _demo_running, _print_info, _held_keys
        if keycode in (_GLFW_UP, _GLFW_DOWN, _GLFW_LEFT, _GLFW_RIGHT, _GLFW_KP8, _GLFW_KP2):
            _held_keys[keycode] = time.time()
            return
        key = chr(keycode) if 0 < keycode < 128 else None
        if key == "R":
            _target[:] = demo_config["init_target"]
            _demo_running = False
            _held_keys.clear()
        elif keycode == 32:  # SPACE
            _demo_running = not _demo_running
        elif key == "P":
            _print_info = True

    return key_callback


def _clamp_target_2dof(target: np.ndarray, cfg: PlanarArmConfig) -> np.ndarray:
    xy = target[:2].copy()
    r = float(np.linalg.norm(xy))
    if r < cfg.r_min + 0.005:
        xy = np.array([cfg.r_min + 0.005, 0.0]) if r < 1e-9 else xy / r * (cfg.r_min + 0.005)
    elif r > cfg.r_max - 0.005:
        xy = xy / r * (cfg.r_max - 0.005)
    return np.array([xy[0], xy[1], target[2]])


def _clamp_target_3dof(target: np.ndarray, cfg: Arm3DOFConfig) -> np.ndarray:
    t = target.copy()
    z_lo = max(0.05, cfg.L0 - cfg.reach_max + 0.01)
    z_hi = cfg.L0 + cfg.reach_max - 0.01
    t[2] = np.clip(t[2], z_lo, z_hi)
    xy = t[:2]
    r_xy = float(np.linalg.norm(xy))
    z_prime = t[2] - cfg.L0
    r_max_at_z = max(0.02, np.sqrt(max(0, cfg.reach_max**2 - z_prime**2)) - 0.01)
    if r_xy > r_max_at_z:
        xy = np.array([r_max_at_z, 0.0]) if r_xy < 1e-9 else xy / r_xy * r_max_at_z
    t[0], t[1] = xy[0], xy[1]
    return t


def _load_ik_model_mujoco(model_type: str, nn_ckpt: str, arm_name: str):
    """加载 IK 模型（MLP 或 ESN）用于 MuJoCo 实时控制。"""
    try:
        import torch

        if not os.path.exists(nn_ckpt):
            return None, None
        ckpt = torch.load(nn_ckpt, map_location="cpu", weights_only=False)
        if model_type == "mlp":
            if isinstance(ckpt, dict) and "config" in ckpt:
                model = IKNN_MLP(**ckpt["config"])
                model.load_state_dict(ckpt["state_dict"])
                info = (
                    f"[{arm_name.upper()}] MLP IK (hidden={ckpt['config'].get('hidden_dim', 256)})"
                )
            else:
                model = IKNN_MLP(input_dim=2, hidden_dim=256, output_dim=4)
                model.load_state_dict(ckpt)
                info = f"[{arm_name.upper()}] MLP IK (旧格式)"
        else:  # esn
            if isinstance(ckpt, dict) and "config" in ckpt:
                model = IKNN_ESN(**ckpt["config"])
                model.load_state_dict(ckpt["state_dict"])
                info = f"[{arm_name.upper()}] ESN IK"
            else:
                model = IKNN_ESN(input_dim=2, output_dim=4)
                model.load_state_dict(ckpt)
                info = f"[{arm_name.upper()}] ESN IK (旧格式)"
        model.eval()
        return model, info
    except Exception as e:
        print(f"[ERROR] 加载模型失败: {e}")
        return None, None


def _run_2dof_viewer(
    xml_path: str,
    cfg: PlanarArmConfig,
    use_nn: bool = False,
    nn_ckpt: str | None = None,
    model_type: str = "mlp",
) -> None:
    """启动 2-DOF 平面臂 MuJoCo 交互可视化。"""
    global _target, _demo_running, _demo_t, _print_info
    import mujoco
    import mujoco.viewer

    mjmodel = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(mjmodel)
    mujoco.mj_forward(mjmodel, data)

    init_target = np.array([0.20, 0.0, cfg.arm_height])
    _target = init_target.copy()
    demo_config = {
        "init_target": init_target.copy(),
        "center": np.array([0.0, 0.0, cfg.arm_height]),
        "radius": 0.20,
    }

    target_body_id = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_BODY, "target")
    target_mocap_id = mjmodel.body_mocapid[target_body_id]

    nn_model = None
    if use_nn and nn_ckpt:
        nn_model, info = _load_ik_model_mujoco(model_type, nn_ckpt, "2dof")
        if nn_model:
            print(f"[2DOF] NN IK 模型已加载：{info}")
        else:
            print("[2DOF] 加载 NN 模型失败，降级为解析 IK")
    else:
        print("[2DOF] 使用解析 IK")

    j1_id = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
    j2_id = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_JOINT, "joint2")

    print("\n=== 2-DOF 平面臂 MuJoCo 可视化 ===")
    print("  ↑/↓: +Y/-Y   ←/→: -X/+X   R: 复位   SPACE: 演示   P: 打印状态\n")

    with mujoco.viewer.launch_passive(
        mjmodel, data, key_callback=_make_key_callback(demo_config)
    ) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -70
        viewer.cam.distance = 0.8
        viewer.cam.lookat[:] = [0, 0, 0.05]

        while viewer.is_running():
            t_loop = time.time()
            if not _demo_running:
                now = time.time()
                if now - _held_keys.get(_GLFW_UP, 0) < _HELD_TIMEOUT:
                    _target[1] += _STEP
                if now - _held_keys.get(_GLFW_DOWN, 0) < _HELD_TIMEOUT:
                    _target[1] -= _STEP
                if now - _held_keys.get(_GLFW_LEFT, 0) < _HELD_TIMEOUT:
                    _target[0] -= _STEP
                if now - _held_keys.get(_GLFW_RIGHT, 0) < _HELD_TIMEOUT:
                    _target[0] += _STEP
            if _demo_running:
                _demo_t += 0.02
                r_wave = demo_config["radius"] * (0.6 + 0.4 * np.cos(3 * _demo_t))
                _target[0] = r_wave * np.cos(_demo_t)
                _target[1] = r_wave * np.sin(_demo_t)
                _target[2] = cfg.arm_height
            _target = _clamp_target_2dof(_target, cfg)
            data.mocap_pos[target_mocap_id, :] = _target

            if nn_model is not None:
                inp = torch.tensor(_target[:2], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    sincos = nn_model(inp).squeeze(0).numpy()
                angles = (
                    float(np.arctan2(sincos[0], sincos[1])),
                    float(np.arctan2(sincos[2], sincos[3])),
                )
            else:
                angles = inverse_kinematics_planar(_target[:2], cfg, elbow_up=True)

            if angles:
                data.qpos[j1_id] = angles[0]
                data.qpos[j2_id] = angles[1]
                mujoco.mj_forward(mjmodel, data)

            if _print_info:
                ee = data.sensor("ee_pos").data.copy()
                print(
                    f"  目标: ({_target[0]:.3f},{_target[1]:.3f})  "
                    f"θ1={np.degrees(data.qpos[j1_id]):.1f}°  "
                    f"θ2={np.degrees(data.qpos[j2_id]):.1f}°  "
                    f"末端: ({ee[0]:.3f},{ee[1]:.3f})"
                )
                _print_info = False  # noqa: F811

            viewer.sync()
            elapsed = time.time() - t_loop
            if elapsed < 1 / 60:
                time.sleep(1 / 60 - elapsed)


def _run_3dof_viewer(
    xml_path: str,
    cfg: Arm3DOFConfig,
    use_nn: bool = False,
    nn_ckpt: str | None = None,
    model_type: str = "mlp",
) -> None:
    """启动 3-DOF 空间臂 MuJoCo 交互可视化。"""
    global _target, _demo_running, _demo_t, _print_info
    import mujoco
    import mujoco.viewer

    mjmodel = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(mjmodel)
    mujoco.mj_forward(mjmodel, data)

    init_target = np.array([0.20, 0.10, cfg.L0 + 0.15])
    _target = init_target.copy()
    demo_config = {
        "init_target": init_target.copy(),
        "center": np.array([0.0, 0.0, cfg.L0 + 0.10]),
        "radius": 0.18,
    }

    target_body_id = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_BODY, "target")
    target_mocap_id = mjmodel.body_mocapid[target_body_id]

    nn_model = None
    if use_nn and nn_ckpt:
        nn_model, info = _load_ik_model_mujoco(model_type, nn_ckpt, "3dof")
        if nn_model:
            print(f"[3DOF] NN IK 模型已加载：{info}")
        else:
            print("[3DOF] 加载失败，降级为解析 IK")
    else:
        print("[3DOF] 使用解析 IK")

    j1_id = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
    j2_id = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_JOINT, "joint2")
    j3_id = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_JOINT, "joint3")

    print("\n=== 3-DOF 空间臂 MuJoCo 可视化 ===")
    print("  ↑/↓: +Y/-Y   ←/→: -X/+X   KP8/KP2: +Z/-Z   R: 复位   SPACE: 演示   P: 打印\n")

    with mujoco.viewer.launch_passive(
        mjmodel, data, key_callback=_make_key_callback(demo_config)
    ) as viewer:
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -30
        viewer.cam.distance = 1.0
        viewer.cam.lookat[:] = [0, 0, 0.25]

        while viewer.is_running():
            t_loop = time.time()
            if not _demo_running:
                now = time.time()
                if now - _held_keys.get(_GLFW_UP, 0) < _HELD_TIMEOUT:
                    _target[1] += _STEP
                if now - _held_keys.get(_GLFW_DOWN, 0) < _HELD_TIMEOUT:
                    _target[1] -= _STEP
                if now - _held_keys.get(_GLFW_LEFT, 0) < _HELD_TIMEOUT:
                    _target[0] -= _STEP
                if now - _held_keys.get(_GLFW_RIGHT, 0) < _HELD_TIMEOUT:
                    _target[0] += _STEP
                if now - _held_keys.get(_GLFW_KP8, 0) < _HELD_TIMEOUT:
                    _target[2] += _STEP
                if now - _held_keys.get(_GLFW_KP2, 0) < _HELD_TIMEOUT:
                    _target[2] -= _STEP
            if _demo_running:
                _demo_t += 0.02
                r = demo_config["radius"]
                cx, cy, cz = demo_config["center"]
                _target[0] = cx + r * np.cos(_demo_t)
                _target[1] = cy + r * np.sin(_demo_t)
                _target[2] = cz + 0.10 * np.sin(2 * _demo_t)
            _target = _clamp_target_3dof(_target, cfg)
            data.mocap_pos[target_mocap_id, :] = _target

            if nn_model is not None:
                theta1 = float(np.arctan2(_target[1], _target[0]))
                r_xy = float(np.sqrt(_target[0] ** 2 + _target[1] ** 2))
                z_prime = float(_target[2] - cfg.L0)
                inp = torch.tensor([[r_xy, z_prime]], dtype=torch.float32)
                with torch.no_grad():
                    sincos = nn_model(inp).squeeze(0).numpy()
                angles = (
                    theta1,
                    float(np.arctan2(sincos[0], sincos[1])),
                    float(np.arctan2(sincos[2], sincos[3])),
                )
            else:
                angles = inverse_kinematics_3dof(_target, cfg, elbow_up=True)

            if angles:
                data.qpos[j1_id] = angles[0]
                data.qpos[j2_id] = angles[1]
                data.qpos[j3_id] = angles[2]
                mujoco.mj_forward(mjmodel, data)

            if _print_info:
                ee = data.sensor("ee_pos").data.copy()
                print(
                    f"  目标: ({_target[0]:.3f},{_target[1]:.3f},{_target[2]:.3f})  "
                    f"θ=[{np.degrees(data.qpos[j1_id]):.1f}°,"
                    f"{np.degrees(data.qpos[j2_id]):.1f}°,"
                    f"{np.degrees(data.qpos[j3_id]):.1f}°]  "
                    f"末端: ({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})"
                )
                _print_info = False  # noqa: F811

            viewer.sync()
            elapsed = time.time() - t_loop
            if elapsed < 1 / 60:
                time.sleep(1 / 60 - elapsed)


def run_mujoco(args) -> None:
    """MuJoCo 交互可视化入口（对应 'mujoco' 子命令）。"""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(src_dir, "..", "checkpoints")

    if args.arm == "2dof":
        xml_path = os.path.join(src_dir, "arm_2dof.xml")
        cfg = PlanarArmConfig()
        ckpt_default = os.path.join(ckpt_dir, f"ik_2dof_{args.model_type}_best.pt")
        nn_ckpt = (
            args.ckpt if args.ckpt else (ckpt_default if os.path.exists(ckpt_default) else None)
        )
        _run_2dof_viewer(
            xml_path, cfg, use_nn=args.use_nn, nn_ckpt=nn_ckpt, model_type=args.model_type
        )
    else:
        xml_path = os.path.join(src_dir, "arm_3dof.xml")
        cfg = Arm3DOFConfig()
        ckpt_default = os.path.join(ckpt_dir, f"ik_3dof_{args.model_type}_best.pt")
        nn_ckpt = (
            args.ckpt if args.ckpt else (ckpt_default if os.path.exists(ckpt_default) else None)
        )
        _run_3dof_viewer(
            xml_path, cfg, use_nn=args.use_nn, nn_ckpt=nn_ckpt, model_type=args.model_type
        )


# =====================================================================
# 主入口（子命令路由）
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="机械臂可视化统一入口（ik / mujoco 两种模式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ------- ik 子命令 -------
    p_ik = subparsers.add_parser("ik", help="IK 误差评估（离线，自动保存图片）")
    p_ik.add_argument(
        "--arm", default="all", choices=["2dof", "3dof", "all"], help="机械臂类型（默认 all）"
    )
    p_ik.add_argument("--n_test", type=int, default=8000, help="随机测试目标点数（默认 8000）")
    p_ik.add_argument("--seed", type=int, default=7777, help="采样随机种子")
    p_ik.add_argument(
        "--no-save", dest="no_save", action="store_true", help="禁用自动保存（仅显示窗口）"
    )

    # ------- mujoco 子命令 -------
    p_mj = subparsers.add_parser("mujoco", help="MuJoCo 实时交互可视化")
    p_mj.add_argument(
        "--arm", default="2dof", choices=["2dof", "3dof"], help="机械臂类型（默认 2dof）"
    )
    p_mj.add_argument("--use_nn", action="store_true", help="使用 NN 计算 IK（需先训练 train.py）")
    p_mj.add_argument(
        "--model_type", default="mlp", choices=["mlp", "esn"], help="NN 模型类型（默认 mlp）"
    )
    p_mj.add_argument("--ckpt", default=None, help="手动指定权重路径（可选）")

    args = parser.parse_args()

    if args.mode == "ik":
        run_ik(args)
    else:
        run_mujoco(args)
