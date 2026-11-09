"""
逆运动学误差对比可视化

对比三种 IK 方案在随机测试目标上的末端误差：
    1. 解析 IK    — 几何精确解（余弦定理），作为基准
    2. MLP IK     — 训练好的多层感知机
    3. ESN IK     — 回声状态网络（储层计算）

评估方式（纯运动学，无 MuJoCo 仿真）：
    - 在工作空间内均匀采样 N 个随机目标点
    - 各方案分别计算 IK 得到关节角 θ
    - 用正运动学将 θ 映射回末端坐标
    - 误差 = ||FK(θ̂) - target||（毫米），不可达点排除

可视化内容：
    (a) 末端位置误差 CDF（累积分布，越靠左上越好）
    (b) 各方案平均末端误差柱状图
    (c) 2D/3D 工作空间散点图（颜色编码误差大小，仅 MLP）
    (d) 误差直方图（线性 x 轴）

用法：
    uv run src/visualize_ik.py --arm 2dof          # 2-DOF 平面臂
    uv run src/visualize_ik.py --arm 3dof          # 3-DOF 空间臂
    uv run src/visualize_ik.py --arm all           # 两者都画
    uv run src/visualize_ik.py --arm all --save    # 保存图片
    uv run src/visualize_ik.py --arm 2dof --n_test 20000  # 更多测试点
"""

import argparse
import os
import sys

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── 中文字体配置 ─────────────────────────────────────────────────────
for _font in ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei"]:
    if _font in [f.name for f in matplotlib.font_manager.fontManager.ttflist]:
        matplotlib.rcParams["font.sans-serif"] = [_font] + matplotlib.rcParams["font.sans-serif"]
        break
matplotlib.rcParams["axes.unicode_minus"] = False
# ─────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.kinematics_2dof import (
    PlanarArmConfig,
    forward_kinematics_planar,
    inverse_kinematics_planar,
    valid_joint_angles_planar,
)
from src.kinematics_3dof import (
    Arm3DOFConfig,
    forward_kinematics_3dof,
    inverse_kinematics_3dof,
    valid_joint_angles_3dof,
)
from src.models import IKNN_ESN, IKNN_MLP

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
EXPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "export")


# =====================================================================
# 测试目标点采样
# =====================================================================


def sample_targets_2dof(cfg: PlanarArmConfig, n: int, seed: int = 42) -> np.ndarray:
    """
    在 2-DOF 平面臂工作空间内均匀采样目标点。

    工作空间：环形区域，半径 [r_min+margin, r_max-margin] 的 XY 圆环。

    返回:
        (n, 2) float64 [X, Y]（米）
    """
    if n < 0:
        raise ValueError("采样点数 n 必须为非负整数")
    if n == 0:
        return np.empty((0, 2), dtype=np.float64)
    rng = np.random.default_rng(seed)
    # 均匀采样环形（拒绝采样简单可靠）
    # 极坐标：r² 均匀 → r 分布正确
    r_min = cfg.r_min + 0.01
    r_max = cfg.r_max - 0.01
    if not np.isfinite([r_min, r_max]).all() or not 0.0 <= r_min < r_max:
        raise ValueError("连杆长度与安全边距形成空采样域，请检查机械臂配置")
    targets = []
    count = 0
    while count < n:
        # 批量采样候选点（XY 均匀正方形内）
        batch = rng.uniform(-r_max, r_max, (min(3 * (n - count), 8192), 2))
        r = np.linalg.norm(batch, axis=1)
        valid = (r >= r_min) & (r <= r_max)
        accepted = batch[valid][: n - count]
        targets.append(accepted)
        count += len(accepted)
    return np.concatenate(targets, axis=0)


def sample_targets_3dof(cfg: Arm3DOFConfig, n: int, seed: int = 42) -> np.ndarray:
    """
    在 3-DOF 空间臂工作空间内均匀采样目标点。

    工作空间：球形 shell，半径 [0.05, reach_max-0.01]。

    返回:
        (n, 3) float64 [X, Y, Z]（米）
    """
    if n < 0:
        raise ValueError("采样点数 n 必须为非负整数")
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)
    rng = np.random.default_rng(seed)
    r_max = cfg.L1 + cfg.L2 - 0.01
    r_min = 0.05  # 避免奇异点
    if (not np.isfinite([cfg.L0, r_min, r_max]).all()
            or not r_min < r_max or cfg.L0 + r_max <= 0.05):
        raise ValueError("连杆长度、安全边距与高度限制形成空采样域，请检查机械臂配置")
    targets = []
    count = 0
    while count < n:
        batch = rng.uniform(-r_max, r_max, (min(3 * (n - count), 8192), 3))
        # Z 必须在合理范围内：Z ≥ L0 - reach_max + margin
        batch[:, 2] += cfg.L0  # 以 L0 为 Z 中心偏置
        r3d = np.linalg.norm(batch - np.array([0, 0, cfg.L0]), axis=1)
        z_ok = (batch[:, 2] >= 0.05) & (batch[:, 2] <= cfg.L0 + cfg.L1 + cfg.L2 - 0.01)
        valid = (r3d >= r_min) & (r3d <= r_max) & z_ok
        accepted = batch[valid][: n - count]
        targets.append(accepted)
        count += len(accepted)
    return np.concatenate(targets, axis=0)


# =====================================================================
# 各 IK 方案批量求解
# =====================================================================


def solve_2dof_all(
    targets: np.ndarray,  # (N, 2)
    cfg: PlanarArmConfig,
    mlp_model: IKNN_MLP | None,
    esn_model: IKNN_ESN | None,
) -> dict:
    """
    对所有目标点批量求解（解析 + MLP + ESN），并用正运动学计算末端位置。

    返回字典：
        {
          "analytical":  {"ee": (N, 2), "err_mm": (N,), "valid": (N,)},
          "mlp":         {...},
          "esn":         {...},
        }
    """
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
        valid_mlp = np.isfinite(sincos).all(axis=1) & valid_joint_angles_planar(
            np.column_stack([theta1, theta2]), cfg
        )
        err_mlp = np.where(valid_mlp, np.linalg.norm(ee_mlp - targets, axis=1) * 1000, np.nan)
        results["mlp"] = {"ee": ee_mlp, "err_mm": err_mlp, "valid": valid_mlp}

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
        valid_esn = np.isfinite(sincos).all(axis=1) & valid_joint_angles_planar(
            np.column_stack([theta1, theta2]), cfg
        )
        err_esn = np.where(valid_esn, np.linalg.norm(ee_esn - targets, axis=1) * 1000, np.nan)
        results["esn"] = {"ee": ee_esn, "err_mm": err_esn, "valid": valid_esn}

    return results


def solve_3dof_all(
    targets: np.ndarray,  # (N, 3)
    cfg: Arm3DOFConfig,
    mlp_model: IKNN_MLP | None,
    esn_model: IKNN_ESN | None,
) -> dict:
    """
    对所有目标点批量求解 3-DOF IK（解析 + MLP + ESN）。

    注意：MLP/ESN 使用几何解耦：
        θ1 = atan2(y, x)（解析精确）
        MLP/ESN 输入: [r_xy, z'] → sincos → θ2, θ3

    返回字典同上，err_mm 为末端三维距离误差（毫米）。
    """
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

    # 3-DOF MLP/ESN 共用的几何解耦特征
    r_xy = np.sqrt(targets[:, 0] ** 2 + targets[:, 1] ** 2)
    z_prime = targets[:, 2] - cfg.L0
    theta1_ana = np.arctan2(targets[:, 1], targets[:, 0])  # 精确 θ1（解析）

    def decode_3dof(sincos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """从网络输出的 sin/cos 编码还原末端坐标。"""
        t2 = np.arctan2(sincos[:, 0], sincos[:, 1])
        t3 = np.arctan2(sincos[:, 2], sincos[:, 3])
        r = cfg.L1 * np.cos(t2) + cfg.L2 * np.cos(t2 + t3)
        z = cfg.L0 + cfg.L1 * np.sin(t2) + cfg.L2 * np.sin(t2 + t3)
        x = r * np.cos(theta1_ana)
        y = r * np.sin(theta1_ana)
        valid = np.isfinite(sincos).all(axis=1) & valid_joint_angles_3dof(
            np.column_stack([theta1_ana, t2, t3]), cfg
        )
        return np.column_stack([x, y, z]), valid

    # ---- MLP ----
    if mlp_model is not None:
        mlp_model.eval()
        feat = np.column_stack([r_xy, z_prime]).astype(np.float32)
        with torch.no_grad():
            sincos = mlp_model(torch.from_numpy(feat)).numpy()
        ee_mlp, valid_mlp = decode_3dof(sincos)
        err_mlp = np.where(valid_mlp, np.linalg.norm(ee_mlp - targets, axis=1) * 1000, np.nan)
        results["mlp"] = {
            "ee": ee_mlp,
            "err_mm": err_mlp,
            "valid": valid_mlp,
            "targets_2d": np.column_stack([r_xy, z_prime]),
        }

    # ---- ESN ----
    if esn_model is not None:
        esn_model.eval()
        feat = np.column_stack([r_xy, z_prime]).astype(np.float32)
        with torch.no_grad():
            sincos = esn_model(torch.from_numpy(feat)).numpy()
        ee_esn, valid_esn = decode_3dof(sincos)
        err_esn = np.where(valid_esn, np.linalg.norm(ee_esn - targets, axis=1) * 1000, np.nan)
        results["esn"] = {"ee": ee_esn, "err_mm": err_esn, "valid": valid_esn}

    return results


# =====================================================================
# 统一绘图
# =====================================================================


def plot_ik_comparison(
    arm_label: str,
    results: dict,
    targets: np.ndarray,
    save_path: str | None = None,
) -> None:
    """
    绘制四子图 IK 误差对比：
        (a) 末端位置误差 CDF（各方案）
        (b) 平均误差柱状图 + 95 百分位误差
        (c) MLP 误差空间分布散点图（颜色 = 误差大小）
        (d) 误差直方图（ln 换算，展示尾部分布）
    """
    # 方案定义：key → (颜色, 显示标签)
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
    legal_rates = " / ".join(
        f"{label} {np.mean(results[key]['valid']) * 100:.1f}%"
        for key, label in [("analytical", "解析"), ("mlp", "MLP"), ("esn", "ESN")]
        if key in results
    )
    fig.text(
        0.5, 0.915, f"误差仅统计合法解；合法解率：{legal_rates}",
        ha="center", va="center", fontsize=18,
    )
    if is_3dof:
        fig.text(
            0.5, 0.875, "球壳广域压力测试：含正肘训练域外目标",
            ha="center", va="center", fontsize=18,
        )
    gs = gridspec.GridSpec(2, 2, top=0.81 if is_3dof else 0.85, hspace=0.44, wspace=0.35)

    # ---- (a) 误差 CDF ----
    ax0 = fig.add_subplot(gs[0, 0])
    for key, color, label in SCHEMES:
        if key not in results:
            continue
        errs = results[key]["err_mm"]
        errs_valid = errs[~np.isnan(errs)]
        if len(errs_valid) == 0:
            continue
        # 最大显示 30 mm（超出按 30mm 截断）
        sorted_e = np.sort(np.clip(errs_valid, 0, 30))
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

    # ---- (b) 柱状图：均值 + 95 百分位 ----
    ax1 = fig.add_subplot(gs[0, 1])
    bar_keys = [k for k, _, _ in SCHEMES if k in results]
    bar_colors = [c for k, c, _ in SCHEMES if k in results]
    bar_labels = [lb for k, _, lb in SCHEMES if k in results]
    means = []
    p95s = []
    for key in bar_keys:
        errs = results[key]["err_mm"]
        errs_v = errs[~np.isnan(errs)]
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
    # 在柱顶标注数值
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

    # ---- (c) MLP 映射到工作空间的误差分布散点 ----
    ax2 = fig.add_subplot(gs[1, 0])
    if "mlp" in results:
        errs_mlp = results["mlp"]["err_mm"]
        valid = ~np.isnan(errs_mlp)
        clamp_errs = np.clip(errs_mlp[valid], 0, 15)  # 截断到 15 mm 便于着色
        if is_3dof:
            # 用竖直平面坐标 (r_xy, z') 展示
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

    # ---- (d) 误差直方图（对数 x 轴） ----
    ax3 = fig.add_subplot(gs[1, 1])
    bins = np.logspace(-2, 2, 60)  # 0.01mm ~ 100mm
    for key, color, label in SCHEMES:
        if key not in results:
            continue
        errs = results[key]["err_mm"]
        errs_v = errs[~np.isnan(errs)]
        if len(errs_v) == 0:
            continue
        # 解析解误差全为 0 时跳过 density 直方图（避免除以零 warning）
        if errs_v.max() < 1e-6:
            ax3.axvline(1e-3, color=color, linewidth=2, linestyle="--", label=f"{label}（误差≈0）")
            continue
        ax3.hist(errs_v, bins=bins, color=color, alpha=0.5, label=label, density=True)
    ax3.set_xscale("log")
    ax3.set_xlabel("末端位置误差 (mm, 对数轴)", fontsize=15)
    ax3.set_ylabel("概率密度", fontsize=15)
    ax3.set_title("(d) 误差分布直方图（对数 x 轴）", fontsize=18)
    ax3.axvline(1.0, color="k", linestyle="--", alpha=0.4, linewidth=1.2)
    ax3.text(
        1.1,
        ax3.get_ylim()[1] * 0.9 if ax3.get_ylim()[1] > 0 else 0.1,
        "1 mm",
        fontsize=13,
        color="k",
        alpha=0.6,
    )
    ax3.legend(fontsize=15)
    ax3.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize_ik] 已保存: {save_path}")
        plt.close(fig)
    else:
        plt.show()


# =====================================================================
# 打印定量结论
# =====================================================================


def print_summary_ik(arm_label: str, results: dict) -> None:
    """打印各方案末端位置误差汇总表。"""
    print(f"\n{'=' * 65}")
    print(f"  {arm_label} IK 误差对比（末端位置误差 mm；低→好）")
    print("=" * 65)
    fmt = "  {:<22}  均值={:7.3f}mm  中位={:7.3f}mm  p95={:7.2f}mm  合法解率={:.1f}%"
    print("  误差仅统计合法解；合法解率检查有限与关节限位，不代表精度成功率。")
    for key, label in [
        ("analytical", "解析 IK（基准）"),
        ("mlp", "MLP IK"),
        ("esn", "ESN IK"),
    ]:
        if key not in results:
            continue
        errs = results[key]["err_mm"]
        valid = results[key]["valid"]
        errs_v = errs[~np.isnan(errs)]
        reach = valid.mean() * 100 if len(valid) else 0.0
        if len(errs_v) == 0:
            print(f"  {label:<22}  合法解率={reach:.1f}%（无合法解，无法统计误差）")
            continue
        print(
            fmt.format(label, np.mean(errs_v), np.median(errs_v), np.percentile(errs_v, 95), reach)
        )
    print("=" * 65)


# =====================================================================
# 主入口
# =====================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IK 模型误差对比可视化")
    parser.add_argument(
        "--arm",
        type=str,
        default="all",
        choices=["2dof", "3dof", "all"],
        help="机械臂类型（默认 all 两种都画）",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=8000,
        help="随机测试目标点数量（默认 8000）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7777,
        help="采样随机种子（与训练数据不重叠）",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="保存图片到 export/ 目录",
    )
    args = parser.parse_args()

    run_2dof = args.arm in ("2dof", "all")
    run_3dof = args.arm in ("3dof", "all")

    # ================================================================
    # 2-DOF
    # ================================================================
    if run_2dof:
        print("\n[vis_ik] === 2-DOF 平面臂 ===")
        cfg2 = PlanarArmConfig()

        # 加载 MLP
        ckpt_mlp2 = os.path.join(CKPT_DIR, "ik_2dof_mlp_best.pt")
        mlp2 = None
        if os.path.exists(ckpt_mlp2):
            ckpt = torch.load(ckpt_mlp2, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "config" in ckpt:
                mlp2 = IKNN_MLP(**ckpt["config"])
                mlp2.load_state_dict(ckpt["state_dict"])
            else:
                mlp2 = IKNN_MLP(input_dim=2, hidden_dim=256, output_dim=4)
                mlp2.load_state_dict(ckpt)
            mlp2.eval()
            print(f"[vis_ik] 已加载 2DOF MLP: {ckpt_mlp2}")
        else:
            print(f"[vis_ik] 未找到 2DOF MLP 检查点（{ckpt_mlp2}），跳过")

        # 加载 ESN
        ckpt_esn2 = os.path.join(CKPT_DIR, "ik_2dof_esn_best.pt")
        esn2 = None
        if os.path.exists(ckpt_esn2):
            ckpt = torch.load(ckpt_esn2, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "config" in ckpt:
                esn2 = IKNN_ESN(**ckpt["config"])
                esn2.load_state_dict(ckpt["state_dict"])
            else:
                esn2 = IKNN_ESN(input_dim=2, output_dim=4)
                esn2.load_state_dict(ckpt)
            esn2.eval()
            print(f"[vis_ik] 已加载 2DOF ESN: {ckpt_esn2}")
        else:
            print(f"[vis_ik] 未找到 2DOF ESN 检查点（{ckpt_esn2}），跳过")

        # 采样目标 + 求解
        print(f"[vis_ik] 采样 {args.n_test} 个 2DOF 目标点...")
        targets_2d = sample_targets_2dof(cfg2, args.n_test, seed=args.seed)
        results_2d = solve_2dof_all(targets_2d, cfg2, mlp2, esn2)
        print_summary_ik("2DOF", results_2d)

        save_path = os.path.join(EXPORT_DIR, "ik_comparison_2dof.png") if args.save else None
        plot_ik_comparison("2DOF", results_2d, targets_2d, save_path=save_path)

    # ================================================================
    # 3-DOF
    # ================================================================
    if run_3dof:
        print("\n[vis_ik] === 3-DOF 空间臂 ===")
        cfg3 = Arm3DOFConfig()

        # 加载 MLP（3DOF 使用 2D 竖直平面输入，input_dim=2, output_dim=4）
        ckpt_mlp3 = os.path.join(CKPT_DIR, "ik_3dof_mlp_best.pt")
        mlp3 = None
        if os.path.exists(ckpt_mlp3):
            ckpt = torch.load(ckpt_mlp3, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "config" in ckpt:
                mlp3 = IKNN_MLP(**ckpt["config"])
                mlp3.load_state_dict(ckpt["state_dict"])
            else:
                mlp3 = IKNN_MLP(input_dim=2, hidden_dim=384, output_dim=4, n_layers=5)
                mlp3.load_state_dict(ckpt)
            mlp3.eval()
            print(f"[vis_ik] 已加载 3DOF MLP: {ckpt_mlp3}")
        else:
            print(f"[vis_ik] 未找到 3DOF MLP 检查点（{ckpt_mlp3}），跳过")

        # 加载 ESN
        ckpt_esn3 = os.path.join(CKPT_DIR, "ik_3dof_esn_best.pt")
        esn3 = None
        if os.path.exists(ckpt_esn3):
            ckpt = torch.load(ckpt_esn3, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "config" in ckpt:
                esn3 = IKNN_ESN(**ckpt["config"])
                esn3.load_state_dict(ckpt["state_dict"])
            else:
                esn3 = IKNN_ESN(input_dim=2, output_dim=4)
                esn3.load_state_dict(ckpt)
            esn3.eval()
            print(f"[vis_ik] 已加载 3DOF ESN: {ckpt_esn3}")
        else:
            print(f"[vis_ik] 未找到 3DOF ESN 检查点（{ckpt_esn3}），跳过")

        # 采样目标 + 求解
        print(f"[vis_ik] 采样 {args.n_test} 个 3DOF 目标点...")
        print("[vis_ik] 球壳广域压力测试，包含正肘训练域外目标；不等同训练域内精度。")
        targets_3d = sample_targets_3dof(cfg3, args.n_test, seed=args.seed)
        results_3d = solve_3dof_all(targets_3d, cfg3, mlp3, esn3)
        print_summary_ik("3DOF", results_3d)

        save_path = os.path.join(EXPORT_DIR, "ik_comparison_3dof.png") if args.save else None
        plot_ik_comparison("3DOF", results_3d, targets_3d, save_path=save_path)
