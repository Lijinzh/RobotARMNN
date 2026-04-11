"""
训练数据生成脚本（统一版）

支持：
  - 平面 2-DOF 机械臂（水平 XY 平面，L1=0.15m，L2=0.12m）
  - 三自由度空间机械臂（3-DOF，L0=0.20m，L1=0.15m，L2=0.12m）

输出格式：
  *.npz : 训练用（inputs / outputs / angles）
  *.csv : 方便查看 / 共享（含列名，角度单位为度）

3-DOF 数据特殊设计（几何解耦）：
  θ1 = atan2(Y, X) 有精确解析解，无需学习。
  只需学习竖直平面 2-link IK：(r_xy, z') → (θ2, θ3)，其中：
    r_xy = sqrt(X² + Y²)，z' = Z - L0
  数据集同时保存：
    inputs      : (N, 3) 原始 XYZ（用于可视化/参考）
    inputs_vert : (N, 2) [r_xy, z']（3DOF MLP 实际训练输入）
    outputs_vert: (N, 4) [sinθ2,cosθ2,sinθ3,cosθ3]（3DOF MLP 实际训练目标）
    angles_vert : (N, 2) [θ2, θ3]（用于角度误差评估）

采样策略：
  - 仅采样 elbow-up（θ3 ∈ [0, π]），保证同一末端坐标只对应唯一解，
    避免双解歧义导致网络训练目标冲突

用法:
    uv run src/data_gen.py --arm 2dof                  # 生成 2-DOF 数据
    uv run src/data_gen.py --arm 3dof                  # 生成 3-DOF 数据
    uv run src/data_gen.py --arm all                   # 同时生成两者
    uv run src/data_gen.py --arm 2dof --n_samples 1000000  # 增加样本量
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.kinematics_2dof import (
    PlanarArmConfig,
    angles_to_sincos_planar,
    forward_kinematics_planar_batch,
)
from src.kinematics_3dof import (
    Arm3DOFConfig,
    angles_to_sincos_3dof,
    forward_kinematics_3dof_batch,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# =====================================================================
# 2-DOF 平面臂数据生成
# =====================================================================


def gen_2dof(
    n_samples: int = 500_000,
    cfg: PlanarArmConfig | None = None,
    seed: int = 42,
) -> None:
    """
    生成平面 2-DOF IK 数据集，保存为 .npz 和 .csv。

    数据来源：均匀采样 (θ1, θ2) → FK 计算末端 XY。
    CSV 列名：X_m, Y_m, theta1_deg, theta2_deg, sin_t1, cos_t1, sin_t2, cos_t2
    """
    if cfg is None:
        cfg = PlanarArmConfig()

    print(f"[data_gen] 2-DOF 平面臂  n={n_samples:,}  L1={cfg.L1}m  L2={cfg.L2}m")
    rng = np.random.default_rng(seed)

    theta1 = rng.uniform(cfg.theta1_min, cfg.theta1_max, n_samples).astype(np.float32)
    # elbow-up：θ2 ∈ [0, π]，确保 XY→角度映射唯一
    theta2 = rng.uniform(0.0, np.pi, n_samples).astype(np.float32)

    xy = forward_kinematics_planar_batch(theta1, theta2, cfg)  # (N, 2)
    angles = np.stack([theta1, theta2], axis=-1)  # (N, 2)
    sincos = angles_to_sincos_planar(angles)  # (N, 4)

    # 过滤 r < 1cm 的奇异区域
    r = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    valid = r >= 0.01
    xy, angles, sincos = xy[valid], angles[valid], sincos[valid]
    print(f"  有效样本: {len(xy):,}（过滤 {n_samples - len(xy):,} 个奇异点）")

    os.makedirs(DATA_DIR, exist_ok=True)
    npz_path = os.path.join(DATA_DIR, "ik_2dof.npz")
    csv_path = os.path.join(DATA_DIR, "ik_2dof.csv")

    # ---- .npz ----
    np.savez(
        npz_path,
        inputs=xy.astype(np.float32),
        outputs=sincos.astype(np.float32),
        angles=angles.astype(np.float32),
    )
    print(f"  已保存 NPZ: {npz_path}")

    # ---- .csv ----
    header = "X_m,Y_m,theta1_deg,theta2_deg,sin_t1,cos_t1,sin_t2,cos_t2"
    csv_data = np.column_stack([
        xy,
        np.degrees(angles),  # 角度转换为度，方便阅读
        sincos,
    ])
    np.savetxt(csv_path, csv_data, delimiter=",", header=header, comments="", fmt="%.6f")
    print(f"  已保存 CSV: {csv_path}  ({len(xy):,} 行 × 8 列)")


# =====================================================================
# 3-DOF 空间臂数据生成
# =====================================================================


def gen_3dof(
    n_samples: int = 500_000,
    cfg: Arm3DOFConfig | None = None,
    seed: int = 42,
) -> None:
    """
    生成 3-DOF 空间臂 IK 数据集，保存为 .npz 和 .csv。

    CSV 列名：X_m, Y_m, Z_m, theta1_deg, theta2_deg, theta3_deg,
              sin_t1, cos_t1, sin_t2, cos_t2, sin_t3, cos_t3
    """
    if cfg is None:
        cfg = Arm3DOFConfig()

    print(f"[data_gen] 3-DOF 空间臂  n={n_samples:,}  L0={cfg.L0}m  L1={cfg.L1}m  L2={cfg.L2}m")
    rng = np.random.default_rng(seed)

    theta1 = rng.uniform(cfg.theta1_min, cfg.theta1_max, n_samples).astype(np.float32)
    theta2 = rng.uniform(cfg.theta2_min, cfg.theta2_max, n_samples).astype(np.float32)
    # elbow-up：θ3 ∈ [0, π]
    theta3 = rng.uniform(0.0, np.pi, n_samples).astype(np.float32)

    xyz = forward_kinematics_3dof_batch(theta1, theta2, theta3, cfg)  # (N, 3)

    # ---- 过滤奇异区域 ----
    # r = L1*cos(θ2) + L2*cos(θ2+θ3) 为有符号水平投影
    # r < 0 表示臂"反向"指，此时 r_xy = |r| 导致竖直平面映射多值→NN无法学习
    # r ≈ 0 为奇异点（Jacobian 退化）
    # 只保留 r > 0.01 的配置，确保 (r_xy, z') → (θ2, θ3) 单值可学习
    r_signed = (cfg.L1 * np.cos(theta2) + cfg.L2 * np.cos(theta2 + theta3)).astype(np.float32)
    valid = r_signed > 0.01
    theta1, theta2, theta3 = theta1[valid], theta2[valid], theta3[valid]
    xyz = xyz[valid]
    r_signed = r_signed[valid]
    n_valid = len(theta1)
    print(
        f"  有效样本: {n_valid:,}（过滤 {n_samples - n_valid:,} 个奇异/反向配置，"
        f"比例 {100 * (n_samples - n_valid) / n_samples:.1f}%）"
    )

    angles = np.stack([theta1, theta2, theta3], axis=-1)  # (N, 3)
    sincos = angles_to_sincos_3dof(angles)  # (N, 6)

    # ---- 竖直平面分解特征（几何解耦后的 MLP 真实训练输入）----
    # θ1 有精确解析解 atan2(Y, X)，只需学习竖直平面 2-link IK
    # 过滤后 r > 0，所以 r_xy = r_signed（无歧义，单值映射）
    r_xy = r_signed  # r > 0，已等于 sqrt(X²+Y²)
    z_prime = (xyz[:, 2] - cfg.L0).astype(np.float32)
    inputs_vert = np.stack([r_xy, z_prime], axis=-1)  # (N, 2) [r_xy, z']
    angles_vert = np.stack([theta2, theta3], axis=-1)  # (N, 2) [θ2, θ3]
    sincos_vert = angles_to_sincos_planar(angles_vert)  # (N, 4) [sinθ2,cosθ2,sinθ3,cosθ3]
    print(
        f"  末端范围  X:[{xyz[:, 0].min():.3f},{xyz[:, 0].max():.3f}]  "
        f"Y:[{xyz[:, 1].min():.3f},{xyz[:, 1].max():.3f}]  "
        f"Z:[{xyz[:, 2].min():.3f},{xyz[:, 2].max():.3f}]"
    )
    print(
        f"  竖直平面  r_xy:[{r_xy.min():.3f},{r_xy.max():.3f}]  "
        f"z':[{z_prime.min():.3f},{z_prime.max():.3f}]"
    )
    os.makedirs(DATA_DIR, exist_ok=True)
    npz_path = os.path.join(DATA_DIR, "ik_3dof.npz")
    csv_path = os.path.join(DATA_DIR, "ik_3dof.csv")

    # ---- .npz ----
    np.savez(
        npz_path,
        inputs=xyz.astype(np.float32),  # (N, 3) 原始 XYZ
        inputs_vert=inputs_vert.astype(np.float32),  # (N, 2) [r_xy, z'] 竖直平面
        outputs=sincos.astype(np.float32),  # (N, 6) 全关节 sin/cos
        outputs_vert=sincos_vert.astype(np.float32),  # (N, 4) [sinθ2,cosθ2,sinθ3,cosθ3]
        angles=angles.astype(np.float32),  # (N, 3) [θ1, θ2, θ3]
        angles_vert=angles_vert.astype(np.float32),  # (N, 2) [θ2, θ3]
    )
    print(f"  已保存 NPZ: {npz_path}")

    # ---- .csv ----
    header = (
        "X_m,Y_m,Z_m,r_xy_m,z_prime_m,"
        "theta1_deg,theta2_deg,theta3_deg,"
        "sin_t1,cos_t1,sin_t2,cos_t2,sin_t3,cos_t3"
    )
    csv_data = np.column_stack([
        xyz,
        r_xy[:, np.newaxis],
        z_prime[:, np.newaxis],
        np.degrees(angles),
        sincos,
    ])
    np.savetxt(csv_path, csv_data, delimiter=",", header=header, comments="", fmt="%.6f")
    print(f"  已保存 CSV: {csv_path}  ({n_valid:,} 行 × 14 列)")


# =====================================================================
# 主入口
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 IK 训练数据（含 CSV 导出）")
    parser.add_argument(
        "--arm",
        type=str,
        default="all",
        choices=["2dof", "3dof", "all"],
        help="生成哪种臂的数据（默认 all）",
    )
    parser.add_argument("--n_samples", type=int, default=500_000, help="样本数量")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.arm in ("2dof", "all"):
        gen_2dof(n_samples=args.n_samples, seed=args.seed)
        print()
    if args.arm in ("3dof", "all"):
        gen_3dof(n_samples=args.n_samples, seed=args.seed)
