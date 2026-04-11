"""
机械臂神经网络控制 — 快速入口

支持：
  平面 2-DOF 臂（水平 XY 平面，L1=0.15m，L2=0.12m）
  空间 3-DOF 臂（底座Z旋转 + 肩部Y俯仰 + 肘部Y俯仰）

用法：
    uv run main.py                               # 运行运动学演示

    # 数据生成
    uv run src/data_gen.py --arm 2dof            # 生成 2-DOF 数据（50万样本 + CSV）
    uv run src/data_gen.py --arm 3dof            # 生成 3-DOF 数据（50万样本 + CSV）
    uv run src/data_gen.py --arm all             # 同时生成两者

    # 模型训练
    uv run src/train.py --arm 2dof --model mlp   # 平面臂 MLP
    uv run src/train.py --arm 2dof --model esn   # 平面臂 ESN（储备池 + MLP 读出）
    uv run src/train.py --arm 3dof --model mlp   # 空间臂 MLP
    uv run src/train.py --arm all --model all    # 全部训练

    # 可视化（需要先运行 data_gen + train）
    uv run src/visualize_mujoco.py --arm 2dof
    uv run src/visualize_mujoco.py --arm 3dof

    # 单元测试
    uv run pytest tests/ -v
"""

import numpy as np

from src.kinematics_2dof import (
    PlanarArmConfig,
    forward_kinematics_planar,
    inverse_kinematics_planar,
)
from src.kinematics_3dof import (
    Arm3DOFConfig,
    forward_kinematics_3dof_batch,
    inverse_kinematics_3dof,
)


def demo_2dof():
    """平面 2-DOF 臂正运动学 → 逆运动学往返演示"""
    cfg = PlanarArmConfig()
    print("=" * 60)
    print("平面 2-DOF 机械臂（水平 XY 平面）— 运动学演示")
    print("=" * 60)
    print(f"  大臂 L1={cfg.L1}m  小臂 L2={cfg.L2}m")
    print(f"  可达半径: r_min={cfg.r_min:.3f}m  r_max={cfg.r_max:.3f}m")
    print()

    test_cases = [
        (0.0, 0.0, "手臂完全伸直 → X 轴"),
        (np.pi / 2, 0.0, "手臂完全伸直 → Y 轴"),
        (0.0, np.pi / 2, "肘部弯曲 90°"),
        (np.pi / 4, np.pi / 3, "任意角度"),
        (np.pi, np.pi / 6, "手臂朝 -X 方向"),
    ]

    print(f"{'θ1':>8}  {'θ2':>8}  {'说明':<18}  {'X(m)':>8}  {'Y(m)':>8}  {'IK往返误差':>12}")
    print("-" * 72)
    for t1, t2, desc in test_cases:
        xy = forward_kinematics_planar(t1, t2, cfg)
        ik = inverse_kinematics_planar(xy, cfg, elbow_up=(t2 >= 0))
        if ik is not None:
            t1_r, t2_r = ik
            err = max(
                abs(np.arctan2(np.sin(t1_r - t1), np.cos(t1_r - t1))),
                abs(np.arctan2(np.sin(t2_r - t2), np.cos(t2_r - t2))),
            )
            err_str = f"{err:.2e} rad"
        else:
            err_str = "不可达"
        print(
            f"  {np.degrees(t1):+6.1f}°  {np.degrees(t2):+6.1f}°  {desc:<18}"
            f"  {xy[0]:+8.4f}  {xy[1]:+8.4f}  {err_str:>12}"
        )


def demo_3dof():
    """空间 3-DOF 臂 FK → IK 往返演示"""
    cfg = Arm3DOFConfig()
    print()
    print("=" * 60)
    print("空间 3-DOF 机械臂 — 运动学演示")
    print("=" * 60)
    print(f"  L0(底座高)={cfg.L0}m  L1={cfg.L1}m  L2={cfg.L2}m")
    print()

    test_cases = [
        ([0.0, 0.0, 0.0], "各关节归零"),
        ([0.0, np.pi / 4, np.pi / 3], "正面仰角"),
        ([np.pi / 2, np.pi / 6, np.pi / 4], "转向左侧"),
    ]

    angles_arr = np.array([[t[0], t[1], t[2]] for t, _ in test_cases], dtype=np.float32)
    xyzs = forward_kinematics_3dof_batch(angles_arr[:, 0], angles_arr[:, 1], angles_arr[:, 2], cfg)

    print(
        f"{'θ1':>8}  {'θ2':>8}  {'θ3':>8}  {'说明':<16}  {'X':>8}  {'Y':>8}  {'Z':>8}  {'IK误差':>10}"
    )
    print("-" * 85)
    for (angles, desc), xyz in zip(test_cases, xyzs):
        t1, t2, t3 = angles
        ik = inverse_kinematics_3dof(xyz, cfg)
        if ik is not None:
            r1, r2, r3 = ik
            errs = [
                abs(np.arctan2(np.sin(r - a), np.cos(r - a)))
                for r, a in zip([r1, r2, r3], [t1, t2, t3])
            ]
            err_str = f"{max(errs):.2e}"
        else:
            err_str = "不可达"
        print(
            f"  {np.degrees(t1):+6.1f}°  {np.degrees(t2):+6.1f}°  {np.degrees(t3):+6.1f}°"
            f"  {desc:<16}  {xyz[0]:+8.4f}  {xyz[1]:+8.4f}  {xyz[2]:+8.4f}  {err_str:>10}"
        )


def print_workflow():
    print()
    print("=" * 60)
    print("完整训练流程")
    print("=" * 60)
    print("  1. uv run src/data_gen.py --arm all       # 生成数据 (~10s)")
    print("  2. uv run src/train.py --arm 2dof --model all  # 训练 2-DOF")
    print("  3. uv run src/train.py --arm 3dof --model mlp  # 训练 3-DOF")
    print("  4. uv run src/visualize_mujoco.py --arm 2dof   # 交互可视化")
    print("  5. uv run pytest tests/ -v                # 单元测试")


if __name__ == "__main__":
    demo_2dof()
    demo_3dof()
    print_workflow()
