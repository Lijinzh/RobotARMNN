"""
三自由度空间机械臂运动学

构型：
  关节1 (θ1)：底座绕 Z 轴旋转（方位角），范围 [-π, π]
  关节2 (θ2)：肩部绕 Y 轴俯仰（仰角），范围 [-π/2, π/2]
  关节3 (θ3)：肘部绕 Y 轴俯仰（相对角），范围 [-π, π]

连杆参数：
  L0 = 0.10 m（底柱高度，固定偏置）
  L1 = 0.15 m（大臂）
  L2 = 0.12 m（小臂）

正运动学 (FK)：
  r  = L1*cos(θ2) + L2*cos(θ2 + θ3)   -- 水平投影半径
  X  = r * cos(θ1)
  Y  = r * sin(θ1)
  Z  = L0 + L1*sin(θ2) + L2*sin(θ2 + θ3)

逆运动学 (IK)——几何解耦法：
  1. θ1 = atan2(Y, X)
  2. 在竖直平面 (r, Z-L0) 内用两连杆平面 IK（余弦定理）求 θ2, θ3
     r    = sqrt(X² + Y²)
     z'   = Z - L0
     D    = (r² + z'² - L1² - L2²) / (2*L1*L2)  ∈ [-1, 1]
     θ3   = ±acos(D)
     θ2   = atan2(z', r) - atan2(L2*sin(θ3), L1 + L2*cos(θ3))

注意：θ3 有两解（elbow-up / elbow-down）。常规分支超限时尝试负水平半径；
θ1 在 r→0 时退化，取限位内最接近 0 的角度。
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Arm3DOFConfig:
    """三自由度空间机械臂参数"""

    L0: float = 0.20  # 底柱高度（固定偏置 Z）= turret(0.10) + link0(0.10) = 0.20m
    L1: float = 0.15  # 大臂长度
    L2: float = 0.12  # 小臂长度

    # 关节限制（弧度）
    theta1_min: float = -np.pi
    theta1_max: float = np.pi
    theta2_min: float = -np.pi / 2
    theta2_max: float = np.pi / 2
    theta3_min: float = -np.pi
    theta3_max: float = np.pi

    @property
    def reach_max(self) -> float:
        """最大可达半径（斜面距离）"""
        return self.L1 + self.L2

    @property
    def z_min(self) -> float:
        return self.L0 - self.L1 - self.L2

    @property
    def z_max(self) -> float:
        return self.L0 + self.L1 + self.L2


def forward_kinematics_3dof(
    theta1: float, theta2: float, theta3: float, cfg: Arm3DOFConfig
) -> np.ndarray:
    """
    正运动学：关节角 → 末端 XYZ 坐标。

    返回:
        np.ndarray [X, Y, Z]（米）
    """
    # 竖直平面内的投影分量
    r = cfg.L1 * np.cos(theta2) + cfg.L2 * np.cos(theta2 + theta3)
    z = cfg.L0 + cfg.L1 * np.sin(theta2) + cfg.L2 * np.sin(theta2 + theta3)
    # 底座旋转展开到 XY
    x = r * np.cos(theta1)
    y = r * np.sin(theta1)
    return np.array([x, y, z], dtype=np.float64)


def forward_kinematics_3dof_batch(
    theta1: np.ndarray,
    theta2: np.ndarray,
    theta3: np.ndarray,
    cfg: Arm3DOFConfig,
) -> np.ndarray:
    """
    批量正运动学，向量化。

    返回:
        (N, 3) [X, Y, Z]
    """
    r = cfg.L1 * np.cos(theta2) + cfg.L2 * np.cos(theta2 + theta3)
    z = cfg.L0 + cfg.L1 * np.sin(theta2) + cfg.L2 * np.sin(theta2 + theta3)
    x = r * np.cos(theta1)
    y = r * np.sin(theta1)
    return np.stack([x, y, z], axis=-1)


def inverse_kinematics_3dof(
    target: np.ndarray,
    cfg: Arm3DOFConfig,
    elbow_up: bool = True,
) -> tuple[float, float, float] | None:
    """
    逆运动学解析解（几何解耦）。

    参数:
        target:   [X, Y, Z] 目标末端坐标
        cfg:      机械臂配置
        elbow_up: True 优先 θ3 >= 0，False 优先 θ3 <= 0；超限时尝试另一分支。

    返回:
        (theta1, theta2, theta3) 弧度；不可达返回 None。
    """
    result = _inverse_kinematics_3dof_array(np.asarray(target)[None, :], cfg, elbow_up)[0]
    return tuple(float(angle) for angle in result) if np.isfinite(result).all() else None


def inverse_kinematics_3dof_batch(
    targets: np.ndarray,
    cfg: Arm3DOFConfig,
    elbow_up: bool = True,
) -> np.ndarray:
    """
    批量 IK，向量化（不可达点填 NaN）。

    参数:
        targets: (N, 3) [X, Y, Z]

    返回:
        (N, 3) [θ1, θ2, θ3]，不可达点为 NaN
    """
    return _inverse_kinematics_3dof_array(np.asarray(targets), cfg, elbow_up).astype(np.float32)


def valid_joint_angles_3dof(angles: np.ndarray, cfg: Arm3DOFConfig) -> np.ndarray:
    """检查输出是否有限且满足全部关节限位；容忍 float32 边界舍入。"""
    angles = np.asarray(angles)
    lower = np.array([cfg.theta1_min, cfg.theta2_min, cfg.theta3_min])
    upper = np.array([cfg.theta1_max, cfg.theta2_max, cfg.theta3_max])
    return np.all(np.isfinite(angles) & (angles >= lower - 1e-6) & (angles <= upper + 1e-6), axis=-1)


def _inverse_kinematics_3dof_array(targets, cfg, elbow_up):
    """单点和批量共用双精度求解，统一距离、限位与分支回退。"""
    x, y, z = (targets[:, index].astype(np.float64) for index in range(3))
    r_xy = np.sqrt(x**2 + y**2)
    singular_yaw = np.clip(0.0, cfg.theta1_min, cfg.theta1_max)
    theta1 = np.where(r_xy < 1e-9, singular_yaw, np.arctan2(y, x))
    z_prime = z - cfg.L0
    D = (r_xy**2 + z_prime**2 - cfg.L1**2 - cfg.L2**2) / (2.0 * cfg.L1 * cfg.L2)
    reachable = np.isfinite(targets).all(axis=1) & (D >= -1.0 - 1e-6) & (D <= 1.0 + 1e-6)
    D = np.clip(D, -1.0, 1.0)
    sine = np.sqrt(np.maximum(0.0, 1.0 - D**2))
    result = np.full((len(targets), 3), np.nan)
    preferred = 1.0 if elbow_up else -1.0
    # 保留正半径的两肘分支优先级；仅无合法解时尝试手臂向后折的等价构型。
    for radius, yaw in ((r_xy, theta1), (-r_xy, theta1 + np.pi), (-r_xy, theta1 - np.pi)):
        if np.all(np.isfinite(result[:, 0]) | ~reachable):
            break
        for direction in (preferred, -preferred):
            sin_t3 = direction * sine
            theta3 = np.arctan2(sin_t3, D)
            theta2 = np.arctan2(z_prime, radius) - np.arctan2(cfg.L2 * sin_t3, cfg.L1 + cfg.L2 * D)
            theta2 = np.arctan2(np.sin(theta2), np.cos(theta2))
            candidate = np.column_stack([yaw, theta2, theta3])
            accept = reachable & valid_joint_angles_3dof(candidate, cfg) & np.isnan(result[:, 0])
            result[accept] = candidate[accept]
    return result


def angles_to_sincos_3dof(angles: np.ndarray) -> np.ndarray:
    """
    (N, 3) 角度 → (N, 6) sin/cos 表示
    输出列：[sin θ1, cos θ1, sin θ2, cos θ2, sin θ3, cos θ3]
    """
    return np.stack(
        [
            np.sin(angles[:, 0]),
            np.cos(angles[:, 0]),
            np.sin(angles[:, 1]),
            np.cos(angles[:, 1]),
            np.sin(angles[:, 2]),
            np.cos(angles[:, 2]),
        ],
        axis=-1,
    ).astype(np.float32)


def sincos_to_angles_3dof(sincos: np.ndarray) -> np.ndarray:
    """
    (N, 6) sin/cos → (N, 3) 角度
    """
    theta1 = np.arctan2(sincos[:, 0], sincos[:, 1])
    theta2 = np.arctan2(sincos[:, 2], sincos[:, 3])
    theta3 = np.arctan2(sincos[:, 4], sincos[:, 5])
    return np.stack([theta1, theta2, theta3], axis=-1).astype(np.float32)
