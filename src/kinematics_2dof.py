"""
平面双自由度机械臂运动学（水平 XY 平面）

构型：
  关节1 (θ1)：底座绕 Z 轴旋转（肩部），范围 [-π, π]
  关节2 (θ2)：肘部绕 Z 轴旋转（相对于大臂），范围 [-π, π]
  L1 = 0.15 m（大臂长度）
  L2 = 0.12 m（小臂长度）

运动在 Z=0.05m 水平面内，Z 坐标固定不参与 IK。

正运动学 (FK)：
  X = L1*cos(θ1) + L2*cos(θ1 + θ2)
  Y = L1*sin(θ1) + L2*sin(θ1 + θ2)

逆运动学 (IK)：
  D = (X² + Y² - L1² - L2²) / (2*L1*L2)  ∈ [-1, 1]
  θ2 = ±acos(D)         （两解：elbow-up / elbow-down）
  θ1 = atan2(Y, X) - atan2(L2*sin(θ2), L1 + L2*cos(θ2))

可达范围：
  r_max = L1 + L2 = 0.27m（臂伸直）
  r_min = |L1 - L2| = 0.03m（臂折叠）
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PlanarArmConfig:
    """平面 2-DOF 机械臂参数"""

    L1: float = 0.15  # 大臂长度（m）
    L2: float = 0.12  # 小臂长度（m）

    # 关节限制（弧度）
    theta1_min: float = -np.pi
    theta1_max: float = np.pi
    theta2_min: float = -np.pi
    theta2_max: float = np.pi

    # 臂的固定高度（MuJoCo 中 Z 轴方向）
    arm_height: float = 0.05  # m

    @property
    def r_max(self) -> float:
        return self.L1 + self.L2

    @property
    def r_min(self) -> float:
        return abs(self.L1 - self.L2)


def forward_kinematics_planar(theta1: float, theta2: float, cfg: PlanarArmConfig) -> np.ndarray:
    """
    正运动学：关节角度 → 末端 XY 坐标。

    参数:
        theta1: 肩部旋转角（弧度）
        theta2: 肘部相对旋转角（弧度）
        cfg:    机械臂配置

    返回:
        np.ndarray [X, Y]（米）
    """
    x = cfg.L1 * np.cos(theta1) + cfg.L2 * np.cos(theta1 + theta2)
    y = cfg.L1 * np.sin(theta1) + cfg.L2 * np.sin(theta1 + theta2)
    return np.array([x, y], dtype=np.float64)


def forward_kinematics_planar_batch(
    theta1: np.ndarray, theta2: np.ndarray, cfg: PlanarArmConfig
) -> np.ndarray:
    """
    批量正运动学，向量化。

    参数:
        theta1: (N,)
        theta2: (N,)

    返回:
        (N, 2) [X, Y]
    """
    x = cfg.L1 * np.cos(theta1) + cfg.L2 * np.cos(theta1 + theta2)
    y = cfg.L1 * np.sin(theta1) + cfg.L2 * np.sin(theta1 + theta2)
    return np.stack([x, y], axis=-1)


def inverse_kinematics_planar(
    target_xy: np.ndarray,
    cfg: PlanarArmConfig,
    elbow_up: bool = True,
) -> tuple[float, float] | None:
    """
    逆运动学解析解（余弦定理方法）。

    参数:
        target_xy: [X, Y] 目标末端坐标（米）
        cfg:       机械臂配置
        elbow_up:  True → 取 θ2 > 0（肘部向上/左），False → θ2 < 0（肘部向下/右）

    返回:
        (theta1, theta2) 弧度；若目标不可达返回 None。
    """
    x, y = float(target_xy[0]), float(target_xy[1])
    r2 = x**2 + y**2  # 目标距底座距离的平方

    # 余弦定理：D = cos(θ2)
    D = (r2 - cfg.L1**2 - cfg.L2**2) / (2.0 * cfg.L1 * cfg.L2)

    # 检查可达性（|D| > 1 表示目标在奇异伸展半径之外）
    if D < -1.0 - 1e-6 or D > 1.0 + 1e-6:
        return None  # 目标超出工作空间

    # 钳制数值误差
    D = float(np.clip(D, -1.0, 1.0))

    # 肘关节角（两解：elbow_up → θ2 正值，elbow_down → θ2 负值）
    sin_theta2 = np.sqrt(max(0.0, 1.0 - D**2))
    if not elbow_up:
        sin_theta2 = -sin_theta2
    theta2 = float(np.arctan2(sin_theta2, D))

    # 肩关节角
    k1 = cfg.L1 + cfg.L2 * D
    k2 = cfg.L2 * sin_theta2
    theta1 = float(np.arctan2(y, x) - np.arctan2(k2, k1))
    # 归一化到 [-π, π]
    theta1 = float(np.arctan2(np.sin(theta1), np.cos(theta1)))

    return (theta1, theta2)


def inverse_kinematics_planar_batch(
    targets: np.ndarray,
    cfg: PlanarArmConfig,
    elbow_up: bool = True,
) -> np.ndarray:
    """
    批量 IK，向量化（不可达点填 NaN）。

    参数:
        targets: (N, 2) [X, Y]
        cfg:     机械臂配置

    返回:
        (N, 2) [θ1, θ2]，不可达点为 NaN
    """
    x = targets[:, 0].astype(np.float64)
    y = targets[:, 1].astype(np.float64)
    r2 = x**2 + y**2

    D = (r2 - cfg.L1**2 - cfg.L2**2) / (2.0 * cfg.L1 * cfg.L2)
    valid = (D >= -1.0 - 1e-6) & (D <= 1.0 + 1e-6)
    D = np.clip(D, -1.0, 1.0)

    sin_t2 = np.sqrt(np.maximum(0.0, 1.0 - D**2))
    if not elbow_up:
        sin_t2 = -sin_t2
    theta2 = np.arctan2(sin_t2, D)

    k1 = cfg.L1 + cfg.L2 * D
    k2 = cfg.L2 * sin_t2
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    theta1 = np.arctan2(np.sin(theta1), np.cos(theta1))  # 归一化

    result = np.stack([theta1, theta2], axis=-1).astype(np.float32)
    result[~valid] = np.nan
    return result


def angles_to_sincos_planar(angles: np.ndarray) -> np.ndarray:
    """
    (N, 2) 角度 → (N, 4) sin/cos 表示
    输出列顺序：[sin θ1, cos θ1, sin θ2, cos θ2]
    """
    return np.stack(
        [np.sin(angles[:, 0]), np.cos(angles[:, 0]), np.sin(angles[:, 1]), np.cos(angles[:, 1])],
        axis=-1,
    ).astype(np.float32)


def sincos_to_angles_planar(sincos: np.ndarray) -> np.ndarray:
    """
    (N, 4) sin/cos → (N, 2) 角度
    """
    theta1 = np.arctan2(sincos[:, 0], sincos[:, 1])
    theta2 = np.arctan2(sincos[:, 2], sincos[:, 3])
    return np.stack([theta1, theta2], axis=-1).astype(np.float32)
