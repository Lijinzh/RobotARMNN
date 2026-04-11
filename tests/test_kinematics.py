"""
平面双自由度机械臂运动学单元测试

测试覆盖：
  1. 正运动学基准值验证（已知输入输出）
  2. FK → IK 往返一致性（误差 < 1e-5 rad）
  3. IK 超出可达范围返回 None
  4. 批量 FK 与单次 FK 结果一致性
  5. sin/cos 表示的可逆性
  6. 可达范围边界测试
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.kinematics_2dof import (
    PlanarArmConfig,
    angles_to_sincos_planar,
    forward_kinematics_planar,
    forward_kinematics_planar_batch,
    inverse_kinematics_planar,
    inverse_kinematics_planar_batch,
    sincos_to_angles_planar,
)

# =====================================================================
# Fixture 配置
# =====================================================================


@pytest.fixture
def cfg():
    """默认平面 2-DOF 机械臂配置"""
    return PlanarArmConfig()


# =====================================================================
# 测试 1：正运动学基准值
# =====================================================================


class TestForwardKinematics:
    def test_zero_angles(self, cfg):
        """θ1=0, θ2=0 时，末端应在 X 轴正方向 (L1+L2, 0)"""
        xy = forward_kinematics_planar(0.0, 0.0, cfg)
        np.testing.assert_allclose(xy, [cfg.L1 + cfg.L2, 0.0], atol=1e-10)

    def test_theta1_90_deg(self, cfg):
        """θ1=π/2, θ2=0 时，末端应在 Y 轴正方向 (0, L1+L2)"""
        xy = forward_kinematics_planar(np.pi / 2, 0.0, cfg)
        np.testing.assert_allclose(xy, [0.0, cfg.L1 + cfg.L2], atol=1e-10)

    def test_theta2_90_deg(self, cfg):
        """θ1=0, θ2=π/2 时，末端在 (L1, L2)"""
        xy = forward_kinematics_planar(0.0, np.pi / 2, cfg)
        np.testing.assert_allclose(xy, [cfg.L1, cfg.L2], atol=1e-10)

    def test_theta1_pi(self, cfg):
        """θ1=π, θ2=0 时，末端应在 X 轴负方向 (-(L1+L2), 0)"""
        xy = forward_kinematics_planar(np.pi, 0.0, cfg)
        np.testing.assert_allclose(xy, [-(cfg.L1 + cfg.L2), 0.0], atol=1e-10)

    def test_output_shape(self, cfg):
        """FK 输出应为长度为 2 的数组"""
        xy = forward_kinematics_planar(0.5, 0.3, cfg)
        assert xy.shape == (2,)

    def test_ee_reachable_radius(self, cfg):
        """任意关节角度，末端到原点的距离应在 [r_min, r_max] 范围内"""
        rng = np.random.default_rng(0)
        for _ in range(200):
            t1 = rng.uniform(-np.pi, np.pi)
            t2 = rng.uniform(-np.pi, np.pi)
            xy = forward_kinematics_planar(t1, t2, cfg)
            r = float(np.linalg.norm(xy))
            assert r <= cfg.r_max + 1e-9, f"超过最大可达半径: r={r:.5f} > r_max={cfg.r_max}"
            assert r >= cfg.r_min - 1e-9, f"低于最小可达半径: r={r:.5f} < r_min={cfg.r_min}"


# =====================================================================
# 测试 2：逆运动学
# =====================================================================


class TestInverseKinematics:
    def test_basic_ik_straight(self, cfg):
        """手臂伸直 (L1+L2, 0) → θ1=0, θ2=0"""
        result = inverse_kinematics_planar(np.array([cfg.L1 + cfg.L2, 0.0]), cfg)
        assert result is not None
        theta1, theta2 = result
        np.testing.assert_allclose([theta1, theta2], [0.0, 0.0], atol=1e-9)

    def test_unreachable_far(self, cfg):
        """距离超过 r_max 的目标应返回 None"""
        result = inverse_kinematics_planar(np.array([10.0, 0.0]), cfg)
        assert result is None

    def test_unreachable_too_close(self, cfg):
        """距离小于 r_min 的目标应返回 None（原点）"""
        result = inverse_kinematics_planar(np.array([0.001, 0.0]), cfg)
        assert result is None

    def test_fk_ik_roundtrip_elbow_up(self, cfg):
        """FK → IK(elbow_up=True) 往返：关节角误差 < 1e-5 rad"""
        rng = np.random.default_rng(42)
        errors = []
        for _ in range(300):
            # 仅测试 elbow_up 区域（θ2 ∈ [0, π]）
            t1 = rng.uniform(-np.pi, np.pi)
            t2 = rng.uniform(0.0, np.pi)
            xy = forward_kinematics_planar(t1, t2, cfg)
            result = inverse_kinematics_planar(xy, cfg, elbow_up=True)
            assert result is not None, f"FK 点 {xy} 应可达"
            t1_r, t2_r = result
            e1 = abs(np.arctan2(np.sin(t1_r - t1), np.cos(t1_r - t1)))
            e2 = abs(np.arctan2(np.sin(t2_r - t2), np.cos(t2_r - t2)))
            errors.append(max(e1, e2))
        max_err = max(errors)
        assert max_err < 1e-5, f"FK→IK 最大误差 {max_err:.2e} rad，超过 1e-5"

    def test_fk_ik_roundtrip_elbow_down(self, cfg):
        """FK → IK(elbow_up=False) 往返：关节角误差 < 1e-5 rad"""
        rng = np.random.default_rng(7)
        errors = []
        for _ in range(200):
            t1 = rng.uniform(-np.pi, np.pi)
            t2 = rng.uniform(-np.pi, 0.0)  # elbow_down 区域
            xy = forward_kinematics_planar(t1, t2, cfg)
            result = inverse_kinematics_planar(xy, cfg, elbow_up=False)
            assert result is not None
            t1_r, t2_r = result
            e1 = abs(np.arctan2(np.sin(t1_r - t1), np.cos(t1_r - t1)))
            e2 = abs(np.arctan2(np.sin(t2_r - t2), np.cos(t2_r - t2)))
            errors.append(max(e1, e2))
        max_err = max(errors)
        assert max_err < 1e-5, f"FK→IK(elbow_down) 最大误差 {max_err:.2e} rad"


# =====================================================================
# 测试 3：批量操作
# =====================================================================


class TestBatchOperations:
    def test_batch_fk_matches_single(self, cfg):
        """批量 FK 结果应与逐个调用完全一致"""
        rng = np.random.default_rng(7)
        N = 500
        t1 = rng.uniform(-np.pi, np.pi, N).astype(np.float32)
        t2 = rng.uniform(-np.pi, np.pi, N).astype(np.float32)

        batch_result = forward_kinematics_planar_batch(t1, t2, cfg)
        single_results = np.array([forward_kinematics_planar(t1[i], t2[i], cfg) for i in range(N)])
        np.testing.assert_allclose(batch_result, single_results, atol=1e-6)

    def test_batch_fk_output_shape(self, cfg):
        """批量 FK 输出形状应为 (N, 2)"""
        N = 100
        t1 = np.zeros(N)
        t2 = np.zeros(N)
        result = forward_kinematics_planar_batch(t1, t2, cfg)
        assert result.shape == (N, 2)

    def test_batch_ik_valid_mask(self, cfg):
        """批量 IK 中超出范围的点应填 NaN"""
        targets = np.array(
            [
                [cfg.L1 + cfg.L2, 0.0],  # 可达（伸直）
                [100.0, 0.0],  # 不可达（太远）
                [0.0, cfg.L1 + cfg.L2],  # 可达
            ],
            dtype=np.float32,
        )
        result = inverse_kinematics_planar_batch(targets, cfg)
        assert not np.any(np.isnan(result[0])), "第 0 行可达，不应有 NaN"
        assert np.all(np.isnan(result[1])), "第 1 行不可达，应全 NaN"
        assert not np.any(np.isnan(result[2])), "第 2 行可达，不应有 NaN"


# =====================================================================
# 测试 4：sin/cos 表示
# =====================================================================


class TestSinCosRepresentation:
    def test_roundtrip(self):
        """任意角度经 sin/cos 表示后还原误差应 < 1e-6"""
        rng = np.random.default_rng(0)
        N = 500
        angles = np.stack(
            [rng.uniform(-np.pi, np.pi, N), rng.uniform(-np.pi, np.pi, N)],
            axis=-1,
        ).astype(np.float32)
        sincos = angles_to_sincos_planar(angles)
        recovered = sincos_to_angles_planar(sincos)
        # 使用角度差绕 [-π, π] 比较
        diff = np.abs(np.arctan2(np.sin(recovered - angles), np.cos(recovered - angles)))
        assert diff.max() < 1e-6, f"sin/cos 往返最大误差: {diff.max():.2e}"

    def test_output_shape(self):
        """angles_to_sincos_planar 输出应为 (N, 4)"""
        N = 50
        angles = np.zeros((N, 2), dtype=np.float32)
        sincos = angles_to_sincos_planar(angles)
        assert sincos.shape == (N, 4)

    def test_sincos_to_angles_shape(self):
        """sincos_to_angles_planar 输出应为 (N, 2)"""
        N = 50
        sincos = np.zeros((N, 4), dtype=np.float32)
        angles = sincos_to_angles_planar(sincos)
        assert angles.shape == (N, 2)

    def test_pi_boundary_continuity(self):
        """θ=π 和 θ=-π 对应相同的 sin/cos，还原后结果相同"""
        angles_pos = np.array([[np.pi, 0.0]], dtype=np.float32)
        angles_neg = np.array([[-np.pi, 0.0]], dtype=np.float32)
        sc_pos = angles_to_sincos_planar(angles_pos)
        sc_neg = angles_to_sincos_planar(angles_neg)
        np.testing.assert_allclose(sc_pos, sc_neg, atol=1e-6)
