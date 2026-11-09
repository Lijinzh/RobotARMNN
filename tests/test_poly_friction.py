"""验证摩擦回归的物理单位、评估划分和误差定义。"""

import numpy as np
import pytest

from src.validate_poly_friction import (
    fit_model,
    load_data,
    metrics,
    predict_model,
    select_candidates,
)


def test_saturation_recovers_known_torque_and_model_roundtrip(tmp_path):
    """在独立留出样本上恢复已知力矩，防止把标准化速度误当 rad/s。"""
    rng = np.random.default_rng(9)
    x = rng.uniform(-1.0, 1.0, (180, 6))
    x[:, 3:] *= [0.07, 0.12, 0.2]
    width = 0.023
    y = 0.4 + x[:, :3] * [0.1, 0.2, -0.3] + x[:, 3:] * [0.2, 0.15, 0.08]
    y += np.clip(x[:, 3:] / width, -1, 1) * [0.5, 0.4, 0.25]
    model = fit_model(x[:120], y[:120], degree=1, friction="saturation", width=width)
    np.testing.assert_allclose(predict_model(model, x[120:]), y[120:], atol=1e-12)
    np.testing.assert_allclose(model["mean"], x[:120].mean(0))
    path = tmp_path / "model.npz"
    np.savez(path, **model)
    with np.load(path) as restored:
        np.testing.assert_allclose(predict_model(restored, x[120:]), y[120:], atol=1e-12)


def test_metrics_use_variance_weighted_normalization():
    y = np.array([[-1.0, -2.0, -4.0], [1.0, 2.0, 4.0]])
    prediction = y + [0.1, 0.2, 0.4]
    result = metrics(y, prediction)
    assert result["nrmse"] == pytest.approx(0.1)
    assert result["variance_weighted_r2"] == pytest.approx(0.99)
    np.testing.assert_allclose(result["rmse_Nm_joint"], [0.1, 0.2, 0.4])


def test_sampling_is_per_trajectory_and_excludes_other_trajectories(tmp_path):
    """交错保存轨迹也必须分别抽样，不能全局每隔 25 点抽样或纳入其余轨迹。"""
    traj_id = np.tile(np.arange(12), 60)
    row = np.arange(len(traj_id))
    x = np.repeat(row[:, None], 6, axis=1)
    y = x[:, :3]
    path = tmp_path / "data.npz"
    np.savez(path, inputs=x, outputs_bg=y, traj_id=traj_id, dt=0.002, label_version=2)
    sampled_x, sampled_y, ids, indices, dt = load_data(path)
    expected = np.concatenate([np.flatnonzero(traj_id == tid)[::25] for tid in range(10)])
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(sampled_x, x[expected])
    np.testing.assert_array_equal(sampled_y, y[expected])
    assert set(ids) == set(range(10))
    assert (ids < 8).sum() == 24
    assert (ids >= 8).sum() == 6
    assert dt == 0.002


def test_legacy_labels_and_test_trajectories_are_rejected(tmp_path):
    path = tmp_path / "old.npz"
    np.savez(path, label_version=1)
    with pytest.raises(ValueError, match="label_version=2"):
        load_data(path)
    with pytest.raises(ValueError, match="0–7"):
        select_candidates(np.zeros((10, 6)), np.zeros((10, 3)), np.arange(10))


def test_saturation_width_must_be_positive():
    with pytest.raises(ValueError, match="正数"):
        fit_model(np.zeros((10, 6)), np.zeros((10, 3)), 1, "saturation", width=0)
