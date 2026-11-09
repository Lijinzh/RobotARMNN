"""逆动力学标签的物理一致性与数据保存回归测试。"""

import mujoco
import numpy as np
import pytest

from src import data_gen_idyn as idyn


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_path(idyn.XML_PATH)


def test_full_labels_respond_to_desired_acceleration(model):
    """相同位置、速度下，远离摩擦换向区的力矩差应等于惯性力矩差。"""
    q = np.tile([0.1, 0.2, -0.3], (2, 1))
    dq = np.tile([0.4, -0.5, 0.6], (2, 1))
    ddq = np.array([[0.0, 0.0, 0.0], [2.0, -3.0, 4.0]])
    full, bg = idyn.compute_inverse_dynamics_labels(q, dq, ddq)

    data = mujoco.MjData(model)
    data.qpos[:] = q[0]
    data.qvel[:] = dq[0]
    mujoco.mj_forward(model, data)
    expected_delta = np.empty(model.nv)
    mujoco.mj_mulM(model, data, expected_delta, ddq[1] - ddq[0])
    np.testing.assert_allclose(full[1] - full[0], expected_delta, atol=1e-7)
    np.testing.assert_array_equal(bg[0], bg[1])


def test_background_compensates_damping_and_coulomb_friction(model):
    """滑动区的补偿力矩应抵消阻尼和 Coulomb 摩擦，而不是叠加阻力。"""
    q = np.array([[0.1, 0.2, -0.3]])
    dq = np.array([[0.4, -0.5, 0.6]])
    _, bg = idyn.compute_inverse_dynamics_labels(q, dq, np.zeros_like(q))

    data = mujoco.MjData(model)
    data.qpos[:] = q[0]
    data.qvel[:] = dq[0]
    mujoco.mj_forward(model, data)
    assert data.ncon == 0
    expected = (
        data.qfrc_bias
        + model.dof_damping * dq[0]
        + model.dof_frictionloss * np.sign(dq[0])
    )
    np.testing.assert_allclose(bg[0], expected, atol=1e-7)


@pytest.mark.parametrize("background", [False, True])
def test_labels_reproduce_requested_acceleration(model, background):
    """把标签作为电机力矩施加后，应恢复指定加速度，包含低速换向区。"""
    rng = np.random.default_rng(7)
    q = rng.uniform(-0.3, 0.3, (30, 3))
    dq = rng.uniform(-0.8, 0.8, (30, 3))
    dq[:10] *= 0.01
    dq[0] = 0.0
    ddq = rng.uniform(-5.0, 5.0, (30, 3))
    full, bg = idyn.compute_inverse_dynamics_labels(q, dq, ddq)
    labels = bg if background else full
    accelerations = np.zeros_like(ddq) if background else ddq

    data = mujoco.MjData(model)
    for i in range(len(q)):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = q[i]
        data.qvel[:] = dq[i]
        data.ctrl[:] = labels[i]
        mujoco.mj_forward(model, data)
        assert data.ncon == 0
        np.testing.assert_allclose(data.qacc, accelerations[i], atol=1e-5)


def test_excitation_scaling_preserves_derivatives():
    """大幅值轨迹限幅时，位置、速度和加速度必须同步缩放。"""
    t, q, dq, ddq = idyn.gen_excitation_traj(
        2.0, 0.0002, idyn.JOINT_RANGES, rng=np.random.default_rng(42), amp_scale=100.0
    )
    for j, (lo, hi) in enumerate(idyn.JOINT_RANGES):
        assert np.all(q[:, j] >= lo)
        assert np.all(q[:, j] <= hi)
    np.testing.assert_allclose(np.gradient(q, t, axis=0)[1:-1], dq[1:-1], atol=1e-4)
    np.testing.assert_allclose(np.gradient(dq, t, axis=0)[1:-1], ddq[1:-1], atol=1e-3)


def test_dataset_saves_acceleration_and_label_version(tmp_path, monkeypatch):
    """九维逆动力学输入应包含对应解析加速度，并标明标签版本。"""
    output_path = tmp_path / "idyn.npz"
    monkeypatch.setattr(idyn, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(idyn, "NPZ_PATH", str(output_path))
    idyn.gen_idyn_dataset(n_traj=2, duration=0.02, seed=42)

    with np.load(output_path) as saved:
        assert saved["label_version"].item() == 2
        assert saved["seed"].item() == 42
        assert saved["inputs"].shape == (20, 6)
        assert saved["inputs_full"].shape == (20, 9)
        np.testing.assert_array_equal(saved["inputs_full"][:, :6], saved["inputs"])
        np.testing.assert_array_equal(saved["inputs_full"][:, 6:], saved["ddq_des"])
        full, bg = idyn.compute_inverse_dynamics_labels(
            saved["q_des"], saved["dq_des"], saved["ddq_des"]
        )
        np.testing.assert_allclose(saved["outputs_inv"], full, atol=2e-7)
        np.testing.assert_allclose(saved["outputs_bg"], bg, atol=2e-7)
