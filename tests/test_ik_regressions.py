"""IK 采样资源、关节限位、评估口径及实时拒绝非法姿态的回归测试。"""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from src import visualize, visualize_ik, visualize_mujoco
from src.kinematics_2dof import (
    PlanarArmConfig,
    forward_kinematics_planar,
    inverse_kinematics_planar,
    inverse_kinematics_planar_batch,
)
from src.kinematics_3dof import (
    Arm3DOFConfig,
    forward_kinematics_3dof,
    inverse_kinematics_3dof,
    inverse_kinematics_3dof_batch,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLERS = [
    (module, dimension)
    for module in (visualize, visualize_ik)
    for dimension in (2, 3)
]


@pytest.mark.parametrize("module,dimension", SAMPLERS)
def test_sampling_counts_points_with_bounded_batches(module, dimension, monkeypatch):
    original_rng = np.random.default_rng
    counts = []

    class CountingRng:
        def __init__(self, seed):
            self.rng = original_rng(seed)

        def uniform(self, low, high, size):
            assert size[0] <= 8192, "单批不应按全部目标数大规模分配"
            counts.append(size[0])
            return self.rng.uniform(low, high, size)

    monkeypatch.setattr(np.random, "default_rng", CountingRng)
    cfg = PlanarArmConfig() if dimension == 2 else Arm3DOFConfig()
    sampler = getattr(module, f"sample_targets_{dimension}dof")
    samples = sampler(cfg, 32, seed=42)
    assert samples.shape == (32, dimension)
    assert sum(counts) <= 320
    np.testing.assert_array_equal(samples, sampler(cfg, 32, seed=42))
    counts.clear()
    assert sampler(cfg, 8000, seed=42).shape == (8000, dimension)
    assert sum(counts) <= 8000 * 8


@pytest.mark.parametrize("module,dimension", SAMPLERS)
def test_sampling_empty_and_negative_counts(module, dimension):
    cfg = PlanarArmConfig() if dimension == 2 else Arm3DOFConfig()
    sampler = getattr(module, f"sample_targets_{dimension}dof")
    assert sampler(cfg, 0).shape == (0, dimension)
    with pytest.raises(ValueError, match="非负"):
        sampler(cfg, -1)


@pytest.mark.parametrize("module,dimension", SAMPLERS)
def test_sampling_rejects_empty_workspace_before_entering_loop(module, dimension, monkeypatch):
    cfg = PlanarArmConfig(L1=0.01, L2=0.01) if dimension == 2 else Arm3DOFConfig(L1=0.02, L2=0.02)

    def unexpected_sampling(*args):
        pytest.fail("空采样域必须在生成候选点前拒绝，避免拒绝采样无限循环")

    monkeypatch.setattr(np.random, "default_rng", lambda _: SimpleNamespace(uniform=unexpected_sampling))
    sampler = getattr(module, f"sample_targets_{dimension}dof")
    assert sampler(cfg, 0).shape == (0, dimension)
    with pytest.raises(ValueError, match="采样域"):
        sampler(cfg, 1)


@pytest.mark.parametrize("module", [visualize, visualize_ik])
def test_sampling_rejects_empty_height_intersection(module, monkeypatch):
    def unexpected_sampling(*args):
        pytest.fail("高度限制与球壳没有交集时，不得进入拒绝采样循环")

    monkeypatch.setattr(np.random, "default_rng", lambda _: SimpleNamespace(uniform=unexpected_sampling))
    with pytest.raises(ValueError, match="采样域"):
        module.sample_targets_3dof(Arm3DOFConfig(L0=-1.0), 1)


@pytest.mark.parametrize("dimension", [2, 3])
def test_xml_joint_ranges_are_radians(dimension):
    model = mujoco.MjModel.from_xml_path(str(ROOT / "src" / f"arm_{dimension}dof.xml"))
    expected = [[-np.pi, np.pi]] * dimension
    if dimension == 3:
        expected[1] = [-np.pi / 2, np.pi / 2]
    np.testing.assert_allclose(model.jnt_range, expected, atol=1e-5)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("elbow_up", [True, False])
def test_single_and_batch_ik_agree_and_obey_limits(dimension, elbow_up):
    rng = np.random.default_rng(42)
    if dimension == 2:
        cfg = PlanarArmConfig()
        targets = rng.uniform(-0.25, 0.25, (100, 2))
        single, batch = inverse_kinematics_planar, inverse_kinematics_planar_batch
        fk = forward_kinematics_planar
    else:
        cfg = Arm3DOFConfig()
        targets = rng.uniform(-0.25, 0.25, (100, 3))
        targets[:, 2] += cfg.L0
        targets[0] = [0.1, 0.0, 0.05]
        single, batch = inverse_kinematics_3dof, inverse_kinematics_3dof_batch
        fk = forward_kinematics_3dof
    combined = batch(targets, cfg, elbow_up=elbow_up)
    for target, angles in zip(targets, combined):
        scalar = single(target, cfg, elbow_up=elbow_up)
        if scalar is None:
            assert np.isnan(angles).all()
        else:
            np.testing.assert_allclose(angles, scalar, atol=1e-6)
            for joint in range(dimension):
                assert getattr(cfg, f"theta{joint + 1}_min") - 1e-6 <= angles[joint]
                assert angles[joint] <= getattr(cfg, f"theta{joint + 1}_max") + 1e-6
            np.testing.assert_allclose(fk(*angles, cfg), target, atol=1e-6)


@pytest.mark.parametrize("dimension", [2, 3])
def test_ik_respects_base_and_elbow_limits_and_uses_backup_branch(dimension):
    if dimension == 2:
        cfg = PlanarArmConfig(theta2_max=0.0)
        target = forward_kinematics_planar(0.3, -0.8, cfg)
        single, batch = inverse_kinematics_planar, inverse_kinematics_planar_batch
        restricted = PlanarArmConfig(theta1_min=-0.1, theta1_max=0.1)
        unreachable = np.array([0.0, 0.2])
    else:
        cfg = Arm3DOFConfig(theta3_max=0.0)
        target = np.array([0.18, 0.02, 0.25])
        single, batch = inverse_kinematics_3dof, inverse_kinematics_3dof_batch
        restricted = Arm3DOFConfig(theta1_min=-0.1, theta1_max=0.1)
        unreachable = np.array([0.1, 0.1, 0.25])
    scalar = single(target, cfg)
    assert scalar is not None
    assert scalar[-1] < 0.0
    np.testing.assert_allclose(batch(target[None], cfg)[0], scalar, atol=1e-6)
    assert single(unreachable, restricted) is None
    assert np.isnan(batch(unreachable[None], restricted)).all()


@pytest.mark.parametrize("elbow_up", [True, False])
def test_3dof_ik_tries_negative_radius_with_restricted_yaw(elbow_up):
    cfg = Arm3DOFConfig(theta1_min=-0.1, theta1_max=0.1)
    target = forward_kinematics_3dof(0.0, 1.4, 0.8, cfg)
    assert target[0] < 0.0
    scalar = inverse_kinematics_3dof(target, cfg, elbow_up=elbow_up)
    assert scalar is not None
    np.testing.assert_allclose(forward_kinematics_3dof(*scalar, cfg), target, atol=1e-12)
    assert cfg.theta1_min <= scalar[0] <= cfg.theta1_max
    np.testing.assert_allclose(
        inverse_kinematics_3dof_batch(target[None], cfg, elbow_up=elbow_up)[0], scalar, atol=1e-6
    )


def test_3dof_ik_chooses_allowed_yaw_at_vertical_singularity():
    cfg = Arm3DOFConfig(theta1_min=0.2, theta1_max=0.4)
    target = np.array([0.0, 0.0, cfg.L0 + 0.2])
    scalar = inverse_kinematics_3dof(target, cfg)
    assert scalar is not None
    assert cfg.theta1_min <= scalar[0] <= cfg.theta1_max
    np.testing.assert_allclose(forward_kinematics_3dof(*scalar, cfg), target, atol=1e-12)
    np.testing.assert_allclose(inverse_kinematics_3dof_batch(target[None], cfg)[0], scalar, atol=1e-6)


class FixedOutputs(torch.nn.Module):
    def __init__(self, angles):
        super().__init__()
        angles = np.asarray(angles)
        self.values = torch.tensor(
            np.stack([np.sin(angles), np.cos(angles)], axis=-1).reshape(len(angles), -1),
            dtype=torch.float32,
        )

    def forward(self, x):
        return self.values.expand(len(x), -1)


@pytest.mark.parametrize("module,dimension", SAMPLERS)
def test_nn_evaluation_rejects_nonfinite_and_out_of_limit_angles(module, dimension):
    if dimension == 2:
        cfg = PlanarArmConfig(theta1_min=-0.5, theta1_max=0.5)
        targets = np.tile([0.18, 0.02], (4, 1))
        angles = [[0.1, 0.4], [1.2, 0.4], [np.nan, 0.4], [0.1, 0.4]]
    else:
        cfg = Arm3DOFConfig()
        targets = np.tile([0.18, 0.02, 0.25], (4, 1))
        angles = [[0.1, 0.4], [-2.0, 0.4], [np.nan, 0.4], [0.1, 0.4]]
    model = FixedOutputs(angles)
    # atan2(Inf, 有限值) 仍可能返回有限角，必须检查原始编码。
    model.values[-1, 0] = float("inf")
    results = getattr(module, f"solve_{dimension}dof_all")(targets, cfg, model, model)
    for name in ("mlp", "esn"):
        np.testing.assert_array_equal(results[name]["valid"], [True, False, False, False])
        assert np.isnan(results[name]["err_mm"][1:]).all()


VIEWERS = [
    (visualize, 2, "_run_2dof_viewer", "_load_ik_model_mujoco"),
    (visualize, 3, "_run_3dof_viewer", "_load_ik_model_mujoco"),
    (visualize_mujoco, 2, "run_2dof_visualizer", "load_ik_model"),
    (visualize_mujoco, 3, "run_3dof_visualizer", "load_ik_model"),
]


@pytest.mark.parametrize("module,dimension,function,loader", VIEWERS)
@pytest.mark.parametrize("invalid_kind", ["limits", "nonfinite"])
@pytest.mark.parametrize("overlay_supported", [True, False])
def test_viewer_holds_pose_and_reports_rejected_nn_output(
    module, dimension, function, loader, invalid_kind, overlay_supported, monkeypatch, capsys
):
    """用单帧假查看器验证入口安全逻辑，不创建原生窗口。"""
    cfg = PlanarArmConfig(theta1_min=-0.5, theta1_max=0.5) if dimension == 2 else Arm3DOFConfig()
    invalid = [1.2, 0.4] if dimension == 2 else [-2.0, 0.4]
    model = FixedOutputs([invalid])
    if invalid_kind == "nonfinite":
        model.values[0, 0] = float("inf")
    monkeypatch.setattr(module, loader, lambda *args: (model, "测试模型"))
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    captured = {}

    class Viewer:
        def __init__(self):
            self.cam = SimpleNamespace(lookat=np.zeros(3))
            self.steps = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def is_running(self):
            self.steps += 1
            return self.steps == 1

        def sync(self):
            pass

    if overlay_supported:
        Viewer.set_texts = lambda self, texts: captured.update(overlay=str(texts))

    def launch(model, data, **kwargs):
        captured["data"] = data
        return Viewer()

    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    getattr(module, function)(
        str(ROOT / "src" / f"arm_{dimension}dof.xml"), cfg, use_nn=True, nn_ckpt="mock.pt"
    )
    np.testing.assert_array_equal(captured["data"].qpos, np.zeros(dimension))
    if overlay_supported:
        assert "REJECTED" in captured["overlay"]
    assert "拒绝更新" in capsys.readouterr().out


@pytest.mark.parametrize("module,dimension,function,loader", VIEWERS)
@pytest.mark.parametrize("checkpoint", [None, "missing.pt"])
def test_nn_viewer_does_not_fall_back_when_checkpoint_missing(
    module, dimension, function, loader, checkpoint, monkeypatch
):
    cfg = PlanarArmConfig() if dimension == 2 else Arm3DOFConfig()

    def unexpected_launch(*args, **kwargs):
        pytest.fail("指定 NN 模式却缺少权重时，不得改用解析模式打开窗口")

    monkeypatch.setattr(mujoco.viewer, "launch_passive", unexpected_launch)
    monkeypatch.setattr(module, loader, lambda *args: (None, None))
    getattr(module, function)(
        str(ROOT / "src" / f"arm_{dimension}dof.xml"), cfg, use_nn=True, nn_ckpt=checkpoint
    )


@pytest.mark.parametrize("module", [visualize, visualize_ik])
def test_summary_reports_zero_legal_rate_when_all_outputs_rejected(module, capsys):
    module.print_summary_ik(
        "3DOF", {"mlp": {"err_mm": np.full(2, np.nan), "valid": np.zeros(2, bool)}}
    )
    printed = capsys.readouterr().out
    assert "合法解率=0.0%" in printed
    assert "无法统计误差" in printed
