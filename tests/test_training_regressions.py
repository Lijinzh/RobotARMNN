"""训练划分、预热、单轮调度、CLI 与 checkpoint 的回归测试。"""

import numpy as np
import pytest
import torch

from src import train, train_idyn
from src.models import IDynESN, IDynGRU, IDynLSTM, IDynMLP, IKNN_ESN, IKNN_MLP


@pytest.fixture
def training_files(tmp_path, monkeypatch):
    """所有训练结果只写临时目录，禁止覆盖项目权重。"""
    rng = np.random.default_rng(17)
    x = rng.normal(size=(600, 6)).astype(np.float32)
    traj_id = np.repeat(np.arange(5), 120)
    x += traj_id[:, None]
    y = (0.2 * x[:, :3] + 0.1 * x[:, 3:]).astype(np.float32)
    idyn_path = tmp_path / "idyn.npz"
    np.savez(
        idyn_path, inputs=x, outputs=y, traj_id=traj_id, n_traj=5,
        dt=np.float32(0.002), label_version=2,
    )
    xy = rng.normal(size=(24, 2)).astype(np.float32)
    angles = rng.uniform(-1, 1, size=(24, 2)).astype(np.float32)
    sincos = np.stack([np.sin(angles), np.cos(angles)], axis=-1).reshape(24, 4)
    np.savez(tmp_path / "ik_2dof.npz", inputs=xy, outputs=sincos, angles=angles)
    monkeypatch.setattr(train, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(train, "CKPT_DIR", str(tmp_path / "ik_checkpoints"))
    monkeypatch.setattr(train_idyn, "DATA_PATH", str(idyn_path))
    monkeypatch.setattr(train_idyn, "CKPT_DIR", str(tmp_path / "idyn_checkpoints"))
    monkeypatch.setattr(torch, "compile", lambda model: model)
    return {"x": x, "xy": xy, "traj_id": traj_id, "path": tmp_path}


def test_ik_esn_uses_training_statistics_only(training_files):
    xy = torch.from_numpy(training_files["xy"])
    indices = torch.randperm(24, generator=torch.Generator().manual_seed(42))[:18]
    model = train.train_esn(
        "2dof", reservoir_dim=8, readout_hidden=4, epochs=1,
        batch_size=12, val_ratio=0.25, device_str="cpu",
    )
    np.testing.assert_allclose(model.x_mean.numpy(), xy[indices].mean(0).numpy())
    np.testing.assert_allclose(model.x_std.numpy(), xy[indices].std(0).numpy())
    assert model.training_metadata["normalization"] == "training_only"


def test_single_epoch_ik_mlp_finishes(training_files):
    model = train.train_mlp(
        "2dof", epochs=1, batch_size=12, hidden_dim=8, n_layers=1, device_str="cpu"
    )
    assert torch.isfinite(model(torch.zeros(1, 2))).all()


@pytest.mark.parametrize("kind", ["mlp", "esn", "lstm", "gru"])
def test_idyn_training_uses_whole_trajectories_and_records_protocol(training_files, kind):
    kwargs = {"epochs": 1, "batch_size": 4, "device_str": "cpu"}
    if kind == "mlp":
        model = train_idyn.train_idyn_mlp(hidden_dim=8, n_layers=1, **kwargs)
    elif kind == "esn":
        model = train_idyn.train_idyn_esn(reservoir_dim=8, readout_hidden=4, **kwargs)
    else:
        model = train_idyn.train_idyn_rnn(
            model_type=kind, hidden_dim=4, n_layers=1, seq_len=40, **kwargs
        )
    expected_x = torch.from_numpy(training_files["x"][training_files["traj_id"] < 4])
    np.testing.assert_allclose(model.x_mean.numpy(), expected_x.mean(0).numpy())
    meta = model.training_metadata
    assert meta["target"] == "outputs"
    assert meta["sampling_dt_seconds"] == pytest.approx(0.002)
    assert meta["train_trajectory_ids"] == [0, 1, 2, 3]
    assert meta["validation_trajectory_ids"] == [4]
    assert meta["warmup_steps"] == 100
    assert meta["train_loss_samples"] == 80
    assert meta["validation_loss_samples"] == 20
    assert len(meta["data_sha256"]) == 64
    if kind in {"lstm", "gru"}:
        assert meta["validation_protocol"] == "continuous_trajectory_after_warmup"


def test_rnn_validation_carries_state_through_the_whole_trajectory():
    """累积器必须连续跑完轨迹，按块重置会得到不同且错误的验证值。"""
    class Accumulator(torch.nn.Module):
        def forward(self, x):
            return x[:, :3].cumsum(0), None

    x = torch.ones(12, 6)
    y = x[:, :3].cumsum(0)
    mse, mae = train_idyn.evaluate_rnn_trajectories(Accumulator(), [(x, y)], warmup_steps=2)
    assert mse == 0.0
    assert mae == 0.0


def test_rnn_validation_keeps_tail_dropped_from_training_chunks(training_files):
    model = train_idyn.train_idyn_rnn(
        model_type="gru", hidden_dim=4, n_layers=1, epochs=1, seq_len=55,
        batch_size=4, device_str="cpu",
    )
    # 每条 120 帧：训练在 110 帧前切成完整块，其中 100–109 为有效损失；
    # 验证连续评分 100–119，不应把末尾 10 帧也丢掉。
    assert model.training_metadata["train_loss_samples"] == 40
    assert model.training_metadata["validation_loss_samples"] == 20


@pytest.mark.parametrize("kind", ["lstm", "gru"])
def test_rnn_cli_honors_batch_size_and_has_its_own_default(monkeypatch, kind):
    calls = []
    monkeypatch.setattr(train_idyn, "train_idyn_rnn", lambda **kwargs: calls.append(kwargs))
    train_idyn.main(["--model", kind, "--epochs", "1", "--batch_size", "7"])
    train_idyn.main(["--model", kind, "--epochs", "1"])
    assert [call["batch_size"] for call in calls] == [7, 256]


@pytest.mark.parametrize("factory", [
    lambda: IKNN_MLP(hidden_dim=8, n_layers=1),
    lambda: IKNN_ESN(reservoir_dim=8, readout_hidden=4),
    lambda: IDynMLP(hidden_dim=8, n_layers=1),
    lambda: IDynESN(reservoir_dim=8, readout_hidden=4),
    lambda: IDynLSTM(hidden_dim=4, n_layers=1),
    lambda: IDynGRU(hidden_dim=4, n_layers=1),
])
def test_checkpoint_metadata_roundtrip_is_optional_and_backward_compatible(tmp_path, factory):
    model = factory()
    path = tmp_path / "new.pt"
    model.save(path, metadata={"target": "test_target", "sampling_dt_seconds": 0.002})
    loaded = type(model).load(path)
    assert loaded.training_metadata["target"] == "test_target"
    legacy = torch.load(path, map_location="cpu", weights_only=False)
    legacy.pop("metadata")
    torch.save(legacy, tmp_path / "legacy.pt")
    restored = type(model).load(tmp_path / "legacy.pt")
    assert restored.training_metadata == {}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name])


def test_legacy_esn_resave_corrects_radius_without_changing_predictions(tmp_path):
    model = IDynESN(reservoir_dim=32, readout_hidden=4, spectral_radius=0.65, sparsity=0.5)
    legacy = {
        "config": {"n_joints": 3, "reservoir_dim": 32, "readout_hidden": 4,
                   "spectral_radius": float(model.Wres.abs().max())},
        "state_dict": model.state_dict(),
    }
    path = tmp_path / "legacy_esn.pt"
    torch.save(legacy, path)
    restored = IDynESN.load(path)
    restored.save(tmp_path / "resaved.pt")
    saved = torch.load(tmp_path / "resaved.pt", map_location="cpu", weights_only=False)
    actual = float(torch.linalg.eigvals(model.Wres).abs().max())
    assert saved["config"]["spectral_radius"] == pytest.approx(actual, abs=1e-6)
    reloaded = IDynESN.load(tmp_path / "resaved.pt")
    x = torch.randn(7, 6)
    torch.testing.assert_close(model(x), reloaded(x), rtol=0, atol=0)
