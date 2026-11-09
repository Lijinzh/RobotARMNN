"""控制方案语义、渲染隔离、参考标记及关闭清理的回归测试。"""

import os
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from src import visualize_idyn as vis


def reference_trajectory(steps=6):
    q = np.tile([0.1, 0.2, -0.3], (steps, 1))
    q[:, 0] += np.arange(steps) * 0.001
    dq = np.tile([0.5, 0.0, 0.0], (steps, 1))
    return q, dq


class FakeViewer:
    """模拟关闭、同步污染与渲染异常，不打开真实窗口。"""

    def __init__(self, model, data, close_after=None, poison=False, fail=False):
        self.model, self.data = model, data
        self.close_after, self.poison, self.fail = close_after, poison, fail
        self.frames = []
        self.frame_times = []
        self.exits = []
        self.texts = []
        self.locked = False
        self.cam = SimpleNamespace(lookat=np.zeros(3))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.exits.append(exc)

    def is_running(self):
        return self.close_after is None or len(self.frames) < self.close_after

    @contextmanager
    def lock(self):
        self.locked = True
        try:
            yield
        finally:
            self.locked = False

    def set_texts(self, texts):
        self.texts.append(texts)

    def sync(self):
        if self.fail:
            raise RuntimeError("render_failure")
        self.frames.append((self.data.qpos.copy(), self.data.mocap_pos.copy()))
        self.frame_times.append(self.data.time)
        if self.poison:
            self.data.qpos[:] = 9.0
            self.model.opt.gravity[:] = 9.0


def install_viewer(monkeypatch, **kwargs):
    handles = []

    def launch(model, data, **launch_options):
        handle = FakeViewer(model, data, **kwargs)
        handle.launch_options = launch_options
        handles.append(handle)
        return handle

    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    return handles


def test_total_torque_gru_cannot_be_added_as_mpc_residual():
    q, dq = reference_trajectory()
    with pytest.raises(ValueError, match="残差"):
        vis.run_closed_loop(q, dq, use_mpc_gru=True)


def test_viewer_model_must_be_an_active_scheme():
    q, dq = reference_trajectory()
    with pytest.raises(ValueError, match="未启用"):
        vis.run_closed_loop(q, dq, viewer_model="gru")


def test_reference_marker_and_rendering_cannot_change_physics(monkeypatch):
    q, dq = reference_trajectory()
    baseline = vis.run_closed_loop(q, dq)
    handles = install_viewer(monkeypatch, poison=True)
    actual = vis.run_closed_loop(q, dq, show_viewer=True)
    np.testing.assert_array_equal(actual["pd"], baseline["pd"])
    handle = handles[0]
    assert len(handle.exits) == 1
    assert len(handle.frames) == 2
    reference_model = mujoco.MjModel.from_xml_path(vis.XML_PATH)
    reference_data = mujoco.MjData(reference_model)
    site = mujoco.mj_name2id(reference_model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    for sim_time, (shown_q, markers) in zip(handle.frame_times, handle.frames):
        i = round(sim_time / vis.DT)
        reference_data.qpos[:] = q[i]
        mujoco.mj_forward(reference_model, reference_data)
        np.testing.assert_allclose(markers[0], reference_data.site_xpos[site], atol=1e-12)
        np.testing.assert_allclose(shown_q, actual["pd"][i], atol=1e-7)


def test_closing_viewer_stops_and_trims_logs(monkeypatch):
    q, dq = reference_trajectory(steps=20)
    handles = install_viewer(monkeypatch, close_after=2)
    result = vis.run_closed_loop(q, dq, show_viewer=True)
    assert result["completed_steps"] == 10
    assert result["interrupted"]
    assert result["pd"].shape == (10, 3)
    np.testing.assert_array_equal(result["q_des"], q[:10])
    assert len(handles[0].exits) == 1


def test_viewer_samples_frames_without_changing_physics_or_accumulating_render_time(monkeypatch):
    q, dq = reference_trajectory(steps=45)
    baseline = vis.run_closed_loop(q, dq)
    handles = install_viewer(monkeypatch)
    clock = [100.0]
    original_sync = FakeViewer.sync

    def slow_sync(handle):
        original_sync(handle)
        clock[0] += 0.01  # 渲染每次耗时 10ms，控制周期仍为 2ms。

    monkeypatch.setattr(FakeViewer, "sync", slow_sync)
    monkeypatch.setattr(vis.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(vis.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    result = vis.run_closed_loop(q, dq, show_viewer=True)
    np.testing.assert_array_equal(result["pd"], baseline["pd"])
    # 500Hz 不能整除 60Hz；每 9 步显示约 56Hz，并强制显示末步。
    indices = [0, 9, 18, 27, 36, 44]
    np.testing.assert_allclose(handles[0].frame_times, np.asarray(indices) * vis.DT)
    shown = np.stack([frame[0] for frame in handles[0].frames])
    np.testing.assert_allclose(shown, result["pd"][indices], atol=1e-7)
    assert 100.0 + len(q) * vis.DT <= clock[0]
    assert clock[0] <= 100.0 + len(q) * vis.DT + 0.01 + 1e-12


def test_viewer_exception_cleans_up(monkeypatch):
    q, dq = reference_trajectory()
    handles = install_viewer(monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match="render_failure"):
        vis.run_closed_loop(q, dq, show_viewer=True)
    assert len(handles[0].exits) == 1
    assert handles[0].exits[0][0] is RuntimeError


def test_viewer_locks_scene_and_labels_the_displayed_dynamics(monkeypatch):
    q, dq = reference_trajectory()
    handles = install_viewer(monkeypatch)
    original_forward = mujoco.mj_forward
    locked_updates = []

    def check_forward(model, data):
        if handles and model is handles[0].model:
            assert handles[0].locked, "已启动的 viewer 场景更新必须持锁"
            locked_updates.append(id(data))
        return original_forward(model, data)

    monkeypatch.setattr(mujoco, "mj_forward", check_forward)
    result = vis.run_closed_loop(q, dq, show_viewer=True, viewer_model="pd")
    handle = handles[0]
    assert len(locked_updates) == 2 * len(handle.frames)
    assert handle.launch_options == {"show_left_ui": False, "show_right_ui": False}
    assert (handle.cam.azimuth, handle.cam.elevation, handle.cam.distance) == (135, -25, 1.05)
    np.testing.assert_array_equal(handle.cam.lookat, [0.12, 0.0, 0.2])
    for sim_time, (font, position, left, right) in zip(handle.frame_times, handle.texts):
        index = round(sim_time / vis.DT)
        assert font == mujoco.mjtFontScale.mjFONTSCALE_200
        assert position == mujoco.mjtGridPos.mjGRID_TOPLEFT
        assert "Dynamics simulation" in left
        assert "PD" in right and "reference EE" in right
        assert f"{index * vis.DT:.3f} s" in right
        expected_error = np.degrees(np.abs(result["pd"][index] - q[index])).mean()
        assert f"{expected_error:.3f} deg" in right
    assert len(handle.texts) == len(handle.frames) == 2


def test_viewer_without_optional_text_overlay_remains_supported(monkeypatch):
    monkeypatch.delattr(FakeViewer, "set_texts")
    handles = install_viewer(monkeypatch)
    q, dq = reference_trajectory()
    result = vis.run_closed_loop(q, dq, show_viewer=True)
    assert result["completed_steps"] == len(q)
    assert len(handles[0].frames) == 2


@pytest.mark.parametrize("scheme", ["pd", "mpc"])
def test_viewer_displays_the_requested_active_scheme(monkeypatch, scheme):
    q, dq = reference_trajectory()
    handles = install_viewer(monkeypatch)
    result = vis.run_closed_loop(q, dq, use_mpc=True, show_viewer=True, viewer_model=scheme)
    shown = np.stack([frame[0] for frame in handles[0].frames])
    indices = np.rint(np.asarray(handles[0].frame_times) / vis.DT).astype(int)
    assert result["viewer_model"] == scheme
    np.testing.assert_allclose(shown, result[scheme][indices], atol=1e-7)
    assert not np.allclose(result["pd"], result["mpc"], atol=1e-8)


def test_phase_plot_uses_reference_velocity_and_readable_fonts(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    q, dq = reference_trajectory()
    dq[:, 0] = np.linspace(-0.2, 0.3, len(q))
    result = {"q_des": q, "dq_des": dq, "pd": q.copy(), "esn": q.copy(),
              "tau_esn": np.ones_like(q)}
    seen_x = []
    figures = []
    original_scatter, original_figure = Axes.scatter, vis.plt.figure

    def scatter(axis, x, *args, **kwargs):
        seen_x.append(np.asarray(x))
        return original_scatter(axis, x, *args, **kwargs)

    def figure(*args, **kwargs):
        created = original_figure(*args, **kwargs)
        figures.append(created)
        return created

    monkeypatch.setattr(Axes, "scatter", scatter)
    monkeypatch.setattr(vis.plt, "figure", figure)
    vis.plot_comparison(result, str(tmp_path / "comparison.png"))
    np.testing.assert_array_equal(seen_x[0], dq[::2, 0])
    for axis in figures[0].axes:
        texts = [axis.title, axis.xaxis.label, axis.yaxis.label]
        texts.extend(axis.get_xticklabels() + axis.get_yticklabels())
        if axis.get_legend() is not None:
            texts.extend(axis.get_legend().get_texts())
        assert all(text.get_fontsize() >= 18 for text in texts)


def test_cli_loads_cuda_serialized_checkpoint_on_cpu():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "src/visualize_idyn.py"), "--model", "esn",
         "--duration", "0.008", "--no-save"],
        cwd=root,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "Agg",
             "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        text=True, capture_output=True, timeout=45,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("arguments,diagnostic", [
    (["--model", "mpc_gru"], "总力矩"),
    (["--model", "mpc", "--viewer-model", "gru"], "未启用"),
])
def test_cli_rejects_invalid_combinations(arguments, diagnostic):
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "src/visualize_idyn.py"), *arguments],
        cwd=root,
        env={**os.environ, "MPLBACKEND": "Agg", "OPENBLAS_NUM_THREADS": "1",
             "OMP_NUM_THREADS": "1"},
        text=True, capture_output=True, timeout=45,
    )
    assert completed.returncode == 2
    assert diagnostic in completed.stderr
