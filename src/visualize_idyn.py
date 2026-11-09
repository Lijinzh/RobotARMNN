"""
逆动力学补偿效果对比可视化

对比独立 PD、MPC 及 PD + MLP/ESN/LSTM/GRU 总力矩前馈的跟踪误差。
现有神经网络学习 PD 示教总力矩，不是独立摩擦残差，不能直接叠加到 MPC。

可视化内容：
    (a) 关节角跟踪误差曲线（3 关节，时域对比）
    (b) MAE 柱状图（定量结论）
    (c) 总前馈力矩时序
    (d) 总前馈力矩与参考速度的相图（不能单独证明摩擦迟滞）

用法：
    uv run src/visualize_idyn.py               # 绘图窗口
    uv run src/visualize_idyn.py --no-save     # 显示图表而不自动保存
    uv run src/visualize_idyn.py --model esn --mujoco --viewer-model esn
"""

import argparse
from contextlib import ExitStack
import os
import sys
import time

# ── 中文字体配置 ────────────────────────────────────────────────────
# 优先使用系统 Noto Sans CJK SC，其次 WenQuanYi Micro Hei，最后 SimHei
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

for _font in ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei"]:
    if _font in [f.name for f in matplotlib.font_manager.fontManager.ttflist]:
        matplotlib.rcParams["font.sans-serif"] = [_font] + matplotlib.rcParams["font.sans-serif"]
        break
matplotlib.rcParams["axes.unicode_minus"] = False  # 负号正确显示
matplotlib.rcParams.update({
    "font.size": 18, "axes.titlesize": 18, "axes.labelsize": 18,
    "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18,
})
# ────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import mujoco
import mujoco.viewer

from src.data_gen_idyn import DT, JOINT_RANGES, KD, KP, XML_PATH, gen_excitation_traj
from src.models import IDynESN, IDynGRU, IDynLSTM, IDynMLP
from src.mpc_controller import LinearMPC, get_horizon_ref

CKPT_MLP = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "idyn_3dof_mlp_best.pt")
CKPT_ESN = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "idyn_3dof_esn_best.pt")
CKPT_LSTM = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "idyn_3dof_lstm_best.pt")
CKPT_GRU = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "idyn_3dof_gru_best.pt")
EXPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "export")

# =====================================================================
# 闭环仿真：PD + 前馈补偿
# =====================================================================


def run_closed_loop(
    q_des: np.ndarray,
    dq_des: np.ndarray,
    ddq_des: np.ndarray | None = None,
    mlp_model: IDynMLP | None = None,
    esn_model: IDynESN | None = None,
    lstm_model: IDynLSTM | None = None,
    gru_model: IDynGRU | None = None,
    xml_path: str = XML_PATH,
    show_viewer: bool = False,
    use_mpc: bool = False,
    use_mpc_gru: bool = False,
    mpc_horizon: int = 10,
    viewer_model: str | None = None,
) -> dict:
    """
    运行闭环仿真，收集各方案的跟踪轨迹。

    控制律：
        纯 PD：     τ = KP*(q_des-q) + KD*(dq_des-dq)
        MPC：       滚动预测优化（预测域 N 步，解析 QP）
        FF 方案：   τ = τ_ff(q_des, dq_des) + KP*(q_des-q) + KD*(dq_des-dq)

    viewer_model 必须是启用的方案；未指定时显示首个前馈/MPC 方案或 PD。
    viewer 的模型和状态均与物理仿真隔离，显示的状态和参考取自同一积分前时刻。
    """
    if use_mpc_gru:
        raise ValueError(
            "现有 GRU 学习 outputs 总力矩，不能直接作为 MPC 摩擦残差叠加；"
            "需要先定义残差目标并重新训练。请分别使用 MPC 或 PD+GRU。"
        )
    if len(q_des) == 0:
        raise ValueError("参考轨迹不能为空")
    mjmodel = mujoco.MjModel.from_xml_path(xml_path)

    def make_data():
        d = mujoco.MjData(mjmodel)
        mujoco.mj_forward(mjmodel, d)
        return d

    j_ids = [
        mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i + 1}") for i in range(3)
    ]
    qpos_ids = mjmodel.jnt_qposadr[j_ids]
    dof_ids = mjmodel.jnt_dofadr[j_ids]

    T = len(q_des)
    device = torch.device("cpu")

    configs = {"pd": make_data()}
    # MPC 方案（滚动预测优化，无 NN 前馈）
    mpc_ctrl = None
    if use_mpc:
        configs["mpc"] = make_data()
        mpc_ctrl = LinearMPC(n_joint=3, dt=DT, horizon=mpc_horizon)
    if mlp_model is not None:
        configs["mlp"] = make_data()
    if esn_model is not None:
        configs["esn"] = make_data()
        H_esn = esn_model.init_hidden(batch_size=1).to(device)
    if lstm_model is not None:
        configs["lstm"] = make_data()
        H_lstm = lstm_model.init_hidden(batch_size=1)
        H_lstm = tuple(h.to(device) for h in H_lstm)
    if gru_model is not None:
        configs["gru"] = make_data()
        H_gru = gru_model.init_hidden(batch_size=1).to(device)

    viewer_display_key = viewer_model or next((k for k in configs if k != "pd"), "pd")
    if viewer_display_key not in configs:
        raise ValueError(f"显示方案 {viewer_display_key!r} 未启用；已启用：{list(configs)}")
    for d in configs.values():
        d.qpos[qpos_ids] = q_des[0]
        d.qvel[dof_ids] = dq_des[0]  # 匹配初始速度，避免瞬态误差
        mujoco.mj_forward(mjmodel, d)

    logs = {k: np.zeros((T, 3), dtype=np.float32) for k in configs}
    tau_logs = {k: np.zeros((T, 3), dtype=np.float32) for k in configs if k not in ("pd", "mpc")}

    completed = 0
    with ExitStack() as stack:
        viewer_ctx = None
        if show_viewer:
            # UI 可修改模型选项，因此连 MjModel 也独立，不能只隔离 MjData。
            render_model = mujoco.MjModel.from_xml_path(xml_path)
            viewer_data = mujoco.MjData(render_model)
            reference_data = mujoco.MjData(render_model)
            target_id = mujoco.mj_name2id(render_model, mujoco.mjtObj.mjOBJ_BODY, "target")
            target_mocap_id = render_model.body_mocapid[target_id]
            ee_site_id = mujoco.mj_name2id(render_model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

            def update_scene(qpos, qvel, q_ref, dq_ref, sim_time):
                reference_data.qpos[qpos_ids] = q_ref
                reference_data.qvel[dof_ids] = dq_ref
                mujoco.mj_forward(render_model, reference_data)
                viewer_data.qpos[:] = qpos
                viewer_data.qvel[:] = qvel
                viewer_data.time = sim_time
                viewer_data.mocap_pos[target_mocap_id] = reference_data.site_xpos[ee_site_id]
                mujoco.mj_forward(render_model, viewer_data)

            initial = configs[viewer_display_key]
            update_scene(initial.qpos, initial.qvel, q_des[0], dq_des[0], initial.time)
            print(f"[vis] MuJoCo 显示方案：{viewer_display_key}；红球为参考末端位置")
            viewer_ctx = stack.enter_context(mujoco.viewer.launch_passive(
                render_model, viewer_data, show_left_ui=False, show_right_ui=False,
            ))
            viewer_label = {
                "pd": "PD", "mpc": "MPC", "mlp": "PD + MLP feedforward",
                "esn": "PD + ESN feedforward", "lstm": "PD + LSTM feedforward",
                "gru": "PD + GRU feedforward",
            }[viewer_display_key]
            with viewer_ctx.lock():
                viewer_ctx.cam.azimuth = 135
                viewer_ctx.cam.elevation = -25
                viewer_ctx.cam.distance = 1.05
                viewer_ctx.cam.lookat[:] = [0.12, 0.0, 0.2]

        # 控制保持每步积分；显示间隔向上取整，避免刷新率超过约 60Hz。
        viewer_interval = max(1, int(np.ceil(1.0 / (60.0 * DT))))
        wall_start = time.perf_counter() if viewer_ctx is not None else 0.0
        for t in range(T):
            if viewer_ctx is not None and not viewer_ctx.is_running():
                break
            render_frame = viewer_ctx is not None and (t % viewer_interval == 0 or t == T - 1)
            frame_state = None
            for key, d in configs.items():
                q_cur = d.qpos[qpos_ids].copy()
                dq_cur = d.qvel[dof_ids].copy()
                logs[key][t] = q_cur
                if render_frame and key == viewer_display_key:
                    frame_state = (d.qpos.copy(), d.qvel.copy(), d.time)

                if key == "pd":
                    tau = KP * (q_des[t] - q_cur) + KD * (dq_des[t] - dq_cur)
                elif key == "mpc":
                    q_h, dq_h = get_horizon_ref(t, q_des, dq_des, mpc_ctrl.N)
                    mujoco.mj_forward(mjmodel, d)
                    tau = mpc_ctrl.compute(mjmodel, d, q_cur, dq_cur, q_h, dq_h, dof_ids)
                else:
                    x = torch.tensor(np.concatenate([q_des[t], dq_des[t]])[None], dtype=torch.float32)
                    tau_pd_corr = KP * (q_des[t] - q_cur) + KD * (dq_des[t] - dq_cur)
                    with torch.no_grad():
                        if key == "mlp":
                            tau_ff = mlp_model(x).squeeze(0).numpy()
                        elif key == "esn":
                            tau_t, H_esn = esn_model.forward_step(x, H_esn)
                            tau_ff = tau_t.squeeze(0).numpy()
                        elif key == "lstm":
                            tau_t, H_lstm = lstm_model.forward_step(x, H_lstm)
                            tau_ff = tau_t.squeeze(0).numpy()
                        elif key == "gru":
                            tau_t, H_gru = gru_model.forward_step(x, H_gru)
                            tau_ff = tau_t.squeeze(0).numpy()
                    tau = tau_ff + tau_pd_corr
                    tau_logs[key][t] = tau_ff
                d.ctrl[:3] = tau
                mujoco.mj_step(mjmodel, d)

            completed = t + 1
            if render_frame:
                qpos, qvel, sim_time = frame_state
                with viewer_ctx.lock():
                    update_scene(qpos, qvel, q_des[t], dq_des[t], sim_time)
                if hasattr(viewer_ctx, "set_texts"):
                    error_deg = np.degrees(np.abs(qpos[qpos_ids] - q_des[t])).mean()
                    viewer_ctx.set_texts((
                        mujoco.mjtFontScale.mjFONTSCALE_200,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "Dynamics simulation\nScheme\nTime\nMean joint error\nRed target",
                        f"MuJoCo mj_step\n{viewer_label}\n{sim_time:.3f} s\n"
                        f"{error_deg:.3f} deg\nreference EE",
                    ))
                viewer_ctx.sync()
            if viewer_ctx is not None:
                # 对齐绝对仿真时间；渲染耗时已包含在内，不逐帧累加等待。
                delay = wall_start + completed * DT - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)

    return {
        **{key: values[:completed] for key, values in logs.items()},
        **{f"tau_{key}": values[:completed] for key, values in tau_logs.items()},
        "q_des": q_des[:completed],
        "dq_des": dq_des[:completed],
        "viewer_model": viewer_display_key if show_viewer else None,
        "completed_steps": completed,
        "requested_steps": T,
        "interrupted": completed < T,
    }


# =====================================================================
# 绘图
# =====================================================================


def plot_comparison(result: dict, save_path: str | None = None) -> None:
    """
    绘制四子图对比图：
        (a) 所有关节平均跟踪误差
        (b) 各关节误差柱状图（定量）
        (c) joint1 前馈力矩时序（最后 500 步）
        (d) joint1 总前馈力矩与对应参考速度的相图
    """
    q_des = result["q_des"]
    T = len(q_des)
    t_arr = np.arange(T) * DT

    # 定义比较方案：key -> (颜色, 标签)
    SCHEMES = [
        ("pd", "#888888", "纯 PD（无前馈）"),
        ("mpc", "#9C27B0", "MPC（滚动预测优化）"),
        ("mlp", "#2196F3", "PD + MLP 前馈"),
        ("esn", "#F44336", "PD + ESN 前馈（有记忆）"),
        ("lstm", "#4CAF50", "PD + LSTM 前馈（门控记忆）"),
        ("gru", "#FF9800", "PD + GRU 前馈（轻量门控）"),
    ]

    errs = {}
    for key, _, _ in SCHEMES:
        if result.get(key) is not None:
            errs[key] = np.degrees(np.abs(result[key] - q_des))

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        "闭环跟踪与总力矩前馈对比：PD / MPC / PD+MLP / PD+ESN / LSTM / GRU",
        fontsize=22,
        fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.52, wspace=0.35)

    # ---- (a) 平均跟踪误差时域曲线 ----
    ax0 = fig.add_subplot(gs[0, 0])
    for key, color, label in SCHEMES:
        if key in errs:
            ax0.plot(
                t_arr,
                errs[key].mean(-1),
                color=color,
                alpha=0.85,
                label=label,
                linewidth=0.9 + (0.3 if key == "pd" else 0),
            )
    ax0.set_xlabel("时间 (s)")
    ax0.set_ylabel("平均跟踪误差 (°)")
    ax0.set_title("(a) 关节角跟踪误差（3关节均值）")
    ax0.legend(fontsize=18)
    ax0.grid(True, alpha=0.3)

    # ---- (b) 各关节误差柱状图 ----
    ax1 = fig.add_subplot(gs[0, 1])
    joint_names = ["Joint 1\n(底座)", "Joint 2\n(肩部)", "Joint 3\n(肘部)"]
    n_schemes = sum(1 for k, _, _ in SCHEMES if k in errs)
    width = 0.8 / n_schemes
    x_pos = np.arange(3)
    offset = 0
    for key, color, label in SCHEMES:
        if key in errs:
            ax1.bar(
                x_pos + offset * width,
                errs[key].mean(0),
                width,
                color=color,
                label=label,
                alpha=0.85,
            )
            offset += 1
    ax1.set_xticks(x_pos + (n_schemes - 1) * width / 2)
    ax1.set_xticklabels(joint_names)
    ax1.set_ylabel("平均误差 (°)")
    ax1.set_title("(b) 各关节平均跟踪误差")
    ax1.legend(fontsize=18)
    ax1.grid(True, alpha=0.3, axis="y")

    # ---- (c) 前馈力矩时序（joint1，最后 500 步） ----
    ax2 = fig.add_subplot(gs[1, 0])
    t_show = min(500, T)
    t_slice = slice(T - t_show, T)
    t_arr_s = t_arr[t_slice]
    tau_keys = [
        ("mlp", "#2196F3", "MLP 前馈"),
        ("esn", "#F44336", "ESN 前馈"),
        ("lstm", "#4CAF50", "LSTM 前馈"),
        ("gru", "#FF9800", "GRU 前馈"),
    ]
    for key, color, label in tau_keys:
        tau_arr = result.get(f"tau_{key}")
        if tau_arr is not None and tau_arr.any():
            ax2.plot(
                t_arr_s, tau_arr[t_slice, 0], color=color, label=label, linewidth=0.9, alpha=0.9
            )
    ax2.set_xlabel("时间 (s)")
    ax2.set_ylabel("力矩 (N·m)")
    ax2.set_title("(c) Joint1 总前馈力矩（最后 500 步）")
    ax2.legend(fontsize=18)
    ax2.grid(True, alpha=0.3)

    # ---- (d) 相图：同一参考状态的速度与网络总前馈力矩，不能据此识别摩擦迟滞。 ----
    ax3 = fig.add_subplot(gs[1, 1])
    step = 2
    for key, color, label in [
        ("mlp", "#2196F3", "MLP"),
        ("esn", "#F44336", "ESN"),
        ("lstm", "#4CAF50", "LSTM"),
        ("gru", "#FF9800", "GRU"),
    ]:
        tau_arr = result.get(f"tau_{key}")
        if tau_arr is not None and tau_arr.any() and result.get(key) is not None:
            dq_arr = result["dq_des"][:, 0]
            ax3.scatter(dq_arr[::step], tau_arr[::step, 0], c=color, s=1, alpha=0.5, label=label)
    ax3.set_xlabel("参考关节角速度 dq₁,des (rad/s)")
    ax3.set_ylabel("总前馈力矩 τ₁ (N·m)")
    ax3.set_title("(d) 总前馈力矩与参考速度相图")
    ax3.legend(fontsize=18, markerscale=6)
    ax3.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize_idyn] 已保存: {save_path}")
        plt.close(fig)
    else:
        plt.show()


# =====================================================================
# 打印定量结论
# =====================================================================


def print_summary(result: dict) -> None:
    q_des = result["q_des"]
    print("\n" + "=" * 70)
    print("  定量对比（平均跟踪误差，度；低→好）")
    print("=" * 70)
    for key, label in [
        ("pd", "纯 PD（无前馈）"),
        ("mpc", "MPC（滚动预测优化）"),
        ("mlp", "PD + MLP 前馈（无记忆）"),
        ("esn", "PD + ESN 前馈（固定储层）"),
        ("lstm", "PD + LSTM 前馈（门控记忆）"),
        ("gru", "PD + GRU 前馈（轻量门控）"),
    ]:
        if result.get(key) is not None:
            err_deg = np.degrees(np.abs(result[key] - q_des)).mean()
            per_joint = np.degrees(np.abs(result[key] - q_des)).mean(0)
            print(
                f"  {label:<30} 总均值={err_deg:.3f}°  "
                f"[{per_joint[0]:.3f}, {per_joint[1]:.3f}, {per_joint[2]:.3f}]°"
            )
    print("=" * 70)


# =====================================================================
# 主入口
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="逆动力学补偿效果对比可视化")
    parser.add_argument(
        "--no-save",
        dest="no_save",
        action="store_true",
        help="禁用自动保存（默认自动保存到 export/）",
    )
    parser.add_argument("--mujoco", action="store_true", help="同时显示 MuJoCo 实时窗口")
    parser.add_argument(
        "--traj_seed", type=int, default=999, help="测试轨迹随机种子（与训练数据不同）"
    )
    parser.add_argument("--duration", type=float, default=6.0, help="测试轨迹时长（秒）")
    parser.add_argument(
        "--amp_scale",
        type=float,
        default=2.0,
        help="测试轨迹幅值缩放（默认 2.0，可视化效果更明显；数据生成时固定为 1.0）",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        choices=["pd", "mlp", "esn", "lstm", "gru", "mpc", "mpc_gru"],
        default=["mlp", "esn", "lstm", "gru", "mpc"],
        help="指定参与对比的方案，PD 始终为基线；旧 mpc_gru 因目标不匹配而拒绝运行",
    )
    parser.add_argument(
        "--viewer-model", choices=["pd", "mlp", "esn", "lstm", "gru", "mpc"],
        help="窗口显示的已启用方案；默认显示 --model 中首个实际启用方案",
    )
    parser.add_argument(
        "--mpc_horizon",
        type=int,
        default=10,
        help="MPC 预测域步数（默认 10）",
    )
    args = parser.parse_args()
    enabled_models = set(args.model)
    if "mpc_gru" in enabled_models:
        parser.error(
            "现有 GRU 学习 outputs 总力矩，不能作为 MPC 摩擦残差叠加；"
            "需要定义残差目标并重新训练。可使用 --model mpc gru 独立对比。"
        )
    if args.duration <= 0 or args.mpc_horizon < 1:
        parser.error("duration 必须为正数，mpc_horizon 必须至少为 1")
    if args.viewer_model is not None and args.viewer_model not in enabled_models | {"pd"}:
        parser.error(f"显示方案 {args.viewer_model!r} 未启用，请通过 --model 选择")

    # ---- 判断是否启用 MPC ----
    use_mpc = "mpc" in enabled_models

    # ---- 加载模型 ----
    mlp_model, esn_model, lstm_model, gru_model = None, None, None, None

    for ckpt_path, model_name, load_fn in [
        (CKPT_MLP, "MLP", lambda p: IDynMLP.load(p, map_location="cpu")),
        (CKPT_ESN, "ESN", lambda p: IDynESN.load(p, map_location="cpu")),
        (CKPT_LSTM, "LSTM", lambda p: IDynLSTM.load(p, map_location="cpu")),
        (CKPT_GRU, "GRU", lambda p: IDynGRU.load(p, map_location="cpu")),
    ]:
        if model_name.lower() not in enabled_models:
            continue  # 用户未指定，跳过
        if os.path.exists(ckpt_path):
            m = load_fn(ckpt_path)
            m.eval()
            print(f"[vis] 已加载 {model_name}: {ckpt_path}")
            if model_name == "MLP":
                mlp_model = m
            elif model_name == "ESN":
                esn_model = m
            elif model_name == "LSTM":
                lstm_model = m
            elif model_name == "GRU":
                gru_model = m
        else:
            print(f"[vis] {model_name} 检查点不存在（{ckpt_path}），跳过")

    # MPC 不依赖 NN，需单独统计
    has_any = (
        use_mpc
        or "pd" in enabled_models
        or any(m is not None for m in [mlp_model, esn_model, lstm_model, gru_model])
    )
    if not has_any:
        print("\n[vis] 请先训练模型：uv run src/train_idyn.py --model all")
        sys.exit(1)
    active_models = {"pd"}
    if use_mpc:
        active_models.add("mpc")
    for name, model in (
        ("mlp", mlp_model), ("esn", esn_model),
        ("lstm", lstm_model), ("gru", gru_model),
    ):
        if model is not None:
            active_models.add(name)
    if args.viewer_model is not None and args.viewer_model not in active_models:
        parser.error(f"显示方案 {args.viewer_model!r} 未启用，请检查其权重文件")
    display_model = args.viewer_model or next(name for name in args.model if name in active_models)

    # ---- 生成测试轨迹（不同 seed，确保与训练数据不重叠） ----
    rng = np.random.default_rng(args.traj_seed)
    t_arr, q_des, dq_des, ddq_des = gen_excitation_traj(
        args.duration, DT, JOINT_RANGES, n_freqs=4, rng=rng, amp_scale=args.amp_scale
    )
    T = len(q_des)
    print(
        f"\n[vis] 测试轨迹: {T:,} 步 ({args.duration}s), seed={args.traj_seed}, amp_scale={args.amp_scale}"
    )

    # ---- 闭环仿真 ----
    n_nn = sum(m is not None for m in [mlp_model, esn_model, lstm_model, gru_model])
    n_mpc = int(use_mpc)
    print(f"[vis] 运行闭环仿真（共 {1 + n_nn + n_mpc} 方案，每方案独立仿真）...")
    if use_mpc:
        print(
            f"[vis] MPC 预测域 = {args.mpc_horizon} 步（{args.mpc_horizon * DT * 1000:.1f} ms），无约束 QP 后首步限幅"
        )
    result = run_closed_loop(
        q_des,
        dq_des,
        ddq_des=ddq_des,
        mlp_model=mlp_model,
        esn_model=esn_model,
        lstm_model=lstm_model,
        gru_model=gru_model,
        show_viewer=args.mujoco,
        use_mpc=use_mpc,
        mpc_horizon=args.mpc_horizon,
        viewer_model=display_model,
    )

    # ---- 结果输出 ----
    if result["completed_steps"] == 0:
        print("[vis] 窗口已关闭，未完成任何仿真步。")
        sys.exit(0)
    if result["interrupted"]:
        print(f"[vis] 窗口关闭，已停止在 {result['completed_steps']}/{T} 步；只汇总已完成数据。")
    print_summary(result)

    # 默认自动保存，除非显式传入 --no-save
    save_path = None if args.no_save else os.path.join(EXPORT_DIR, "idyn_comparison.png")
    if save_path:
        print(f"[vis] 图片将保存至: {save_path}")
    plot_comparison(result, save_path=save_path)
