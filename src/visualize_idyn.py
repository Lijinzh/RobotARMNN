"""
逆动力学补偿效果对比可视化

对比三种控制方案在相同测试轨迹上的跟踪误差：
    1. 纯 PD 控制          — 无摩擦补偿（基线）
    2. PD + MLP 前馈补偿   — 无记忆模型，学不到迟滞方向
    3. PD + ESN 前馈补偿   — 递归记忆，能感知运动历史

可视化内容：
    (a) 关节角跟踪误差曲线（3 关节，时域对比）
    (b) 力矩预测误差（ESN vs MLP，时域对比）
    (c) MAE 柱状图（定量结论）
    (d) 迟滞相图：τ vs dq（展示 ESN 学到了迟滞环）

用法：
    uv run src/visualize_idyn.py               # 绘图窗口
    uv run src/visualize_idyn.py --save        # 保存为 export/idyn_comparison.png
    uv run src/visualize_idyn.py --mujoco      # 同时打开 MuJoCo 实时可视化
"""

import argparse
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
) -> dict:
    """
    运行闭环仿真，收集各方案的跟踪轨迹。

    控制律：
        纯 PD：     τ = KP*(q_des-q) + KD*(dq_des-dq)
        MPC：       滚动预测优化（预测域 N 步，解析 QP）
        MPC+GRU：   MPC 基础上叠加 GRU 摩擦补偿前馈
        FF 方案：   τ = τ_ff(q_des, dq_des) + KP*(q_des-q) + KD*(dq_des-dq)
    """
    mjmodel = mujoco.MjModel.from_xml_path(xml_path)

    def make_data():
        d = mujoco.MjData(mjmodel)
        mujoco.mj_forward(mjmodel, d)
        return d

    j_ids = [
        mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i + 1}") for i in range(3)
    ]

    T = len(q_des)
    device = torch.device("cpu")

    configs = {"pd": make_data()}
    # MPC 方案（滚动预测优化，无 NN 前馈）
    mpc_ctrl = None
    mpc_gru_ctrl = None
    if use_mpc:
        configs["mpc"] = make_data()
        mpc_ctrl = LinearMPC(n_joint=3, dt=DT, horizon=mpc_horizon)
    if use_mpc_gru and gru_model is not None:
        configs["mpc_gru"] = make_data()
        mpc_gru_ctrl = LinearMPC(n_joint=3, dt=DT, horizon=mpc_horizon)
        H_mpc_gru = gru_model.init_hidden(batch_size=1).to(device)
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

    for d in configs.values():
        for i, jid in enumerate(j_ids):
            d.qpos[jid] = q_des[0, i]
            d.qvel[jid] = dq_des[0, i]  # 匹配初始速度，避免瞬态误差
        mujoco.mj_forward(mjmodel, d)

    logs = {k: np.zeros((T, 3), dtype=np.float32) for k in configs}
    tau_logs = {k: np.zeros((T, 3), dtype=np.float32) for k in configs if k not in ("pd", "mpc")}

    if show_viewer:
        # 创建独立的渲染专用 data，与仿真 data 完全隔离
        # 直接绑定仿真 data 会导致 viewer.sync() 内部的 mj_forward 篡改仿真状态
        viewer_data = mujoco.MjData(mjmodel)
        for i, jid in enumerate(j_ids):
            viewer_data.qpos[jid] = q_des[0, i]
        mujoco.mj_forward(mjmodel, viewer_data)
        viewer_display_key = next(reversed(configs))  # 跟随精度最好的方案显示
        viewer_ctx = mujoco.viewer.launch_passive(mjmodel, viewer_data)
        viewer_ctx.__enter__()
    else:
        viewer_data = None
        viewer_display_key = None
        viewer_ctx = None

    for t in range(T):
        for key, d in configs.items():
            q_cur = np.array([d.qpos[jid] for jid in j_ids])
            dq_cur = np.array([d.qvel[jid] for jid in j_ids])
            logs[key][t] = q_cur

            if key == "pd":
                tau = KP * (q_des[t] - q_cur) + KD * (dq_des[t] - dq_cur)

            elif key == "mpc":
                # 纯 MPC：滚动预测优化，无 NN 前馈
                # 每步在当前工作点线性化动力学，预测 N 步，解析 QP
                q_h, dq_h = get_horizon_ref(t, q_des, dq_des, mpc_ctrl.N)
                mujoco.mj_forward(mjmodel, d)
                tau = mpc_ctrl.compute(mjmodel, d, q_cur, dq_cur, q_h, dq_h, j_ids)

            elif key == "mpc_gru":
                # MPC + GRU 前馈：MPC 优化得到动力学最优轨迹，GRU 补偿残差摩擦
                q_h, dq_h = get_horizon_ref(t, q_des, dq_des, mpc_gru_ctrl.N)
                mujoco.mj_forward(mjmodel, d)
                tau_mpc = mpc_gru_ctrl.compute(mjmodel, d, q_cur, dq_cur, q_h, dq_h, j_ids)
                x_t = torch.tensor(np.concatenate([q_des[t], dq_des[t]])[None], dtype=torch.float32)
                with torch.no_grad():
                    tau_ff_t, H_mpc_gru = gru_model.forward_step(x_t, H_mpc_gru)
                tau_ff = tau_ff_t.squeeze(0).numpy()
                tau = tau_mpc + tau_ff
                tau_logs[key][t] = tau_ff

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
                    else:
                        tau_ff = np.zeros(3)

                tau = tau_ff + tau_pd_corr
                tau_logs[key][t] = tau_ff

            d.ctrl[:3] = tau
            mujoco.mj_step(mjmodel, d)

        if show_viewer and viewer_ctx is not None:
            # 将跟踪方案的最新状态复制到渲染 data，不影响仿真 data
            d_show = configs[viewer_display_key]
            for jid in j_ids:
                viewer_data.qpos[jid] = d_show.qpos[jid]
                viewer_data.qvel[jid] = d_show.qvel[jid]
            mujoco.mj_forward(mjmodel, viewer_data)
            viewer_ctx.sync()
            time.sleep(DT * 0.5)

    if show_viewer and viewer_ctx is not None:
        viewer_ctx.__exit__(None, None, None)

    return {
        "pd": logs["pd"],
        "mpc": logs.get("mpc"),
        "mpc_gru": logs.get("mpc_gru"),
        "mlp": logs.get("mlp"),
        "esn": logs.get("esn"),
        "lstm": logs.get("lstm"),
        "gru": logs.get("gru"),
        "q_des": q_des,
        "tau_mpc_gru": tau_logs.get("mpc_gru", np.zeros((T, 3))),
        "tau_mlp": tau_logs.get("mlp", np.zeros((T, 3))),
        "tau_esn": tau_logs.get("esn", np.zeros((T, 3))),
        "tau_lstm": tau_logs.get("lstm", np.zeros((T, 3))),
        "tau_gru": tau_logs.get("gru", np.zeros((T, 3))),
    }


# =====================================================================
# 绘图
# =====================================================================


def plot_comparison(result: dict, save_path: str | None = None) -> None:
    """
    绘制四子图对比图：
        (a) 所有关节平均跟踪误差（时域曲线，5 方案）
        (b) 各关节误差柱状图（定量）
        (c) joint1 前馈力矩时序（最后 500 步）
        (d) joint1 摩擦迟滞相图：dq vs τ
    """
    q_des = result["q_des"]
    T = len(q_des)
    t_arr = np.arange(T) * DT

    # 定义比较方案：key -> (颜色, 标签)
    SCHEMES = [
        ("pd", "#888888", "纯 PD（无补偿）"),
        ("mpc", "#9C27B0", "MPC（滚动预测优化）"),
        ("mpc_gru", "#E91E63", "MPC + GRU 前馈"),
        ("mlp", "#2196F3", "PD + MLP 前馈"),
        ("esn", "#F44336", "PD + ESN 前馈（有记忆）"),
        ("lstm", "#4CAF50", "PD + LSTM 前馈（门控记忆）"),
        ("gru", "#FF9800", "PD + GRU 前馈（轻量门控）"),
    ]

    errs = {}
    for key, _, _ in SCHEMES:
        if result.get(key) is not None:
            errs[key] = np.degrees(np.abs(result[key] - q_des))

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "控制方案对比：PD / MPC / MPC+GRU / PD+MLP / PD+ESN / LSTM / GRU",
        fontsize=22,
        fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

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
    ax0.legend(fontsize=14)
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
    ax1.legend(fontsize=14)
    ax1.grid(True, alpha=0.3, axis="y")

    # ---- (c) 前馈力矩时序（joint1，最后 500 步） ----
    ax2 = fig.add_subplot(gs[1, 0])
    t_show = min(500, T)
    t_slice = slice(T - t_show, T)
    t_arr_s = t_arr[t_slice]
    tau_keys = [
        ("mpc_gru", "#E91E63", "MPC+GRU 前馈分量"),
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
    ax2.set_title("(c) Joint1 前馈力矩预测（最后 500 步）")
    ax2.legend(fontsize=14)
    ax2.grid(True, alpha=0.3)

    # ---- (d) 迟滞相图：dq vs τ_ff（LSTM/GRU 应展现更清晰的迟滞环） ----
    ax3 = fig.add_subplot(gs[1, 1])
    step = 2
    # 画 MLP/ESN 低对比度，突出 GRU 系列
    for key, color, label in [
        ("mpc_gru", "#E91E63", "MPC+GRU 摩擦分量"),
        ("mlp", "#AAAAFF", "MLP"),
        ("esn", "#FFAAAA", "ESN"),
        ("lstm", "#4CAF50", "LSTM"),
        ("gru", "#FF9800", "GRU"),
    ]:
        tau_arr = result.get(f"tau_{key}")
        if tau_arr is not None and tau_arr.any() and result.get(key) is not None:
            dq_arr = np.gradient(result[key][:, 0], DT)
            ax3.scatter(dq_arr[::step], tau_arr[::step, 0], c=color, s=1, alpha=0.5, label=label)
    ax3.set_xlabel("关节角速度 dq₁ (rad/s)")
    ax3.set_ylabel("前馈力矩 τ₁ (N·m)")
    ax3.set_title("(d) 摩擦迟滞相图（dq₁ vs τ₁）\nLSTM/GRU 应输出更清晰的迟滞环")
    ax3.legend(fontsize=14, markerscale=6)
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
        ("pd", "纯 PD（无补偿）"),
        ("mpc", "MPC（滚动预测优化）"),
        ("mpc_gru", "MPC + GRU 前馈"),
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
        choices=["mlp", "esn", "lstm", "gru", "mpc", "mpc_gru"],
        default=["mlp", "esn", "lstm", "gru", "mpc", "mpc_gru"],
        help="指定参与对比的方案（默认全部），例：--model gru mpc mpc_gru",
    )
    parser.add_argument(
        "--mpc_horizon",
        type=int,
        default=10,
        help="MPC 预测域步数（默认 10）",
    )
    args = parser.parse_args()
    enabled_models = set(args.model)

    # ---- 判断是否启用 MPC ----
    use_mpc = "mpc" in enabled_models
    use_mpc_gru = "mpc_gru" in enabled_models

    # ---- 加载模型 ----
    mlp_model, esn_model, lstm_model, gru_model = None, None, None, None

    for ckpt_path, model_name, load_fn in [
        (CKPT_MLP, "MLP", lambda p: IDynMLP.load(p)),
        (CKPT_ESN, "ESN", lambda p: IDynESN.load(p)),
        (CKPT_LSTM, "LSTM", lambda p: IDynLSTM.load(p)),
        (CKPT_GRU, "GRU", lambda p: IDynGRU.load(p)),
    ]:
        # mpc_gru 需要 gru 权重
        need_gru_for_mpc = (model_name == "GRU") and use_mpc_gru
        if model_name.lower() not in enabled_models and not need_gru_for_mpc:
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
        or use_mpc_gru
        or any(m is not None for m in [mlp_model, esn_model, lstm_model, gru_model])
    )
    if not has_any:
        print("\n[vis] 请先训练模型：uv run src/train_idyn.py --model all")
        sys.exit(1)

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
    n_mpc = int(use_mpc) + int(use_mpc_gru)
    print(f"[vis] 运行闭环仿真（共 {1 + n_nn + n_mpc} 方案，每方案独立仿真）...")
    if use_mpc:
        print(f"[vis] MPC 预测域 = {args.mpc_horizon} 步（{args.mpc_horizon * DT * 1000:.1f} ms），每步内部解析 QP优化")
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
        use_mpc_gru=use_mpc_gru,
        mpc_horizon=args.mpc_horizon,
    )

    # ---- 结果输出 ----
    print_summary(result)

    # 默认自动保存，除非显式传入 --no-save
    save_path = None if args.no_save else os.path.join(EXPORT_DIR, "idyn_comparison.png")
    if save_path:
        print(f"[vis] 图片将保存至: {save_path}")
    plot_comparison(result, save_path=save_path)
