"""
逆动力学训练数据生成（MuJoCo 仿真）

任务设计：
    在带有 Coulomb 摩擦（frictionloss）的机械臂上跑激励轨迹，
    让 PD 控制器跟踪轨迹，记录：
        q(t)     — 实际关节角（磁编码器真值）
        dq(t)    — 实际关节速度
        tau(t)   — PD 控制器命令，作为闭环示教标签 outputs

    学习目标：
        inputs = [q_des, dq_des] → outputs 或 outputs_bg
        inputs_full = [q_des, dq_des, ddq_des] → outputs_inv
        outputs_inv 是指定加速度的逆动力学；outputs_bg 是零加速度补偿。
        outputs 是闭环 PD 示教命令，不是精确逆动力学真值。

激励轨迹设计：
    傅里叶级数叠加（3-5个不同频率），覆盖关节空间
    多条独立轨迹（不同随机相位频率），提升数据多样性
    避免关节极限（在 80% 范围内激励）

用法：
    uv run src/data_gen_idyn.py                # 默认生成 20 条轨迹
    uv run src/data_gen_idyn.py --n_traj 50    # 生成 50 条
    uv run src/data_gen_idyn.py --preview       # 播放一条轨迹预览
"""

import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ---- 路径 ----
XML_PATH = os.path.join(os.path.dirname(__file__), "arm_3dof_friction.xml")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
NPZ_PATH = os.path.join(DATA_DIR, "idyn_3dof.npz")

# ---- 仿真参数 ----
DT = 0.002  # XML 里设置的 timestep
TRAJ_DURATION = 8.0  # 每条轨迹时长（秒）
T_STEPS = int(TRAJ_DURATION / DT)  # 时间步数

# ---- 关节范围（80% 安全内激励）----
JOINT_RANGES = [
    (-0.8 * np.pi, 0.8 * np.pi),  # joint1: 底座 [-π, π]
    (-0.4 * np.pi, 0.4 * np.pi),  # joint2: 肩部 [-π/2, π/2]
    (-0.6 * np.pi, 0.6 * np.pi),  # joint3: 肘部 [-π, π]
]

# ---- PD 控制增益 ----
# 数据生成用高刚度 PD；实际跟踪误差取决于轨迹，不能视为精确逆动力学真值。
# KD 选取：临界阻尼 KD = 2*sqrt(KP*J)，J≈0.02 → KD≈5，避免速度项削波
KP_DATA = np.array([300.0, 300.0, 200.0])  # 数据生成 PD 位置增益
KD_DATA = np.array([5.0, 5.0, 4.0])  # 数据生成 PD 速度增益（临界阻尼）

# 测试/评估用软刚度（体现前馈补偿的价值）
KP = np.array([8.0, 8.0, 6.0])  # 测试 PD 位置增益
KD = np.array([1.5, 1.5, 1.0])  # 测试 PD 速度增益


def gen_excitation_traj(
    duration: float,
    dt: float,
    joint_ranges: list,
    n_freqs: int = 4,
    rng: np.random.Generator | None = None,
    amp_scale: float = 1.0,
) -> tuple:
    """
    生成多频率正弦叠加激励轨迹（3 关节）。

    参数:
        duration    : 轨迹时长（秒）
        dt          : 时间步长
        joint_ranges: 每个关节的 (min, max) 范围
        n_freqs     : 频率分量数
        rng         : 随机生成器（用于可重复性）
        amp_scale   : 幅值缩放因子（默认 1.0；可视化时可调大，如 2.0）

    返回:
        t_arr : (T,) 时间数组
        q_des : (T, 3) 期望关节角
        dq_des: (T, 3) 期望关节速度（解析微分）
        ddq_des: (T, 3) 期望关节加速度（解析二阶导）
    """
    if rng is None:
        rng = np.random.default_rng()

    t_arr = np.arange(0, duration, dt)
    T = len(t_arr)
    q_des = np.zeros((T, 3))
    dq_des = np.zeros((T, 3))
    ddq_des = np.zeros((T, 3))

    for j in range(3):
        q_min, q_max = joint_ranges[j]
        amp_max = (q_max - q_min) / 2 * 0.7  # 幅值不超过范围的 70%
        q_center = (q_max + q_min) / 2

        # 随机频率（0.2 ~ 1.5 Hz，避免太高频导致加速度过大）
        freqs = rng.uniform(0.2, 1.5, n_freqs)
        # 幅值按频率衰减（高频贡献小，避免超限）
        amps = rng.uniform(0, amp_max / n_freqs, n_freqs) / (freqs + 0.5) * 0.5 * amp_scale
        phases = rng.uniform(0, 2 * np.pi, n_freqs)

        ω = 2 * np.pi * freqs  # 角频率数组
        for i in range(n_freqs):
            q_des[:, j] += amps[i] * np.sin(ω[i] * t_arr + phases[i])
            dq_des[:, j] += amps[i] * ω[i] * np.cos(ω[i] * t_arr + phases[i])
            ddq_des[:, j] -= amps[i] * ω[i] ** 2 * np.sin(ω[i] * t_arr + phases[i])  # 解析二阶导

        # 整条轨迹同步缩放，避免仅裁剪位置后破坏解析导数的一致性。
        peak = np.max(np.abs(q_des[:, j]))
        half_range = (q_max - q_min) / 2
        if peak > half_range:
            scale = half_range / peak
            q_des[:, j] *= scale
            dq_des[:, j] *= scale
            ddq_des[:, j] *= scale

        q_des[:, j] += q_center

    return t_arr, q_des, dq_des, ddq_des.astype(np.float32)


def simulate_trajectory(
    q_des: np.ndarray,
    dq_des: np.ndarray,
    kp: np.ndarray = KP_DATA,
    kd: np.ndarray = KD_DATA,
    xml_path: str = XML_PATH,
) -> dict:
    """
    在 MuJoCo 中仿真一条轨迹，用 PD 控制器跟踪 q_des/dq_des。

    参数:
        q_des  : (T, 3) 期望关节角度
        dq_des : (T, 3) 期望关节速度
        kp, kd : PD 增益
        xml_path: MuJoCo MJCF 路径

    返回:
        dict 包含：
            q    : (T, 3) 实际关节角
            dq   : (T, 3) 实际关节速度
            tau  : (T, 3) PD 控制器命令（可能受执行器限幅影响）
            q_err: (T, 3) 跟踪误差
    """
    mjmodel = mujoco.MjModel.from_xml_path(xml_path)
    mjdata = mujoco.MjData(mjmodel)
    mujoco.mj_forward(mjmodel, mjdata)

    # 获取关节 ID
    j_ids = [
        mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i + 1}") for i in range(3)
    ]

    T = len(q_des)
    log_q = np.zeros((T, 3), dtype=np.float32)
    log_dq = np.zeros((T, 3), dtype=np.float32)
    log_tau = np.zeros((T, 3), dtype=np.float32)

    # 初始化到第一个期望位置 + 速度（避免速度初始不连续引起瞬态震荡）
    for i, jid in enumerate(j_ids):
        mjdata.qpos[jid] = q_des[0, i]
        mjdata.qvel[jid] = dq_des[0, i]  # 匹配期望初速度
    mujoco.mj_forward(mjmodel, mjdata)

    for t in range(T):
        # 读取当前状态
        q_cur = np.array([mjdata.qpos[jid] for jid in j_ids])
        dq_cur = np.array([mjdata.qvel[jid] for jid in j_ids])

        # PD 控制律：tau = kp*(q_des - q) + kd*(dq_des - dq)
        tau = kp * (q_des[t] - q_cur) + kd * (dq_des[t] - dq_cur)

        # 写控制量并步进
        mjdata.ctrl[:3] = tau
        mujoco.mj_step(mjmodel, mjdata)

        # 记录
        log_q[t] = q_cur
        log_dq[t] = dq_cur
        log_tau[t] = tau

    return {
        "q": log_q,
        "dq": log_dq,
        "tau": log_tau,
        "q_err": log_q - q_des,
    }


def compute_inverse_dynamics_labels(
    q_des: np.ndarray,
    dq_des: np.ndarray,
    ddq_des: np.ndarray,
    xml_path: str = XML_PATH,
) -> tuple:
    """
    计算两种逆动力学力矩标签。

    给定期望轨迹 (q_des, dq_des, ddq_des)，使用 MuJoCo 的逆动力学：
        tau_full = mj_inverse(q_des, dq_des, ddq_des).qfrc_inverse
        tau_bg   = mj_inverse(q_des, dq_des, 0).qfrc_inverse

    力平衡为 tau = M*ddq + bias - passive - constraint。
    阻尼属于 passive；Coulomb frictionloss 由约束求解器处理。
    当前 XML 的电机 gear=1，且没有额外外力，因此结果直接对应电机力矩。

    说明：
        - tau_full 的完整状态输入需为 [q_des, dq_des, ddq_des]（9D）。
        - tau_bg 表示当前位置、速度下保持零加速度所需的力矩，输入为 6D。
        - 约束力可能随指定加速度变化，不能一般化为 tau_full = M*ddq + tau_bg。
        - 不在设置 qacc 后调用 mj_forward，避免期望加速度被前向动力学覆盖。

    参数:
        q_des   : (T, 3) 期望关节角
        dq_des  : (T, 3) 期望关节速度
        ddq_des : (T, 3) 期望关节加速度（gen_excitation_traj 解析计算）
    返回:
        tau_full : (T, 3) 指定加速度的逆动力学力矩
        tau_bg   : (T, 3) 零加速度补偿力矩
    """
    mjmodel = mujoco.MjModel.from_xml_path(xml_path)
    mjdata = mujoco.MjData(mjmodel)

    j_ids = [
        mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i + 1}") for i in range(3)
    ]
    qpos_ids = mjmodel.jnt_qposadr[j_ids]
    dof_ids = mjmodel.jnt_dofadr[j_ids]
    T = len(q_des)
    tau_full = np.zeros((T, 3), dtype=np.float32)
    tau_bg = np.zeros((T, 3), dtype=np.float32)

    for t in range(T):
        mjdata.qpos[qpos_ids] = q_des[t]
        mjdata.qvel[dof_ids] = dq_des[t]

        # 指定加速度，由逆动力学同时处理惯性、偏置、被动力和约束力。
        mjdata.qacc[dof_ids] = ddq_des[t]
        mujoco.mj_inverse(mjmodel, mjdata)
        tau_full[t] = mjdata.qfrc_inverse[dof_ids]

        # 在相同位置、速度下单独求零加速度补偿，包含 frictionloss。
        mjdata.qacc[dof_ids] = 0.0
        mujoco.mj_inverse(mjmodel, mjdata)
        tau_bg[t] = mjdata.qfrc_inverse[dof_ids]

    return tau_full, tau_bg


def gen_idyn_dataset(
    n_traj: int = 20,
    duration: float = TRAJ_DURATION,
    n_freqs: int = 4,
    seed: int = 42,
    xml_path: str = XML_PATH,
) -> None:
    """
    批量生成逆动力学数据集，保存为 .npz。

    文件结构（不同轨迹以 traj_id 区分，训练/测试应按轨迹划分）：
        inputs     : (N_total, 6) — [q_des, dq_des]
        inputs_full: (N_total, 9) — [q_des, dq_des, ddq_des]
        outputs    : (N_total, 3) — PD 示教命令
        outputs_inv: (N_total, 3) — 指定加速度的逆动力学力矩
        outputs_bg : (N_total, 3) — 零加速度补偿力矩
        ddq_des    : (N_total, 3) — 期望加速度
        label_version: 2         — 修正加速度覆盖、阻尼符号和摩擦遗漏

    输出路径为 NPZ_PATH；重新生成前应先归档旧数据。
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[data_gen_idyn] 生成 {n_traj} 条激励轨迹，每条 {duration}s ({int(duration / DT)} 步)")
    print(f"  XML: {xml_path}")
    print(f"  总样本数（约）: {n_traj * int(duration / DT):,}")

    rng = np.random.default_rng(seed)
    t0 = time.time()

    all_q, all_dq, all_tau, all_traj_id, all_q_des, all_dq_des = [], [], [], [], [], []
    all_ddq_des = []
    all_tau_inv = []  # 完整逆动力学标签（含惯性项 M*ddq）
    all_tau_bg = []  # 零加速度补偿标签，依赖位置和速度

    for i in range(n_traj):
        _, q_des, dq_des, ddq_des = gen_excitation_traj(
            duration, DT, JOINT_RANGES, n_freqs=n_freqs, rng=rng
        )
        result = simulate_trajectory(q_des, dq_des, xml_path=xml_path)

        # 计算两种精确标签
        tau_inv, tau_bg = compute_inverse_dynamics_labels(q_des, dq_des, ddq_des, xml_path=xml_path)

        T = len(result["q"])
        all_q.append(result["q"])
        all_dq.append(result["dq"])
        all_tau.append(result["tau"])
        all_tau_inv.append(tau_inv[:T])
        all_tau_bg.append(tau_bg[:T])
        all_traj_id.append(np.full(T, i, dtype=np.int32))
        all_q_des.append(q_des[:T].astype(np.float32))
        all_dq_des.append(dq_des[:T].astype(np.float32))
        all_ddq_des.append(ddq_des[:T].astype(np.float32))

        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            q_err_deg = np.degrees(np.abs(result["q_err"])).mean()
            tau_diff = np.abs(tau_bg[:T] - result["tau"]).mean()
            print(
                f"  轨迹 {i + 1:3d}/{n_traj} | "
                f"跟踪误差均值={q_err_deg:.2f}° | "
                f"PD↔bg 差={tau_diff:.4f} N·m | "
                f"耗时 {elapsed:.1f}s"
            )

    # 拼接所有轨迹
    q_all = np.concatenate(all_q, axis=0)
    dq_all = np.concatenate(all_dq, axis=0)
    tau_all = np.concatenate(all_tau, axis=0)  # PD 示教命令
    tau_inv_all = np.concatenate(all_tau_inv, axis=0)  # 完整逆动力学，输入需包含加速度
    tau_bg_all = np.concatenate(all_tau_bg, axis=0)  # 零加速度补偿
    tid_all = np.concatenate(all_traj_id)
    qd_all = np.concatenate(all_q_des, axis=0)
    dqd_all = np.concatenate(all_dq_des, axis=0)
    ddqd_all = np.concatenate(all_ddq_des, axis=0)

    # inputs：[q_des, dq_des] — 6D 前馈模型输入
    X_des = np.concatenate([qd_all, dqd_all], axis=-1).astype(np.float32)  # (N, 6)
    X_full = np.concatenate([X_des, ddqd_all], axis=-1).astype(np.float32)  # (N, 9)
    X_actual = np.concatenate([q_all, dq_all], axis=-1).astype(np.float32)  # (N, 6)

    tau_bg_diff = np.abs(tau_bg_all - tau_all).mean()
    print(f"\n  总样本: {len(X_des):,}")
    print(f"  PD 标签 vs bg 标签 平均差异: {tau_bg_diff:.4f} N·m")
    print(f"  bg 标签 std/joint: {tau_bg_all.std(0)}")
    print(f"  bg 标签 |max|/joint: {np.abs(tau_bg_all).max(0)}")

    np.savez(
        NPZ_PATH,
        inputs=X_des,  # (N, 6) [q_des, dq_des] — 推荐推理输入
        inputs_full=X_full,  # (N, 9) [q_des, dq_des, ddq_des] — 完整逆动力学输入
        inputs_actual=X_actual,  # (N, 6) [q_actual, dq_actual] — 备用
        outputs=tau_all,  # (N, 3) PD 示教命令，不是精确逆动力学真值
        outputs_inv=tau_inv_all,  # (N, 3) 指定加速度的逆动力学力矩
        outputs_bg=tau_bg_all,  # (N, 3) 零加速度补偿力矩
        q=q_all,
        dq=dq_all,
        q_des=qd_all,
        dq_des=dqd_all,
        ddq_des=ddqd_all,
        traj_id=tid_all,
        n_traj=np.int32(n_traj),
        dt=np.float32(DT),
        label_version=np.int32(2),
        seed=np.int64(seed),
        mujoco_version=np.array(mujoco.__version__),
    )
    print(f"  已保存: {NPZ_PATH}  ({time.time() - t0:.1f}s)")


def preview_trajectory(seed: int = 0, xml_path: str = XML_PATH) -> None:
    """
    用 MuJoCo Viewer 可视化一条激励轨迹（调试用）
    """
    rng = np.random.default_rng(seed)
    _, q_des, dq_des, _ = gen_excitation_traj(TRAJ_DURATION, DT, JOINT_RANGES, rng=rng)

    mjmodel = mujoco.MjModel.from_xml_path(xml_path)
    mjdata = mujoco.MjData(mjmodel)
    mujoco.mj_forward(mjmodel, mjdata)

    j_ids = [
        mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i + 1}") for i in range(3)
    ]

    T = len(q_des)
    t = 0

    print("[preview] 播放激励轨迹（关闭窗口退出）")
    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -30
        viewer.cam.distance = 1.0
        while viewer.is_running() and t < T:
            q_cur = np.array([mjdata.qpos[jid] for jid in j_ids])
            dq_cur = np.array([mjdata.qvel[jid] for jid in j_ids])
            tau = KP * (q_des[t] - q_cur) + KD * (dq_des[t] - dq_cur)
            mjdata.ctrl[:3] = tau
            mujoco.mj_step(mjmodel, mjdata)
            t += 1
            viewer.sync()
            time.sleep(DT)


# =====================================================================
# 主入口
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="逆动力学数据生成")
    parser.add_argument("--n_traj", type=int, default=20, help="轨迹条数（默认 20）")
    parser.add_argument("--duration", type=float, default=TRAJ_DURATION, help="每条轨迹时长（秒）")
    parser.add_argument("--n_freqs", type=int, default=4, help="激励轨迹频率分量数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", action="store_true", help="可视化一条轨迹后退出")
    args = parser.parse_args()

    if args.preview:
        preview_trajectory(seed=args.seed)
    else:
        gen_idyn_dataset(
            n_traj=args.n_traj,
            duration=args.duration,
            n_freqs=args.n_freqs,
            seed=args.seed,
        )
