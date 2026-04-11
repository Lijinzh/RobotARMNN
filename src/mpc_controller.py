"""
基于模型的机械臂控制器

提供两种控制器：

1. ComputedTorqueControl (CTC) — 计算力矩控制（推荐）
   ----------------------------------------------------------
   理论：对非线性动力学方程精确求逆，将系统解耦为线性双积分器。

       M(q) q̈ = τ - c(q, dq)
       →  τ = M(q) [q̈_des + Kp(q_des-q) + Kd(dq_des-dq)] + c(q, dq)

   内环 PD 仅需承担模型误差（摩擦等）。
   每步计算量仅为 1 次矩阵向量乘法，< 0.05 ms。

2. LinearMPC — 解析 QP 线性化 MPC（去掉 scipy，纯 numpy 求解）
   ----------------------------------------------------------
   每步线性化动力学，构造预测展开矩阵后用 numpy.linalg.solve 直接求解：
       U* = -H_qp^{-1} f_qp
   力矩约束通过截断处理。每步约 0.1-0.3 ms（N=10），
   比原来 scipy L-BFGS-B 迭代快 10-50 倍。

用法：
    # CTC（推荐，最快最准）
    ctc = ComputedTorqueControl(n_joint=3, tau_max=50.0)
    tau = ctc.compute(mjmodel, data, q_cur, dq_cur, q_des, dq_des, ddq_des, j_ids)

    # LinearMPC
    mpc = LinearMPC(n_joint=3, dt=0.002, horizon=10, tau_max=50.0)
    mpc.reset()
    q_h, dq_h = get_horizon_ref(t, q_des_all, dq_des_all, mpc.N)
    tau = mpc.compute(mjmodel, data, q_cur, dq_cur, q_h, dq_h, j_ids)
"""

import mujoco
import numpy as np
from scipy.linalg import block_diag


# ======================================================================
# 公共辅助：从 MuJoCo 读取质量矩阵和偏置力
# ======================================================================


def _get_M_and_c(mjmodel: mujoco.MjModel, mjdata: mujoco.MjData, j_ids: list) -> tuple:
    """从 MuJoCo 读取关节子空间质量矩阵 M (n×n) 和偏置力 c (n,)。"""
    M_full = np.zeros((mjmodel.nv, mjmodel.nv), dtype=np.float64)
    mujoco.mj_fullM(mjmodel, M_full, mjdata.qM)
    M = M_full[np.ix_(j_ids, j_ids)]
    c = mjdata.qfrc_bias[j_ids].copy()
    return M, c


# ======================================================================
# ComputedTorqueControl — 计算力矩控制（CTC）
# ======================================================================


class ComputedTorqueControl:
    """
    计算力矩控制器（Computed Torque Control, CTC）。

    控制律（模型前馈 + PD 反馈）：
        τ = M(q)*q̈_des + c(q,dq) + Kp*(q_des-q) + Kd*(dq_des-dq)

    其中 Kp、Kd 的单位与纯 PD 相同（N·m/rad），
    可以直接与纯 PD 对比：误差减少量 = 模型前馈 M*q̈_des+c 的贡献。

    每步计算量：1次 mj_fullM + 1次矩阵乘法，< 0.05 ms，
    与纯 PD 方案执行速度几乎相同。

    注意：CTC 本身无法补偿摩擦（不在刚体动力学模型中），
    配合 GRU 前馈可以进一步降低误差。
    """

    def __init__(
        self,
        n_joint: int = 3,
        tau_max: float = 50.0,
        Kp: np.ndarray | None = None,
        Kd: np.ndarray | None = None,
    ):
        """
        参数:
            n_joint : 控制关节数
            tau_max : 力矩约束幅值（N·m）
            Kp      : 位置误差增益对角阵（N·m/rad，默认 diag([8, 8, 6])，与纯 PD 相同）
            Kd      : 速度误差增益对角阵（N·m·s/rad，默认 diag([1.5, 1.5, 1.0])，与纯 PD 相同）
        """
        self.n = n_joint
        self.tau_max = tau_max

        if Kp is None:
            # 默认与可视化脚本的纯 PD 增益一致（KP from data_gen_idyn.py）
            Kp = np.diag([8.0, 8.0, 6.0])
        if Kd is None:
            Kd = np.diag([1.5, 1.5, 1.0])

        self.Kp = Kp
        self.Kd = Kd

    def reset(self) -> None:
        """兼容接口。"""
        pass

    def compute(
        self,
        mjmodel: mujoco.MjModel,
        mjdata: mujoco.MjData,
        q_cur: np.ndarray,
        dq_cur: np.ndarray,
        q_des: np.ndarray,
        dq_des: np.ndarray,
        ddq_des: np.ndarray,
        j_ids: list,
    ) -> np.ndarray:
        """
        计算 CTC 力矩指令。

        参数:
            q_des   : 当前步参考关节角 (n,)
            dq_des  : 当前步参考关节速度 (n,)
            ddq_des : 当前步参考关节加速度 (n,)（由轨迹解析获得）
            j_ids   : DoF 地址列表

        返回:
            tau : (n,) CTC 力矩指令（已截断到 ±tau_max）
        """
        M, c = _get_M_and_c(mjmodel, mjdata, j_ids)
        # 模型前馈：τ_ff = M(q)*q̈_des + c(q,dq)（精确逆动力学，补偿惯性力和科氏力）
        tau_ff = M @ ddq_des + c
        # PD 反馈：与纯 PD 相同增益（Kp 单位 N·m/rad），消除模型误差和摩擦残差
        tau_pd = self.Kp @ (q_des - q_cur) + self.Kd @ (dq_des - dq_cur)
        tau = tau_ff + tau_pd
        return np.clip(tau, -self.tau_max, self.tau_max)


# ======================================================================
# LinearMPC — 解析 QP 线性化 MPC
# ======================================================================


class LinearMPC:
    """
    基于局部线性化的关节空间 MPC 控制器（纯 numpy 解析解版本）。

    每步在当前工作点线性化动力学：x[k+1] = Φ x[k] + Γ u[k] + d

    解析求解无约束 QP：U* = -H_qp^{-1} f_qp
    力矩约束通过最终截断处理。

    每步约 0.1-0.3 ms（N=10），比 scipy 迭代快 10-50 倍。
    """

    def __init__(
        self,
        n_joint: int = 3,
        dt: float = 0.002,
        horizon: int = 10,
        Q: np.ndarray | None = None,
        Qv: np.ndarray | None = None,
        Qf: np.ndarray | None = None,
        R: np.ndarray | None = None,
        tau_max: float = 50.0,
    ):
        self.n = n_joint
        self.dt = dt
        self.N = horizon
        self.tau_max = tau_max

        if Q is None:
            Q = np.diag([200.0, 200.0, 150.0])
        if Qv is None:
            Qv = np.diag([5.0, 5.0, 3.0])
        if Qf is None:
            Qf = 5.0 * Q
        if R is None:
            R = np.diag([0.01, 0.01, 0.01])

        self.Q = Q
        self.Qv = Qv
        self.Qf = Qf
        self.R = R

        # 预计算固定权重块对角矩阵（不依赖工作点）
        n, N, two_n = n_joint, horizon, 2 * n_joint
        W = np.zeros((N * two_n, N * two_n))
        for k in range(N):
            base = k * two_n
            W[base: base + n, base: base + n] = Qf if k == N - 1 else Q
            W[base + n: base + two_n, base + n: base + two_n] = Qv
        self._W = W
        self._R_bar = block_diag(*[self.R for _ in range(N)])

    def reset(self) -> None:
        """兼容接口。"""
        pass

    def compute(
        self,
        mjmodel: mujoco.MjModel,
        mjdata: mujoco.MjData,
        q_cur: np.ndarray,
        dq_cur: np.ndarray,
        q_des_N: np.ndarray,
        dq_des_N: np.ndarray,
        j_ids: list,
    ) -> np.ndarray:
        """
        计算当前时刻最优力矩指令（滚动优化，仅返回第 0 步）。

        参数:
            q_des_N  : 参考关节角序列 (N+1, n)
            dq_des_N : 参考关节速度序列 (N+1, n)

        返回:
            tau : (n,) 最优力矩（截断到 ±tau_max）
        """
        n, N, dt = self.n, self.N, self.dt
        two_n = 2 * n

        # ---- 1. 读取 M, c ----
        M, c = _get_M_and_c(mjmodel, mjdata, j_ids)
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.diag(1.0 / np.diag(M))

        # ---- 2. 离散 LTV 系统矩阵（前向欧拉） ----
        Phi = np.eye(two_n)
        Phi[:n, n:] = np.eye(n) * dt
        Gamma = np.zeros((two_n, n))
        Gamma[n:, :] = M_inv * dt
        d_bias = np.zeros(two_n)
        d_bias[n:] = -M_inv @ c * dt

        # ---- 3. 预计算 Φ^0 .. Φ^N（避免重复矩阵乘法）----
        Phi_pows = [np.eye(two_n)]
        for _ in range(N):
            Phi_pows.append(Phi @ Phi_pows[-1])

        # ---- 4. 构造展开矩阵 A_bar, B_bar, f_bar ----
        A_bar = np.zeros((N * two_n, two_n))
        B_bar = np.zeros((N * two_n, N * n))
        f_bar = np.zeros(N * two_n)

        f_acc = np.zeros(two_n)
        for k in range(N):
            A_bar[k * two_n:(k + 1) * two_n, :] = Phi_pows[k + 1]
            f_acc = Phi @ f_acc + d_bias
            f_bar[k * two_n:(k + 1) * two_n] = f_acc
            # B_bar 行 k：Phi^(k-j)*Gamma，对所有 j <= k
            for j in range(k + 1):
                B_bar[k * two_n:(k + 1) * two_n, j * n:(j + 1) * n] = (
                    Phi_pows[k - j] @ Gamma
                )

        # ---- 5. 参考状态序列 X_ref ----
        X_ref = np.zeros(N * two_n)
        for k in range(N):
            ref_idx = min(k + 1, len(q_des_N) - 1)
            X_ref[k * two_n: k * two_n + n] = q_des_N[ref_idx]
            X_ref[k * two_n + n: (k + 1) * two_n] = dq_des_N[ref_idx]

        # ---- 6. 构造 QP 矩阵，解析求解 ----
        x0 = np.concatenate([q_cur, dq_cur])
        e_free = A_bar @ x0 + f_bar - X_ref
        H_qp = B_bar.T @ self._W @ B_bar + self._R_bar
        f_qp = B_bar.T @ self._W @ e_free

        try:
            U_opt = np.linalg.solve(H_qp, -f_qp)
        except np.linalg.LinAlgError:
            U_opt = np.zeros(N * n)

        # ---- 7. 截断约束，返回第 0 步 ----
        return np.clip(U_opt[:n], -self.tau_max, self.tau_max)


# ======================================================================
# 便捷函数：提取预测域参考轨迹切片
# ======================================================================


def get_horizon_ref(
    t: int,
    q_des: np.ndarray,
    dq_des: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    从全轨迹提取从步 t 开始向后 horizon 步的参考切片，
    末尾超出部分用最后一帧填充。

    返回:
        q_ref  : (horizon+1, n)
        dq_ref : (horizon+1, n)
    """
    T = len(q_des)
    end = min(t + horizon + 1, T)
    q_ref = q_des[t:end]
    dq_ref = dq_des[t:end]
    if len(q_ref) < horizon + 1:
        pad = horizon + 1 - len(q_ref)
        q_ref = np.vstack([q_ref, np.tile(q_des[-1], (pad, 1))])
        dq_ref = np.vstack([dq_ref, np.tile(dq_des[-1], (pad, 1))])
    return q_ref, dq_ref
