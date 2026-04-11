"""
线性化 MPC 控制器（机械臂关节空间跟踪）

设计思路：
    在每个控制步，利用 MuJoCo 获取关节空间质量矩阵 M(q) 和
    偏置力矩 c(q,dq)（重力 + 科里奥利），对动力学方程线性化：

        M(q) * q̈ = τ - c(q, dq)
        ⇒  q̈ = M⁻¹(q) * τ - M⁻¹(q) * c(q, dq)
        ⇒  q̈ ≈ A_dyn * τ + b_dyn  （当前时刻线性化，M, c 固定）

    离散化（前向欧拉，步长 dt）：
        x[k+1] = Φ * x[k] + Γ * τ[k] + d
        x = [q; dq] ∈ R^6,  τ ∈ R^3

    滚动优化问题（预测域 N 步）：
        min₍U₎  Σₖ₌₀ᴺ⁻¹ [ Δq[k]ᵀQ Δq[k] + Δdq[k]ᵀQv Δdq[k] + τ[k]ᵀR τ[k] ]
                + Δq[N]ᵀ Qf Δq[N]
        s.t.    -τ_max ≤ τ[k] ≤ τ_max,  ∀k ∈ [0,N-1]

    其中 Δq[k] = q[k] - q_des[k]，仅滚动施加第一步 τ[0]。

    求解器：scipy.optimize.minimize（SLSQP 方法），无需额外依赖。
    计算复杂度：N=10, n_joint=3 → 30 变量, 60 约束，
    每步求解约 0.2~0.5 ms（远小于 dt=2 ms），
    通过上步解热启动（warm-start）进一步加速。

用法（作为控制器）：
    mpc = LinearMPC(
        n_joint=3, dt=0.002,
        horizon=10,
        Q=np.diag([200, 200, 150]),   # 位置误差权重
        Qv=np.diag([5, 5, 3]),        # 速度误差权重
        Qf=np.diag([400, 400, 300]),  # 终端位置误差权重
        R=np.diag([0.01, 0.01, 0.01]),# 控制量权重（防止力矩突变）
        tau_max=50.0,
    )
    mpc.reset()

    # 在仿真循环内：
    tau = mpc.compute(
        mjdata,      # MuJoCo MjData（用于读取 M, c）
        mjmodel,     # MuJoCo MjModel
        q_cur,       # 当前关节角 (3,)
        dq_cur,      # 当前关节速度 (3,)
        q_des_N,     # 后续 N+1 步参考关节角 (N+1, 3)
        dq_des_N,    # 后续 N+1 步参考关节速度 (N+1, 3)
        j_ids,       # joint id 列表 [id1, id2, id3]
    )
"""

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import block_diag

# ======================================================================
# 辅助：从 MuJoCo 读取质量矩阵和偏置力的 3×3 关节子空间版本
# ======================================================================


def _get_M_and_c(mjmodel: mujoco.MjModel, mjdata: mujoco.MjData, j_ids: list) -> tuple:
    """
    从 MuJoCo 读取关节空间质量矩阵 M (n×n) 和偏置力 c (n,)。

    参数:
        mjmodel : MuJoCo 模型（用于 mj_fullM）
        mjdata  : 当前仿真数据（已 mj_forward）
        j_ids   : 控制关节的 dof 地址列表

    返回:
        M  : (n, n) 质量矩阵，正定对称
        c  : (n,)  偏置力（重力 + 科里奥利 + 阻尼）
    """
    n = len(j_ids)
    # MuJoCo 的 qM 是稀疏存储，用 mj_fullM 展开为密集矩阵（需传 (nv, nv) 二维数组）
    M_full = np.zeros((mjmodel.nv, mjmodel.nv), dtype=np.float64)
    mujoco.mj_fullM(mjmodel, M_full, mjdata.qM)
    # 取关节子空间（j_ids 视为 dof 地址，对 3DOF 臂通常与 qpos id 一致）
    M = M_full[np.ix_(j_ids, j_ids)]

    # qfrc_bias: 重力 + 科里奥利（不含 damping，damping 在 qfrc_passive 里）
    # 这里取 qfrc_bias 作为 c（最常用的偏置组合）
    c = mjdata.qfrc_bias[j_ids].copy()
    return M, c


# ======================================================================
# LinearMPC 控制器类
# ======================================================================


class LinearMPC:
    """
    基于局部线性化的关节空间 MPC 控制器。

    每步对当前工作点执行一次线性化，将非线性机械臂近似为
    线性时变（LTV）系统，并求解滚动优化问题。

    关键特性：
        - 支持力矩硬约束（τ_max）
        - 用上步最优解热启动减少迭代次数
        - 增量式 QP 构造：批量展开预测轨迹后转化为标准 QP
        - 不依赖 casadi / osqp 等额外库，只需 scipy
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
        """
        参数:
            n_joint : 控制关节数（默认 3）
            dt      : 控制步长（秒），需与仿真 timestep 一致
            horizon : 预测域步数 N（默认 10）
            Q       : (n,n) 位置误差权重矩阵（默认 200*I）
            Qv      : (n,n) 速度误差权重矩阵（默认 5*I）
            Qf      : (n,n) 终端位置权重矩阵（默认 5*Q）
            R       : (n,n) 控制量权重矩阵（默认 0.01*I）
            tau_max : 力矩约束幅值（N·m，默认 50）
        """
        self.n = n_joint
        self.dt = dt
        self.N = horizon
        self.tau_max = tau_max

        # 默认权重（可由调用方覆盖）
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

        # 上步最优控制序列（热启动用），形状 (N*n,)
        self._u_warm = np.zeros(horizon * n_joint)

    def reset(self) -> None:
        """重置热启动缓存（开始新轨迹时调用）。"""
        self._u_warm = np.zeros(self.N * self.n)

    # ------------------------------------------------------------------
    # 核心方法
    # ------------------------------------------------------------------

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
        计算当前时刻的最优力矩指令（滚动优化，仅返回第 0 步）。

        参数:
            mjmodel  : MuJoCo MjModel
            mjdata   : 最新的 MjData（已完成 mj_forward）
            q_cur    : 当前关节角 (n,)
            dq_cur   : 当前关节速度 (n,)
            q_des_N  : 参考关节角序列 (N+1, n)，q_des_N[0] = 下一步目标
            dq_des_N : 参考关节速度序列 (N+1, n)
            j_ids    : DoF 地址列表（与 q_cur 对应）

        返回:
            tau : (n,) 最优力矩指令（已施加约束）
        """
        n, N, dt = self.n, self.N, self.dt

        # ---- 1. 获取当前工作点的 M, c ----
        M, c = _get_M_and_c(mjmodel, mjdata, j_ids)
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            # 奇异时退化到对角近似
            M_inv = np.diag(1.0 / np.diag(M))

        # ---- 2. 构造离散 LTV 系统矩阵 ----
        # 状态 x = [q; dq] ∈ R^(2n)
        # x[k+1] = Φ x[k] + Γ u[k] + d
        #   Φ = [[I, I*dt], [0, I]]
        #   Γ = [[0       ], [M_inv*dt]]
        #   d = [[0       ], [-M_inv*c*dt]]
        Phi = np.eye(2 * n)
        Phi[:n, n:] = np.eye(n) * dt
        Gamma = np.zeros((2 * n, n))
        Gamma[n:, :] = M_inv * dt  # 只取对角会更快，但用全матрица
        d_bias = np.zeros(2 * n)
        d_bias[n:] = -M_inv @ c * dt

        # ---- 3. 批量展开预测：X = A_bar * x0 + B_bar * U + f_bar ----
        # X ∈ R^((N)*2n)，U ∈ R^(N*n)
        # A_bar: (N*2n, 2n)
        # B_bar: (N*2n, N*n)（下三角 Toeplitz 结构）
        # f_bar: (N*2n,)（累积偏置）
        two_n = 2 * n
        A_bar = np.zeros((N * two_n, two_n))
        B_bar = np.zeros((N * two_n, N * n))
        f_bar = np.zeros(N * two_n)

        # 递推构造 A_bar 和  f_bar
        Phi_pow = np.eye(two_n)  # Φ^k
        f_acc = np.zeros(two_n)  # 累积偏置
        for k in range(N):
            Phi_pow = Phi_pow @ Phi if k > 0 else Phi
            A_bar[k * two_n : (k + 1) * two_n, :] = Phi_pow

            # f_bar[k] = Σ_{j=0}^{k-1} Φ^(k-1-j) * d  +  Φ^(k-1) * d
            # 等价：f_bar[k] -= A_bar[k-1] * (I-Phi)^{-1}... 更简单直接递推
            f_acc = Phi @ f_acc + d_bias
            f_bar[k * two_n : (k + 1) * two_n] = f_acc

        # 构造 B_bar（每列为 从 j+1 步到 N 步的 Γ 贡献）
        for j in range(N):
            # u[j] 对步 k (k > j) 的贡献为 Φ^(k-j-1) * Γ
            Phi_pow_local = np.eye(two_n)
            for k in range(j, N):
                if k == j:
                    B_col = Gamma
                else:
                    Phi_pow_local = Phi @ Phi_pow_local
                    B_col = Phi_pow_local @ Gamma
                B_bar[k * two_n : (k + 1) * two_n, j * n : (j + 1) * n] = B_col

        # ---- 4. 构造 QP 代价矩阵 ----
        # 代价 = Σ_{k=0}^{N-1} [q_err^T Q q_err + dq_err^T Qv dq_err + u^T R u]
        #       + q_err[N]^T Qf q_err[N]
        # 注：X 从 k=1 开始（k=0 为 x[1]，即第一步预测后的状态）

        x0 = np.concatenate([q_cur, dq_cur])  # (2n,)
        # 参考状态序列 X_ref（对 k=1..N）
        X_ref = np.zeros(N * two_n)
        for k in range(N):
            idx = (k + 1) % len(q_des_N)  # 防止索引超 horizon 时溯界
            if k + 1 < len(q_des_N):
                X_ref[k * two_n : k * two_n + n] = q_des_N[k + 1]
                X_ref[k * two_n + n : (k + 1) * two_n] = dq_des_N[k + 1]
            else:
                # 超出参考长度则用末端目标
                X_ref[k * two_n : k * two_n + n] = q_des_N[-1]
                X_ref[k * two_n + n : (k + 1) * two_n] = dq_des_N[-1]

        # 构造块对角权重矩阵 W
        W = np.zeros((N * two_n, N * two_n))
        for k in range(N):
            base = k * two_n
            if k == N - 1:
                # 终端步用 Qf
                W[base : base + n, base : base + n] = self.Qf
            else:
                W[base : base + n, base : base + n] = self.Q
            # 速度误差权重（所有步）
            W[base + n : base + two_n, base + n : base + two_n] = self.Qv

        # 控制量权重
        R_bar = block_diag(*[self.R for _ in range(N)])

        # QP 矩阵: H = B_bar^T W B_bar + R_bar
        #         f = B_bar^T W (A_bar x0 + f_bar - X_ref)
        A_x0 = A_bar @ x0 + f_bar  # 自由响应 (N*2n,)
        e_free = A_x0 - X_ref
        H_qp = B_bar.T @ W @ B_bar + R_bar
        f_qp = B_bar.T @ W @ e_free

        # ---- 5. 求解 QP（scipy SLSQP / L-BFGS-B） ----
        u_bound = [(-self.tau_max, self.tau_max)] * (N * n)

        def cost(u: np.ndarray) -> float:
            return 0.5 * float(u @ H_qp @ u) + float(f_qp @ u)

        def grad(u: np.ndarray) -> np.ndarray:
            return H_qp @ u + f_qp

        # 热启动：右移上步解，最后一步复制末端
        u0 = np.roll(self._u_warm, -n)
        u0[-n:] = self._u_warm[-n:]

        res = minimize(
            cost,
            u0,
            jac=grad,
            method="L-BFGS-B",
            bounds=u_bound,
            options={"maxiter": 50, "ftol": 1e-8},
        )

        U_opt = res.x
        self._u_warm = U_opt  # 缓存供下步热启动

        # ---- 6. 返回第 0 步控制量 ----
        tau_opt = U_opt[:n]
        # 再次截断确保不越界（防止数值误差）
        return np.clip(tau_opt, -self.tau_max, self.tau_max)


# ======================================================================
# 便捷函数：为 N 步预测准备参考轨迹切片
# ======================================================================


def get_horizon_ref(
    t: int,
    q_des: np.ndarray,
    dq_des: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    从目标轨迹中提取当前步往后 horizon 步的参考切片，
    末尾超出的部分用最后一帧填充。

    参数:
        t       : 当前时间步
        q_des   : 全轨迹参考关节角 (T, n)
        dq_des  : 全轨迹参考关节速度 (T, n)
        horizon : 预测域步数

    返回:
        q_ref   : (horizon+1, n)
        dq_ref  : (horizon+1, n)
    """
    T = len(q_des)
    end = min(t + horizon + 1, T)
    q_ref = q_des[t:end]
    dq_ref = dq_des[t:end]
    # 不足 horizon+1 步时用末尾填充
    if len(q_ref) < horizon + 1:
        pad = horizon + 1 - len(q_ref)
        q_ref = np.vstack([q_ref, np.tile(q_des[-1], (pad, 1))])
        dq_ref = np.vstack([dq_ref, np.tile(dq_des[-1], (pad, 1))])
    return q_ref, dq_ref
