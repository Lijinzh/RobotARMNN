"""
神经网络模型定义

模型列表：
  IKNN_MLP   — 通用逆运动学 MLP（2-DOF / 3-DOF 均可用）
                 2-DOF: input_dim=2, output_dim=4
                 3-DOF: input_dim=3, output_dim=6
  IKNN_ESN   — 增强 ESN：随机储备池特征提取 + MLP 读出网络
                 相比纯线性读出（岭回归）新增 MLP 非线性层，
                 消除线性映射在 IK 边界区域失效导致的 max_error≈180° 问题
  IDynMLP    — 逆动力学 MLP（无记忆基线）：[q(t),dq(t)] → τ(t)
  IDynESN    — 逆动力学递归 ESN（有时序记忆）
                 H(t) = tanh(Win@x(t) + Wres@H(t-1))
                 τ(t) = readout(H(t))
                 Wres 谱半径 < 1 保证 Echo State Property（回声状态条件）
                 迟滞摩擦必须知道"从哪个方向来"，ESN 天然记住运动历史
  CurrentNN  — 电流控制 MLP（供 ESP32 部署）

sin/cos 输出设计：
  所有 IK 模型输出 [sin θ, cos θ] 而非原始角度，
  消除 atan2(Y,X) 在 ±π 处的不连续跳变。
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# =====================================================================
# IKNN_MLP — 通用逆运动学多层感知机
# =====================================================================


class IKNN_MLP(nn.Module):
    """
    逆运动学 MLP，支持 2-DOF 和 3-DOF。

    2-DOF 平面臂：input_dim=2 (XY)，output_dim=4 [sinθ1,cosθ1,sinθ2,cosθ2]
    3-DOF 空间臂：input_dim=3 (XYZ)，output_dim=6 [sinθ1,cosθ1,...,sinθ3,cosθ3]

    输入归一化：
      调用 set_input_stats(X_train) 后，forward 会自动对输入做零均值单位方差归一化。
      3-DOF 因 Z 轴有 L0=0.20m 偏置，输入三维分布不对齐，归一化是提升精度的关键。
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 256,
        output_dim: int = 4,
        n_layers: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_joints = output_dim // 2

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

        # 输入归一化参数（set_input_stats 时写入；默认 mean=0, std=1 为恒等变换）
        self.register_buffer("x_mean", torch.zeros(input_dim))
        self.register_buffer("x_std", torch.ones(input_dim))

    def set_input_stats(self, X: Tensor) -> None:
        """根据训练数据设置输入归一化参数（训练前调用一次）。"""
        self.x_mean.copy_(X.mean(0).to(self.x_mean.device))
        self.x_std.copy_(X.std(0).clamp(min=1e-8).to(self.x_std.device))

    def forward(self, x: Tensor) -> Tensor:
        """(batch, input_dim) → (batch, output_dim) sin/cos 表示

        始终应用输入归一化。未调用 set_input_stats 时，x_mean=0、x_std=1，归一化为恒等变换。
        """
        x = (x - self.x_mean) / self.x_std
        return self.net(x)

    def predict_angles(self, x: Tensor) -> Tensor:
        """直接返回 (batch, n_joints) 角度（弧度）"""
        sincos = self.forward(x)
        return torch.stack(
            [torch.atan2(sincos[:, 2 * i], sincos[:, 2 * i + 1]) for i in range(self.n_joints)],
            dim=-1,
        )

    def save(self, path: str) -> None:
        """保存至 .pt（含架构配置 + 权重，与 IKNN_ESN 格式一致）。"""
        n_layers = (len(self.net) - 1) // 2
        torch.save(
            {
                "config": {
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "output_dim": self.output_dim,
                    "n_layers": n_layers,
                },
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location=None) -> "IKNN_MLP":
        """从 .pt 文件恢复（支持新格式 dict，也向后兼容仅 state_dict 的旧格式）。"""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(ckpt, dict) and "config" in ckpt:
            model = cls(**ckpt["config"])
            model.load_state_dict(ckpt["state_dict"])
        else:
            # 旧格式：仅 state_dict，使用默认架构参数
            input_dim = next(iter(ckpt.values())).shape[-1] if ckpt else 2
            model = cls(input_dim=2, hidden_dim=256, output_dim=4)
            model.load_state_dict(ckpt)
        return model


# =====================================================================
# IKNN_ESN — 增强型 ESN（随机储备池 + MLP 读出）
# =====================================================================


class IKNN_ESN(nn.Module):
    """
    增强型 ESN：随机储备池特征提取 + MLP 读出网络。

    架构：
      输入 X (N, input_dim)
        ↓  输入归一化（set_input_stats 时写入，固定）
        ↓  Win @ X + b_in  [固定随机权重，不训练]
        ↓  tanh  → 储备池激活 H (N, reservoir_dim)
        ↓  Linear(reservoir_dim, readout_hidden) → Tanh → Linear(readout_hidden, output_dim)
        ↓  输出 sincos (N, output_dim)
      θ = atan2(sin, cos) 还原角度

    训练策略（4090 优化）：
      1. set_input_stats(X_train)：计算并存储归一化参数
      2. precompute_features(X_all)：一次性批量计算全量 H，储备池固定无需重复算
      3. 在 (H_batch → sincos) 上训练 readout MLP，每步仅小网络反向传播
    """

    def __init__(
        self,
        input_dim: int = 2,
        reservoir_dim: int = 512,
        readout_hidden: int = 128,
        output_dim: int = 4,
        input_scaling: float = 1.0,
        seed: int = 0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.reservoir_dim = reservoir_dim
        self.output_dim = output_dim
        self.n_joints = output_dim // 2

        rng = np.random.default_rng(seed)
        Win = rng.uniform(-input_scaling, input_scaling, (reservoir_dim, input_dim)).astype(
            np.float32
        )
        b_in = rng.uniform(-0.1, 0.1, (reservoir_dim,)).astype(np.float32)
        self.register_buffer("Win", torch.from_numpy(Win))
        self.register_buffer("b_in", torch.from_numpy(b_in))

        # 输入归一化参数（fit 阶段写入，推理阶段只读）
        self.register_buffer("x_mean", torch.zeros(input_dim))
        self.register_buffer("x_std", torch.ones(input_dim))

        # MLP 读出网络：唯一可训练部分
        self.readout = nn.Sequential(
            nn.Linear(reservoir_dim, readout_hidden),
            nn.Tanh(),
            nn.Linear(readout_hidden, output_dim),
        )

    def set_input_stats(self, X: Tensor) -> None:
        """根据训练数据计算输入归一化参数（训练前调用一次）"""
        self.x_mean.copy_(X.mean(0))
        self.x_std.copy_(X.std(0).clamp(min=1e-8))

    def precompute_features(self, X: Tensor) -> Tensor:
        """
        预计算储备池激活 H（固定无梯度），高效训练用。

        参数:
            X: (N, input_dim) GPU 张量
        返回:
            H: (N, reservoir_dim)
        """
        with torch.no_grad():
            X_n = (X - self.x_mean) / self.x_std
            return torch.tanh(X_n @ self.Win.T + self.b_in)

    def forward(self, x: Tensor) -> Tensor:
        """完整前向传播：坐标 → sin/cos"""
        return self.readout(self.precompute_features(x))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测关节角度 (N, n_joints) 弧度，numpy 接口"""
        self.eval()
        device = self.x_mean.device
        with torch.no_grad():
            sincos = self.forward(torch.from_numpy(X.astype(np.float32)).to(device)).cpu().numpy()
        return np.stack(
            [np.arctan2(sincos[:, 2 * i], sincos[:, 2 * i + 1]) for i in range(self.n_joints)],
            axis=-1,
        ).astype(np.float32)

    def save(self, path: str) -> None:
        """保存到 .pt（含架构配置 + 权重）"""
        torch.save(
            {
                "config": {
                    "input_dim": self.input_dim,
                    "reservoir_dim": self.reservoir_dim,
                    "readout_hidden": self.readout[0].out_features,
                    "output_dim": self.output_dim,
                },
                "state_dict": self.state_dict(),
            },
            path,
        )
        print(f"[IKNN_ESN] 已保存至: {path}")

    @classmethod
    def load(cls, path: str) -> "IKNN_ESN":
        """从 .pt 文件恢复"""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        esn = cls(**ckpt["config"])
        esn.load_state_dict(ckpt["state_dict"])
        print(f"[IKNN_ESN] 已从 {path} 加载")
        return esn


# =====================================================================
# CurrentNN — 电流控制 MLP
# =====================================================================


class CurrentNN(nn.Module):
    """
    电流控制 MLP（ESP32 部署）。

    输入:  [θ1, θ2, dθ1, dθ2, θ1_des, θ2_des]
    输出:  [I1, I2]（安培）
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 256,
        output_dim: int = 2,
        n_layers: int = 3,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# =====================================================================
# IDynMLP — 逆动力学 MLP（无记忆基线）
# =====================================================================


class IDynMLP(nn.Module):
    """
    逆动力学 MLP — 无记忆对照组。

    只看当前帧 [q(t), dq(t)] 预测所需关节力矩 τ(t)。
    无法建模摩擦迟滞（不知道从哪个方向运动而来），作为 ESN 的对照基线。

    输入: (batch, n_joints*2)  — [q1,q2,q3, dq1,dq2,dq3]
    输出: (batch, n_joints)    — [τ1, τ2, τ3]
    """

    def __init__(
        self,
        n_joints: int = 3,
        hidden_dim: int = 256,
        n_layers: int = 4,
    ):
        super().__init__()
        self.n_joints = n_joints
        input_dim = n_joints * 2  # [q, dq]
        output_dim = n_joints  # [τ]

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

        # 输入归一化（q 和 dq 量纲不同，归一化是必要的）
        self.register_buffer("x_mean", torch.zeros(input_dim))
        self.register_buffer("x_std", torch.ones(input_dim))

    def set_input_stats(self, X: Tensor) -> None:
        """根据训练数据设置归一化参数（训练前调用一次）"""
        self.x_mean.copy_(X.mean(0))
        self.x_std.copy_(X.std(0).clamp(min=1e-8))

    def forward(self, x: Tensor) -> Tensor:
        """(batch, n_joints*2) → (batch, n_joints)"""
        return self.net((x - self.x_mean) / self.x_std)

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": {
                    "n_joints": self.n_joints,
                    "hidden_dim": self.net[0].out_features,
                    "n_layers": (len(self.net) - 1) // 2,
                },
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location=None) -> "IDynMLP":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model


# =====================================================================
# IDynESN — 逆动力学递归 ESN（有时序记忆）
# =====================================================================


class IDynESN(nn.Module):
    """
    递归 ESN 逆动力学模型 — 核心对照实验组。

    储层动态（真递归，物理器件时分复用对应此结构）：
        H(t) = tanh(Win @ x(t) + Wres @ H(t-1) + b_in)
        τ(t) = readout(H(t))

    为什么比 MLP 更适合逆动力学（含摩擦迟滞）：
        摩擦迟滞取决于前一步运动方向，ESN 的 H(t) 记住了这段历史，
        MLP 只看 x(t) = [q(t), dq(t)]，无法区分"从正方向来"还是"从负方向来"。

    关键超参：
        spectral_radius < 1  保证 Echo State Property（储层状态唯一由输入决定）
        spectral_radius ≈ 0.9 记忆较长；≈ 0.5 遗忘较快
        sparsity ≈ 0.9       Wres 90% 稀疏，接近真实神经网络连接稀疏性

    输入: x(t) = [q(t), dq(t)]  (n_joints*2 维，逐步输入)
    输出: τ(t)                   (n_joints 维)

    训练策略（固定储层，只训练读出）：
        precompute_sequence_features(X_seq) 递归无梯度预计算全序列 H
        → 在 (H → τ) 上训练 readout MLP
    """

    def __init__(
        self,
        n_joints: int = 3,
        reservoir_dim: int = 512,
        readout_hidden: int = 128,
        spectral_radius: float = 0.9,
        sparsity: float = 0.9,
        input_scaling: float = 1.0,
        seed: int = 0,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.reservoir_dim = reservoir_dim
        input_dim = n_joints * 2

        rng = np.random.default_rng(seed)

        # 输入投影矩阵（固定随机，不训练）
        Win = rng.uniform(-input_scaling, input_scaling, (reservoir_dim, input_dim)).astype(
            np.float32
        )
        b_in = rng.uniform(-0.1, 0.1, reservoir_dim).astype(np.float32)

        # 储层循环矩阵（稀疏 + 缩放谱半径）
        Wres = rng.normal(0, 1, (reservoir_dim, reservoir_dim)).astype(np.float32)
        Wres[rng.random((reservoir_dim, reservoir_dim)) > (1 - sparsity)] = 0.0
        eigvals = np.linalg.eigvals(Wres)
        sr = np.max(np.abs(eigvals))
        if sr > 1e-8:
            Wres *= spectral_radius / sr  # 缩放到目标谱半径

        self.register_buffer("Win", torch.from_numpy(Win))
        self.register_buffer("Wres", torch.from_numpy(Wres))
        self.register_buffer("b_in", torch.from_numpy(b_in))

        # 输入归一化参数
        self.register_buffer("x_mean", torch.zeros(input_dim))
        self.register_buffer("x_std", torch.ones(input_dim))

        # MLP 读出（唯一可训练部分）
        self.readout = nn.Sequential(
            nn.Linear(reservoir_dim, readout_hidden),
            nn.Tanh(),
            nn.Linear(readout_hidden, n_joints),
        )

    def set_input_stats(self, X: Tensor) -> None:
        """根据训练数据设置输入归一化参数（训练前调用一次）"""
        self.x_mean.copy_(X.mean(0))
        self.x_std.copy_(X.std(0).clamp(min=1e-8))

    def precompute_sequence_features(self, X_seq: Tensor, warmup: int = 50) -> tuple:
        """
        递归预计算序列的所有储层状态 H（固定储层，无梯度）。

        参数:
            X_seq  : (T, input_dim) 或 (B, T, input_dim) 的输入序列
            warmup : 前 warmup 步的 H 不用于训练（让储层充分激励）
        返回:
            H_valid: (T-warmup, reservoir_dim) 或 (B, T-warmup, reservoir_dim)
            X_valid: 对应时间步的 X（用于读出训练时对齐标签）
        """
        with torch.no_grad():
            if X_seq.dim() == 2:
                # 单条序列 (T, input_dim)
                T = X_seq.shape[0]
                X_n = (X_seq - self.x_mean) / self.x_std  # 归一化
                H = torch.zeros(self.reservoir_dim, device=X_seq.device)
                Hs = []
                for t in range(T):
                    H = torch.tanh(X_n[t] @ self.Win.T + H @ self.Wres.T + self.b_in)
                    if t >= warmup:
                        Hs.append(H)
                return torch.stack(Hs, dim=0), X_seq[warmup:]
            else:
                # 批量序列 (B, T, input_dim)
                B, T, _ = X_seq.shape
                X_n = (X_seq - self.x_mean) / self.x_std
                H = torch.zeros(B, self.reservoir_dim, device=X_seq.device)
                Hs = []
                for t in range(T):
                    H = torch.tanh(X_n[:, t, :] @ self.Win.T + H @ self.Wres.T + self.b_in)
                    if t >= warmup:
                        Hs.append(H)
                return torch.stack(Hs, dim=1), X_seq[:, warmup:, :]

    def forward_step(self, x: Tensor, H_prev: Tensor) -> tuple:
        """
        单步推理（在线/实时使用）。

        参数:
            x     : (batch, input_dim) 当前步输入
            H_prev: (batch, reservoir_dim) 上一步储层状态
        返回:
            tau   : (batch, n_joints) 预测力矩
            H     : (batch, reservoir_dim) 当前储层状态
        """
        x_n = (x - self.x_mean) / self.x_std
        H = torch.tanh(x_n @ self.Win.T + H_prev @ self.Wres.T + self.b_in)
        return self.readout(H), H

    def forward(self, x: Tensor, H_prev: Tensor | None = None) -> Tensor:
        """
        序列前向传播（训练或批量评估），返回每步的力矩预测。

        参数:
            x     : (T, input_dim) 或 (B, T, input_dim)
            H_prev: 初始储层状态，None 则从零初始化
        返回:
            tau   : (T, n_joints) 或 (B, T, n_joints)
        """
        if x.dim() == 2:
            T = x.shape[0]
            H = torch.zeros(self.reservoir_dim, device=x.device) if H_prev is None else H_prev
            x_n = (x - self.x_mean) / self.x_std
            taus = []
            for t in range(T):
                H = torch.tanh(x_n[t] @ self.Win.T + H @ self.Wres.T + self.b_in)
                taus.append(self.readout(H))
            return torch.stack(taus, dim=0)
        else:
            B, T, _ = x.shape
            H = torch.zeros(B, self.reservoir_dim, device=x.device) if H_prev is None else H_prev
            x_n = (x - self.x_mean) / self.x_std
            taus = []
            for t in range(T):
                H = torch.tanh(x_n[:, t, :] @ self.Win.T + H @ self.Wres.T + self.b_in)
                taus.append(self.readout(H))
            return torch.stack(taus, dim=1)

    def init_hidden(self, batch_size: int = 1) -> Tensor:
        """返回零初始化的储层状态"""
        return torch.zeros(batch_size, self.reservoir_dim, device=self.Win.device)

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": {
                    "n_joints": self.n_joints,
                    "reservoir_dim": self.reservoir_dim,
                    "readout_hidden": self.readout[0].out_features,
                    "spectral_radius": float((self.Wres.abs().max()).item()),  # 近似记录
                },
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location=None) -> "IDynESN":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = ckpt["config"]
        # spectral_radius 在 load 时无需重新生成 Wres，直接从 state_dict 恢复
        model = cls(
            n_joints=cfg["n_joints"],
            reservoir_dim=cfg["reservoir_dim"],
            readout_hidden=cfg["readout_hidden"],
        )
        model.load_state_dict(ckpt["state_dict"])
        print(f"[IDynESN] 已从 {path} 加载")
        return model


# =====================================================================
# IDynLSTM — 逆动力学 LSTM（全参数门控记忆网络）
# =====================================================================


class IDynLSTM(nn.Module):
    """
    逆动力学 LSTM — 全参数可训练的门控记忆网络。

    相比 ESN（固定储层）的优势：
        - 遗忘门可主动学会"记住速度方向"，忘掉无关历史
        - 全参数可训练（BPTT），容量远大于 ESN 固定读出
        - 输入门 / 遗忘门 / 输出门三重门控，更精确建模摩擦迟滞动态

    网络结构：
        x(t) → [归一化] → LSTM(n_layers, hidden_dim) → Linear → τ(t)

    在线推理接口：
        hidden = model.init_hidden(1)
        for t in range(T):
            tau, hidden = model.forward_step(x[t:t+1], hidden)
    """

    def __init__(
        self,
        n_joints: int = 3,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        input_dim = n_joints * 2  # [q, dq]

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.readout = nn.Linear(hidden_dim, n_joints)

        # 输入归一化缓冲（q 和 dq 量纲差异大）
        self.register_buffer("x_mean", torch.zeros(input_dim))
        self.register_buffer("x_std", torch.ones(input_dim))

    def set_input_stats(self, X: Tensor) -> None:
        """根据训练数据设置归一化参数（训练前调用一次）"""
        self.x_mean.copy_(X.mean(0))
        self.x_std.copy_(X.std(0).clamp(min=1e-8))

    def _norm(self, x: Tensor) -> Tensor:
        return (x - self.x_mean) / self.x_std

    def forward(self, x: Tensor, hidden=None):
        """
        序列前向传播（训练 / 批量评估）。

        参数:
            x      : (seq_len, input_dim) 或 (batch, seq_len, input_dim)
            hidden : (h0, c0) 各 (n_layers, batch, hidden_dim)；None→零初始化
        返回:
            tau    : (seq_len, n_joints) 或 (batch, seq_len, n_joints)
            hidden : 更新后的 (h, c)
        """
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)  # (1, T, input_dim)

        out, hidden = self.lstm(self._norm(x), hidden)  # (batch, T, hidden_dim)
        tau = self.readout(out)  # (batch, T, n_joints)

        return (tau.squeeze(0), hidden) if squeeze else (tau, hidden)

    def forward_step(self, x: Tensor, hidden) -> tuple:
        """
        单步推理（在线使用）。

        参数:
            x      : (batch, input_dim)
            hidden : (h, c) 各 (n_layers, batch, hidden_dim)
        返回:
            tau    : (batch, n_joints)
            hidden : 更新后的 (h, c)
        """
        x_n = self._norm(x).unsqueeze(1)  # (batch, 1, input_dim)
        out, hidden = self.lstm(x_n, hidden)
        return self.readout(out.squeeze(1)), hidden

    def init_hidden(self, batch_size: int = 1):
        """返回零初始化的 (h0, c0)"""
        dev = self.x_mean.device
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=dev)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=dev)
        return (h, c)

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": {
                    "n_joints": self.n_joints,
                    "hidden_dim": self.hidden_dim,
                    "n_layers": self.n_layers,
                },
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location=None) -> "IDynLSTM":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        print(f"[IDynLSTM] 已从 {path} 加载")
        return model


# =====================================================================
# IDynGRU — 逆动力学 GRU（轻量门控记忆网络）
# =====================================================================


class IDynGRU(nn.Module):
    """
    逆动力学 GRU — 轻量门控记忆网络。

    相比 LSTM：参数约少 1/4，速度更快，记忆容量略小（无 cell state）。
    相比 ESN ：全参数可训练，容量更大，能主动学习相关历史。

    在线推理接口：
        hidden = model.init_hidden(1)
        for t in range(T):
            tau, hidden = model.forward_step(x[t:t+1], hidden)
    """

    def __init__(
        self,
        n_joints: int = 3,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        input_dim = n_joints * 2

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.readout = nn.Linear(hidden_dim, n_joints)

        self.register_buffer("x_mean", torch.zeros(input_dim))
        self.register_buffer("x_std", torch.ones(input_dim))

    def set_input_stats(self, X: Tensor) -> None:
        self.x_mean.copy_(X.mean(0))
        self.x_std.copy_(X.std(0).clamp(min=1e-8))

    def _norm(self, x: Tensor) -> Tensor:
        return (x - self.x_mean) / self.x_std

    def forward(self, x: Tensor, hidden=None):
        """
        x      : (seq_len, input_dim) 或 (batch, seq_len, input_dim)
        hidden : (n_layers, batch, hidden_dim)；None→零初始化
        返回: (tau, hidden)
        """
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)

        out, hidden = self.gru(self._norm(x), hidden)
        tau = self.readout(out)
        return (tau.squeeze(0), hidden) if squeeze else (tau, hidden)

    def forward_step(self, x: Tensor, hidden) -> tuple:
        """
        单步推理。
        x: (batch, input_dim); hidden: (n_layers, batch, hidden_dim)
        """
        x_n = self._norm(x).unsqueeze(1)
        out, hidden = self.gru(x_n, hidden)
        return self.readout(out.squeeze(1)), hidden

    def init_hidden(self, batch_size: int = 1) -> Tensor:
        """返回零初始化的隐状态"""
        dev = self.x_mean.device
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=dev)

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": {
                    "n_joints": self.n_joints,
                    "hidden_dim": self.hidden_dim,
                    "n_layers": self.n_layers,
                },
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location=None) -> "IDynGRU":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        print(f"[IDynGRU] 已从 {path} 加载")
        return model


# =====================================================================
# 工具函数
# =====================================================================


def count_parameters(model: nn.Module) -> int:
    """返回模型可训练参数总数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=== IKNN_MLP (2-DOF) ===")
    mlp2 = IKNN_MLP(input_dim=2, output_dim=4)
    print(f"  参数量: {count_parameters(mlp2):,}  输出: {mlp2(torch.randn(4, 2)).shape}")

    print("\n=== IKNN_MLP (3-DOF) ===")
    mlp3 = IKNN_MLP(input_dim=3, output_dim=6)
    print(f"  参数量: {count_parameters(mlp3):,}  输出: {mlp3(torch.randn(4, 3)).shape}")

    print("\n=== IKNN_ESN (2-DOF) ===")
    esn = IKNN_ESN(input_dim=2, reservoir_dim=512, output_dim=4)
    X_d = torch.randn(100, 2)
    esn.set_input_stats(X_d)
    H = esn.precompute_features(X_d)
    print(f"  储备池 H{H.shape}  readout 可训练参数: {count_parameters(esn):,}")
