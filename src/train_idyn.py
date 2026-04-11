"""
逆动力学模型训练脚本（MLP 对照 vs 递归 ESN）

训练策略：
    MLP（无记忆基线）：
        输入当前帧 [q(t), dq(t)]，批量梯度下降，torch.compile 加速

    IDynESN（递归储层，有时序记忆）：
        1. 按轨迹切分序列，不跨轨迹截断（避免储层状态污染）
        2. 对每条轨迹递归预计算全序列储层状态 H（固定储层，无梯度）
        3. 对预热步 (warmup) 后的 H → τ 训练读出 MLP
        4. 对照关键：ESN 的 H(t) 记住了运动历史，
           MLP 的输入只有当前帧，无法区分正向/反向运动（摩擦迟滞方向）

用法：
    uv run src/train_idyn.py --model mlp           # 训练 MLP 基线
    uv run src/train_idyn.py --model esn           # 训练 ESN
    uv run src/train_idyn.py --model all           # 两者都训练
    uv run src/train_idyn.py --model all --epochs 500
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models import IDynESN, IDynGRU, IDynLSTM, IDynMLP, count_parameters

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "idyn_3dof.npz")

# =====================================================================
# 工具：力矩预测误差（N·m 平均绝对误差）
# =====================================================================


def tau_mae(pred: torch.Tensor, true: torch.Tensor) -> float:
    """平均绝对力矩误差（N·m）"""
    return float(pred.sub(true).abs().mean().item())


# =====================================================================
# Part 1：训练 IDynMLP（无记忆基线）
# =====================================================================


def train_idyn_mlp(
    epochs: int = 500,
    batch_size: int = 65536,
    lr: float = 3e-4,
    hidden_dim: int = 256,
    n_layers: int = 4,
    val_ratio: float = 0.1,
    device_str: str = "auto",
) -> IDynMLP:
    """
    训练逆动力学 MLP（无记忆基线）。
    输入：当前帧 [q, dq]（6 维）；输出：τ（3 维）
    """
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"[IDynMLP] GPU: {torch.cuda.get_device_name(0)}")

    # ---- 加载数据 ----
    if not os.path.exists(DATA_PATH):
        print("[IDynMLP] 数据集不存在，自动生成（默认 20 条轨迹）...")
        from src.data_gen_idyn import gen_idyn_dataset

        gen_idyn_dataset()

    raw = np.load(DATA_PATH)
    X_all = torch.from_numpy(raw["inputs"]).to(device)  # (N, 6) [q_des, dq_des]
    # 使用高质量 PD 标签（KP_DATA=300 确保 <0.1° 跟踪误差，含重力/Coulomb摩擦/阻尼）
    label_key = "outputs"
    Y_all = torch.from_numpy(raw[label_key].astype(np.float32)).to(device)  # (N, 3) τ

    n_total = len(X_all)
    n_val = int(n_total * val_ratio)
    perm = torch.randperm(
        n_total, device=device, generator=torch.Generator(device=device).manual_seed(42)
    )
    ti, vi = perm[: n_total - n_val], perm[n_total - n_val :]
    X_tr, Y_tr = X_all[ti], Y_all[ti]
    X_v, Y_v = X_all[vi], Y_all[vi]
    print(f"[IDynMLP] 训练: {len(ti):,}  验证: {n_val:,}")

    # ---- 模型 ----
    model = IDynMLP(n_joints=3, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    model.set_input_stats(X_tr)
    print(f"[IDynMLP] 参数量: {count_parameters(model):,}")
    compiled = torch.compile(model)

    # ---- AdamW + Cosine LR ----
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warm = max(1, epochs // 20)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [
            torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, warm),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs - warm),
        ],
        milestones=[warm],
    )

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_ckpt = os.path.join(CKPT_DIR, "idyn_3dof_mlp_best.pt")
    best_loss = float("inf")
    t0 = time.time()
    n_tr = len(ti)

    for ep in range(1, epochs + 1):
        model.train()
        perm_e = torch.randperm(n_tr, device=device)
        loss_sum, nb = 0.0, 0
        for s in range(0, n_tr, batch_size):
            idx = perm_e[s : s + batch_size]
            pred = compiled(X_tr[idx])
            loss = nn.functional.mse_loss(pred, Y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            vp = compiled(X_v)
            vl = nn.functional.mse_loss(vp, Y_v).item()
            ve = tau_mae(vp, Y_v)
        if vl < best_loss:
            best_loss = vl
            model.save(best_ckpt)

        if ep % 50 == 0 or ep <= 5:
            print(
                f"  Ep {ep:4d}/{epochs} | loss={loss_sum / nb:.5f} | "
                f"val={vl:.5f} | MAE={ve:.4f} N·m | {time.time() - t0:.0f}s"
            )

    model = IDynMLP.load(best_ckpt, map_location=device)
    model.to(device).eval()
    with torch.no_grad():
        fe = tau_mae(model(X_v), Y_v)
    print(f"[IDynMLP] 完成！验证 MAE: {fe:.4f} N·m  →  {best_ckpt}")
    return model


# =====================================================================
# Part 2：训练 IDynESN（递归储层，有时序记忆）
# =====================================================================


def train_idyn_esn(
    reservoir_dim: int = 512,
    readout_hidden: int = 128,
    spectral_radius: float = 0.9,
    epochs: int = 300,
    batch_size: int = 4096,
    lr: float = 1e-3,
    warmup_steps: int = 100,
    seq_len: int = 500,
    val_traj_ratio: float = 0.2,
    device_str: str = "auto",
) -> IDynESN:
    """
    训练逆动力学递归 ESN。

    训练流程（固定储层，只训练读出）：
        1. 按轨迹切分序列（不跨轨迹）
        2. 对每条轨迹递归预计算储层状态序列 H（无梯度）
        3. 截掉 warmup 步后的 (H, τ) 对打包成 mini-batch
        4. 仅训练 readout MLP（梯度不回传到 Win/Wres）

    关键：H(t) = tanh(Win@x(t) + Wres@H(t-1)) 记住了运动方向历史，
    因此 readout 在预测摩擦力矩时，能知道"从哪个方向来"。
    """
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"[IDynESN] GPU: {torch.cuda.get_device_name(0)}")

    # ---- 加载数据 ----
    if not os.path.exists(DATA_PATH):
        print("[IDynESN] 数据集不存在，自动生成...")
        from src.data_gen_idyn import gen_idyn_dataset

        gen_idyn_dataset()

    raw = np.load(DATA_PATH)
    X_np = raw["inputs"].astype(np.float32)  # (N, 6)
    label_key = "outputs"  # 高质量 PD 标签（含 Coulomb 摩擦）
    Y_np = raw[label_key].astype(np.float32)  # (N, 3)
    traj_id = raw["traj_id"]  # (N,)
    n_traj = int(raw["n_traj"])
    n_val_traj = max(1, int(n_traj * val_traj_ratio))
    n_tr_traj = n_traj - n_val_traj
    print(f"[IDynESN] 轨迹: 训练 {n_tr_traj}  验证 {n_val_traj}")

    def get_traj_data(traj_indices):
        """提取指定轨迹的完整序列列表"""
        seqs = []
        for tid in traj_indices:
            mask = traj_id == tid
            seqs.append((X_np[mask], Y_np[mask]))
        return seqs

    tr_seqs = get_traj_data(range(n_tr_traj))
    val_seqs = get_traj_data(range(n_tr_traj, n_traj))

    # ---- 构建模型 ----
    esn = IDynESN(
        n_joints=3,
        reservoir_dim=reservoir_dim,
        readout_hidden=readout_hidden,
        spectral_radius=spectral_radius,
    ).to(device)

    # 用全部训练数据估计归一化参数
    X_tr_flat = np.concatenate([s[0] for s in tr_seqs], axis=0)
    esn.set_input_stats(torch.from_numpy(X_tr_flat).to(device))
    print(
        f"[IDynESN] 储层: {reservoir_dim}  谱半径: {spectral_radius}  "
        f"readout 参数: {count_parameters(esn):,}"
    )

    # ---- 预计算训练序列储层状态 H（核心优化） ----
    print(f"[IDynESN] 预计算所有训练轨迹储层状态 (warmup={warmup_steps})...")
    t_pre = time.time()
    tr_H_list, tr_Y_list = [], []
    for X_seq, Y_seq in tr_seqs:
        X_t = torch.from_numpy(X_seq).to(device)  # (T, 6)
        Y_t = torch.from_numpy(Y_seq).to(device)  # (T, 3)
        H_valid, _ = esn.precompute_sequence_features(X_t, warmup=warmup_steps)
        tr_H_list.append(H_valid)
        tr_Y_list.append(Y_t[warmup_steps:])

    H_tr = torch.cat(tr_H_list, dim=0)  # (N_tr_valid, reservoir_dim)
    Y_tr = torch.cat(tr_Y_list, dim=0)  # (N_tr_valid, 3)
    print(
        f"[IDynESN] 预计算耗时 {time.time() - t_pre:.1f}s  "
        f"训练样本: {len(H_tr):,}  H shape: {tuple(H_tr.shape)}"
    )

    # 验证集也预计算（一次性）
    val_H_list, val_Y_list = [], []
    for X_seq, Y_seq in val_seqs:
        X_t = torch.from_numpy(X_seq).to(device)
        Y_t = torch.from_numpy(Y_seq).to(device)
        H_valid, _ = esn.precompute_sequence_features(X_t, warmup=warmup_steps)
        val_H_list.append(H_valid)
        val_Y_list.append(Y_t[warmup_steps:])
    H_v = torch.cat(val_H_list, dim=0)
    Y_v = torch.cat(val_Y_list, dim=0)

    # ---- 仅训练读出 MLP ----
    opt = torch.optim.AdamW(esn.readout.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_ckpt = os.path.join(CKPT_DIR, "idyn_3dof_esn_best.pt")
    best_loss = float("inf")
    t0 = time.time()
    n_tr_samples = len(H_tr)

    for ep in range(1, epochs + 1):
        esn.readout.train()
        perm_e = torch.randperm(n_tr_samples, device=device)
        loss_sum, nb = 0.0, 0
        for s in range(0, n_tr_samples, batch_size):
            idx = perm_e[s : s + batch_size]
            pred = esn.readout(H_tr[idx])
            loss = nn.functional.mse_loss(pred, Y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            nb += 1
        sched.step()

        esn.readout.eval()
        with torch.no_grad():
            vp = esn.readout(H_v)
            vl = nn.functional.mse_loss(vp, Y_v).item()
            ve = tau_mae(vp, Y_v)
        if vl < best_loss - 1e-5:  # min_delta：避免每 epoch 都写磁盘
            best_loss = vl
            esn.save(best_ckpt)

        if ep % 50 == 0 or ep <= 5:
            print(
                f"  Ep {ep:4d}/{epochs} | loss={loss_sum / nb:.5f} | "
                f"val={vl:.5f} | MAE={ve:.4f} N·m | {time.time() - t0:.0f}s"
            )

    esn = IDynESN.load(best_ckpt, map_location=device)
    esn.to(device).eval()
    with torch.no_grad():
        vp = esn.readout(H_v)
        fe = tau_mae(vp, Y_v)
    print(f"[IDynESN] 完成！验证 MAE: {fe:.4f} N·m  →  {best_ckpt}")
    return esn


# =====================================================================
# Part 3：训练 IDynLSTM（全参数 BPTT 门控记忆）
# =====================================================================


def train_idyn_rnn(
    model_type: str = "lstm",  # "lstm" 或 "gru"
    hidden_dim: int = 256,
    n_layers: int = 2,
    epochs: int = 300,
    seq_len: int = 200,  # 每块长度（步）；摩擦迟滞时间尺度 < 0.4s = 200步
    batch_size: int = 256,  # 预切块后大批量训练，榨干 GPU
    lr: float = 1e-3,
    val_traj_ratio: float = 0.2,
    device_str: str = "auto",
) -> "IDynLSTM | IDynGRU":
    """
    训练逆动力学 LSTM 或 GRU — 批量切块训练（速度最优）。

    速度优化：
        旧方案 batch_size=1：30 轨迹 × 20 块 = 600 次反向传播/epoch，GPU 利用率 <5%
        新方案：预切块 → (N_chunks, seq_len, 6)，batch_size=256
            → 每 epoch 仅 ~2 次反向传播，GPU 利用率接近 100%，速度提升 ~100×

        stateless LSTM：每批从零隐状态开始，对于摩擦迟滞（时间尺度 <200步）影响极小。
    """
    assert model_type in ("lstm", "gru"), f"model_type 须为 lstm/gru，得到 {model_type}"
    tag = f"IDyn{model_type.upper()}"

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"[{tag}] GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(DATA_PATH):
        print(f"[{tag}] 数据集不存在，自动生成...")
        from src.data_gen_idyn import gen_idyn_dataset

        gen_idyn_dataset()

    raw = np.load(DATA_PATH)
    X_np = raw["inputs"].astype(np.float32)
    label_key = "outputs"  # 高质量 PD 标签（含 Coulomb 摩擦）
    Y_np = raw[label_key].astype(np.float32)
    traj_id = raw["traj_id"]
    n_traj = int(raw["n_traj"])
    print(f"[{tag}] 标签: {label_key}  轨迹数: {n_traj}")

    n_val_traj = max(1, int(n_traj * val_traj_ratio))
    n_tr_traj = n_traj - n_val_traj
    print(f"[{tag}] 轨迹: 训练 {n_tr_traj}  验证 {n_val_traj}")

    tr_seqs = [(X_np[traj_id == i], Y_np[traj_id == i]) for i in range(n_tr_traj)]
    val_seqs = [(X_np[traj_id == i], Y_np[traj_id == i]) for i in range(n_tr_traj, n_traj)]

    # ---- 预切块：所有训练轨迹 → (N_chunks, seq_len, dim) ----
    # 每条轨迹均匀切块（末尾不足 seq_len 的丢弃），保证批次内长度一致
    X_chunks, Y_chunks = [], []
    for X_seq, Y_seq in tr_seqs:
        T = len(X_seq)
        for start in range(0, T - seq_len + 1, seq_len):
            X_chunks.append(X_seq[start : start + seq_len])
            Y_chunks.append(Y_seq[start : start + seq_len])

    X_tr = torch.from_numpy(np.stack(X_chunks)).to(device)  # (N_chunks, seq_len, 6)
    Y_tr = torch.from_numpy(np.stack(Y_chunks)).to(device)  # (N_chunks, seq_len, 3)
    n_chunks = len(X_tr)
    iters_per_ep = (n_chunks + batch_size - 1) // batch_size
    print(
        f"[{tag}] 训练块: {n_chunks}块 × {seq_len}步  "
        f"batch={batch_size}  → {iters_per_ep} iter/epoch"
    )

    X_val_chunks, Y_val_chunks = [], []
    for X_seq, Y_seq in val_seqs:
        T = len(X_seq)
        for start in range(0, T - seq_len + 1, seq_len):
            X_val_chunks.append(X_seq[start : start + seq_len])
            Y_val_chunks.append(Y_seq[start : start + seq_len])
    X_v = torch.from_numpy(np.stack(X_val_chunks)).to(device)
    Y_v = torch.from_numpy(np.stack(Y_val_chunks)).to(device)

    # ---- 构建模型 ----
    X_tr_flat = X_tr.reshape(-1, X_tr.shape[-1])
    ModelClass = IDynLSTM if model_type == "lstm" else IDynGRU
    model = ModelClass(n_joints=3, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    model.set_input_stats(X_tr_flat)
    print(f"[{tag}] 参数量: {count_parameters(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warm = max(1, epochs // 10)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [
            torch.optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, warm),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs - warm),
        ],
        milestones=[warm],
    )

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_ckpt = os.path.join(CKPT_DIR, f"idyn_3dof_{model_type}_best.pt")
    best_loss = float("inf")
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_chunks, device=device)
        loss_sum, nb = 0.0, 0

        for s in range(0, n_chunks, batch_size):
            idx = perm[s : s + batch_size]
            x_b = X_tr[idx]  # (B, seq_len, 6)
            y_b = Y_tr[idx]  # (B, seq_len, 3)

            # stateless：每批从零初始化（无跨块状态传递）
            tau_pred, _ = model(x_b)
            loss = nn.functional.mse_loss(tau_pred, y_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            loss_sum += loss.item()
            nb += 1

        sched.step()

        model.eval()
        with torch.no_grad():
            vp_list = [model(X_v[s : s + batch_size])[0] for s in range(0, len(X_v), batch_size)]
            vp = torch.cat(vp_list, dim=0)
            vl = nn.functional.mse_loss(vp, Y_v).item()
            ve = tau_mae(vp, Y_v)

        if vl < best_loss - 1e-5:
            best_loss = vl
            model.save(best_ckpt)

        if ep % 50 == 0 or ep <= 5:
            print(
                f"  Ep {ep:4d}/{epochs} | loss={loss_sum / nb:.5f} | "
                f"val={vl:.5f} | MAE={ve:.4f} N·m | {time.time() - t0:.0f}s"
            )

    model = ModelClass.load(best_ckpt, map_location=device)
    model.to(device).eval()
    with torch.no_grad():
        vp_list = [model(X_v[s : s + batch_size])[0] for s in range(0, len(X_v), batch_size)]
        fe = tau_mae(torch.cat(vp_list), Y_v)
    print(f"[{tag}] 完成！验证 MAE: {fe:.4f} N·m  →  {best_ckpt}")
    return model


# =====================================================================
# 主入口
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="逆动力学模型训练（MLP vs 递归 ESN vs LSTM vs GRU）"
    )
    parser.add_argument("--model", default="all", choices=["mlp", "esn", "lstm", "gru", "all"])
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="手动指定训练轮数（0 = 用默认值：MLP 500 / ESN 300）",
    )
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument(
        "--lr", type=float, default=0.0, help="学习率（0 = 用默认值：MLP 3e-4 / ESN 1e-3）"
    )
    parser.add_argument("--hidden_dim", type=int, default=256, help="MLP 隐藏层宽度")
    parser.add_argument("--n_layers", type=int, default=4, help="MLP 层数")
    parser.add_argument("--reservoir", type=int, default=512, help="ESN 储层维度")
    parser.add_argument("--readout_hidden", type=int, default=128)
    parser.add_argument(
        "--spectral_radius", type=float, default=0.9, help="Wres 谱半径（0.5~0.99，越大记忆越长）"
    )
    parser.add_argument(
        "--warmup", type=int, default=100, help="ESN warmup 步数（预热储层状态，不用于训练）"
    )
    args = parser.parse_args()

    models = ["mlp", "esn", "lstm", "gru"] if args.model == "all" else [args.model]

    for m in models:
        print(f"\n{'=' * 55}")
        print(f"  训练逆动力学 — {m.upper()}")
        print(f"{'=' * 55}")
        if m == "mlp":
            ep = args.epochs if args.epochs > 0 else 500
            lr = args.lr if args.lr > 0 else 3e-4
            train_idyn_mlp(
                epochs=ep,
                batch_size=args.batch_size,
                lr=lr,
                hidden_dim=args.hidden_dim,
                n_layers=args.n_layers,
            )
        elif m == "esn":
            ep = args.epochs if args.epochs > 0 else 300
            lr = args.lr if args.lr > 0 else 1e-3
            train_idyn_esn(
                reservoir_dim=args.reservoir,
                readout_hidden=args.readout_hidden,
                spectral_radius=args.spectral_radius,
                epochs=ep,
                batch_size=args.batch_size,
                lr=lr,
                warmup_steps=args.warmup,
            )
        else:  # lstm 或 gru
            ep = args.epochs if args.epochs > 0 else 300
            lr = args.lr if args.lr > 0 else 1e-3
            train_idyn_rnn(
                model_type=m,
                hidden_dim=args.hidden_dim,
                n_layers=args.n_layers,
                epochs=ep,
                lr=lr,
            )
