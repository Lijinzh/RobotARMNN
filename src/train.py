"""
统一训练脚本（2-DOF + 3-DOF，MLP + ESN）

用法:
    uv run src/train.py --arm 2dof --model mlp             # 平面臂 MLP
    uv run src/train.py --arm 2dof --model esn             # 平面臂 ESN
    uv run src/train.py --arm 3dof --model mlp             # 空间臂 MLP
    uv run src/train.py --arm 3dof --model esn             # 空间臂 ESN
    uv run src/train.py --arm all --model all              # 全部训练

GPU 加速策略（RTX 4090 专项）：
  MLP：全量数据加载到 GPU，torch.compile，AdamW + Warmup Cosine LR，batch=65536
  ESN：一次性预计算全量储备池激活 H（固定），再训练 MLP 读出，每步仅小网络反向传播
"""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models import IKNN_ESN, IKNN_MLP, count_parameters

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# =====================================================================
# 工具：角度误差（全 GPU）
# =====================================================================


def angle_error_deg(pred_sincos: torch.Tensor, true_angles: torch.Tensor) -> float:
    """
    在 GPU 上从 sin/cos 预测计算关节角度误差（度）。

    参数:
        pred_sincos : (N, 2k)  — [sin θ1, cos θ1, ..., sin θk, cos θk]
        true_angles : (N, k)   — 真实角度（弧度）
    """
    n_joints = true_angles.shape[1]
    preds = [torch.atan2(pred_sincos[:, 2 * i], pred_sincos[:, 2 * i + 1]) for i in range(n_joints)]
    pred = torch.stack(preds, dim=-1)
    diff = torch.abs(pred - true_angles)
    diff = torch.minimum(diff, 2 * torch.pi - diff)
    return float(diff.mean().item() * 180.0 / torch.pi)


# =====================================================================
# Part 1: 训练 MLP
# =====================================================================


def train_mlp(
    arm: str,
    epochs: int = 500,
    batch_size: int = 65536,
    lr: float = 3e-4,
    hidden_dim: int = 256,
    n_layers: int = 3,
    val_ratio: float = 0.1,
    device_str: str = "auto",
) -> IKNN_MLP:
    """
    训练逆运动学 MLP（2-DOF 或 3-DOF）。

    策略：全量数据加载到 GPU，torch.compile 融合内核，AdamW + Warmup Cosine LR。
    """
    assert arm in ("2dof", "3dof"), f"arm 必须是 2dof 或 3dof，得到 {arm!r}"

    # ---- 设备 ----
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"[MLP-{arm}] GPU: {torch.cuda.get_device_name(0)}")

    # ---- 加载数据 ----
    data_path = os.path.join(DATA_DIR, f"ik_{arm}.npz")
    if not os.path.exists(data_path):
        print(f"[MLP-{arm}] 数据集不存在，自动生成...")
        _auto_gen(arm)
    raw = np.load(data_path)

    if arm == "3dof":
        # 几何解耦：θ1 用解析解，NN 只学竖直平面 2-link IK
        # 输入 [r_xy, z'] (2D)，输出 [sinθ2, cosθ2, sinθ3, cosθ3] (4D)
        X_all = torch.from_numpy(raw["inputs_vert"]).to(device)  # (N, 2)
        Y_all = torch.from_numpy(raw["outputs_vert"]).to(device)  # (N, 4)
        A_all = torch.from_numpy(raw["angles_vert"]).to(device)  # (N, 2)
        input_dim, output_dim = 2, 4
        print(
            f"[MLP-{arm}] 使用竖直平面分解特征: 输入 [r_xy, z']  输出 [sinθ2, cosθ2, sinθ3, cosθ3]"
        )
    else:
        X_all = torch.from_numpy(raw["inputs"]).to(device)
        Y_all = torch.from_numpy(raw["outputs"]).to(device)
        A_all = torch.from_numpy(raw["angles"]).to(device)
        input_dim, output_dim = 2, 4

    n_total = len(X_all)
    n_val = int(n_total * val_ratio)
    perm = torch.randperm(
        n_total, device=device, generator=torch.Generator(device=device).manual_seed(42)
    )
    ti, vi = perm[: n_total - n_val], perm[n_total - n_val :]
    X_tr, Y_tr = X_all[ti], Y_all[ti]
    X_v, Y_v, A_v = X_all[vi], Y_all[vi], A_all[vi]
    print(f"[MLP-{arm}] 训练: {len(ti):,}  验证: {n_val:,}")

    # ---- 模型 ----
    model = IKNN_MLP(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers
    ).to(device)
    # 设置输入归一化（消除量纲差异，提升收敛速度）
    model.set_input_stats(X_tr)
    print(f"[MLP-{arm}] 参数量: {count_parameters(model):,}  已设置输入归一化")
    compiled = torch.compile(model)

    # ---- AdamW + Warmup Cosine LR ----
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warm = max(1, epochs // 20)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [
            torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, warm),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, epochs - warm)),
        ],
        milestones=[warm],
    )

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_ckpt = os.path.join(CKPT_DIR, f"ik_{arm}_mlp_best.pt")
    best_loss = float("inf")
    metadata = {
        "task": "inverse_kinematics", "arm": arm,
        "input_definition": "[r_xy, z-L0]" if arm == "3dof" else "[x, y]",
        "target": "outputs_vert" if arm == "3dof" else "outputs",
        "target_definition": "joint_angle_sin_cos", "normalization": "training_only",
        "split_seed": 42, "train_samples": len(ti), "validation_samples": len(vi),
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
    }
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
            ve = angle_error_deg(vp, A_v)
        if vl < best_loss:
            best_loss = vl
            model.save(best_ckpt, metadata=metadata)

        if ep % 50 == 0 or ep <= 5:
            print(
                f"  Ep {ep:4d}/{epochs} | loss={loss_sum / nb:.5f} | "
                f"val={vl:.5f} | err={ve:.3f}° | {time.time() - t0:.0f}s"
            )

    model = IKNN_MLP.load(best_ckpt, map_location=device)
    model.to(device).eval()
    with torch.no_grad():
        fe = angle_error_deg(model(X_v), A_v)
    print(f"[MLP-{arm}] 完成！最终误差: {fe:.4f}°  最优模型: {best_ckpt}")
    return model


# =====================================================================
# Part 2: 训练 ESN（储备池预计算 + MLP 读出训练）
# =====================================================================


def train_esn(
    arm: str,
    reservoir_dim: int = 512,
    readout_hidden: int = 128,
    epochs: int = 300,
    batch_size: int = 65536,
    lr: float = 1e-3,
    val_ratio: float = 0.1,
    device_str: str = "auto",
) -> IKNN_ESN:
    """
    训练增强 ESN（随机储备池 + MLP 读出）。

    高效策略：
      1. 一次性预计算全量储备池激活 H（固定无梯度，快）
      2. 仅在 (H → sincos) 上训练小型 MLP 读出，每步梯度只通过读出网络传播
    """
    assert arm in ("2dof", "3dof"), f"arm 必须是 2dof 或 3dof，得到 {arm!r}"

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"[ESN-{arm}] GPU: {torch.cuda.get_device_name(0)}")

    # ---- 加载数据 ----
    data_path = os.path.join(DATA_DIR, f"ik_{arm}.npz")
    if not os.path.exists(data_path):
        print(f"[ESN-{arm}] 数据集不存在，自动生成...")
        _auto_gen(arm)
    raw = np.load(data_path)

    if arm == "3dof":
        # 几何解耦：与 MLP 保持一致，使用竖直平面特征
        # θ1 解析精确，ESN 只学 (r_xy, z') → (θ2, θ3)
        X_all = torch.from_numpy(raw["inputs_vert"]).to(device)  # (N, 2)
        Y_all = torch.from_numpy(raw["outputs_vert"]).to(device)  # (N, 4)
        A_all = torch.from_numpy(raw["angles_vert"]).to(device)  # (N, 2)
        input_dim, output_dim = 2, 4
        print(
            f"[ESN-{arm}] 使用竖直平面分解特征: 输入 [r_xy, z']  输出 [sinθ2, cosθ2, sinθ3, cosθ3]"
        )
    else:
        X_all = torch.from_numpy(raw["inputs"]).to(device)
        Y_all = torch.from_numpy(raw["outputs"]).to(device)
        A_all = torch.from_numpy(raw["angles"]).to(device)
        input_dim, output_dim = 2, 4

    n_total = len(X_all)
    n_val = int(n_total * val_ratio)
    perm = torch.randperm(
        n_total, device=device, generator=torch.Generator(device=device).manual_seed(42)
    )
    ti, vi = perm[: n_total - n_val], perm[n_total - n_val :]
    X_tr, Y_tr = X_all[ti], Y_all[ti]
    X_v, A_v = X_all[vi], A_all[vi]
    print(f"[ESN-{arm}] 训练: {len(ti):,}  验证: {n_val:,}")

    # ---- 构建 ESN ----
    esn = IKNN_ESN(
        input_dim=input_dim,
        reservoir_dim=reservoir_dim,
        readout_hidden=readout_hidden,
        output_dim=output_dim,
    ).to(device)

    # 设置归一化参数
    esn.set_input_stats(X_tr)

    # ---- 一次性预计算全量储备池激活 H（核心优化）----
    print(f"[ESN-{arm}] 预计算储备池激活 (reservoir_dim={reservoir_dim})...")
    t_pre = time.time()
    H_tr = esn.precompute_features(X_tr)  # (N_tr, reservoir_dim) - 固定不变
    H_v = esn.precompute_features(X_v)  # (N_v, reservoir_dim)
    print(f"[ESN-{arm}] 预计算耗时 {time.time() - t_pre:.1f}s  H_train shape={tuple(H_tr.shape)}")
    print(f"[ESN-{arm}] readout 可训练参数: {count_parameters(esn):,}")

    # ---- 仅训练 MLP 读出 ----
    opt = torch.optim.AdamW(esn.readout.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    os.makedirs(CKPT_DIR, exist_ok=True)
    best_ckpt = os.path.join(CKPT_DIR, f"ik_{arm}_esn_best.pt")
    best_loss = float("inf")
    metadata = {
        "task": "inverse_kinematics", "arm": arm,
        "input_definition": "[r_xy, z-L0]" if arm == "3dof" else "[x, y]",
        "target": "outputs_vert" if arm == "3dof" else "outputs",
        "target_definition": "joint_angle_sin_cos", "normalization": "training_only",
        "split_seed": 42, "train_samples": len(ti), "validation_samples": len(vi),
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
    }
    t0 = time.time()
    n_tr = len(ti)

    for ep in range(1, epochs + 1):
        esn.readout.train()
        perm_e = torch.randperm(n_tr, device=device)
        loss_sum, nb = 0.0, 0
        for s in range(0, n_tr, batch_size):
            idx = perm_e[s : s + batch_size]
            pred = esn.readout(H_tr[idx])  # 直接用预计算的 H
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
            vl = nn.functional.mse_loss(vp, Y_all[vi]).item()
            ve = angle_error_deg(vp, A_v)
        if vl < best_loss:
            best_loss = vl
            esn.save(best_ckpt, metadata=metadata)

        if ep % 50 == 0 or ep <= 5:
            print(
                f"  Ep {ep:4d}/{epochs} | loss={loss_sum / nb:.5f} | "
                f"val={vl:.5f} | err={ve:.3f}° | {time.time() - t0:.0f}s"
            )

    final_esn = IKNN_ESN.load(best_ckpt).to(device)
    # 输入统计量已经随最佳 checkpoint 保存，不再用验证数据重新估计。
    final_esn.eval()
    with torch.no_grad():
        H_v_final = final_esn.precompute_features(X_v)
        fp = final_esn.readout(H_v_final)
        fe = angle_error_deg(fp, A_v)
    print(f"[ESN-{arm}] 完成！最终误差: {fe:.4f}°  最优模型: {best_ckpt}")
    return final_esn


# =====================================================================
# 自动生成数据集
# =====================================================================


def _auto_gen(arm: str) -> None:
    """若数据集文件不存在，自动调用 data_gen.py 生成"""
    from src.data_gen import gen_2dof, gen_3dof

    if arm == "2dof":
        gen_2dof()
    else:
        gen_3dof()


# =====================================================================
# 主入口
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 IK 逆运动学网络")
    parser.add_argument(
        "--arm", default="2dof", choices=["2dof", "3dof", "all"], help="机械臂类型（默认 2dof）"
    )
    parser.add_argument(
        "--model", default="mlp", choices=["mlp", "esn", "all"], help="模型类型（默认 mlp）"
    )
    parser.add_argument("--epochs", type=int, default=500, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=256, help="MLP 隐藏层宽度")
    parser.add_argument("--n_layers", type=int, default=3, help="MLP 层数")
    parser.add_argument("--reservoir", type=int, default=512, help="ESN 储备池维度")
    parser.add_argument("--readout_hidden", type=int, default=128, help="ESN MLP 读出隐藏维度")
    args = parser.parse_args()

    arms = ["2dof", "3dof"] if args.arm == "all" else [args.arm]
    models = ["mlp", "esn"] if args.model == "all" else [args.model]

    for arm in arms:
        for model in models:
            print(f"\n{'=' * 55}")
            print(f"  训练 {arm.upper()} — {model.upper()}")
            print(f"{'=' * 55}")
            if model == "mlp":
                # 几何解耦后，3DOF 竖直平面子问题等价于 2DOF，
                # 无需大网络，使用与 2DOF 相同的默认配置
                train_mlp(
                    arm=arm,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    hidden_dim=args.hidden_dim,
                    n_layers=args.n_layers,
                )
            else:
                train_esn(
                    arm=arm,
                    reservoir_dim=args.reservoir,
                    readout_hidden=args.readout_hidden,
                    epochs=min(args.epochs, 300),  # ESN 收敛更快
                    batch_size=args.batch_size,
                    lr=1e-3,  # ESN readout 使用更大 lr
                )
