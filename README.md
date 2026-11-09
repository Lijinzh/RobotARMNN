# 🤖 RobotARMNN — 机械臂神经网络运动学与逆动力学

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.x-green)](https://mujoco.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

基于 **MuJoCo** 物理仿真的机械臂神经网络学习 Demo，包含两个完整实验模块：

| 模块 | 任务 | 网络 |
|------|------|------|
| 🦾 **逆运动学 (IK)** | 目标末端坐标 → 关节角（2-DOF 平面臂 + 3-DOF 空间臂） | MLP、ESN |
| ⚡ **逆动力学 (IDyn)** | 期望轨迹 → 前馈力矩（3-DOF 带摩擦空间臂） | MLP、ESN、LSTM、GRU |

### 🖼️ 演示结果一览

**逆运动学误差对比（2-DOF 平面臂）**
![IK 2DOF 对比图](docs/ik_comparison_2dof.png)

**逆运动学误差对比（3-DOF 空间臂）**
![IK 3DOF 对比图](docs/ik_comparison_3dof.png)

**逆动力学补偿效果对比（MLP / ESN / LSTM / GRU vs 纯 PD）**
![IDyn 对比图](docs/idyn_comparison.png)

---

## 📋 目录

- [🦿 机械臂构型](#机械臂构型)
- [📁 项目结构](#项目结构)
- [⚙️ 环境配置](#环境配置)
- [🚀 快速开始](#快速开始)
- [🦾 模块一：逆运动学 IK](#模块一逆运动学-ik)
- [⚡ 模块二：逆动力学 IDyn](#模块二逆动力学-idyn)
- [📊 训练结果](#训练结果)
- [🎮 可视化操作说明](#可视化操作说明)
- [🔬 技术细节](#技术细节)

---

## 🦿 机械臂构型

### 平面 2-DOF 臂

```
关节1 (θ1)：底座绕 Z 轴旋转（肩部）  范围 [-π, π]
关节2 (θ2)：肘部绕 Z 轴旋转（相对大臂）范围 [-π, π]
L1 = 0.15 m（大臂）  L2 = 0.12 m（小臂）
正运动学：X = L1·cos(θ1)+L2·cos(θ1+θ2)
         Y = L1·sin(θ1)+L2·sin(θ1+θ2)
可达半径：r_min = 0.03 m，r_max = 0.27 m
```

### 空间 3-DOF 臂

```
关节1 (θ1)：底座绕 Z 轴旋转
关节2 (θ2)：肩部绕 Y 轴俯仰   范围 [-π/2, π/2]
关节3 (θ3)：肘部绕 Y 轴俯仰   范围 [-π, π]
L0 = 0.20 m（底柱高度）  L1 = 0.15 m（大臂）  L2 = 0.12 m（小臂）
正运动学：r = L1·cos(θ2)+L2·cos(θ2+θ3)
         X = r·cos(θ1)，Y = r·sin(θ1)
         Z = L0+L1·sin(θ2)+L2·sin(θ2+θ3)
```

---

## 📁 项目结构

```
RobotARMNN/
│
├── src/                            # 所有源代码
│   ├── __init__.py
│   │
│   ├── # ─── 运动学 ───────────────────────────────
│   ├── kinematics_2dof.py          # 平面 2-DOF 正/逆运动学 + sin/cos 工具
│   ├── kinematics_3dof.py          # 空间 3-DOF 正/逆运动学（几何解耦法）
│   │
│   ├── # ─── MuJoCo 模型 ──────────────────────────
│   ├── arm_2dof.xml                # 平面臂 MJCF（含 mocap 目标球）
│   ├── arm_3dof.xml                # 空间臂 MJCF（含三轴标记）
│   ├── arm_3dof_friction.xml       # 带 Coulomb 摩擦的 3-DOF 臂（逆动力学用）
│   │
│   ├── # ─── 神经网络模型定义 ──────────────────────
│   ├── models.py                   # IKNN_MLP / IKNN_ESN / IDynMLP /
│   │                               # IDynESN / IDynLSTM / IDynGRU / CurrentNN
│   │
│   ├── # ─── 模块一：逆运动学 IK ───────────────────
│   ├── data_gen.py                 # IK 训练数据生成（随机采样工作空间，50 万点）
│   ├── train.py                    # IK 训练脚本（MLP + ESN，2-DOF + 3-DOF）
│   ├── visualize.py                # 🆕 可视化统一入口（子命令 ik / mujoco）
│   ├── visualize_mujoco.py         # MuJoCo 交互可视化（保留兼容）
│   ├── visualize_ik.py             # IK 误差对比图（保留兼容）
│   │
│   └── # ─── 模块二：逆动力学 IDyn ─────────────────
│       ├── data_gen_idyn.py        # IDyn 数据生成（MuJoCo 高刚度 PD 激励，30 条轨迹）
│       ├── train_idyn.py           # IDyn 训练脚本（MLP / ESN / LSTM / GRU）
        ├── mpc_controller.py       # 🆕 MPC 控制器（滚动预测优化 / 解析 QP，去除 scipy 迭代）
        └── visualize_idyn.py       # IDyn 控制误差对比图（PD/MPC/NN，自动保存）
│
├── tests/
│   ├── __init__.py
│   ├── test_kinematics.py          # 运动学单元测试
│   └── test_*regressions.py        # IK、控制展示和训练协议回归
│
├── checkpoints/                    # 已训练模型权重（*.pt / *.npz）
│   ├── ik_2dof_mlp_best.pt         # 2-DOF IK MLP
│   ├── ik_2dof_esn_best.pt         # 2-DOF IK ESN
│   ├── ik_3dof_mlp_best.pt         # 3-DOF IK MLP
│   ├── ik_3dof_esn_best.pt         # 3-DOF IK ESN
│   ├── idyn_3dof_mlp_best.pt       # 3-DOF IDyn MLP
│   ├── idyn_3dof_esn_best.pt       # 3-DOF IDyn ESN
│   ├── idyn_3dof_lstm_best.pt      # 3-DOF IDyn LSTM
│   └── idyn_3dof_gru_best.pt       # 3-DOF IDyn GRU
│
├── data/                           # 训练数据（.gitignore 忽略，需自行生成）
├── export/                         # 可视化输出图片（.gitignore 忽略）
│
├── main.py                         # 快速入口：运动学解析演示
├── pyproject.toml                  # 依赖声明（uv 管理）
├── uv.lock                         # 依赖锁定文件
├── .python-version                 # Python 版本锁定（3.10）
├── LICENSE                         # MIT 许可证
└── README.md
```

---

## ⚙️ 环境配置

### 前置要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.10 – 3.12 | 推荐 3.10 |
| uv | 最新版 | 替代 pip 的包管理器，速度快 10× |
| CUDA | 12.x （可选） | GPU 训练。无 CUDA 时自动退到 CPU |
| Git | ≥ 2.x | 克隆仓库 |

> **不需要手动创建虚拟环境**，`uv sync` 会自动创建并安装所有依赖。

---

### Linux / macOS

```bash
# 1. 安装 uv（若已安装可跳过）
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env        # 或重开终端让 PATH 生效

# 2. 克隆项目
git clone https://github.com/Lijinzh/RobotARMNN.git
cd RobotARMNN

# 3. 初始化环境（自动创建 .venv，安装所有依赖）
uv sync

# 4. 验证安装
uv run python -c "import torch, mujoco; print('torch:', torch.__version__); print('mujoco:', mujoco.__version__)"
```

**MuJoCo 依赖（Linux）**：`apt install libgl1-mesa-glx libglu1-mesa`（已有桌面环境则无需）

---

### Windows

```powershell
# 1. 安装 uv（PowerShell，需要 Execution Policy 允许运行脚本）
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# 或通过 winget：winget install --id=astral-sh.uv -e

# 2. 重开 PowerShell 让 PATH 生效，克隆项目
git clone https://github.com/Lijinzh/RobotARMNN.git
cd RobotARMNN

# 3. 初始化环境
uv sync

# 4. 验证安装
uv run python -c "import torch, mujoco; print('torch:', torch.__version__); print('mujoco:', mujoco.__version__)"
```

**Windows 注意事项**：
- MuJoCo 3.x 在 Windows 10/11 64-bit 原生支持，无需额外依赖。
- GPU 版 PyTorch：若 `uv sync` 安装的是 CPU 版本，可执行：
  ```powershell
  uv add torch --index https://download.pytorch.org/whl/cu124
  ```
- 可视化脚本中的 `MPLBACKEND=Agg` 在 PowerShell 中写法为：
  ```powershell
  $env:MPLBACKEND="Agg"; uv run src/visualize_idyn.py
  ```

---

### GPU 说明

训练脚本会自动检测并使用 GPU（如有）。时长取决于数据量、网络层数、编译缓存和批量；
完整重训的实际参数、耗时与结果见 [2026-11-10 重训记录](docs/retrain_2026-11-10.md)。

---

## 🚀 快速开始

```bash
# 运动学解析演示（无需训练数据或模型）
uv run main.py

# 完整工作流：逆运动学（IK）
uv run src/data_gen.py --arm all          # 生成 IK 训练数据（50 万点）
uv run src/train.py --arm all --model all  # 训练全部 IK 模型
uv run src/visualize.py ik --arm all      # 误差对比图（🆕 自动保存）

# 完整工作流：逆动力学（IDyn）
# 已有 data/idyn_3dof.npz 时，先改名归档；生成命令会覆盖该文件。
uv run src/data_gen_idyn.py --n_traj 30   # 生成 30 条激励轨迹（约 2 min）
uv run src/train_idyn.py --model all       # 训练全部 IDyn 模型，默认每种 1000 轮
uv run src/visualize_idyn.py               # 对比可视化（🆕 自动保存，含 MPC）

# 单元测试
uv run pytest tests/ -v
```

---

## 🦾 模块一：逆运动学 IK

### 📦 数据生成

```bash
uv run src/data_gen.py --arm 2dof   # 平面臂（50 万点，约 10 s）
uv run src/data_gen.py --arm 3dof   # 空间臂（50 万点，含 r<0 过滤，约 15 s）
uv run src/data_gen.py --arm all    # 两种臂都生成
```

生成文件保存到 `data/`（`.npz` 格式 + `.csv` 方便查看，已加入 `.gitignore`）。

### 🏋️ 训练

```bash
uv run src/train.py --arm 2dof --model mlp        # 2-DOF MLP  (~7 s / 500 ep)
uv run src/train.py --arm 2dof --model esn        # 2-DOF ESN  (~3 s / 300 ep)
uv run src/train.py --arm 3dof --model mlp        # 3-DOF MLP  (~9 s / 500 ep)
uv run src/train.py --arm 3dof --model esn        # 3-DOF ESN  (~5 s / 300 ep)
uv run src/train.py --arm all --model all          # 全部训练
```

训练完成后权重自动保存到 `checkpoints/ik_{arm}_{model}_best.pt`。

### 🎨 可视化

**误差对比图（无需 MuJoCo 窗口）：**

```bash
uv run src/visualize_ik.py --arm 2dof           # 2-DOF：弹出交互窗口
uv run src/visualize_ik.py --arm 3dof
uv run src/visualize_ik.py --arm all --save     # 保存到 export/

# Windows PowerShell 保存：
$env:MPLBACKEND="Agg"; uv run src/visualize_ik.py --arm all --save
```

**MuJoCo 交互可视化（方向键控制目标点）：**

```bash
uv run src/visualize_mujoco.py --arm 2dof                             # 解析 IK
uv run src/visualize_mujoco.py --arm 2dof --use_nn                    # MLP IK
uv run src/visualize_mujoco.py --arm 2dof --use_nn --model_type esn   # ESN IK
uv run src/visualize_mujoco.py --arm 3dof --use_nn                    # 3-DOF MLP
uv run src/visualize_mujoco.py --arm 3dof --use_nn --model_type esn
```

### IK 模型架构

| 模型 | 输入维度 | 输出维度 | 网络结构 | 参数量 |
|------|---------|---------|---------|--------|
| IKNN_MLP (2-DOF) | 2 `[X,Y]` | 4 `[sinθ₁,cosθ₁,sinθ₂,cosθ₂]` | 3 × Linear(256) + ReLU | ~133K |
| IKNN_MLP (3-DOF) | 2 `[r_xy, z']` | 4 `[sinθ₂,cosθ₂,sinθ₃,cosθ₃]` | 3 × Linear(256) + ReLU | ~133K |
| IKNN_ESN (2-DOF) | 2 | 4 | 储层(512) + MLP读出(128) | ~66K 可训练 |
| IKNN_ESN (3-DOF) | 2 | 4 | 储层(512) + MLP读出(128) | ~66K 可训练 |

> 3-DOF 在正水平半径训练域内使用**几何解耦**：θ₁ 由 `atan2(Y,X)` 求解，MLP/ESN 只学竖直平面子问题 `(r_xy, z') → (θ₂, θ₃)`，减少回归维度。
>
> 输出用 `[sinθ, cosθ]` 编码而非直接输出角度，避免 ±π 跳变导致的训练不稳定。

---

## ⚡ 模块二：逆动力学 IDyn

**控制律**：τ = τ_ff(q_des, dq_des) + KP·(q_des−q) + KD·(dq_des−dq)

前馈力矩 τ_ff 由神经网络预测，PD 项负责误差纠偏（弱刚度：KP=[8,8,6]，KD=[1.5,1.5,1]）。

这是通过力矩控制实现位置轨迹跟踪：参考位置/速度进入前馈，实际位置/速度进入 PD。
当前 MLP、ESN、LSTM、GRU 都学习高增益 PD 的总命令 `outputs`；ESN 使用参考状态历史，
不直接读取实际状态或误差。它不是专门学习摩擦残差的模型，也不能仅凭环状相图证明摩擦迟滞记忆。

**MPC 滚动预测优化**：每步在当前工作点线性化动力学，预测未来 N=10 步，
用 numpy 求解无约束二次优化，再截断输出力矩，只执行第 0 步并滚动重复。
它不依赖 NN 训练，但也不是严格的含力矩约束 QP。
现有 GRU 输出不能直接作为 MPC 的摩擦残差叠加；原 `mpc_gru` 路径已禁用，
需要先定义一致的残差目标并训练对应模型，才能重新评估组合方案。
### 数据生成

```bash
# 已有数据请先归档，避免覆盖旧实验所用的数据。
uv run src/data_gen_idyn.py --n_traj 30   # 生成 30 条激励轨迹（推荐最小值）
uv run src/data_gen_idyn.py --n_traj 60   # 更多轨迹，精度更高
```

每条轨迹用高刚度 PD（KP=[300,300,200]、KD=[5,5,4]）驱动，实际跟踪误差随轨迹变化。
数据保存于 `data/idyn_3dof.npz`（已加入 `.gitignore`），重新生成会覆盖该文件。

**标签版本 2：三个目标需要分别使用。**

| 标签 | 定义 | 对应输入 |
|------|------|----------|
| `outputs` | 高增益 PD 示教命令，包含闭环跟踪误差的影响，不是精确逆动力学真值 | `inputs`：`[q_des, dq_des]`，6 维 |
| `outputs_inv` | `mj_inverse(q_des, dq_des, ddq_des)` 所需力矩 | `inputs_full`：`[q_des, dq_des, ddq_des]`，9 维 |
| `outputs_bg` | `mj_inverse(q_des, dq_des, 0)` 的零加速度补偿力矩 | `inputs`：`[q_des, dq_des]`，6 维 |

旧标签生成代码在设置期望加速度后调用 `mj_forward`，覆盖了加速度；旧 `outputs_bg`
还错误地使用 `bias + passive`，反转了阻尼补偿符号，并漏掉由约束求解器处理的
`frictionloss`。版本 2 直接调用 `mj_inverse`，其力平衡为
`tau = M*ddq + bias - passive - constraint`。约束力可能随指定加速度变化，尤其在低速区，
因此不能一般化为 `outputs_inv = M*ddq + outputs_bg`。

新文件同时保存 `ddq_des`、`label_version=2`、`seed` 和 `mujoco_version`。
没有 `label_version` 的旧文件应重新生成，改代码不会自动修复现有 `.npz`。
轨迹超限时同步缩放位置、速度、加速度，保持解析导数一致。

现有 `train_idyn.py` 仍默认训练 `inputs → outputs`，不会自动选择 `bg` 或九维 `inv`。
器件/多项式回归应显式选用对应字段，按 `traj_id` 划分训练、验证、测试轨迹，
并只使用训练集估计标准化参数。旧 `bg` 上的低误差不能代表正确补偿的精度；
修正 `bg` 包含速度换向附近的摩擦变化，可另行评估分段、饱和或速度方向特征。

物理回归测试（含加速度响应、摩擦方向、低速正反动力学往返及轨迹导数一致性）：

```bash
uv run pytest tests/test_idyn_labels.py -v
```

### 多项式与摩擦特征回归

```bash
uv run src/validate_poly_friction.py
# 可选：--data 数据路径 --output-dir 结果目录
```

该脚本固定使用版本 2 的 `inputs → outputs_bg`：轨迹 0–7 训练、8–9 测试，
每条轨迹从索引 0 每隔 25 点直接抽样，从 500 Hz 降为 20 Hz，共 1280/320 点；
保留启动段，无滤波、无插值。其余轨迹不参与。先用训练轨迹 0–5 拟合、6–7 验证，
选择阶数和摩擦参数，再用全部 8 条训练轨迹重拟合。标准化仅使用当次训练数据，
采用含截距和全部交叉项的无正则普通最小二乘。

方向特征为三个关节的 `sign(dq)`；低速饱和特征为三个关节的
`clip(dq/v0, -1, 1)`，其中 `dq` 使用原始 rad/s 单位，不先标准化。
三个关节共用一个 `v0`，从 0.002–0.2 rad/s 的 31 个对数间隔候选中，
仅按内部验证 NRMSE 选择。它是对当前仿真摩擦的回归近似，不是通用精确摩擦公式。

2026-11-09，seed=42 修正数据上的离线结果：

| 模型 | 内部验证 NRMSE | 测试 NRMSE | 每关节系数 |
|------|----------------|------------|------------|
| 一阶多项式 | 0.554547 | 0.476943 | 7 |
| 三阶多项式 | 0.549483 | 0.576879 | 84 |
| 三阶＋方向特征 | 0.129050 | 0.094332 | 87 |
| 一阶＋低速饱和 | 0.010226 | 0.006329 | 10 |
| 二阶＋低速饱和（内部验证选中） | 0.010082 | 0.006014 | 31 |
| 三阶＋低速饱和 | 0.010145 | 0.007054 | 87 |

三种低速饱和模型均由内部验证选中 `v0=0.0233183 rad/s`。
整体 NRMSE 定义为 `sqrt(sum_j MSE_j / sum_j Var(y_j))`，各关节分别中心化，`ddof=0`。
这里只评估两条固定测试轨迹上的仿真力矩；结果不代表器件精度、闭环跟踪精度或跨工况泛化。

结果默认保存到 `export/idyn_poly_friction/`：`report.json` 包含完整选参记录、
各关节 NRMSE/RMSE、低速区误差和数据 SHA-256；`selected_model.npz` 保存选中模型的
标准化、单项式指数和系数；`test_predictions.npz` 保存测试预测和原始样本索引。
模型已重新加载并核对预测。可以用 `src.validate_poly_friction.predict_model` 推理：

```python
import numpy as np
from src.validate_poly_friction import predict_model

with np.load("export/idyn_poly_friction/selected_model.npz") as model:
    # x 的形状为 (N, 6)，列顺序为 q_des[3]、dq_des[3]，单位为 rad、rad/s。
    tau_bg = predict_model(model, x)
```

### 训练

```bash
uv run src/train_idyn.py --model all     # 训练全部 4 种模型（MLP/ESN/LSTM/GRU）
uv run src/train_idyn.py --model gru     # 只训练 GRU
uv run src/train_idyn.py --model lstm
uv run src/train_idyn.py --model mlp
uv run src/train_idyn.py --model esn
```

权重保存至 `checkpoints/idyn_3dof_{model}_best.pt`；重新训练前应归档需要保留的同名权重。

四种模型统一使用末尾 20% 的完整轨迹验证；标准化只使用训练轨迹输入。
默认 `--warmup 100`：每条轨迹前 100 步（500 Hz 下为 0.2 秒）不计训练损失和验证分数，
递归模型仍可用这些输入建立状态。MLP/ESN 的 CLI 默认 batch 为 65536，LSTM/GRU 为 256，
显式 `--batch_size` 对所有模型生效。CLI 默认训练 1000 轮；`--epochs 0` 使用各模型函数默认轮数。

LSTM/GRU 保留独立块、零状态训练，末尾不足完整块的训练样本丢弃；验证改为完整轨迹连续推理，
每条轨迹只在开头清零，预热后的尾部全部参与评分。此训练策略不等同于整轨迹 BPTT。
新 checkpoint 的 `model.training_metadata` 记录目标、输入定义、采样周期、数据哈希、轨迹划分和预热协议。
IK ESN 同样只用训练集估计标准化。仓库中的八个在用模型已按修复后的协议完整重训，
使用的数据、随机种子、训练参数及新旧结果见 [重训记录](docs/retrain_2026-11-10.md)。

### 可视化

```bash
# 默认自动保存到 export/idyn_comparison.png（无需 --save）
uv run src/visualize_idyn.py

# 禁用保存，仅显示窗口
uv run src/visualize_idyn.py --no-save

# 在相同轨迹上比较总力矩前馈，窗口明确显示 ESN 方案
uv run src/visualize_idyn.py --model mlp esn lstm gru mpc --amp_scale 1 --mujoco --viewer-model esn

# 仅 MPC，不加 GRU 前馈，观察纯预测控制效果
uv run src/visualize_idyn.py --model mpc --mujoco --viewer-model mpc

# 改变预测域步数，评估跟踪误差与计算量（更长不保证更准确）
uv run src/visualize_idyn.py --model mpc --mpc_horizon 20 --mujoco --viewer-model mpc

# 增幅测试应与训练幅值 amp_scale=1 分别报告
uv run src/visualize_idyn.py --model esn gru --amp_scale 2 --duration 10 --mujoco --viewer-model gru

# 全方案对比（所有 NN + MPC，不打开 MuJoCo，自动保存图片）
uv run src/visualize_idyn.py

# Windows PowerShell（无 MuJoCo 窗口，仅保存图片）：
$env:MPLBACKEND="Agg"; uv run src/visualize_idyn.py --model mpc esn gru --amp_scale 1
```

`--mujoco` 打开真实力矩控制＋`mj_step` 的动力学窗口，`--viewer-model` 明确选择显示方案，
红色目标球跟随期望末端位置；窗口不会自动判定哪个算法最好。
控制积分保持 500 Hz，窗口约 56 Hz 刷新，避免屏幕垂直同步把控制循环拖慢；画面显示当前方案和即时关节误差。
可用鼠标拖动旋转/平移视角，滚轮缩放，关闭窗口退出。
未指定幅值时默认 `amp_scale=2`，高于训练生成时的幅值设置，不能混作训练幅值下的测试。

### IDyn 模型架构

| 模型 | 架构 | 记忆 | 参数量 |
|------|------|------|--------|
| IDynMLP | 4×Linear(256, Tanh) + 输入归一化 | 无 | 199,939 |
| IDynESN | 固定储层(512) + MLP读出 | 递归状态 | ~66K 可训练 |
| IDynLSTM | 4层LSTM(256) + Linear | 遗忘/输入/输出门 | 1,850,115 |
| IDynGRU | 4层GRU(256) + Linear | 重置/更新门 | 1,387,779 |

输入：`[q₁,q₂,q₃, dq₁,dq₂,dq₃]`（6维）  输出：`[τ₁, τ₂, τ₃]`（N·m）
表中架构和参数量取自现有 checkpoint；训练函数与 CLI 的默认值可不同。

---

## 📊 训练结果

### 🎯 IK 重训结果（2026-11-10，8000 随机测试点，seed=42）

误差只统计有限且满足全部关节限位的输出，合法解率单独报告，不等于精度成功率。
3-DOF 为球壳广域压力测试，包含需要负肘分支的正肘训练域外目标，不等同于训练域内精度。

| 模型 | 2-DOF 合法解率 | 2-DOF 均值／p95 | 3-DOF 合法解率 | 3-DOF 均值／p95 |
|------|-----------|-----------|-----------|-----------|
| 解析 IK（基准） | 100% | 0.000／0.00 mm | 100% | 0.000／0.00 mm |
| MLP IK | 100% | 1.015／2.32 mm | 94.4625% | 0.637／1.45 mm |
| ESN IK | 100% | 6.504／12.24 mm | 95.2125% | 4.218／8.73 mm |

共同域另外固定 8000 点、seed=7777。3DOF MLP/ESN 合法解率分别为 99.925%/99.85%，
合法样本均值为 0.644/4.172 mm。旧、新权重的网络规模保持一致，重训并未整体改善 IK；
完整共同域定义、旧权重对照及以全部目标为分母的成功率见 [重训记录](docs/retrain_2026-11-10.md)。

两个评估入口使用累计样本数停止，默认 8000 点仅需约 1.3 万／1.8 万个候选；
旧循环按批次数停止，理论上会生成 1.92 亿个候选。单点与批量解析 IK 共用限位与分支逻辑。
实时 NN 超限或非有限输出会被拒绝，窗口保留上一有效姿态并提示，缺失权重不会自动冒充解析结果。

### ⚡ IDyn 完整重训结果（2026-11-10）

八个在用模型已完整重训。以下使用新的 `outputs` 权重、CPU 单线程推理、500 Hz、每条 6 秒，
统计全部时刻与三个关节的平均绝对角误差；离线验证则按协议排除每轨迹前100步。
新旧闭环共用冻结参考轨迹，PD/MPC与旧结果逐点一致，训练/验证输入没有精确行或整段参考前缀重叠。
这里只评估三条常规幅值、一条增幅轨迹，不代表 `outputs_bg/inv` 离线精度或普遍泛化。

统一轨迹验证集上的力矩 MAE（新权重，CPU连续推理）：MLP **0.048702**、ESN **0.052703**、
LSTM **0.011552**、GRU **0.011052 N·m**。该验证集用于选最佳权重，不属于独立测试集。

| seed／幅值 | PD | MPC | PD+MLP | PD+ESN | PD+LSTM | PD+GRU |
|------|------:|------:|------:|------:|------:|------:|
| 999／1 | 2.1170° | 0.6890° | 0.1656° | 0.1726° | 0.0364° | 0.0470° |
| 1000／1 | 2.1862° | 0.7082° | 0.1933° | 0.2059° | 0.0588° | 0.0534° |
| 1001／1 | 2.2973° | 0.7865° | 0.1710° | 0.2020° | 0.0424° | 0.0455° |
| 三条幅值1均值 | 2.2002° | 0.7279° | 0.1766° | 0.1935° | 0.0459° | 0.0486° |
| 999／2（增幅测试） | 2.4330° | 0.7724° | 0.2644° | 0.2880° | 0.1079° | 0.0618° |

相较旧权重，三条幅值1均值中 MLP/ESN/GRU 分别改善约1.09%/6.36%/3.19%，LSTM变差约12.51%。
ESN改善PD，但仍没有优于MLP；不能据此宣称摩擦记忆优势或硬件泛化。
四组六方案均数值有限，积分后无关节越界、无命令饱和；`mj_step` 返回时记录的接触数为零，未监测全部 RK4 内部子阶段。
完整新旧比较见 [重训记录](docs/retrain_2026-11-10.md)；缺陷修复与旧权重基线见 [审查记录](docs/audit_2026-11-10.md)。

---

## 🎮 可视化操作说明

以下键盘操作属于 `visualize.py mujoco` / `visualize_mujoco.py` 的 IK 几何查看器。
它们直接设置关节姿态并调用 `mj_forward`，适合检查 IK 几何运动，不能验证摩擦或动力学补偿。
IDyn 动力学窗口使用独立参考轨迹，不支持这些按键移动参考目标。

| 按键 | 动作 |
|------|------|
| ↑ / ↓ | 目标 +Y / -Y（长按连续移动） |
| ← / → | 目标 -X / +X（长按连续移动） |
| 小键盘 8 / 2 | 目标 +Z / -Z（仅 3-DOF）|
| R | 复位目标点 |
| Space | 启动 / 停止自动轨迹演示 |
| P | 打印当前坐标、关节角、末端位置 |
| 鼠标左键拖动 | 旋转视角 |
| 鼠标右键拖动 | 平移视角 |
| 鼠标滚轮 | 缩放 |

---

## 🔬 技术细节

### 为什么选 sin/cos 输出编码？

直接回归角度 θ 存在 ±π 跳变（如 -π 和 +π 是同一方向但数值差 2π），网络难以学习。  
改为输出 `[sinθ, cosθ]` 后圆周空间连续，再用 `atan2(sin, cos)` 还原，精度显著提升。

### 3-DOF IK 几何解耦

同一末端坐标可能对应多个关节构型，训练数据通过以下限制固定映射：

1. **θ₁ 解析求解**：在正水平半径域取 `θ₁ = atan2(Y, X)`
2. **降维**：计算 `r_xy = √(X²+Y²)`, `z' = Z - L0`，将问题降为 2D 平面 IK
3. **固定分支**：训练只用正肘分支，并排除悬臂向后伸的 `r < 0` 区域（本数据约占原候选的 20.7%）

关节角误差使用度，末端位置误差使用毫米；两种指标不能混用。当前结果分别报告训练域验证与广域压力测试。

### IDyn 关键参数

| 参数 | 值 | 说明 |
|------|----|------|
| KP_DATA / KD_DATA | [300,300,200] / [5,5,4] | 数据生成高刚度 PD，实际跟踪误差需测量 |
| KP / KD（测试） | [8,8,6] / [1.5,1.5,1] | 软刚度 PD，凸显前馈效果 |
| ctrlrange | ±50 N·m | 支持峰值前馈幅值 |
| `<compiler angle="radian"/>` | MJCF 必需 | 否则关节范围默认解释为角度 |
| Coulomb 摩擦 | frictionloss | 由约束求解器处理；不能仅凭此参数认定存在独立迟滞状态或递归模型优势 |

### 技术栈

| 包 | 版本 | 用途 |
|----|------|------|
| Python | 3.10 | 语言版本 |
| PyTorch | 2.11+ | GPU 训练 + `torch.compile` |
| MuJoCo | 3.6+ | 物理仿真 + `launch_passive` 可视化 |
| NumPy | 2.x | 运动学计算 |
| Matplotlib | 3.8+ | 可视化对比图 |
| SciPy | 1.13+ | 信号处理（激励轨迹） |
| uv | 最新 | 包管理 / 运行（禁用 pip）|

---

## 📄 许可证

[MIT License](LICENSE) © 2026 Lijinzh
