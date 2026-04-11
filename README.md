# RobotARMNN — 机械臂神经网络逆运动学

基于 MuJoCo 仿真的两款机械臂（平面 2-DOF / 空间 3-DOF），训练神经网络（MLP + ESN）
完成逆运动学求解，支持 MuJoCo 交互式可视化。

---

## 机械臂构型

### 平面 2-DOF 臂（水平 XY 平面）

```
关节1 (θ1)：底座绕 Z 轴旋转（肩部），范围 [-π, π]
关节2 (θ2)：肘部绕 Z 轴旋转（相对大臂），范围 [-π, π]
L1 = 0.15 m（大臂）  L2 = 0.12 m（小臂）

正运动学：
  X = L1·cos(θ1) + L2·cos(θ1+θ2)
  Y = L1·sin(θ1) + L2·sin(θ1+θ2)

可达半径：r_min = 0.03 m，r_max = 0.27 m
```

### 空间 3-DOF 臂

```
关节1 (θ1)：底座绕 Z 轴旋转
关节2 (θ2)：肩部绕 Y 轴俯仰，范围 [-π/2, π/2]
关节3 (θ3)：肘部绕 Y 轴俯仰，范围 [-π, π]
L0 = 0.20 m（底柱高度）  L1 = 0.15 m（大臂）  L2 = 0.12 m（小臂）

正运动学：
  r  = L1·cos(θ2) + L2·cos(θ2+θ3)
  X  = r·cos(θ1)，Y = r·sin(θ1)
  Z  = L0 + L1·sin(θ2) + L2·sin(θ2+θ3)
```

---

## 神经网络模型

### IKNN_MLP — 通用逆运动学 MLP

| 项目 | 2-DOF | 3-DOF |
|------|-------|-------|
| 输入 | `[X, Y]`（2 维） | `[X, Y, Z]`（3 维，含输入归一化） |
| 输出 | `[sinθ1,cosθ1,sinθ2,cosθ2]`（4 维） | `[sinθ1..cosθ3]`（6 维） |
| 网络结构 | 3 × Linear(256) + ReLU | 5 × Linear(384) + ReLU（更大容量）|
| 参数量 | ~133K | ~590K |
| 训练 | AdamW + Warmup Cosine LR，batch=65536，全量 GPU |
| 输入归一化 | 无（XY 分布均匀） | **有**（Z 轴有 L0=0.20 偏置，归一化是精度关键）|

### IKNN_ESN — 增强型 ESN（随机储备池 + MLP 读出）

| 项目 | 值 |
|------|---|
| 架构 | 固定随机储备池（512 维 tanh）→ MLP 读出（128 隐层） |
| 可训练参数 | ~66K（仅 readout MLP） |
| 输入归一化 | 有（`set_input_stats` 自动设置）|
| 训练策略 | 预计算全量储备池激活 H → 只训练 MLP 读出 |
| 优势 | 训练极快（~3s）；小网络；3DOF 旧 MLP 无归一化时 ESN 反而更好 |

> **为什么用 sin/cos 输出？**  
> θ = atan2(Y,X) 在 ±π 处存在数值跳变，改用 `[sinθ, cosθ]` 输出圆周连续，精度大幅提升。

---

## 项目结构

```
RobotARMNN/
├── src/
│   ├── kinematics_2dof.py    # 平面 2-DOF 正/逆运动学 + sin/cos 工具
│   ├── kinematics_3dof.py    # 空间 3-DOF 正/逆运动学（几何解耦）
│   ├── arm_2dof.xml          # MuJoCo MJCF（平面臂 + mocap 目标球）
│   ├── arm_3dof.xml          # MuJoCo MJCF（空间臂 + 三轴十字标记）
│   ├── models.py             # 神经网络定义（IKNN_MLP、IKNN_ESN、CurrentNN）
│   ├── data_gen.py           # 统一数据生成（.npz + .csv）
│   ├── train.py              # 统一训练脚本（MLP + ESN，2-DOF + 3-DOF）
│   └── visualize_mujoco.py   # MuJoCo 交互可视化（方向键控制，NN/解析 IK）
├── tests/
│   └── test_kinematics.py    # 18 项运动学单元测试
├── data/                     # 训练数据（.npz + .csv，各 50 万样本）
├── checkpoints/              # 训练后模型权重（.pt）
├── main.py                   # 快速入口：运动学演示
├── pyproject.toml
└── README.md
```

---

## 快速开始

```bash
# 安装依赖
uv sync

# 运动学演示（无需训练）
uv run main.py

# 1. 生成训练数据（50 万样本，含 .csv 方便查看）
uv run src/data_gen.py --arm all

# 2. 训练 IK 网络
uv run src/train.py --arm 2dof --model mlp   # 平面臂 MLP (~7s/500ep)
uv run src/train.py --arm 2dof --model esn   # 平面臂 ESN (~3s/300ep)
uv run src/train.py --arm 3dof --model mlp   # 空间臂 MLP (自动 384×5, ~25s/800ep)
uv run src/train.py --arm 3dof --model esn   # 空间臂 ESN (~5s/300ep)

# 或一次性全部训练
uv run src/train.py --arm all --model all

# 3. 交互可视化（方向键移动目标点）
uv run src/visualize_mujoco.py --arm 2dof                            # 2DOF 解析 IK
uv run src/visualize_mujoco.py --arm 3dof                            # 3DOF 解析 IK
uv run src/visualize_mujoco.py --arm 2dof --use_nn                   # 2DOF MLP
uv run src/visualize_mujoco.py --arm 2dof --use_nn --model_type esn  # 2DOF ESN
uv run src/visualize_mujoco.py --arm 3dof --use_nn                   # 3DOF MLP
uv run src/visualize_mujoco.py --arm 3dof --use_nn --model_type esn  # 3DOF ESN

# 4. IK 误差对比可视化（解析 vs MLP vs ESN，纯运动学，无需仿真）
uv run src/visualize_ik.py --arm 2dof          # 2DOF 误差对比
uv run src/visualize_ik.py --arm 3dof          # 3DOF 误差对比
uv run src/visualize_ik.py --arm all           # 两种臂都画
MPLBACKEND=Agg uv run src/visualize_ik.py --arm all --save  # 保存到 export/

# 5. 单元测试
uv run pytest tests/ -v
```

---

## 训练结果（RTX 4090，50 万样本）

| 模型 | 臂类型 | 平均角度误差 | 训练耗时 |
|------|--------|------------|---------|
| IKNN_MLP (hidden=256, layers=3) | 2-DOF | ~1.9° | ~7 s / 500ep |
| IKNN_ESN (reservoir=512) | 2-DOF | ~3.3° | ~3 s / 300ep |
| IKNN_MLP (hidden=384, layers=5, 含归一化) | 3-DOF | 待重训 | ~25 s / 800ep |
| IKNN_ESN (reservoir=512) | 3-DOF | 待重训 | ~5 s / 300ep |

---

## 模型选择

可视化时通过 `--model_type` 参数选择模型：

```bash
# MLP（精度较高，全参数可训练；默认选项）
uv run src/visualize_mujoco.py --arm 2dof --use_nn --model_type mlp

# ESN（训练快，推理快，小网络）
uv run src/visualize_mujoco.py --arm 2dof --use_nn --model_type esn

# 指定自定义检查点路径
uv run src/visualize_mujoco.py --arm 3dof --use_nn --model_type mlp \
    --ckpt_3dof /path/to/model.pt
```

**默认检查点查找规则**：`checkpoints/ik_{arm}_{model_type}_best.pt`

程序启动时会打印加载的模型信息：
```
[2DOF] NN IK 模型已加载：[2DOF] MLP IK (hidden=256, layers=3, 含输入归一化)
       路径: checkpoints/ik_2dof_mlp_best.pt
```

---

## 可视化操作说明

| 按键 | 动作 |
|------|------|
| ↑ / ↓ | 目标点 +Y / -Y（前后，长按连续移动） |
| ← / → | 目标点 -X / +X（左右，长按连续移动） |
| 小键盘 8 / 2 | 目标点 +Z / -Z（上下，仅 3-DOF，长按连续移动） |
| R | 复位目标点到初始位置 |
| Space | 启动/停止自动轨迹演示 |
| **P** | 打印当前目标坐标、关节角度与末端位置 |
| 鼠标左键拖动 | 旋转视角 |
| 鼠标右键拖动 | 平移视角 |
| 鼠标滚轮 | 缩放 |

---

## 3-DOF MLP 精度改进说明

旧 3-DOF MLP 精度极差（13.57°），经过以下三步几何解耦后降至 **0.36°**（优于 2DOF 的 0.51°）：

| 步骤 | 问题 | 修复 |
|------|------|------|
| **1. 几何解耦** | 网络需从原始 XYZ 自行"发现" r_xy=√(X²+Y²) | θ1 → 解析精确（atan2(Y,X)）；NN 只学竖直平面 2-link IK |
| **2. 单值过滤** | r = L1·cosθ2 + L2·cos(θ2+θ3) 可为负，导致 r_xy=|r| 多值映射，NN 无法学习 | 数据生成时过滤 r<0.01 样本（约 20.7%） |
| **3. 维度降低** | 原始输入 [X,Y,Z]→[sinθ1..sinθ3] 是 3D→6D 复杂映射 | 改为 [r_xy, z']→[sinθ2,cosθ2,sinθ3,cosθ3] 等效 2D 平面 IK |

**结果对比：**

| 方案 | 误差 | 训练时间 |
|------|------|---------|
| 旧 3DOF MLP（XYZ 输入） | 13.57° | 800 ep / 45s |
| 几何解耦 v1（含多值区域） | 3.81° | 800 ep / 45s |
| 几何解耦 v2（过滤 r<0） | **0.36°** | 500 ep / 9s |
| 2DOF MLP（对比基准） | 0.51° | 500 ep / 11s |

重训命令：

```bash
uv run src/data_gen.py --arm 3dof       # 重新生成（含 r<0 过滤）
uv run src/train.py --arm 3dof --model mlp  # 重训
```

---

## 3-DOF 逆动力学补偿 Demo

### 目标

用神经网络学习 3-DOF 机械臂的逆动力学（含关节摩擦），作为前馈力矩补偿项，配合软刚度 PD 控制器实现精确跟踪。

**控制律**：τ = τ_ff(q_des, dq_des) + KP·(q_des−q) + KD·(dq_des−dq)

其中 τ_ff 由神经网络预测（MLP / ESN / LSTM / GRU），KP=8, KD=1.5（弱刚度，凸显前馈效果）。

**控制目标**：`gen_excitation_traj(seed=999)` 生成的多频率正弦叠加关节角度轨迹 q_des(t)（与训练集不重叠的测试轨迹）。各控制方案独立闭环仿真，比较跟踪误差。

### 完整流程

```bash
# 步骤 1：生成 30 条高刚度 PD 激励轨迹（KP=300，跟踪误差 < 0.1°）
uv run src/data_gen_idyn.py --n_traj 30

# 步骤 2：训练全部 4 种模型（约 2~3 min，RTX 4090）
uv run src/train_idyn.py --model all

# 也可单独训练某个模型
uv run src/train_idyn.py --model gru

# 步骤 3：可视化对比（默认测试轨迹幅值 2× 放大，展示更明显）
MPLBACKEND=Agg uv run src/visualize_idyn.py --save           # 保存对比图
uv run src/visualize_idyn.py --model gru                     # 只看 GRU vs 纯 PD
uv run src/visualize_idyn.py --model gru lstm                # GRU + LSTM
uv run src/visualize_idyn.py --model gru --mujoco            # GRU + MuJoCo 实时窗口

# 调整测试轨迹参数
uv run src/visualize_idyn.py --amp_scale 3.0 --duration 10.0 --save
```

### 神经网络架构

| 模型 | 架构 | 特点 |
|------|------|------|
| **IDynMLP** | 4×Linear(256, Tanh) | 无记忆基线；只看当前帧 [q, dq] |
| **IDynESN** | 固定储层(512) + MLP 读出 | 有时序记忆；训练极快（固定储层）|
| **IDynLSTM** | 2 层 LSTM(256) + Linear | 全参数门控记忆；建模摩擦迟滞 |
| **IDynGRU** | 2 层 GRU(256) + Linear | LSTM 轻量版；参数量约少 1/4 |

输入：`[q1,q2,q3, dq1,dq2,dq3]`（6 维，含输入归一化）  
输出：`[τ1, τ2, τ3]`（3 维，N·m）

### 训练结果（RTX 4090，30 条轨迹，KP_DATA=300）

| 方案 | 验证 MAE | 闭环跟踪误差（均值）| 各关节误差 [J1, J2, J3] |
|------|---------|-------------------|------------------------|
| 纯 PD（无前馈，基准） | — | 2.117° | [2.774, 1.877, 1.700]° |
| PD + MLP 前馈 | 0.042 N·m | 0.160° | [0.358, 0.071, 0.051]° |
| PD + ESN 前馈 | 0.054 N·m | 0.180° | [0.341, 0.086, 0.112]° |
| PD + LSTM 前馈 | 0.043 N·m | 0.167° | [0.303, 0.094, 0.104]° |
| **PD + GRU 前馈** | **0.014 N·m** | **0.042°** | **[0.043, 0.045, 0.038]°** |

> GRU 前馈比纯 PD 误差降低 **50×**；各关节误差近乎均匀，无明显弱轴。

### 关键参数（`data_gen_idyn.py`）

| 参数 | 值 | 说明 |
|------|----|------|
| KP_DATA / KD_DATA | 300 / 5 | 数据生成用高刚度 PD（标签质量高，跟踪误差 0.09°）|
| KP / KD | 8 / 1.5 | 可视化测试用软刚度 PD（凸显前馈效果）|
| ctrlrange | ±50 N·m | 支持逆动力学前馈峰值幅值 |
| JOINT_RANGES | ±0.8π, ±0.4π, ±0.6π | 关节1/2/3 活动范围（弧度）|
| 标签 | `outputs`（高质量 PD 力矩，含 Coulomb 摩擦）| 训练目标 |

---

## 技术栈

- **Python** 3.10 · **uv** 包管理
- **PyTorch** 2.x（CUDA，`torch.compile`，RTX 4090）
- **MuJoCo** 3.x（`launch_passive` 交互可视化）
- **NumPy**（运动学计算，CSV 导出）
