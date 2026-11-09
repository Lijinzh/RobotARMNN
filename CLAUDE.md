# CLAUDE.md

本机通用协作规则已上收到全局 `~/.claude/CLAUDE.md`；该文件不随仓库分发。

## 项目地图（重大变更时 AI 自动更新）
- **目标**：2/3DOF 逆运动学与 3DOF 逆动力学神经网络学习 Demo（MuJoCo 仿真）
- **入口**：IK 为 `src/data_gen.py` → `src/train.py` → `src/visualize.py`；IDyn 为 `src/data_gen_idyn.py` → `src/train_idyn.py` → `src/visualize_idyn.py`
- **数据流**：激励轨迹生成（高刚度 PD 采样）→ 总力矩示教命令 `outputs` → MLP/ESN/LSTM/GRU 训练 → PD＋总力矩前馈闭环对比；模型不是纯摩擦残差
- **关键参数**：`KP_DATA=[300,300,200]`，`KD_DATA=[5,5,4]`，`ctrlrange=±50 N·m`，角度单位为弧度；仅排除相邻 body 对碰撞，其他接触仍启用
- **Checkpoints**：`checkpoints/idyn_3dof_{mlp,esn,lstm,gru}_best.pt`

## 项目专属规范
- 可视化文字元素（标题/标签/图例）至少 `18pt`
- 如果修改训练流程、结果输出或使用方式，需同步更新 `README.md` 中的说明和训练结果表格；仓库现有文风偏活泼，可保持一致

## 常用指令
```bash
uv sync
uv add <pkg>
uv run <script.py>
uv run --with <pkg> <script.py>
uv run pytest
git status --short
git diff
uv cache prune
```
