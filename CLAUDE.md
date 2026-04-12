# CLAUDE.md

## 身份规约
- 每次回复必须以 **"金哲大人"** 开头，全程中文（含代码注释）
- 硬件：RTX 4090 24G，GPU 训练代码务必榨干性能

## 互动协议
- **需求模糊** → 先提问澄清，不猜测
- **编码前** → 先描述方案，待批准后动手
- **改动 > 3 个文件** → 必须拆分小任务逐步执行
- **被纠正后** → 回复含"反思"并制定"不再犯计划"

## 技术栈
- 包管理 / 运行：**uv**（禁止 pip，禁止兼容性代码）
- 版本控制：git（改坏必须能回滚）
- 写完代码后自动执行运行检查，验证逻辑无误

## 项目地图（重大变更时 AI 自动更新）
- **目标**：3DOF 机械臂逆动力学神经网络学习 Demo（MuJoCo 仿真）
- **入口**：`src/data_gen_idyn.py` → `src/train_idyn.py` → `src/visualize_idyn.py`
- **数据流**：激励轨迹生成（高刚度PD采样）→ 训练标签（outputs: τ含Coulomb摩擦）→ MLP/ESN/LSTM/GRU 训练 → 闭环仿真对比
- **关键参数**：KP_DATA=300, KD_DATA=5, ctrlrange=±50 N·m, angle=radian, 自碰撞已排除
- **Checkpoints**：`checkpoints/idyn_3dof_{mlp,esn,lstm,gru}_best.pt`

## 编码规范
- 关键函数 / 复杂逻辑加详细中文注释
- 可视化文字元素（标题/标签/图例）≥ 18pt
- Bug 修复：先写能复现的测试用例，再修复，再验证
- 修改代码,同步更新 README.md 中的使用说明和训练结果表格,README里面要有表情,显得生动活泼

## 常用指令
```bash
uv sync                          # 同步环境
uv add <pkg>                     # 添加依赖
uv run <script.py>               # 运行脚本（自动激活虚拟环境）
uv run --with <pkg> <script.py>  # 临时依赖运行
pytest                           # 执行测试
git checkout . / git reset --hard HEAD  # 回滚
uv cache prune                   # 清理无用缓存
```
