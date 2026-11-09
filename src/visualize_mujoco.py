"""
MuJoCo 交互式机械臂可视化

功能：
  - 平面 2-DOF 机械臂（水平 XY 平面）和 3-DOF 空间机械臂 MuJoCo 实时渲染
  - 键盘控制目标点位置，以 IK 直接设置合法关节角；这是运动学摆位，不执行动力学积分
  - 默认使用解析 IK；显式 NN 模式缺权重时终止，非法输出保持上一有效姿态并提示

键盘操作：
  ↑ / ↓      — 目标点 +Y / -Y（前后）
  ← / →      — 目标点 -X / +X（左右）
  KP8 / KP2  — 目标点 +Z / -Z（上下，仅 3-DOF 有效，小键盘 8/2）
  R           — 复位目标点到初始位置
  SPACE       — 启动/暂停自动圆形轨迹演示
  P           — 显示当前目标坐标和关节角

鼠标操作（MuJoCo viewer 原生）：
  左键拖动   — 旋转视角
  右键拖动   — 平移视角
  滚轮       — 缩放

用法:
    # 2-DOF 平面臂（无需训练好的模型，使用解析 IK）
    uv run src/visualize_mujoco.py --arm 2dof

    # 3-DOF 空间臂
    uv run src/visualize_mujoco.py --arm 3dof

    # 使用已训练的 NN 做 IK（更快，精度取决于训练效果）
    uv run src/visualize_mujoco.py --arm 2dof --use_nn
    uv run src/visualize_mujoco.py --arm 3dof --use_nn
"""

import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.kinematics_2dof import (
    PlanarArmConfig,
    inverse_kinematics_planar,
    valid_joint_angles_planar,
)
from src.kinematics_3dof import (
    Arm3DOFConfig,
    inverse_kinematics_3dof,
    valid_joint_angles_3dof,
)

# =====================================================================
# GLFW 方向键码（> 128，不能用 chr()）
# =====================================================================
_GLFW_RIGHT = 262  # → 目标 +X
_GLFW_LEFT = 263  # ← 目标 -X
_GLFW_DOWN = 264  # ↓ 目标 -Y
_GLFW_UP = 265  # ↑ 目标 +Y
_GLFW_KP8 = 328  # 小键盘 8 目标 +Z（仅 3-DOF）
_GLFW_KP2 = 322  # 小键盘 2 目标 -Z（仅 3-DOF）

# 长按检测超时：主循环 ~60fps，GLFW 按键 repeat 约 30fps，
# 超过此时间无回调则视为键已松开
_HELD_TIMEOUT = 0.12  # 秒

# =====================================================================
# 控制状态（全局，供 key_callback 修改）
# =====================================================================
_STEP = 0.01  # 每次按键的目标点移动步长（米）

# 当前目标点（可被 key_callback 修改）
_target = np.zeros(3, dtype=np.float64)
_demo_running = False  # 是否在播放圆形轨迹演示
_demo_t = 0.0  # 演示轨迹时间参数
_print_info = False  # 是否在下一帧打印信息
_held_keys: dict[int, float] = {}  # keycode → 最后一次收到 callback 的时间戳


def make_key_callback(arm_type: str, demo_config: dict):
    """
    返回 key_callback 闭包，捕获目标点和演示状态。

    长按连续移动实现原理：
      - 方向键每次 callback（PRESS + REPEAT）均更新 _held_keys[keycode] 时间戳
      - 主循环每帧检查时间戳，若未超时则持续施加 _STEP 移动
      - 键松开后超过 _HELD_TIMEOUT 无回调，主循环自动停止移动

    参数:
        arm_type    : "2dof" 或 "3dof"
        demo_config : 包含演示轨迹参数（圆心、半径等）
    """
    global _target, _demo_running, _print_info, _held_keys

    def key_callback(keycode: int) -> None:
        global _target, _demo_running, _print_info, _held_keys

        # 方向键 / PgUp / PgDn：仅更新长按时间戳，移动逻辑在主循环
        if keycode in (_GLFW_UP, _GLFW_DOWN, _GLFW_LEFT, _GLFW_RIGHT, _GLFW_KP8, _GLFW_KP2):
            _held_keys[keycode] = time.time()
            return

        # 字符键（ASCII 0-127）
        key = chr(keycode) if 0 < keycode < 128 else None

        if key == "R":
            _target[:] = demo_config["init_target"]
            _demo_running = False
            _held_keys.clear()
        elif keycode == 32:  # SPACE
            _demo_running = not _demo_running
        elif key == "P":
            _print_info = True

    return key_callback


def clamp_target_2dof(target: np.ndarray, cfg: PlanarArmConfig) -> np.ndarray:
    """将目标点钳制到 2-DOF 平面臂可达范围内。"""
    xy = target[:2].copy()
    r = float(np.linalg.norm(xy))
    if r < cfg.r_min + 0.005:
        # 太近，推到最小可达半径
        if r < 1e-9:
            xy = np.array([cfg.r_min + 0.005, 0.0])
        else:
            xy = xy / r * (cfg.r_min + 0.005)
    elif r > cfg.r_max - 0.005:
        # 太远，缩到最大可达半径内
        xy = xy / r * (cfg.r_max - 0.005)
    return np.array([xy[0], xy[1], target[2]])


def clamp_target_3dof(target: np.ndarray, cfg: Arm3DOFConfig) -> np.ndarray:
    """将目标点钳制到 3-DOF 空间臂可达范围内（粗略球形限制）。"""
    t = target.copy()
    # Z 不能低于地面（0.05m）
    z_lo = max(0.05, cfg.L0 - cfg.reach_max + 0.01)
    z_hi = cfg.L0 + cfg.reach_max - 0.01
    t[2] = np.clip(t[2], z_lo, z_hi)
    xy = t[:2]
    r_xy = float(np.linalg.norm(xy))
    # 水平可达半径随 Z 变化（简单球形约束）
    z_prime = t[2] - cfg.L0
    r_max_at_z = np.sqrt(max(0, cfg.reach_max**2 - z_prime**2))
    if r_max_at_z < 0.02:
        r_max_at_z = 0.02
    if r_xy > r_max_at_z - 0.01:
        if r_xy < 1e-9:
            xy = np.array([r_max_at_z - 0.01, 0.0])
        else:
            xy = xy / r_xy * (r_max_at_z - 0.01)
    t[0], t[1] = xy[0], xy[1]
    return t


# =====================================================================
# 模型加载辅助函数
# =====================================================================


def load_ik_model(model_type: str, nn_ckpt: str, arm_name: str) -> tuple:
    """
    根据模型类型动态加载 IK 模型（MLP 或 ESN）。

    参数:
        model_type: "mlp" 或 "esn"
        nn_ckpt   : 检查点路径
        arm_name  : "2dof" 或 "3dof"（用于推断维度）

    返回:
        (nn_model, model_info_str) 或 (None, None) 如果加载失败
    """
    try:
        import torch

        from src.models import IKNN_ESN, IKNN_MLP

        if not os.path.exists(nn_ckpt):
            return None, None

        # 根据 arm_name 推断输入/输出维度
        # 注意：3DOF MLP 使用几何解耦后的竖直平面输入，维度与 2DOF 相同
        if arm_name == "2dof":
            input_dim, output_dim = 2, 4
            info_prefix = f"[{arm_name.upper()}] MLP IK"
        else:  # 3dof — 竖直平面子问题: (r_xy, z') → (sinθ2,cosθ2,sinθ3,cosθ3)
            input_dim, output_dim = 2, 4
            info_prefix = f"[{arm_name.upper()}] MLP IK (θ1=解析+NN竖直平面)"

        if model_type == "mlp":
            # MLP 支持新格式（config+state_dict）和旧格式（仅 state_dict）
            ckpt = torch.load(nn_ckpt, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "config" in ckpt:
                model = IKNN_MLP(**ckpt["config"])
                model.load_state_dict(ckpt["state_dict"])
                hd = ckpt["config"].get("hidden_dim", 256)
                nl = ckpt["config"].get("n_layers", 3)
                info_str = f"{info_prefix} (hidden={hd}, layers={nl}, 含输入归一化)"
            else:
                # 旧格式向后兼容
                model = IKNN_MLP(input_dim=input_dim, hidden_dim=256, output_dim=output_dim)
                model.load_state_dict(ckpt)
                info_str = f"{info_prefix} (旧格式, hidden=256)"
            model.eval()
            return model, info_str
        elif model_type == "esn":
            # ESN 从带配置和 state_dict 的 dict 加载
            ckpt = torch.load(nn_ckpt, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "config" in ckpt:
                model = IKNN_ESN(**ckpt["config"])
                model.load_state_dict(ckpt["state_dict"])
                info_str = (
                    f"[{arm_name.upper()}] ESN IK "
                    f"(reservoir={ckpt['config'].get('reservoir_dim', '?')}, "
                    f"readout={ckpt['config'].get('readout_hidden', '?')}, "
                    f"params≈66K 可训练)"
                )
            else:
                # 旧版本直接存 state_dict
                model = IKNN_ESN(input_dim=input_dim, output_dim=output_dim)
                model.load_state_dict(ckpt)
                info_str = f"[{arm_name.upper()}] ESN IK (默认配置)"
            model.eval()
            return model, info_str
        else:
            print(f"[ERROR] 未知模型类型: {model_type}")
            return None, None
    except Exception as e:
        print(f"[ERROR] 加载模型失败: {e}")
        return None, None


# =====================================================================


def run_2dof_visualizer(
    xml_path: str,
    cfg: PlanarArmConfig,
    use_nn: bool = False,
    nn_ckpt: str | None = None,
    model_type: str = "mlp",
) -> None:
    """
    启动 2-DOF 平面臂 MuJoCo 交互可视化。

    参数:
        xml_path: MJCF 文件路径
        cfg:      平面臂运动学配置
        use_nn:   是否使用 NN 做 IK（需要已训练的模型）
        nn_ckpt:  NN 权重路径
    """
    global _target, _demo_running, _demo_t, _print_info

    # ---- 加载模型 ----
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # ---- 初始目标点 ----
    init_target = np.array([0.20, 0.0, cfg.arm_height])
    _target = init_target.copy()

    demo_config = {
        "init_target": init_target.copy(),
        "center": np.array([0.0, 0.0, cfg.arm_height]),
        "radius": 0.20,
    }

    # ---- 获取 mocap body id ----
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    target_mocap_id = model.body_mocapid[target_body_id]

    # ---- 可选：加载 NN 模型 ----
    nn_model = None
    if use_nn:
        if not nn_ckpt:
            print("[2DOF] NN 模式缺少检查点，终止显示；未切换为解析 IK。")
            return
        nn_model, model_info = load_ik_model(model_type, nn_ckpt, "2dof")
        if nn_model is not None:
            print(f"[2DOF] NN IK 模型已加载：{model_info}")
            print(f"       路径: {nn_ckpt}")
        else:
            print("[2DOF] 加载 NN 模型失败，终止显示；未切换为解析 IK。")
            return
    else:
        print("[2DOF] 使用解析 IK（解析解精度最优）")

    # ---- 关节 id ----
    j1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
    j2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint2")

    key_cb = make_key_callback("2dof", demo_config)

    print("\n=== 2-DOF 平面臂 MuJoCo 运动学摆位（不执行动力学积分） ===")
    print("  ↑/↓: 目标 +Y/-Y    ←/→: 目标 -X/+X  （长按连续移动）")
    print("  R: 复位             SPACE: 圆形演示")
    print("  P: 打印当前状态     鼠标拖动: 旋转视角\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        # 初始视角（俯视，适合水平 XY 平面臂）
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -70
        viewer.cam.distance = 0.8
        viewer.cam.lookat[:] = [0, 0, 0.05]
        last_status = None

        while viewer.is_running():
            t_loop = time.time()

            # ---- 长按方向键连续移动目标点（60fps）----
            if not _demo_running:
                now = time.time()
                if now - _held_keys.get(_GLFW_UP, 0) < _HELD_TIMEOUT:
                    _target[1] += _STEP
                if now - _held_keys.get(_GLFW_DOWN, 0) < _HELD_TIMEOUT:
                    _target[1] -= _STEP
                if now - _held_keys.get(_GLFW_LEFT, 0) < _HELD_TIMEOUT:
                    _target[0] -= _STEP
                if now - _held_keys.get(_GLFW_RIGHT, 0) < _HELD_TIMEOUT:
                    _target[0] += _STEP

            # ---- 自动轨迹演示 ----
            if _demo_running:
                _demo_t += 0.02  # 每帧推进（约 50 Hz）
                r_base = demo_config["radius"]
                cx, cy = demo_config["center"][:2]
                # 玫瑰花曲线：半径随角度做 3 瓣正弦调制
                # r(t) = r_base * (0.6 + 0.4 * cos(3*t))
                # 轨迹在 XY 平面内呈带弧波的花瓣形
                r_wave = r_base * (0.6 + 0.4 * np.cos(3 * _demo_t))
                _target[0] = cx + r_wave * np.cos(_demo_t)
                _target[1] = cy + r_wave * np.sin(_demo_t)
                _target[2] = cfg.arm_height

            # ---- 钳制目标点 ----
            _target = clamp_target_2dof(_target, cfg)
            data.mocap_pos[target_mocap_id, :] = _target

            # ---- 计算 IK ----
            target_xy = _target[:2]
            if nn_model is not None:
                import torch

                with torch.no_grad():
                    inp = torch.tensor(target_xy, dtype=torch.float32).unsqueeze(0)
                    sincos = nn_model(inp).squeeze(0).numpy()
                theta1 = float(np.arctan2(sincos[0], sincos[1]))
                theta2 = float(np.arctan2(sincos[2], sincos[3]))
                angles = (theta1, theta2)
                if not np.isfinite(sincos).all():
                    angles = None
            else:
                angles = inverse_kinematics_planar(target_xy, cfg, elbow_up=True)

            valid = angles is not None and bool(valid_joint_angles_planar(angles, cfg))
            if valid:
                # 仅接受合法运动学姿态；不执行力矩控制或动力学积分。
                data.qpos[j1_id] = angles[0]
                data.qpos[j2_id] = angles[1]
            mujoco.mj_forward(model, data)
            source = "NN IK" if nn_model is not None else "Analytical IK"
            status = f"{source}: valid" if valid else f"{source}: REJECTED; holding last pose"
            if hasattr(viewer, "set_texts"):
                viewer.set_texts((None, None, "Kinematic pose only (no dynamics)", status))
            if not valid and status != last_status:
                print(f"[2DOF] {source} 无合法有限解，拒绝更新，保持上一有效姿态。")
            last_status = status

            # ---- 打印信息 ----
            if _print_info:
                ee_pos = data.sensor("ee_pos").data.copy()
                print(
                    f"  目标: ({_target[0]:.3f}, {_target[1]:.3f})  "
                    f"θ1={np.degrees(data.qpos[j1_id]):.1f}°  "
                    f"θ2={np.degrees(data.qpos[j2_id]):.1f}°  "
                    f"末端: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f})"
                )
                _print_info = False

            viewer.sync()

            # ---- 限制帧率 ~60fps ----
            elapsed = time.time() - t_loop
            if elapsed < 1 / 60:
                time.sleep(1 / 60 - elapsed)


def run_3dof_visualizer(
    xml_path: str,
    cfg: Arm3DOFConfig,
    use_nn: bool = False,
    nn_ckpt: str | None = None,
    model_type: str = "mlp",
) -> None:
    """
    启动 3-DOF 空间臂 MuJoCo 交互可视化。

    参数:
        xml_path: MJCF 文件路径
        cfg:      3-DOF 运动学配置
        use_nn:   是否使用 NN 做 IK
        nn_ckpt:  NN 权重路径
    """
    global _target, _demo_running, _demo_t, _print_info

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # ---- 初始目标点 ----
    init_target = np.array([0.20, 0.10, cfg.L0 + 0.15])
    _target = init_target.copy()

    demo_config = {
        "init_target": init_target.copy(),
        "center": np.array([0.0, 0.0, cfg.L0 + 0.10]),
        "radius": 0.18,
    }

    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    target_mocap_id = model.body_mocapid[target_body_id]

    # ---- 可选：加载 NN ----
    nn_model = None
    if use_nn:
        if not nn_ckpt:
            print("[3DOF] NN 模式缺少检查点，终止显示；未切换为解析 IK。")
            return
        nn_model, model_info = load_ik_model(model_type, nn_ckpt, "3dof")
        if nn_model is not None:
            print(f"[3DOF] NN IK 模型已加载：{model_info}")
            print(f"       路径: {nn_ckpt}")
        else:
            print("[3DOF] 加载 NN 模型失败，终止显示；未切换为解析 IK。")
            return
    else:
        print("[3DOF] 使用解析 IK")

    j1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
    j2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint2")
    j3_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint3")

    key_cb = make_key_callback("3dof", demo_config)

    print("\n=== 3-DOF 空间臂 MuJoCo 运动学摆位（不执行动力学积分） ===")
    print("  ↑/↓: 目标 +Y/-Y    ←/→: 目标 -X/+X    KP8/KP2: 目标 +Z/-Z  （长按连续移动）")
    print("  R: 复位             SPACE: 圆形演示")
    print("  P: 打印当前状态     鼠标拖动: 旋转视角\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        # 初始视角（斜侧视，适合观察空间臂）
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -30
        viewer.cam.distance = 1.0
        viewer.cam.lookat[:] = [0, 0, 0.25]
        last_status = None

        while viewer.is_running():
            t_loop = time.time()

            # ---- 长按方向键连续移动目标点（60fps）----
            if not _demo_running:
                now = time.time()
                if now - _held_keys.get(_GLFW_UP, 0) < _HELD_TIMEOUT:
                    _target[1] += _STEP
                if now - _held_keys.get(_GLFW_DOWN, 0) < _HELD_TIMEOUT:
                    _target[1] -= _STEP
                if now - _held_keys.get(_GLFW_LEFT, 0) < _HELD_TIMEOUT:
                    _target[0] -= _STEP
                if now - _held_keys.get(_GLFW_RIGHT, 0) < _HELD_TIMEOUT:
                    _target[0] += _STEP
                if now - _held_keys.get(_GLFW_KP8, 0) < _HELD_TIMEOUT:
                    _target[2] += _STEP
                if now - _held_keys.get(_GLFW_KP2, 0) < _HELD_TIMEOUT:
                    _target[2] -= _STEP

            if _demo_running:
                _demo_t += 0.02
                r = demo_config["radius"]
                cx, cy, cz = demo_config["center"]
                # 圆形 XY + Z 正弦波浪（周期与 XY 圆不同步，形成螺旋感）
                _target[0] = cx + r * np.cos(_demo_t)
                _target[1] = cy + r * np.sin(_demo_t)
                _target[2] = cz + 0.10 * np.sin(2 * _demo_t)

            _target = clamp_target_3dof(_target, cfg)
            data.mocap_pos[target_mocap_id, :] = _target

            if nn_model is not None:
                import torch

                # θ1 用精确解析解（等价于正运动学的逆操作）
                theta1 = float(np.arctan2(_target[1], _target[0]))
                # 竖直平面特征：r_xy = √(X²+Y²)，z' = Z - L0
                r_xy = float(np.sqrt(_target[0] ** 2 + _target[1] ** 2))
                z_prime = float(_target[2] - cfg.L0)
                with torch.no_grad():
                    inp = torch.tensor([[r_xy, z_prime]], dtype=torch.float32)
                    sincos = nn_model(inp).squeeze(0).numpy()
                theta2 = float(np.arctan2(sincos[0], sincos[1]))
                theta3 = float(np.arctan2(sincos[2], sincos[3]))
                angles = (theta1, theta2, theta3)
                if not np.isfinite(sincos).all():
                    angles = None
            else:
                angles = inverse_kinematics_3dof(_target, cfg, elbow_up=True)

            valid = angles is not None and bool(valid_joint_angles_3dof(angles, cfg))
            if valid:
                data.qpos[j1_id] = angles[0]
                data.qpos[j2_id] = angles[1]
                data.qpos[j3_id] = angles[2]
            mujoco.mj_forward(model, data)
            source = "NN IK" if nn_model is not None else "Analytical IK"
            status = f"{source}: valid" if valid else f"{source}: REJECTED; holding last pose"
            if hasattr(viewer, "set_texts"):
                viewer.set_texts((None, None, "Kinematic pose only (no dynamics)", status))
            if not valid and status != last_status:
                print(f"[3DOF] {source} 无合法有限解，拒绝更新，保持上一有效姿态。")
            last_status = status

            if _print_info:
                ee_pos = data.sensor("ee_pos").data.copy()
                print(
                    f"  目标: ({_target[0]:.3f}, {_target[1]:.3f}, {_target[2]:.3f})  "
                    f"θ1={np.degrees(data.qpos[j1_id]):.1f}°  "
                    f"θ2={np.degrees(data.qpos[j2_id]):.1f}°  "
                    f"θ3={np.degrees(data.qpos[j3_id]):.1f}°  "
                    f"末端: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})"
                )
                _print_info = False

            viewer.sync()
            elapsed = time.time() - t_loop
            if elapsed < 1 / 60:
                time.sleep(1 / 60 - elapsed)


# 主入口
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuJoCo 机械臂交互可视化")
    parser.add_argument(
        "--arm",
        type=str,
        default="2dof",
        choices=["2dof", "3dof"],
        help="选择机械臂类型：2dof（平面 XY）或 3dof（空间）",
    )
    parser.add_argument(
        "--use_nn",
        action="store_true",
        help="使用训练好的 NN 做 IK（需要先运行 train.py 生成权重）",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="mlp",
        choices=["mlp", "esn"],
        help="NN 模型类型：mlp（多层感知机）或 esn（回声状态网络）",
    )
    parser.add_argument(
        "--ckpt_2dof",
        type=str,
        default=None,
        help="平面 2-DOF NN 权重路径（默认自动查找 ik_2dof_[mlp|esn]_best.pt）",
    )
    parser.add_argument(
        "--ckpt_3dof",
        type=str,
        default=None,
        help="3-DOF NN 权重路径（默认自动查找 ik_3dof_[mlp|esn]_best.pt）",
    )
    args = parser.parse_args()

    src_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(src_dir, "..", "checkpoints")

    # 自动查找默认检查点
    if args.ckpt_2dof is None:
        default_ckpt_2dof = os.path.join(ckpt_dir, f"ik_2dof_{args.model_type}_best.pt")
        args.ckpt_2dof = default_ckpt_2dof if os.path.exists(default_ckpt_2dof) else None

    if args.ckpt_3dof is None:
        default_ckpt_3dof = os.path.join(ckpt_dir, f"ik_3dof_{args.model_type}_best.pt")
        args.ckpt_3dof = default_ckpt_3dof if os.path.exists(default_ckpt_3dof) else None

    if args.arm == "2dof":
        xml_path = os.path.join(src_dir, "arm_2dof.xml")
        cfg = PlanarArmConfig()
        run_2dof_visualizer(
            xml_path=xml_path,
            cfg=cfg,
            use_nn=args.use_nn,
            nn_ckpt=args.ckpt_2dof if args.use_nn else None,
            model_type=args.model_type,
        )
    else:
        xml_path = os.path.join(src_dir, "arm_3dof.xml")
        cfg = Arm3DOFConfig()
        run_3dof_visualizer(
            xml_path=xml_path,
            cfg=cfg,
            use_nn=args.use_nn,
            nn_ckpt=args.ckpt_3dof if args.use_nn else None,
            model_type=args.model_type,
        )
