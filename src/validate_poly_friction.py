"""在修正的 bg 标签上比较多项式、方向特征和低速饱和特征。

固定协议：轨迹 0–7 训练、8–9 测试；每条轨迹从第 0 点每隔 25 点抽样。
只在训练轨迹内部用 0–5 拟合、6–7 验证来选择特征参数，再用全部 8 条重拟合。
本实验是离线软件回归代理，不代表器件实测或闭环控制效果。
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SATURATION_WIDTHS = np.geomspace(0.002, 0.2, 31)  # 单位 rad/s，仅由内部验证选择
STRIDE = 25


def design_matrix(x, model):
    """多项式使用训练集标准化；摩擦特征使用原始速度和物理单位阈值。"""
    z = (np.asarray(x, dtype=np.float64) - model["mean"]) / model["scale"]
    polynomial = np.prod(z[:, None, :] ** model["powers"][None, :, :], axis=2)
    friction = str(model["friction"])
    if friction == "none":
        return polynomial
    velocity = np.asarray(x)[:, 3:6]
    if friction == "sign":
        extra = np.sign(velocity)
    elif friction == "saturation":
        extra = np.clip(velocity / float(model["width"]), -1.0, 1.0)
    else:
        raise ValueError(f"未知摩擦特征：{friction}")
    return np.column_stack([polynomial, extra])


def fit_model(x, y, degree, friction="none", width=0.0):
    """带截距和全部交叉项的普通最小二乘；不使用测试数据或正则化。"""
    if friction == "saturation" and width <= 0:
        raise ValueError("低速饱和阈值必须为正数")
    terms = [()]
    for order in range(1, degree + 1):
        terms.extend(itertools.combinations_with_replacement(range(x.shape[1]), order))
    scale = x.std(axis=0)
    model = {
        "mean": x.mean(axis=0),
        "scale": np.where(scale > 0, scale, 1.0),
        "powers": np.array([np.bincount(term, minlength=x.shape[1]) for term in terms]),
        "degree": degree,
        "friction": friction,
        "width": width,
    }
    model["coefficients"] = np.linalg.lstsq(design_matrix(x, model), y, rcond=None)[0]
    return model


def predict_model(model, x):
    return design_matrix(x, model) @ model["coefficients"]


def metrics(y, prediction):
    """按各关节方差之和归一化，ddof=0；同时保存有单位的 RMSE。"""
    mse = np.mean((prediction - y) ** 2, axis=0)
    variance = np.var(y, axis=0)
    if np.any(variance <= 0):
        raise ValueError("NRMSE 评估需要各关节标签具有非零方差")
    nrmse = float(np.sqrt(mse.sum() / variance.sum()))
    return {
        "nrmse": nrmse,
        "nrmse_joint": np.sqrt(mse / variance).tolist(),
        "rmse_Nm_joint": np.sqrt(mse).tolist(),
        "variance_weighted_r2": 1.0 - nrmse**2,
    }


def load_data(path):
    """明确排除其余轨迹，且在每条轨迹内部从索引 0 开始直接抽样。"""
    with np.load(path) as saved:
        if "label_version" not in saved or saved["label_version"].item() != 2:
            raise ValueError("需要 label_version=2 的修正数据，请先归档并重新生成数据")
        traj_id = saved["traj_id"]
        if any(not np.any(traj_id == tid) for tid in range(10)):
            raise ValueError("固定评估协议需要编号 0–9 的 10 条轨迹")
        indices = np.concatenate([np.flatnonzero(traj_id == tid)[::STRIDE] for tid in range(10)])
        return (
            saved["inputs"][indices].astype(np.float64),
            saved["outputs_bg"][indices].astype(np.float64),
            traj_id[indices],
            indices,
            float(saved["dt"]),
        )


def select_candidates(x, y, traj_id):
    """仅接收 8 条训练轨迹；为每个阶数选择饱和阈值，再比较全部候选。"""
    if not np.array_equal(np.unique(traj_id), np.arange(8)):
        raise ValueError("参数选择只能接收编号 0–7 的训练轨迹")
    fit = traj_id < 6
    validation = ~fit
    candidates = []
    search = []
    for friction, degrees in (
        ("none", range(1, 6)),
        ("sign", range(1, 4)),
        ("saturation", range(1, 4)),
    ):
        for degree in degrees:
            trials = []
            widths = SATURATION_WIDTHS if friction == "saturation" else [0.0]
            for width in widths:
                model = fit_model(x[fit], y[fit], degree, friction, float(width))
                score = metrics(y[validation], predict_model(model, x[validation]))["nrmse"]
                trial = {
                    "degree": degree, "friction": friction, "width_rad_s": float(width),
                    "validation_nrmse": score,
                }
                trials.append(trial)
                search.append(trial)
            candidates.append(min(trials, key=lambda trial: trial["validation_nrmse"]))
    selected = min(candidates, key=lambda trial: trial["validation_nrmse"])
    return candidates, selected, search


def run_experiment(data_path, output_dir):
    x, y, traj_id, indices, dt = load_data(data_path)
    train = traj_id < 8
    test = ~train
    candidates, selected, search = select_candidates(x[train], y[train], traj_id[train])
    results = []
    selected_model = None
    selected_prediction = None
    for candidate in candidates:
        model = fit_model(
            x[train], y[train], candidate["degree"], candidate["friction"],
            candidate["width_rad_s"],
        )
        prediction = predict_model(model, x[test])
        result = {
            **candidate,
            "coefficients_per_joint": int(model["coefficients"].shape[0]),
            "train": metrics(y[train], predict_model(model, x[train])),
            "test": metrics(y[test], prediction),
        }
        results.append(result)
        if candidate is selected:
            selected_model, selected_prediction = model, prediction

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "selected_model.npz", **selected_model, label_version=2)
    with np.load(output_dir / "selected_model.npz") as restored:
        np.testing.assert_allclose(predict_model(restored, x[test]), selected_prediction, atol=1e-12)
    np.savez(
        output_dir / "test_predictions.npz", inputs=x[test], outputs_bg=y[test],
        prediction=selected_prediction, traj_id=traj_id[test], sample_indices=indices[test],
    )
    errors = selected_prediction - y[test]
    speed_errors = []
    for joint in range(3):
        near = np.abs(x[test, joint + 3]) < 0.05
        speed_errors.append({
            "joint": joint + 1,
            "low_speed_samples": int(near.sum()),
            "low_speed_rmse_Nm": float(np.sqrt(np.mean(errors[near, joint] ** 2)))
            if near.any() else None,
            "other_samples": int((~near).sum()),
            "other_rmse_Nm": float(np.sqrt(np.mean(errors[~near, joint] ** 2)))
            if (~near).any() else None,
        })
    report = {
        "data_path": str(Path(data_path).resolve()),
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "protocol": {
            "label": "outputs_bg", "label_version": 2,
            "input": "inputs = [q_des, dq_des]", "stride": STRIDE,
            "source_hz": 1.0 / dt, "sampled_hz": 1.0 / (dt * STRIDE),
            "train_trajectories": list(range(8)), "test_trajectories": [8, 9],
            "inner_fit_trajectories": list(range(6)), "validation_trajectories": [6, 7],
            "train_samples": int(train.sum()), "test_samples": int(test.sum()),
            "sampling": "每条轨迹从索引 0 起直接抽样，无滤波、无插值；保留启动段",
            "fit": "含截距和全部交叉项的普通最小二乘，无正则化",
            "selection": "仅按内部验证 NRMSE 选择阶数、特征和饱和阈值，再用 8 条重拟合",
            "normalization": "每次拟合只使用当次训练数据，摩擦特征使用原始 rad/s 速度",
            "saturation_width_candidates_rad_s": SATURATION_WIDTHS.tolist(),
            "nrmse_definition": "sqrt(sum_j MSE_j / sum_j Var(y_j)), ddof=0",
        },
        "selected": selected,
        "results": results,
        "validation_search": search,
        "selected_errors_by_speed": speed_errors,
        "limitations": "仅 2 条固定测试轨迹、仿真 bg 标签和离线软件代理；未验证器件或闭环",
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print("阶数  摩擦特征       阈值(rad/s)  内部验证NRMSE  测试NRMSE  每关节系数")
    for result in results:
        print(
            f"{result['degree']:>4}  {result['friction']:<12}  "
            f"{result['width_rad_s']:>10.6f}  {result['validation_nrmse']:>13.6f}  "
            f"{result['test']['nrmse']:>9.6f}  {result['coefficients_per_joint']:>10}"
        )
    print(f"内部验证选中：{selected}")
    print(f"结果已保存至：{output_dir.resolve()}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "idyn_3dof.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "export" / "idyn_poly_friction")
    args = parser.parse_args()
    run_experiment(args.data, args.output_dir)
