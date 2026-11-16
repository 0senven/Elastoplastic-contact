"""在 PyCharm 直接运行，加载最新 operator.pt 并绘制第 26–30 组验收场。"""

import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat

from train_elastic import ELASTIC_WEIGHTS, load_elastic
from physics import BilinearPhysics, load_inputs, surfaces
from train import DATA, MAX_VALIDATION_STEPS, ROOT, TRAINING_METHOD, Features, PlasticFNO, evaluate, predict_case


CHECKPOINT = None  # 留空自动选择最新 operator.pt，也可填 Path(r"D:\...\operator.pt")
PLOT_FIELDS = (
    ("pressure", "Pressure", "MPa", 1.0),
    ("elastic", "Elastic displacement", "µm", 1000.0),
    ("plastic", "Plastic displacement", "µm", 1000.0),
)


def plot_case(case, evaluation_dir, output, extent, matlab_parameters):
    last = case["iterations"][-1]
    number, step = case["case"], last["correction_round"]
    fields_path = evaluation_dir / f"case_{number:02d}_round_{step}.npz"
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), layout="constrained")
    status = "converged" if last["plastic_converged"] else "plastic iteration did not converge"
    if "model_solve_s" in last:
        model_time = f"Model solve {last['model_solve_s']:.2f} s"
    else:
        model_time = f"Model validation {last['elapsed_s']:.2f} s"
    matlab_s = matlab_parameters.get("compute_time_s")
    matlab_time = f"MATLAB solve {matlab_s:.2f} s" if matlab_s is not None else "MATLAB timing unavailable"
    rounds = last.get("prediction_iterations")
    steps = f"{rounds} prediction rounds" if rounds is not None else f"{step} plastic steps"
    fig.suptitle(f"Case {number} | {steps} | {status}\n{model_time} | {matlab_time}",
                 fontsize=14)
    try:
        with np.load(fields_path) as fields:
            for row, (key, name, unit, scale) in enumerate(PLOT_FIELDS):
                predicted = fields[key] * scale
                reference = fields[f"reference_{key}"] * scale
                difference = predicted - reference
                low = min(predicted.min(), reference.min())
                high = max(predicted.max(), reference.max())
                error_limit = max(float(np.abs(difference).max()), np.finfo(float).eps)
                mae = np.mean(np.abs(difference), dtype=np.float64)
                for col, (field, title, cmap, vmin, vmax) in enumerate((
                    (predicted, "Model", "viridis", low, high),
                    (reference, "MATLAB", "viridis", low, high),
                    (difference, f"Model − MATLAB | MAE {mae:.3g} {unit}", "RdBu_r", -error_limit, error_limit),
                )):
                    ax = axes[row, col]
                    image = ax.imshow(field, origin="lower", extent=extent, interpolation="nearest",
                                      cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.set_title(title)
                    ax.set_xlabel("x (mm)")
                    ax.set_ylabel(f"{name}\ny (mm)" if col == 0 else "y (mm)")
                    if col == 1:
                        fig.colorbar(image, ax=axes[row, :2], label=unit, shrink=0.86)
                    elif col == 2:
                        fig.colorbar(image, ax=ax, label=unit, shrink=0.86)
        fig.savefig(output / f"case_{number:02d}.png", dpi=180)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, help="图片输出目录")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-plastic-steps", type=int, default=MAX_VALIDATION_STEPS)
    parser.add_argument("--case", type=int, choices=range(1, 31), metavar="1–30",
                        help="仅预测指定工况，输出最终场及状态")
    parser.add_argument("--load", type=float, help="覆盖单工况的法向载荷（N）")
    parser.add_argument("--saved", action="store_true", help="直接绘制所选模型已有的完整验收结果")
    parser.add_argument("--elastic-checkpoint", type=Path, default=ELASTIC_WEIGHTS)
    args = parser.parse_args()
    if not args.saved and not args.elastic_checkpoint.is_file():
        parser.error("请先运行 train_elastic.py 训练弹性模型，或用 --elastic-checkpoint 指定权重。")
    if args.max_plastic_steps < 1:
        parser.error("最大塑性步数须为正整数。")
    if args.case is not None and args.saved:
        parser.error("--case 与 --saved 不能同时使用。")
    if args.load is not None and (args.case is None or not np.isfinite(args.load) or args.load <= 0):
        parser.error("--load 需要配合 --case 使用，载荷须为有限正数。")

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = max((ROOT / "runs").glob("*/operator.pt"),
                         key=lambda path: path.stat().st_mtime_ns, default=None)
    if checkpoint is None or not checkpoint.is_file():
        parser.error("找不到 operator.pt，请通过 --checkpoint 指定模型。")
    if args.saved and checkpoint.name != "operator.pt":
        parser.error("--saved 读取最终 operator.pt 对应的验收结果。")
    print(f"模型文件 {checkpoint.resolve()}", flush=True)

    output = args.output or checkpoint.parent / "plots"
    evaluation_dir = checkpoint.parent if args.saved else output / "evaluation"
    report_path = evaluation_dir / "validation.json"
    if args.saved and not report_path.is_file():
        parser.error("模型目录缺少 validation.json，请直接运行以生成验收结果。")
    output.mkdir(parents=True, exist_ok=True)

    if args.saved:
        # 已验收的 NPZ 同时保存了对应的预测和同网格 MATLAB 参考场。
        raw_h, _, dx, dy = surfaces(DATA / "fixed_surfaces")
        domain_x, domain_y = raw_h.shape[1] * dx, raw_h.shape[0] * dy
    else:
        torch.set_num_threads(4)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("training_method") != TRAINING_METHOD:
            parser.error("此权重与新版输入输出格式不兼容，请重新训练。")
        rows, h, dx, dy, dz = load_inputs(DATA, saved["size"])
        domain_x, domain_y = h.shape[1] * dx, h.shape[0] * dy
        elastic = load_elastic(args.elastic_checkpoint, h, dx, dy, args.device)
        physics = BilinearPhysics(elastic, dx, dy, dz, saved["depth_stride"], build_stress_kernels=False)
        features = Features(h, dx * dy, args.device,
                            use_updated_surface=saved.get("use_updated_surface", False),
                            material_channels=saved.get("material_channels", 6))
        model = PlasticFNO(features.channels, saved["width"], saved["modes"], saved["blocks"],
                           depth_count=len(physics.weights),
                           unet_every_step=saved.get("unet_every_step", False),
                           field_channels=features.field_channels)
        model.load_state_dict(saved["model"])
        model.to(args.device)
        if args.case is not None:
            row = rows[args.case - 1].copy()
            if args.load is not None:
                row["F"] = args.load
            fields, info = predict_case(model, features, physics, row, args.case, args.max_plastic_steps)
            path = output / f"prediction_case_{args.case:02d}.npz"
            np.savez_compressed(path, **fields, x_extent_mm=(-domain_x / 2, domain_x / 2),
                                y_extent_mm=(-domain_y / 2, domain_y / 2))
            path.with_suffix(".json").write_text(
                json.dumps(dict(case=args.case, inputs=row, **info), ensure_ascii=False, indent=2),
                encoding="utf-8")
            status = "收敛" if info["plastic_converged"] else "达到轮数上限"
            print(f"预测完成，{info['prediction_iterations']} 轮，{status}。结果 {path.resolve()}", flush=True)
            return
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        evaluate(model, features, physics, rows, DATA, evaluation_dir, args.max_plastic_steps)

    cases = json.loads(report_path.read_text(encoding="utf-8"))
    if [case["case"] for case in cases] != list(range(26, 31)):
        parser.error("验收报告尚未包含完整的第 26–30 组。")
    extent = (-domain_x / 2, domain_x / 2, -domain_y / 2, domain_y / 2)
    with ZipFile(DATA / "case_验证集解析解.zip") as archive:
        for case in cases:
            number = case["case"]
            raw = archive.read(f"case_{number:06d}/parameters.mat")
            parameters = loadmat(BytesIO(raw), simplify_cells=True)["parameters"]
            plot_case(case, evaluation_dir, output, extent, parameters)
    print(f"对照图已保存到 {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
