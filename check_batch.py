"""比较 batch=1 和 batch=4 的 Adam 训练，并检查工况状态是否串用。"""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import train as solver


def read_log(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def train_and_check(model, rows, features, physics, epochs, batch, output, size):
    """使用正式训练入口，并记录它真正提交的工况及塑性状态。"""
    case_ids = {id(row): i for i, row in enumerate(rows, 1)}
    trials = {}
    expected = {}
    committed = []
    resets = Counter()
    stopped_with_law_above_tol = Counter()
    evaluate_state = solver.evaluate_state
    advance = solver.Trial.advance
    cfg = SimpleNamespace(batch=batch, physics_workers=1, max_plastic_steps=solver.MAX_PLASTIC_STEPS,
                          size=size, width=solver.WIDTH, modes=solver.MODES, blocks=solver.BLOCKS,
                          depth_stride=4, smoke=False)

    def observe_trial(row, state, physics):
        trial = evaluate_state(row, state, physics)
        trials[id(trial)] = case_ids[id(row)], row
        return trial

    def observe_advance(trial, increment):
        case_id, row = trials.pop(id(trial))
        updated = advance(trial, increment)
        if increment is trial.law_increment:
            # 固定监测点沿纯本构推进，只用于评估。
            return updated
        saved = expected.get(case_id)
        if saved is None:
            assert trial.state.step == 0 and trial.state.previous_pressure is None, (batch, case_id)
            assert np.all(trial.state.displacement == 0), (batch, case_id)
            assert np.all(trial.state.hardening == row["Q1"]), (batch, case_id)
        else:
            assert trial.state is saved, ("塑性状态串用或丢失", batch, case_id, trial.state.step)
        assert updated.step == trial.state.step + 1, (batch, case_id)
        committed.append((case_id, trial.state.step))
        small_increment = float(increment.max()) <= solver.PLASTIC_TOL_MM
        if small_increment or updated.step >= cfg.max_plastic_steps:
            resets[case_id] += 1
            expected[case_id] = None
            if small_increment and float(trial.law_increment.max()) > solver.PLASTIC_TOL_MM:
                stopped_with_law_above_tol[case_id] += 1
        else:
            expected[case_id] = updated
        return updated

    updates_per_epoch = (len(rows) + batch - 1) // batch
    output.mkdir()
    with patch.object(solver, "evaluate_state", observe_trial), patch.object(solver.Trial, "advance", observe_advance):
        solver.train(model, rows, features, physics, ((epochs * updates_per_epoch, 0),), output, cfg)
    log = read_log(output / "training.jsonl")
    visits = [(case, step) for record in log for case, step in zip(record["cases"], record["plastic_steps"])]
    assert visits == committed, ("采样记录与状态提交不一致", batch)
    order = [case for case, _ in visits]
    for start in range(0, len(order), len(rows)):
        assert sorted(order[start:start + len(rows)]) == list(range(1, len(rows) + 1)), batch
    assert Counter(order) == Counter({case: epochs for case in case_ids.values()}), batch

    probes = read_log(output / "training_probe.jsonl")
    return {
        "adam_updates": len(log),
        "case_visits": len(visits),
        "resets_by_case": dict(resets),
        "stopped_with_law_above_tol_by_case": dict(stopped_with_law_above_tol),
        "initial_probe": probes[0]["fields"],
        "final_probe": probes[-1]["fields"],
    }, order


@torch.no_grad()
def check_batch_math(model, rows, features, physics):
    """冻结权重和输入，核对批量预测及批量损失的归约。"""
    trials = [solver.evaluate_state(row, physics.initial_state(row), physics) for row in rows]
    x = torch.stack([features(row, trial.state, trial.contact) for row, trial in zip(rows, trials)])
    scale = x.new_tensor([trial.increment_scale for trial in trials])[:, None, None]
    law = torch.as_tensor(np.stack([trial.law_increment for trial in trials]), device=x.device)
    high = torch.as_tensor(np.stack([solver.high_pressure_mask(trial.contact.pressure) for trial in trials]),
                           device=x.device)
    divisor = law.amax(dim=(1, 2), keepdim=True).clamp_min(5 * solver.PLASTIC_TOL_MM)
    together = model(x, tuple(range(len(rows)))) * scale
    separately = torch.cat([model(x[i:i + 1], (0,)) for i in range(len(rows))]) * scale
    torch.testing.assert_close(together, separately, rtol=2e-4, atol=1e-9)
    residual = (together - law) / divisor
    batch_loss = solver.loss_parts(residual, high)
    individual = [solver.loss_parts(residual[i:i + 1], high[i:i + 1]) for i in range(len(rows))]
    torch.testing.assert_close(batch_loss, torch.stack(individual).mean(0), rtol=2e-5, atol=1e-8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=64, help="对照网格尺寸")
    parser.add_argument("--epochs", type=int, default=8, help="每组工况的访问次数")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs 必须为正整数。")
    torch.set_num_threads(4)
    rows, h, dx, dy, dz = solver.load_inputs(solver.DATA, args.size)
    rows = rows[:25]
    physics = solver.BilinearPhysics(solver.ElasticContact(h, dx, dy, workers=1), dx, dy, dz, 4)
    features = solver.Features(h, dx * dy, args.device, len(physics.weights))
    output = args.output or solver.ROOT / "batch_checks" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    orders = []
    for batch in (1, 4):
        torch.manual_seed(42)
        model = solver.PlasticFNO(features.channels, field_channels=features.field_channels).to(args.device)
        print(f"测试 batch={batch}，每个训练工况访问 {args.epochs} 次", flush=True)
        summaries[batch], order = train_and_check(model, rows, features, physics, args.epochs, batch,
                                                 output / f"batch_{batch}", args.size)
        orders.append(order)
    assert orders[0] == orders[1], "两个设置的工况访问顺序不同。"
    check_batch_math(model, rows[:4], features, physics)
    report = {"grid_size": args.size, "epochs": args.epochs, "sampling_and_state_checks_passed": True,
              "batch_math_check_passed": True, "batches": summaries}
    path = output / "summary.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for batch, result in summaries.items():
        mse = [f"{f['plastic_step']}={f['constitutive_residual_mse_mm2']:.3e}"
               for f in result["final_probe"]]
        early_stops = sum(result["stopped_with_law_above_tol_by_case"].values())
        print(f"batch={batch}，Adam 更新 {result['adam_updates']} 次，工况访问 {result['case_visits']} 次，"
              f"本构仍超阈值时重置 {early_stops} 次，固定状态 MSE (mm^2) {'，'.join(mse)}")
    print(f"工况抽样、状态推进和批量损失检查通过。报告 {path.resolve()}")


if __name__ == "__main__":
    main()
