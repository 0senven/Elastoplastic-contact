"""PyCharm 入口。前 25 组在线本构残差训练，后 5 组参考场仅用于验收。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter

# 在 PyCharm 点运行前可在这里改路径及训练配置。
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SIZE = 512
STAGES = ((3000, 0), (1000, 200), (0, 200))  # Adam 更新数，L-BFGS 外层步数
MAX_PLASTIC_STEPS = 500
MAX_VALIDATION_STEPS = 10000
PLASTIC_TOL_MM = 1e-6
TRAINING_METHOD = "online_coupled_hardening_elastic_mlp_v2"
ROLLOUT_REVISION = 4
ELASTIC_REL_L2_LIMIT = 0.05
VALIDATION_MSE_FACTOR = 0.0025  # 压力、塑性位移的 MSE 门槛占对应参考场均方值的比例
WIDTH, MODES, BLOCKS = 12, 8, 3
ADAM_LR = 1e-3
PROBE_EARLY_STEPS = 3
PROBE_LATE_STEP = 32  # 完成 32 次物理更新后的固定监测状态

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from train_elastic import ELASTIC_WEIGHTS, load_elastic
from physics import (
    BilinearPhysics, Contact, PlasticState, effective_modulus, load_inputs,
    references, surfaces,
)


class Spectral2D(nn.Module):
    """标准二维 FNO 频域层。"""

    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        self.weights = nn.Parameter(torch.randn(2, width, width, modes, modes, 2) * .02)

    def forward(self, x):
        spectrum = torch.fft.rfft2(x)
        filtered = torch.zeros_like(spectrum)
        modes = min(self.modes, x.shape[-2] // 2, spectrum.shape[-1])
        for index, y_band in enumerate((slice(0, modes), slice(-modes, None))):
            weight = torch.view_as_complex(self.weights[index, :, :, :modes, :modes].contiguous())
            filtered[:, :, y_band, :modes] = torch.einsum(
                "biyx,ioyx->boyx", spectrum[:, :, y_band, :modes], weight
            )
        return torch.fft.irfft2(filtered, s=x.shape[-2:])


class PlasticFNO(nn.Module):
    """标准二维 FNO，每个塑性步叠加浅层 U-Net 的修正。"""

    def __init__(self, channels, width=WIDTH, modes=MODES, blocks=BLOCKS, *,
                 depth_count, unet_every_step=True, field_channels=2):
        super().__init__()
        self.unet_every_step = unet_every_step
        self.lift = nn.Conv2d(channels, width, 1)
        self.spectral = nn.ModuleList(Spectral2D(width, modes) for _ in range(blocks))
        self.pointwise = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(blocks))
        self.head = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(),
                                  nn.Conv2d(width, depth_count, 1))
        self.encode = nn.Sequential(nn.Conv2d(field_channels, 8, 3, padding=1), nn.GELU())
        self.low = nn.Sequential(nn.Conv2d(8, 12, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(12, 8, 3, padding=1), nn.GELU())
        self.decode = nn.Sequential(nn.Conv2d(16, 8, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(8, depth_count, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, 0.05)
        nn.init.zeros_(self.decode[-1].weight)
        nn.init.zeros_(self.decode[-1].bias)

    def forward(self, inputs, initial_indices):
        x = self.lift(inputs)
        for spectral, pointwise in zip(self.spectral, self.pointwise):
            x = F.gelu(spectral(x) + pointwise(x))
        prediction = self.head(x)
        if self.unet_every_step or initial_indices:
            fields = inputs[:, :self.encode[0].in_channels]
            if not self.unet_every_step:
                # 旧权重验收时复现其训练时的首步分支。
                indices = torch.as_tensor(initial_indices, device=inputs.device)
                fields = fields.index_select(0, indices)
            fine = self.encode(fields)
            coarse = self.low(F.avg_pool2d(fine, 2))
            up = F.interpolate(coarse, size=fine.shape[-2:], mode="bilinear", align_corners=False)
            correction = self.decode(torch.cat((fine, up), dim=1))
            if self.unet_every_step:
                prediction = prediction + correction
            else:
                prediction = prediction.index_add(0, indices, correction)
        return F.softplus(prediction, beta=8.0)

    def increments(self, inputs, initial_indices, depth_weights, modulus, tangent_ratio, stress_scale):
        """同一逐深度屈服超量驱动两个增量，单位分别为 mm 和 MPa。"""
        excess = self(inputs, initial_indices) * stress_scale[:, None, None, None]
        integrated = (excess * depth_weights[None, :, None, None]).sum(dim=1)
        plastic = (0.1 * (1 - tangent_ratio) / modulus)[:, None, None] * integrated
        hardening = tangent_ratio[:, None, None, None] * excess
        return plastic, hardening


class Features:
    """物理场、材料参数和同网格复用的 72 通道坐标编码。"""

    def __init__(self, h, area, device, *, use_updated_surface=True,
                 material_channels=4):
        self.h, self.device, self.area = h, device, area
        self.use_updated_surface = use_updated_surface
        self.field_channels = 2 if use_updated_surface else 4
        self.material_channels = material_channels
        self.length = max(float(np.ptp(h)), 1e-10)
        rng = np.random.default_rng(42)
        self.b = np.concatenate([rng.normal(size=(2, 12)) * s for s in (.5, 2., 8.)], axis=1)
        grid = (np.arange(len(h)) + 0.5) / len(h)
        xy = np.stack(np.meshgrid(grid, grid, indexing="xy"), axis=-1)
        phase = 2 * np.pi * (xy @ self.b)
        position = np.concatenate((np.sin(phase), np.cos(phase)), axis=-1).transpose(2, 0, 1)
        self.position = torch.as_tensor(position, dtype=torch.float32, device=device)
        self.channels = self.field_channels + material_channels + len(self.position)

    def __call__(self, row, plastic, pressure, elastic=None):
        scale = self.length
        material = [effective_modulus(row) / 1e5, row["n1"], row["Q1"] / 1e3, row["T1"] / 1e4]
        if self.material_channels == 6:
            # 旧权重的剪切模量和名义压力保留原来的通道位置。
            material.insert(2, row["E1"] / (2 * (1 + row["n1"]) * 1e5))
            material.append(row["F"] / (self.area * plastic.size * 1e3))
        constants = np.broadcast_to(np.asarray(material)[:, None, None],
                                    (self.material_channels, *plastic.shape))
        pressure = pressure / 1e3
        if self.use_updated_surface:
            fields = np.stack(((self.h + plastic) / scale, pressure))
        else:
            # 旧权重按其训练时的输入顺序读取四个物理场。
            fields = np.stack((self.h / scale, plastic / scale, pressure, elastic / scale))
        physical = np.concatenate((fields, constants), axis=0)
        channels = torch.as_tensor(physical, dtype=torch.float32, device=self.device)
        return torch.cat((channels, self.position))


@dataclass
class Trial:
    """当前预测形貌上的接触解与单步本构更新，均在当前优化步内固定。"""

    state: PlasticState
    contact: Contact
    law_increment: np.ndarray
    next_hardening: np.ndarray

    def advance(self, increment):
        return PlasticState(self.state.displacement + increment, self.next_hardening,
                            step=self.state.step + 1)


def evaluate_state(row, state, physics):
    contact = physics.elastic.predict_contact(row, state.displacement)
    law, next_hardening = physics.constitutive_step(row, contact.pressure, state.hardening)
    return Trial(state, contact, law, next_hardening)


def increment_parameters(inputs, rows, physics):
    weights = inputs.new_tensor(physics.weights)
    modulus = inputs.new_tensor([effective_modulus(row) for row in rows])
    tangent = inputs.new_tensor([row["T1"] for row in rows]) / modulus
    stress_scale = inputs.new_tensor([
        max(row["Q1"], row["F"] / (physics.elastic.area * physics.elastic.h.size))
        for row in rows
    ])
    return weights, modulus, tangent, stress_scale


def coupled_loss(plastic, hardening, law, law_hardening, high):
    plastic_scale = law.amax(dim=(1, 2), keepdim=True).clamp_min(5 * PLASTIC_TOL_MM)
    hardening_scale = law_hardening.amax(dim=(1, 2, 3), keepdim=True).clamp_min(0.1)
    return (loss_parts((plastic - law) / plastic_scale, high)
            + loss_parts((hardening - law_hardening) / hardening_scale,
                         high[:, None].expand_as(hardening)))


def loss_parts(residual, high_pressure):
    """全场残差均方及承压点中最高压 5% 的残差均方。"""
    square = residual.square().flatten(1)
    mask = high_pressure.flatten(1)
    local = (square * mask).sum(1) / mask.sum(1).clamp_min(1)
    return torch.stack((square.mean(), local.mean()))


def high_pressure_mask(pressure):
    flat = pressure.ravel()
    active = np.flatnonzero(flat > 0)
    mask = np.zeros(flat.shape, dtype=bool)
    if active.size:
        count = max(1, int(np.ceil(0.05 * active.size)))
        top = np.argpartition(flat[active], -count)[-count:]
        mask[active[top]] = True
    return mask.reshape(pressure.shape)


def field_mse(error):
    """全场均方值，单位是输入场单位的平方。"""
    return float(np.mean(np.square(error, dtype=np.float64)))


def plastic_update_mse(trial, predicted_increment):
    law_increment = trial.law_increment
    return (field_mse(law_increment), field_mse(predicted_increment),
            field_mse(predicted_increment - law_increment))


class AdaptiveWeights:
    def __init__(self):
        self.average = None
        self.weights = np.ones(2)

    def update(self, losses, prior):
        values = losses.detach().cpu().numpy()
        self.average = values if self.average is None else 0.98 * self.average + 0.02 * values
        raw = np.array([1., prior]) / np.sqrt(self.average + 1e-10)
        if prior:
            self.weights = 0.1 + 1.8 * raw / raw.sum()
        else:
            self.weights = np.array([1., 0.])


def train(model, rows, features, physics, stages, out, cfg):
    """在各训练工况保存的塑性状态上训练在线本构残差。"""
    rng = np.random.default_rng(42)
    weights = AdaptiveWeights()
    device = next(model.parameters()).device
    batch_size = min(cfg.batch, len(rows))
    started = perf_counter()
    log = out / "training.jsonl"
    initial_states = [physics.initial_state(row) for row in rows]
    states = initial_states.copy()

    def prepare(indices):
        def trial_for(index):
            return evaluate_state(rows[index], states[index], physics)

        workers = min(cfg.physics_workers, len(indices))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                trials = list(pool.map(trial_for, indices))
        else:
            trials = [trial_for(index) for index in indices]
        inputs = torch.stack([features(rows[i], trial.state.displacement,
                                       trial.contact.pressure, trial.contact.elastic)
                              for i, trial in zip(indices, trials)])
        initial_indices = tuple(j for j, trial in enumerate(trials) if trial.state.step == 0)
        law = torch.as_tensor(np.stack([t.law_increment for t in trials]), device=device)
        law_hardening = torch.as_tensor(
            np.stack([t.next_hardening - t.state.hardening for t in trials]), device=device)
        parameters = increment_parameters(inputs, [rows[i] for i in indices], physics)
        high = torch.as_tensor(np.stack([high_pressure_mask(t.contact.pressure) for t in trials]),
                               device=device)

        def predict():
            return model.increments(inputs, initial_indices, *parameters)

        def compute():
            # 物理求解结果固定，梯度只穿过这一状态上的网络预测。
            plastic, hardening = predict()
            return coupled_loss(plastic, hardening, law, law_hardening, high)

        @torch.no_grad()
        def advance():
            plastic, hardening = predict()
            for i, trial, increment, delta_q in zip(indices, trials, plastic.cpu().numpy(), hardening.cpu().numpy()):
                if not (np.isfinite(increment).all() and np.isfinite(delta_q).all()):
                    raise RuntimeError(f"工况 {i + 1} 的塑性更新出现 NaN 或 Inf。")
                next_state = trial.advance(increment)
                finished = float(increment.max()) <= PLASTIC_TOL_MM
                states[i] = (initial_states[i] if finished or next_state.step >= cfg.max_plastic_steps
                             else next_state)

        return compute, advance, trials

    # 只从训练工况中取一个固定工况。按本构生成固定监测状态，只用于测量。
    probe_index = len(rows) - 1
    probe_row = rows[probe_index]
    probe_limit = PROBE_EARLY_STEPS if cfg.smoke else PROBE_LATE_STEP
    last_probe = min(probe_limit, cfg.max_plastic_steps - 1)
    probe_steps = set(range(min(PROBE_EARLY_STEPS, cfg.max_plastic_steps))) | {last_probe}
    probe_state = initial_states[probe_index]
    probes = []
    print(f"准备固定监测工况 {probe_index + 1}，计算到第 {last_probe + 1} 个塑性步", flush=True)
    for step in range(last_probe + 1):
        trial = evaluate_state(probe_row, probe_state, physics)
        if step in probe_steps:
            probes.append(trial)
        if step < last_probe:
            probe_state = trial.advance(trial.law_increment)

    @torch.no_grad()
    def record_probe(stage, optimizer, step):
        fields = []
        for trial in probes:
            inputs = features(probe_row, trial.state.displacement,
                              trial.contact.pressure, trial.contact.elastic)[None]
            initial_indices = (0,) if trial.state.step == 0 else ()
            plastic, hardening = model.increments(
                inputs, initial_indices, *increment_parameters(inputs, [probe_row], physics))
            prediction = plastic[0].cpu().numpy()
            hardening_error = hardening[0].cpu().numpy() - (trial.next_hardening - trial.state.hardening)
            law_mse, prediction_mse, residual_mse = plastic_update_mse(trial, prediction)
            fields.append(dict(plastic_step=trial.state.step + 1, law_increment_mse_mm2=law_mse,
                               predicted_increment_mse_mm2=prediction_mse,
                               constitutive_residual_mse_mm2=residual_mse,
                               hardening_residual_mse_mpa2=field_mse(hardening_error)))
        row = dict(stage=stage, optimizer=optimizer, step=step, case=probe_index + 1, fields=fields)
        with (out / "training_probe.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(row) + "\n")
        summary = "，".join(f"第 {field['plastic_step']} 步={field['constitutive_residual_mse_mm2']:.3e}"
                            for field in fields)
        print(f"固定工况 {probe_index + 1} 本构残差 MSE (mm^2) {summary}", flush=True)

    def save(stage, tag):
        torch.save({"model": model.state_dict(), "size": cfg.size, "width": cfg.width,
                    "modes": cfg.modes, "blocks": cfg.blocks, "stage": stage,
                    "training_method": TRAINING_METHOD, "depth_stride": cfg.depth_stride,
                    "rollout_revision": ROLLOUT_REVISION,
                    "unet_every_step": model.unet_every_step,
                    "use_updated_surface": features.use_updated_surface,
                    "material_channels": features.material_channels}, out / tag)

    def record(stage, optimizer, step, terms, lr, indices, trials):
        row = dict(stage=stage, optimizer=optimizer, step=step, global_mse=float(terms[0]),
                   local_mse=float(terms[1]), weights=weights.weights.tolist(), lr=lr,
                   seconds=perf_counter() - started,
                   cases=[int(i) + 1 for i in indices],
                   sample_kind="current",
                   plastic_steps=[trial.state.step for trial in trials],
                   law_increment_max_mm=max(float(trial.law_increment.max()) for trial in trials))
        with log.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row) + "\n")
        if step == 1 or step % 25 == 0:
            print(f"阶段{stage} {optimizer} {step}: 全局={row['global_mse']:.3e} "
                  f"局部={row['local_mse']:.3e}", flush=True)

    record_probe(0, "initial", 0)
    for stage, (adam_steps, lbfgs_steps) in enumerate(stages, 1):
        prior = (0., 1., .3)[stage - 1]
        # 阶段二引入高压区损失，给新目标自己的学习率调整过程。
        adam = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            adam, factor=.5, patience=5, min_lr=1e-6)
        order = rng.permutation(len(rows))
        start = 0
        monitor = None
        for step in range(1, adam_steps + 1):
            if start == len(order):
                order, start = rng.permutation(len(rows)), 0
            indices = order[start:start + batch_size]
            start += len(indices)
            compute, advance, trials = prepare(indices)
            adam.zero_grad(set_to_none=True)
            terms = compute()
            weights.update(terms, prior)
            coeff = torch.as_tensor(weights.weights, device=device, dtype=terms.dtype)
            (terms * coeff).sum().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            adam.step()
            advance()
            metric = float(terms[0].detach())
            monitor = metric if monitor is None else .95 * monitor + .05 * metric
            if step % 25 == 0:
                scheduler.step(monitor)
            record(stage, "Adam", step, terms.detach(), adam.param_groups[0]["lr"],
                   indices, trials)
            if step % 250 == 0:
                save(stage, "latest.pt")
            if step % 250 == 0 or step == adam_steps:
                record_probe(stage, "Adam", step)
        # 线搜索期间保持接触状态、本构右端和权重固定。完成工况块才推进。
        lbfgs_order = rng.permutation(len(rows)) if lbfgs_steps else ()
        lbfgs_start = 0
        for step in range(1, lbfgs_steps + 1):
            if (step - 1) % 5 == 0:
                if lbfgs_start == len(lbfgs_order):
                    lbfgs_order, lbfgs_start = rng.permutation(len(rows)), 0
                selected = lbfgs_order[lbfgs_start:lbfgs_start + batch_size]
                lbfgs_start += len(selected)
                compute, advance, trials = prepare(selected)
                with torch.no_grad():
                    weights.update(compute(), prior)
                coeff = torch.as_tensor(weights.weights, device=device, dtype=torch.float32)
                lbfgs = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=1,
                                          history_size=5, line_search_fn="strong_wolfe")

            def closure():
                lbfgs.zero_grad(set_to_none=True)
                loss = (compute() * coeff).sum()
                loss.backward()
                return loss

            lbfgs.step(closure)
            with torch.no_grad():
                terms = compute()
            record(stage, "L-BFGS", step, terms, None, selected, trials)
            if step % 5 == 0 or step == lbfgs_steps:
                advance()
            if step % 50 == 0:
                save(stage, "latest.pt")
            if step % 50 == 0 or step == lbfgs_steps:
                record_probe(stage, "L-BFGS", step)
        save(stage, f"stage_{stage}.pt")
    save(len(stages), "operator.pt")


@torch.no_grad()
def predict_case(model, features, physics, row, number, max_steps=MAX_VALIDATION_STEPS):
    """根据给定工况预测最终接触场和塑性位移。"""
    if max_steps < 1:
        raise ValueError("预测轮数必须为正整数。")
    if not features.use_updated_surface:
        raise ValueError("此塑性权重需要逐轮弹性位移输入，请使用等效形貌和压力输入的权重。")
    elastic = physics.elastic
    model.eval()
    started = perf_counter()
    displacement = np.zeros_like(elastic.h)
    steps = 0
    pressure = elastic.predict_pressure(row, displacement)
    contact_s = perf_counter() - started
    prediction_s = 0.0
    parameters = None
    for iteration in range(1, max_steps + 1):
        prediction_started = perf_counter()
        features_k = features(row, displacement, pressure.cpu().numpy())[None]
        if parameters is None:
            parameters = increment_parameters(features_k, [row], physics)
        plastic, hardening = model.increments(features_k, (0,) if steps == 0 else (), *parameters)
        increment, delta_q = plastic[0].cpu().numpy(), hardening[0].cpu().numpy()
        prediction_s += perf_counter() - prediction_started
        if not (np.isfinite(increment).all() and np.isfinite(delta_q).all()):
            raise RuntimeError(f"工况 {number} 的塑性更新出现 NaN 或 Inf。")
        increment_max = float(increment.max())
        converged = increment_max < PLASTIC_TOL_MM
        if iteration == 1 or iteration % 25 == 0 or converged or iteration == max_steps:
            print(f"工况 {number}，第 {iteration} 轮，最大塑性增量 "
                  f"{increment_max:.3e} mm，阈值 {PLASTIC_TOL_MM:.1e} mm，"
                  f"累计弹性推理 {contact_s:.2f} s，网络及输入处理 {prediction_s:.2f} s", flush=True)
        if converged:
            break
        displacement += increment
        steps += 1
        contact_started = perf_counter()
        pressure = elastic.predict_pressure(row, displacement)
        contact_s += perf_counter() - contact_started
    recovery_started = perf_counter()
    contact = elastic.predict_contact(row, displacement, pressure=pressure)
    recovery_s = perf_counter() - recovery_started
    solve_s = perf_counter() - started
    item = {"correction_round": steps, "prediction_iterations": iteration,
            "elapsed_s": solve_s, "model_solve_s": solve_s,
            "elastic_contact_s": contact_s + recovery_s, "network_prediction_s": prediction_s,
            "pressure_prediction_s": contact_s, "final_elastic_recovery_s": recovery_s,
            "fb_max": contact.fb_max, "penetration_mm": contact.penetration,
            "target_load_N": row["F"], "computed_load_N": contact.computed_load,
            "load_rel_error": contact.load_rel_error,
            "predicted_increment_max_mm": increment_max,
            "plastic_update_max_mm": increment_max, "plastic_update_limit_mm": PLASTIC_TOL_MM,
            "predicted_hardening_increment_max_mpa": float(delta_q.max()),
            "plastic_converged": converged}
    fields = dict(pressure=contact.pressure, elastic=contact.elastic,
                  plastic=displacement, gap=contact.gap)
    return fields, item


@torch.no_grad()
def evaluate(model, features, physics, rows, data, output, max_steps):
    """只在训练结束后打开参考 ZIP，比较纯网络塑性迭代的最终结果。"""
    if max_steps < 1:
        raise ValueError("验收轮数必须为正整数。")
    elastic = physics.elastic
    archive = data / "case_验证集解析解.zip"
    h1, h2, _, _ = surfaces(data / "fixed_surfaces")
    original = h1 - h2
    original -= original.min()
    results = []
    report_path = output / "validation.json"
    model.eval()
    for number, ref in references(archive, elastic.n, rows, original):
        row = rows[number - 1]
        mse_limits = {name: VALIDATION_MSE_FACTOR * field_mse(ref[name])
                      for name in ("pressure", "plastic")}
        fields, item = predict_case(model, features, physics, row, number, max_steps)
        for name in ("pressure", "elastic", "plastic"):
            field = fields[name]
            reference = ref[name]
            error = field - reference
            mse = field_mse(error)
            item[name + "_mse"] = mse
            if name == "elastic":
                relative_l2 = float(np.linalg.norm(error) / max(np.linalg.norm(reference), 1e-20))
                item["elastic_rel_l2"] = relative_l2
                item["elastic_rel_l2_limit"] = ELASTIC_REL_L2_LIMIT
                item["elastic_passed"] = relative_l2 <= ELASTIC_REL_L2_LIMIT
            else:
                item[name + "_mae"] = float(np.mean(np.abs(error), dtype=np.float64))
                item[name + "_mse_limit"] = mse_limits[name]
                item[name + "_passed"] = mse <= mse_limits[name]
            item[name + "_peak_abs"] = float(abs(field.max() - reference.max()))
        item["all_fields_passed"] = all(item[name + "_passed"]
                                         for name in ("pressure", "elastic", "plastic"))
        np.savez_compressed(output / f"case_{number:02d}_round_{item['correction_round']}.npz",
                            **fields, **{"reference_" + key: value for key, value in ref.items()})
        end = item
        passed = end["all_fields_passed"] and end["plastic_converged"]
        results.append({"case": number, "passed": passed, "iterations": [end]})
        report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        if passed:
            status = "通过"
        elif end["plastic_converged"] and end["correction_round"] == 0:
            status = "首轮增量低于阈值，未推进且场误差超限"
        else:
            status = "场误差超限" if end["plastic_converged"] else "塑性未收敛"
        print(f"验收工况 {number}：{status}，{end['correction_round']} 步。"
              f"模型求解耗时 {end['model_solve_s']:.2f} s，"
              f"最大塑性更新增量 {end['plastic_update_max_mm']:.3e} mm，"
              f"压力 MSE={end['pressure_mse']:.3e} MPa^2（门槛 {end['pressure_mse_limit']:.3e}），"
              f"压力 MAE={end['pressure_mae']:.3e} MPa，"
              f"塑性位移 MSE={end['plastic_mse']:.3e} mm^2（门槛 {end['plastic_mse_limit']:.3e}），"
              f"塑性位移 MAE={end['plastic_mae']:.3e} mm，"
              f"弹性位移相对 L2={end['elastic_rel_l2']:.3%}", flush=True)
    passed_cases = sum(case["passed"] for case in results)
    print(f"最终验收：{passed_cases}/{len(results)} 组塑性收敛且压力、塑性位移 MSE "
          f"及弹性位移相对 L2 均达到各自门槛。", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="两组工况、每阶段仅一两个优化步")
    parser.add_argument("--pilot", action="store_true", help="25 组，缩短阶段供快速验收")
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--modes", type=int, default=MODES)
    parser.add_argument("--blocks", type=int, default=BLOCKS)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--depth-stride", type=int, default=4)
    parser.add_argument("--physics-workers", type=int, default=3)
    parser.add_argument("--max-plastic-steps", type=int, default=MAX_PLASTIC_STEPS,
                        help="训练中单条塑性轨迹的最大步数")
    parser.add_argument("--max-validation-steps", type=int, default=MAX_VALIDATION_STEPS,
                        help="验收的最大网络预测轮数")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="仅做验收，不运行训练")
    parser.add_argument("--elastic-checkpoint", type=Path, default=ELASTIC_WEIGHTS)
    args = parser.parse_args()
    if not args.elastic_checkpoint.is_file():
        parser.error("请先运行 train_elastic.py 训练弹性模型，或用 --elastic-checkpoint 指定权重。")
    if args.batch < 1 or args.physics_workers < 1 or args.max_plastic_steps < 1 or args.depth_stride < 1:
        parser.error("批量、物理线程数、塑性步数和深度步幅都必须为正整数。")
    if args.max_validation_steps < 1:
        parser.error("验收轮数必须为正整数。")
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(42)
    output = args.output or Path(__file__).resolve().parent / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        if saved.get("training_method") != TRAINING_METHOD:
            parser.error("此权重与新版输入输出格式不兼容，请重新训练。")
        for key in ("size", "width", "modes", "blocks", "depth_stride"):
            setattr(args, key, saved[key])
    output.mkdir(parents=True, exist_ok=True)
    rows, h, dx, dy, dz = load_inputs(DATA, args.size)
    elastic = load_elastic(args.elastic_checkpoint, h, dx, dy, args.device)
    kernels = "弹性 MLP" if args.checkpoint else "弹性 MLP 和体应力影响核"
    print(f"{args.size}×{args.size}，设备 {args.device}，准备{kernels}，输出 {output}", flush=True)
    physics = BilinearPhysics(elastic, dx, dy, dz, args.depth_stride,
                              build_stress_kernels=args.checkpoint is None)
    use_updated_surface = True if args.checkpoint is None else saved.get("use_updated_surface", False)
    material_channels = 4 if args.checkpoint is None else saved.get("material_channels", 6)
    features = Features(h, dx * dy, args.device,
                        use_updated_surface=use_updated_surface, material_channels=material_channels)
    unet_every_step = True if args.checkpoint is None else saved.get("unet_every_step", False)
    model = PlasticFNO(features.channels, args.width, args.modes, args.blocks,
                       depth_count=len(physics.weights), unet_every_step=unet_every_step,
                       field_channels=features.field_channels).to(args.device)
    np.save(output / "coordinate_frequencies.npy", features.b)
    if args.checkpoint:
        model.load_state_dict(saved["model"])
        if not features.use_updated_surface:
            print("载入旧版四物理场权重，按其训练时的输入格式验收。", flush=True)
        if not model.unet_every_step:
            print("载入的是首步 U-Net 权重，按其训练配置进行验收。", flush=True)
        if saved.get("rollout_revision") != ROLLOUT_REVISION:
            print("此权重使用旧版状态推进训练，本次验收使用新版推进与最大增量判据。", flush=True)
    else:
        stages = ((2, 0), (1, 1), (0, 1)) if args.smoke else (
            ((300, 0), (100, 20), (0, 20)) if args.pilot else STAGES)
        count = 2 if args.smoke else 25
        print(f"在线本构残差训练工况 1–{count}，留出 26–30 组，阶段 {stages}", flush=True)
        train(model, rows[:count], features, physics, stages, output, args)
    max_steps = min(args.max_validation_steps, 3) if args.smoke else args.max_validation_steps
    evaluate(model, features, physics, rows, DATA, output, max_steps)


if __name__ == "__main__":
    main()
