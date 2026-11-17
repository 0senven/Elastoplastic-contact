"""Train the elastic contact MLP using only contact equations."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from physics import Contact, displacement_kernel, effective_modulus, load_inputs

ROOT = Path(__file__).resolve().parent
ELASTIC_WEIGHTS = ROOT / "runs" / "elastic" / "elastic.pt"
STAGES = ((3000, 150), (1000, 200), (0, 200))
ELASTIC_REVISION = 3
VALIDATION_LIMITS = {"fb_max": 2e-4, "penetration_mm": 2e-6, "load_rel_error": 1e-8}


def fourier_coordinates(shape, device):
    rng = np.random.default_rng(42)
    bands = np.concatenate([rng.normal(size=(2, 12)) * s for s in (.5, 2., 8.)], axis=1)
    ny, nx = shape
    xy = np.stack(np.meshgrid((np.arange(nx) + .5) / nx, (np.arange(ny) + .5) / ny), axis=-1)
    return (torch.as_tensor(2 * xy - 1, dtype=torch.float32, device=device),
            torch.as_tensor(bands, dtype=torch.float32, device=device))


class ElasticMLP(nn.Module):
    """Predict pressure with an MLP and recover displacement with the Green operator."""

    def __init__(self, h, dx, dy, device, *, coordinate_shape=None):
        super().__init__()
        self.h, self.n, self.area, self.workers = h, h.shape[0], dx * dy, 1
        self.fft_shape = (2 * h.shape[0], 2 * h.shape[1])
        self.length = max(float(np.ptp(h)), 1e-12)
        self.register_buffer("surface", torch.as_tensor(h, dtype=torch.float32, device=device), persistent=False)
        coordinates, bands = fourier_coordinates(h.shape, device)
        self.register_buffer("coordinates", coordinates, persistent=False)
        self.register_buffer("bands", bands, persistent=False)
        self.register_buffer("half_cell", coordinates.new_tensor((1 / h.shape[1], 1 / h.shape[0])),
                             persistent=False)
        self.coordinate_offsets = nn.Parameter(coordinates.new_zeros((1, 2, *(coordinate_shape or h.shape))))
        self.local_field = nn.Parameter(coordinates.new_zeros((1, 1, *(coordinate_shape or h.shape))))
        kernel = displacement_kernel(h.shape, dx, dy)
        # Uniform pressure gives the row sum of the mirrored Green operator.
        self.uniform_influence = float(kernel[0, 0].real)
        self.register_buffer("kernel", torch.as_tensor(kernel, dtype=torch.complex128, device=device),
                             persistent=False)
        self.mlp = nn.Sequential(nn.Linear(73, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
                                 nn.Linear(32, 1)).to(device)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, shape):
        offsets = self.coordinate_offsets
        if offsets.shape[-2:] != shape.shape:
            offsets = nn.functional.interpolate(offsets, size=shape.shape, mode="bilinear", align_corners=False)
        coordinates = self.coordinates + offsets[0].permute(1, 2, 0).tanh() * self.half_cell
        phase = 2 * torch.pi * (coordinates @ self.bands)
        position = torch.cat((phase.sin(), phase.cos()), dim=-1)
        local = self.local_field
        if local.shape[-2:] != shape.shape:
            local = nn.functional.interpolate(local, size=shape.shape, mode="bilinear", align_corners=False)
        inputs = torch.cat(((shape / self.length).unsqueeze(-1), position), dim=-1)
        return self.mlp(inputs).squeeze(-1) + local[0, 0]

    def project_pressure(self, scores, load):
        """Project dimensionless scores onto the simplex with mean one."""
        flat = scores.flatten()
        ordered = flat.sort(descending=True).values
        ranks = torch.arange(1, flat.numel() + 1, dtype=flat.dtype, device=flat.device)
        thresholds = (ordered.cumsum(0) - flat.numel()) / ranks
        active_count = (ordered > thresholds).sum().clamp_min(1)
        threshold = thresholds.gather(0, (active_count - 1).reshape(1))
        nominal = load / (self.area * flat.numel())
        return ((flat - threshold).clamp_min(0) * nominal).reshape_as(scores)

    def projection_scale(self, row, shape):
        nominal = row["F"] / (self.area * shape.numel())
        return self.uniform_influence * nominal / (effective_modulus(row) * self.length)

    def pressure_from_offset(self, shape, row, offset):
        # This project stores initial gap, so asperity height has the opposite sign.
        scale = self.projection_scale(row, shape)
        return self.project_pressure((-shape / self.length - offset) / scale, row["F"])

    def elastic_displacement(self, row, pressure):
        wide = torch.cat((pressure, pressure.flip((-1,))), dim=-1)
        mirrored = torch.cat((wide, wide.flip((-2,))), dim=-2)
        elastic = torch.fft.irfft2(torch.fft.rfft2(mirrored) * self.kernel.to(
            torch.complex128 if pressure.dtype == torch.float64 else torch.complex64), s=self.fft_shape)
        return elastic[:pressure.shape[0], :pressure.shape[1]] / effective_modulus(row)

    def contact_fields(self, row, shape, pressure):
        elastic = self.elastic_displacement(row, pressure)
        z = shape + elastic
        pressure_sum = pressure.sum(dtype=torch.float64)
        offset = torch.where(pressure_sum > 0,
                             (pressure * z).sum() / pressure_sum.clamp_min(1e-12), z.min())
        gap = z - offset
        nominal = row["F"] / (self.area * shape.numel())
        p, g = pressure / nominal, gap / self.length
        fb = torch.linalg.vector_norm(torch.stack((p, g)), dim=0) - p - g
        penetration = (-gap.min()).clamp_min(0)
        load = pressure_sum * self.area
        relative_load_error = (load - row["F"]) / row["F"]
        return pressure, elastic, gap, fb, penetration, load, relative_load_error

    @torch.no_grad()
    def predict_pressure(self, row, plastic):
        shape = self.updated_surface(plastic)
        return self.pressure_from_offset(shape, row, self(shape))

    def updated_surface(self, plastic):
        return self.surface + torch.as_tensor(plastic, dtype=self.surface.dtype, device=self.surface.device)

    @torch.no_grad()
    def predict_contact(self, row, plastic, pressure=None):
        """Recover contact fields, reusing pressure when it has already been predicted."""
        shape = self.updated_surface(plastic)
        if pressure is None:
            pressure = self.pressure_from_offset(shape, row, self(shape))
        pressure, elastic, gap, fb, penetration, load, error = self.contact_fields(row, shape, pressure)
        return Contact(pressure.cpu().numpy(), elastic.cpu().numpy(), gap.cpu().numpy(),
                       float(fb.abs().max()), float(penetration), 1, float(load), float(error.abs()))


def load_elastic(path, h, dx, dy, device):
    weights = torch.load(path, map_location=device, weights_only=True)
    if weights.get("elastic_revision") != ELASTIC_REVISION:
        raise ValueError("旧弹性权重与当前压力投影尺度不兼容，请重新训练。")
    metrics = weights.get("validation", {})
    if not all(0 <= metrics.get(key, float("inf")) <= limit
               for key, limit in VALIDATION_LIMITS.items()):
        raise ValueError(f"弹性权重 {path} 缺少达标的物理验证记录，请先训练并保存通过验证的权重。")
    local_field = weights["model"]["local_field"]
    model = ElasticMLP(h, dx, dy, device, coordinate_shape=local_field.shape[-2:])
    if local_field.dtype == torch.float64:
        model.double()
    model.load_state_dict(weights["model"])
    return model.eval().requires_grad_(False)


def elastic_loss(model, row, shape, stage):
    offset = model(shape)
    pressure = model.pressure_from_offset(shape, row, offset)
    _, elastic, gap, fb, max_penetration, _, _ = model.contact_fields(row, shape, pressure)
    nominal = row["F"] / (model.area * pressure.numel())
    normalized_pressure = pressure / nominal
    scale = model.projection_scale(row, shape)
    residual = (offset - (elastic / model.length - scale * normalized_pressure)).square()
    contact_loss = fb.square().mean() + (max_penetration / model.length).square()
    global_loss = residual.mean() + contact_loss
    if stage == 3:
        return global_loss
    if stage == 2:
        weights = normalized_pressure / (normalized_pressure + 0.1)
    else:
        weights = normalized_pressure > 0
    local_loss = (residual * weights).sum() / weights.sum().clamp_min(1)
    worst = residual.flatten().topk(min(128, residual.numel())).values.mean()
    penetration = (-gap / model.length).clamp_min(0).square()
    penetration_loss = penetration.flatten().topk(min(32, penetration.numel())).values.mean()
    return global_loss + local_loss + worst + penetration_loss


def training_surfaces(model):
    """Create unlabeled gap shapes by flattening high asperities at two scales."""
    shape = model.surface
    surfaces = [shape]
    for fraction, window in ((0.10, 5), (0.25, 13)):
        flattening = (fraction * model.length - shape).clamp_min(0)
        smooth = nn.functional.avg_pool2d(flattening[None, None], window, stride=1,
                                          padding=window // 2, count_include_pad=False)
        surfaces.append(shape + smooth[0, 0])
    return surfaces


@torch.no_grad()
def validation_metrics(model, rows):
    """Measure the worst contact violation across fixed, unlabeled validation samples."""
    shape = model.surface
    flattening = (0.175 * model.length - shape).clamp_min(0)
    smooth = nn.functional.avg_pool2d(flattening[None, None], 9, stride=1,
                                     padding=4, count_include_pad=False)
    worst = {key: 0.0 for key in VALIDATION_LIMITS}
    for surface in (shape, shape + smooth[0, 0]):
        offset = model(surface)
        for row in rows:
            pressure = model.pressure_from_offset(surface, row, offset)
            _, _, _, fb, penetration, _, load_error = model.contact_fields(row, surface, pressure)
            for key, value in zip(worst, (float(fb.abs().max()), float(penetration), float(load_error.abs()))):
                worst[key] = max(worst[key], value) if np.isfinite(value) else float("inf")
    return worst


def save_candidate(model, rows, path, best_score, stage, optimizer, step):
    metrics = validation_metrics(model, rows)
    eligible = all(metrics[key] <= limit for key, limit in VALIDATION_LIMITS.items())
    score = tuple(metrics[key] for key in VALIDATION_LIMITS)
    print(f"Elastic validation stage={stage} {optimizer}={step} "
          f"FB={score[0]:.3e} penetration={score[1]:.3e} mm load error={score[2]:.3e} "
          f"{'passed' if eligible else 'failed'}", flush=True)
    if eligible and (best_score is None or score < best_score):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"elastic_revision": ELASTIC_REVISION, "model": model.state_dict(),
                    "validation": metrics, "stage": stage, "optimizer": optimizer, "step": step}, path)
        print(f"Best elastic checkpoint: {path.resolve()}", flush=True)
        return score
    return best_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ELASTIC_WEIGHTS)
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.set_num_threads(4)
    rows, h, dx, dy, _ = load_inputs(ROOT / "data", args.size)
    rows = rows[:25]
    validation_rows = rows[4::5]
    model = ElasticMLP(h, dx, dy, args.device)
    best_score = None
    for stage, (adam_steps, lbfgs_steps) in enumerate(STAGES, start=1):
        if stage == 3:
            model.mlp.requires_grad_(False)
            model.coordinate_offsets.requires_grad_(False)
            model.double()
        samples = [(row, shape) for shape in training_surfaces(model) for row in rows]
        parameters = [p for p in model.parameters() if p.requires_grad]
        if adam_steps:
            adam = torch.optim.Adam(parameters, lr=1e-3)
            for step in range(1, adam_steps + 1):
                adam.zero_grad()
                row, shape = samples[(step - 1) % len(samples)]
                loss = elastic_loss(model, row, shape, stage)
                loss.backward()
                adam.step()
                if step == 1 or step % 100 == 0 or step == adam_steps:
                    print(f"Elastic stage {stage} Adam {step}/{adam_steps} loss={float(loss.detach()):.6e}",
                          flush=True)
                    best_score = save_candidate(model, validation_rows, args.output,
                                                best_score, stage, "Adam", step)

        lbfgs = torch.optim.LBFGS(parameters, max_iter=1, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            total = model.surface.new_zeros(())
            for row, shape in samples:
                loss = elastic_loss(model, row, shape, stage) / len(samples)
                loss.backward()
                total += loss.detach()
            return total

        for step in range(1, lbfgs_steps + 1):
            loss = lbfgs.step(closure)
            if step == 1 or step % 25 == 0 or step == lbfgs_steps:
                print(f"Elastic stage {stage} L-BFGS {step}/{lbfgs_steps} loss={float(loss):.6e}", flush=True)
                best_score = save_candidate(model, validation_rows, args.output,
                                            best_score, stage, "L-BFGS", step)

    if best_score is None:
        parser.exit(1, "弹性模型未通过物理保存门槛，本次未保存权重，已有文件未被覆盖。\n")


if __name__ == "__main__":
    main()
