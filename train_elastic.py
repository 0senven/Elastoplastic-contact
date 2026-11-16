"""Train the elastic contact MLP using only contact equations."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from physics import Contact, displacement_kernel, effective_modulus, load_inputs

ROOT = Path(__file__).resolve().parent
ELASTIC_WEIGHTS = ROOT / "runs" / "elastic" / "elastic.pt"


def fourier_coordinates(shape, device):
    rng = np.random.default_rng(42)
    bands = np.concatenate([rng.normal(size=(2, 12)) * s for s in (.5, 2., 8.)], axis=1)
    ny, nx = shape
    xy = np.stack(np.meshgrid((np.arange(nx) + .5) / nx, (np.arange(ny) + .5) / ny), axis=-1)
    phase = 2 * np.pi * (xy @ bands)
    return torch.as_tensor(np.concatenate((np.sin(phase), np.cos(phase)), axis=-1),
                           dtype=torch.float32, device=device)


class ElasticMLP(nn.Module):
    """Predict pressure with an MLP and recover displacement with the Green operator."""

    def __init__(self, h, dx, dy, device):
        super().__init__()
        self.h, self.n, self.area, self.workers = h, h.shape[0], dx * dy, 1
        self.fft_shape = (2 * h.shape[0], 2 * h.shape[1])
        self.length = max(float(np.ptp(h)), 1e-12)
        self.register_buffer("surface", torch.as_tensor(h, dtype=torch.float32, device=device), persistent=False)
        self.register_buffer("position", fourier_coordinates(h.shape, device), persistent=False)
        self.register_buffer("kernel", torch.as_tensor(displacement_kernel(h.shape, dx, dy),
                                                       dtype=torch.complex64, device=device), persistent=False)
        self.mlp = nn.Sequential(nn.Linear(74, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
                                 nn.Linear(32, 1)).to(device)
        nn.init.constant_(self.mlp[-1].bias, 1.0)

    def forward(self, shape, load):
        nominal = load / (self.area * shape.numel())
        load_channel = torch.full_like(shape, nominal / 1e3)
        inputs = torch.cat((torch.stack((shape / self.length, load_channel), dim=-1), self.position), dim=-1)
        return self.mlp(inputs).squeeze(-1).clamp_min(0) * nominal

    def elastic_displacement(self, row, pressure):
        wide = torch.cat((pressure, pressure.flip((-1,))), dim=-1)
        mirrored = torch.cat((wide, wide.flip((-2,))), dim=-2)
        elastic = torch.fft.irfft2(torch.fft.rfft2(mirrored) * self.kernel, s=self.fft_shape)
        return elastic[:pressure.shape[0], :pressure.shape[1]] / effective_modulus(row)

    def contact_fields(self, row, shape, pressure):
        elastic = self.elastic_displacement(row, pressure)
        z = shape + elastic
        pressure_sum = pressure.sum()
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
        return self(self.updated_surface(plastic), row["F"])

    def updated_surface(self, plastic):
        return self.surface + torch.as_tensor(plastic, dtype=self.surface.dtype, device=self.surface.device)

    @torch.no_grad()
    def predict_contact(self, row, plastic, pressure=None):
        """Recover contact fields, reusing pressure when it has already been predicted."""
        shape = self.updated_surface(plastic)
        if pressure is None:
            pressure = self(shape, row["F"])
        pressure, elastic, gap, fb, penetration, load, error = self.contact_fields(row, shape, pressure)
        return Contact(pressure.cpu().numpy(), elastic.cpu().numpy(), gap.cpu().numpy(),
                       float(fb.abs().max()), float(penetration), 1, float(load), float(error.abs()))


def load_elastic(path, h, dx, dy, device):
    model = ElasticMLP(h, dx, dy, device)
    weights = torch.load(path, map_location=device, weights_only=True)
    if weights["mlp.4.weight"].shape[0] != 1:
        raise ValueError("此弹性权重是双输出版本，请用压力单输出模型的权重或重新训练。")
    model.load_state_dict(weights)
    return model.eval().requires_grad_(False)


def elastic_loss(model, row):
    pressure = model(model.surface, row["F"])
    _, _, _, fb, penetration, _, load_error = model.contact_fields(row, model.surface, pressure)
    return fb.square().mean() + (penetration / model.length).square() + load_error.square()


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
    model = ElasticMLP(h, dx, dy, args.device)
    adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(1, 3001):
        adam.zero_grad()
        loss = elastic_loss(model, rows[(step - 1) % len(rows)])
        loss.backward()
        adam.step()
        if step == 1 or step % 100 == 0:
            print(f"Elastic Adam {step}/3000 loss={float(loss.detach()):.6e}", flush=True)

    lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=1, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        total = model.surface.new_zeros(())
        for row in rows:
            loss = elastic_loss(model, row) / len(rows)
            loss.backward()
            total += loss.detach()
        return total

    for step in range(1, 501):
        loss = lbfgs.step(closure)
        if step == 1 or step % 25 == 0:
            print(f"Elastic L-BFGS {step}/500 loss={float(loss):.6e}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Elastic checkpoint: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
