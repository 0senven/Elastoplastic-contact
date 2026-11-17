"""Supervised terminal-field prediction on the random-surface MATLAB data."""

import argparse
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import h5py
import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from torch.nn import functional as F

from physics import cell_average, displacement_kernel, effective_modulus
from train import Spectral2D
from train_elastic import fourier_coordinates

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "data" / "random surface" / "run_001.zip"
FIELDS = ("pressure", "elastic", "plastic")
RAW_INPUT_KEYS = ("F", "E1", "E2", "n1", "n2", "Q1", "T1")


def equivalent_parameters(raw):
    if not all(np.isfinite(raw[key]) for key in RAW_INPUT_KEYS):
        raise ValueError("Load and material parameters must be finite.")
    if min(raw["F"], raw["E1"], raw["E2"], raw["Q1"]) <= 0 or raw["T1"] < 0:
        raise ValueError("Invalid load, modulus, yield strength or tangent modulus.")
    if not (-1 < raw["n1"] < 0.5 and -1 < raw["n2"] < 0.5):
        raise ValueError("Both Poisson ratios must be between -1 and 0.5.")
    return np.array([raw["F"], effective_modulus(raw), raw["n1"], raw["Q1"], raw["T1"]],
                    dtype=np.float32)


def read_cases(archive, size):
    files = (("equivalent_surface", "h_equivalent"),
             ("pressure_distribution", "P"),
             ("equivalent_elastic_displacement", "u_elastic_equivalent"),
             ("equivalent_plastic_displacement", "u_plastic_equivalent"))
    with ZipFile(archive) as source:
        for name in sorted(source.namelist()):
            if not name.endswith("/parameters.mat"):
                continue
            folder = name.rsplit("/", 1)[0]
            parameter = loadmat(BytesIO(source.read(name)), simplify_cells=True)["parameters"]
            fields = []
            original_grid = None
            for file, key in files:
                with h5py.File(BytesIO(source.read(f"{folder}/{file}.mat"))) as mat:
                    field = np.asarray(mat[key]).T
                if original_grid is None:
                    original_grid = field.shape
                if field.shape != original_grid or not np.isfinite(field).all():
                    raise ValueError(f"Invalid grid or nonfinite values in {folder}/{file}.mat")
                fields.append(cell_average(field, size).astype(np.float32))
            ny, nx = original_grid
            grid = parameter["grid"]
            spacing = (float(grid["dx"]) * nx / size, float(grid["dy"]) * ny / size)
            values = equivalent_parameters(parameter)
            if not np.isclose(effective_modulus(parameter), parameter["E"], rtol=1e-10, atol=0):
                raise ValueError(f"Stored equivalent modulus disagrees with two-material compliance in {folder}")
            if not np.isfinite(spacing).all() or min(spacing) <= 0:
                raise ValueError(f"Invalid grid in {folder}")
            yield int(parameter["case_index"]), torch.from_numpy(np.stack(fields)), torch.from_numpy(values), spacing


class TerminalModel(nn.Module):
    """Learn pressure and plastic displacement, with elastic displacement fixed by Green."""

    def __init__(self, size, dx, dy, scales, device):
        super().__init__()
        self.area = dx * dy
        xy, bands = fourier_coordinates((size, size), device)
        phase = 2 * torch.pi * (xy @ bands)
        self.register_buffer("coordinates", torch.cat((phase.sin(), phase.cos()), dim=-1).permute(2, 0, 1),
                             persistent=False)
        kernel = displacement_kernel((size, size), dx, dy)
        self.register_buffer("green", torch.as_tensor(kernel, dtype=torch.complex64, device=device),
                             persistent=False)
        self.register_buffer("scales", torch.as_tensor(scales, dtype=torch.float32, device=device))
        self.register_buffer("parameter_scale", self.scales.new_tensor([1000., 1e5, 1., 1e3, 1e4]),
                             persistent=False)
        self.lift = nn.Conv2d(78, 12, 1)
        self.spectral = nn.ModuleList(Spectral2D(12, 8) for _ in range(3))
        self.pointwise = nn.ModuleList(nn.Conv2d(12, 12, 1) for _ in self.spectral)
        self.fine = nn.Conv2d(6, 8, 3, padding=1)
        self.coarse = nn.Conv2d(8, 8, 3, padding=1)
        self.head = nn.Conv2d(12 + 8 + 8, 2, 1)
        self.to(device)

    def forward(self, shape, parameters):
        batch, ny, nx = shape.shape
        material = (parameters / self.parameter_scale)[:, :, None, None].expand(-1, -1, ny, nx)
        physical = torch.cat((shape[:, None] / self.scales[0], material), dim=1)
        x = self.lift(torch.cat((physical, self.coordinates[None].expand(batch, -1, -1, -1)), dim=1))
        for spectral, pointwise in zip(self.spectral, self.pointwise):
            x = F.gelu(spectral(x) + pointwise(x))
        fine = F.gelu(self.fine(physical))
        coarse = F.gelu(self.coarse(F.avg_pool2d(fine, 2)))
        up = F.interpolate(coarse, size=(ny, nx), mode="bilinear", align_corners=False)
        relative_pressure, plastic = F.softplus(self.head(torch.cat((x, fine, up), dim=1))).unbind(dim=1)
        pressure = relative_pressure * (parameters[:, 0, None, None] /
                                        (self.area * relative_pressure.sum(dim=(-2, -1), keepdim=True)))
        wide = torch.cat((pressure, pressure.flip((-1,))), dim=-1)
        mirrored = torch.cat((wide, wide.flip((-2,))), dim=-2)
        displacement = torch.fft.irfft2(torch.fft.rfft2(mirrored) * self.green, s=(2 * ny, 2 * nx))
        elastic = displacement[:, :ny, :nx] / parameters[:, 1, None, None]
        return torch.stack((pressure, elastic, plastic * self.scales[3]), dim=1)


def data_loss(predicted, target, scales):
    errors = ((predicted - target) / scales[None, 1:, None, None]).square()
    active = target[:, 0] > 0
    contact = (errors[:, 0] * active).sum() / active.sum().clamp_min(1)
    return errors.mean(dim=(0, 2, 3)).sum() + contact


@torch.no_grad()
def score(model, samples, device):
    result = []
    for number, fields, parameters, _ in samples:
        fields, parameters = fields.to(device), parameters.to(device)
        predicted = model(fields[None, 0], parameters[None])[0]
        rel_l2 = (predicted - fields[1:]).flatten(1).norm(dim=1) / fields[1:].flatten(1).norm(dim=1).clamp_min(1e-12)
        load_error = abs(float(predicted[0].sum(dtype=torch.float64)) * model.area / float(parameters[0]) - 1)
        result.append((number, float(data_loss(predicted[None], fields[None, 1:], model.scales)),
                       rel_l2.tolist(), load_error))
    return result


def train(args):
    cases = list(read_cases(args.archive, args.size))
    if len(cases) < 5 or len({case[0] for case in cases}) != len(cases):
        raise ValueError("At least five distinct cases are required.")
    spacing = cases[0][3]
    if any(case[3] != spacing for case in cases):
        raise ValueError("Training cases must share the same physical grid spacing.")
    generator = np.random.default_rng(42)
    indices = generator.permutation(len(cases))
    held_out = max(1, len(cases) // 10)
    test = [cases[i] for i in indices[:held_out]]
    validation = [cases[i] for i in indices[held_out:2 * held_out]]
    training = [cases[i] for i in indices[2 * held_out:]]
    scales = torch.stack([c[1].square().mean(dim=(1, 2)) for c in training]).mean(0).sqrt().clamp_min(1e-12)
    model = TerminalModel(args.size, *spacing, scales, args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for index in generator.permutation(len(training)):
            _, fields, parameters, _ = training[index]
            fields, parameters = fields.to(args.device), parameters.to(args.device)
            optimizer.zero_grad()
            loss = data_loss(model(fields[None, 0], parameters[None]), fields[None, 1:], model.scales)
            loss.backward()
            optimizer.step()
        model.eval()
        validation_loss = float(np.mean([entry[1] for entry in score(model, validation, args.device)]))
        print(f"Epoch {epoch}/{args.epochs} validation loss={validation_loss:.6g}", flush=True)
        if validation_loss < best:
            best = validation_loss
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(model=model.state_dict(), size=args.size, spacing=spacing,
                            validation_ids=[c[0] for c in validation], test_ids=[c[0] for c in test],
                            validation_loss=best), args.output)
    if not np.isfinite(best):
        raise RuntimeError("Training produced no finite validation score.")
    model.load_state_dict(torch.load(args.output, map_location=args.device, weights_only=True)["model"])
    for number, _, rel_l2, load_error in score(model, test, args.device):
        print(f"Test case {number} relative L2 (p, ue, up)={rel_l2}; load error={load_error:.3e}", flush=True)


@torch.no_grad()
def predict(args):
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    size = saved["size"]
    model = TerminalModel(size, *saved["spacing"], saved["model"]["scales"], args.device)
    model.load_state_dict(saved["model"])
    model.eval()
    with np.load(args.input) as data:
        shape = np.asarray(data["h_equivalent"], dtype=np.float32)
        values = equivalent_parameters({key: float(data[key]) for key in RAW_INPUT_KEYS})
    if shape.shape != (size, size) or not np.isfinite(shape).all():
        raise ValueError(f"Expected finite h_equivalent with shape {(size, size)} on the checkpoint grid.")
    output = model(torch.as_tensor(shape, device=args.device)[None],
                   torch.as_tensor(values, device=args.device)[None])[0].cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **dict(zip(FIELDS, output)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--input", type=Path, help="Prediction NPZ with h_equivalent, F, E1, E2, n1, n2, Q1 and T1.")
    args = parser.parse_args()
    if (args.checkpoint is None) != (args.input is None):
        parser.error("--checkpoint and --input must be supplied together.")
    if args.size < 2 or args.epochs < 1:
        parser.error("--size must be at least 2 and --epochs must be positive.")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    if args.input:
        args.output = args.output or ROOT / "runs" / "terminal" / "prediction.npz"
        predict(args)
    else:
        args.output = args.output or ROOT / "runs" / "terminal" / "operator.pt"
        train(args)


if __name__ == "__main__":
    main()
