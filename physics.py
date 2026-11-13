"""固定形貌、矩形半空间核、力控接触和 MATLAB 双线性本构。"""

import csv
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import h5py
import numpy as np
from scipy.fft import irfft2, rfft2
from scipy.io import loadmat

FIELDS = "F E1 E2 n1 n2 Q1 Q2 T1 T2".split()
CELL_CORNERS = ((1, 1, 1), (1, -1, -1), (-1, -1, 1), (-1, 1, -1))


def cases(path):
    """读取工况 CSV，按原始编号保留训练与验收切分。"""
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if [row.get("CaseID") for row in rows] != [str(i) for i in range(1, 31)]:
        raise ValueError("cases.csv 的 CaseID 必须按 1–30 顺序排列。")

    result = []
    for index, row in enumerate(rows, 2):
        try:
            item = {key: float(row[key]) for key in FIELDS}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"工况表第 {index} 行需要数值列 {FIELDS}") from error
        if not all(np.isfinite(list(item.values()))) or item["F"] <= 0:
            raise ValueError(f"工况表第 {index} 行存在无效数值或非正载荷。")
        for material in (1, 2):
            e, nu, q, t = (item[f"{key}{material}"] for key in ("E", "n", "Q", "T"))
            if not (e > 0 and -1 < nu < .5 and q > 0 and 0 <= t < e):
                raise ValueError(f"工况表第 {index} 行材料 {material} 的参数无效。")
        result.append(item)
    return result


def mat_array(path, key):
    """读取 MAT v5 或 v7.3，恢复 MATLAB 的行列顺序。"""
    if h5py.is_hdf5(path):
        with h5py.File(path) as mat:
            return np.asarray(mat[key]).T.copy()
    return np.asarray(loadmat(path)[key])


def surfaces(folder):
    h1, h2 = (mat_array(folder / f"h{i}.mat", f"h{i}") for i in (1, 2))
    if h1.shape != h2.shape or h1.ndim != 2:
        raise ValueError("两张形貌应是尺寸一致的二维数组。")
    grid = loadmat(folder / "grid.mat", simplify_cells=True)["grid"]
    return h1, h2, float(grid["dx"]), float(grid["dy"])


def simplex(v, total):
    """投影到非负且总和固定的压力集合。"""
    sorted_values = np.sort(v.ravel())[::-1]
    cumulative = np.cumsum(sorted_values) - total
    active = sorted_values > cumulative / np.arange(1, sorted_values.size + 1)
    last = np.flatnonzero(active)[-1]
    return np.maximum(v - cumulative[last] / (last + 1), 0)


def displacement_kernel(shape, dx, dy):
    """单位等效模量下矩形压力单元的法向位移核。"""
    ny, nx = shape
    y, x = np.arange(-ny + 1, ny) * dy, np.arange(-nx + 1, nx) * dx
    xx, yy = np.meshgrid(x, y)
    influence = np.zeros_like(xx)
    for a, b, sign in CELL_CORNERS:
        xc, yc = xx + a * dx / 2, yy + b * dy / 2
        r = np.hypot(xc, yc)
        influence += sign * (yc * np.log(xc + r) + xc * np.log(yc + r))
    padded = np.zeros((2 * ny, 2 * nx))
    padded[np.ix_(np.arange(-ny + 1, ny) % (2 * ny),
                  np.arange(-nx + 1, nx) % (2 * nx))] = influence / np.pi
    return rfft2(padded)


def stress_kernel(shape, dx, dy, z, nu):
    """矩形压力单元在指定深度的六分量应力核。"""
    ny, nx = shape
    x, y = np.arange(-nx + 1, nx) * dx, np.arange(-ny + 1, ny) * dy
    xx, yy = np.meshgrid(x, y)
    coefficients = np.zeros((6, 2 * ny - 1, 2 * nx - 1))
    for a, b, sign in CELL_CORNERS:
        xc, yc = xx + a * dx / 2, yy + b * dy / 2
        r = np.sqrt(xc * xc + yc * yc + z * z)
        t = np.pi / 2 * np.sign(xc * yc) if z == 0 else np.arctan(xc * yc / (r * z))
        values = (
            -2 * nu * t + 2 * (1 - 2 * nu) * np.arctan(xc / (r + yc + z)) - xc * z / (r * (r + yc)),
            -2 * nu * t + 2 * (1 - 2 * nu) * np.arctan(yc / (r + xc + z)) - yc * z / (r * (r + xc)),
            -t + xc * z / (r * (r + yc)) + yc * z / (r * (r + xc)),
            (2 * nu - 1) * np.log(r + z) - z / r,
            -z * z / (r * (r + yc)),
            -z * z / (r * (r + xc)),
        )
        coefficients += sign / (2 * np.pi) * np.stack(values)
    padded = np.zeros((6, 2 * ny, 2 * nx))
    padded[:, np.arange(-ny + 1, ny)[:, None] % (2 * ny),
           np.arange(-nx + 1, nx)[None, :] % (2 * nx)] = coefficients
    return rfft2(padded).astype(np.complex64)


@dataclass
class Contact:
    pressure: np.ndarray
    elastic: np.ndarray
    gap: np.ndarray
    fb_max: float
    penetration: float
    iterations: int
    computed_load: float
    load_rel_error: float


@dataclass
class PlasticState:
    """当前网络轨迹的表面位移与各深度点的硬化阈值。"""

    displacement: np.ndarray
    hardening: np.ndarray
    previous_pressure: np.ndarray | None = None
    step: int = 0


def cell_average(field, size):
    """矩形单元面积平均，保持物理域宽度和压力积分。"""
    ny, nx = field.shape
    if ny % size or nx % size:
        raise ValueError("目标网格尺寸必须能整除两个原始网格尺寸。")
    return field.reshape(size, ny // size, size, nx // size).mean(axis=(1, 3))


def load_inputs(data, size):
    data = Path(data)
    h1, h2, dx, dy = surfaces(data / "fixed_surfaces")
    raw = h1 - h2
    h = cell_average(raw, size)
    h -= h.min()
    rows = cases(data / "cases.csv")
    if len(rows) != 30 or any(r["Q1"] > r["Q2"] for r in rows):
        raise ValueError("预期 30 组，且每组材料 1 为软材料。")
    if any(all(r[f"{key}1"] == r[f"{key}2"] for key in ("E", "n", "Q", "T")) for r in rows):
        raise ValueError("本试训按不同材料的等效模量和软侧塑性分支实现。")
    return rows, h, dx * raw.shape[1] / size, dy * raw.shape[0] / size, dx


class ElasticContact:
    """镜像延拓与 MATLAB 一致，压力投影到固定总力的单纯形。"""

    def __init__(self, h, dx, dy, fb_tol=2e-4, penetration_tol=2e-6, workers=2,
                 load_tol=1e-8):
        self.h = h
        self.n = len(h)
        self.area = dx * dy
        self.fb_tol, self.penetration_tol, self.workers = fb_tol, penetration_tol, workers
        self.load_tol = load_tol
        self.fft_shape = (2 * self.n, 2 * self.n)
        self.kernel = displacement_kernel(h.shape, dx, dy)
        self.row_bound = self.apply(np.ones_like(h)).max()
        self.length = max(np.ptp(h), 1e-12)

    def apply(self, p):
        wide = np.concatenate((p, p[:, ::-1]), axis=1)
        mirrored = np.concatenate((wide, wide[::-1]), axis=0)
        field = irfft2(rfft2(mirrored, workers=self.workers) * self.kernel,
                       s=self.fft_shape, workers=self.workers)
        return field[:self.n, :self.n]

    def solve(self, row, plastic, previous=None, max_iter=2400):
        e_star = effective_modulus(row)
        target_load = row["F"]
        total = target_load / self.area
        p0 = total / plastic.size
        p = np.full_like(self.h, p0) if previous is None else previous.copy()
        y = p.copy()
        t = 1.0
        base = self.h + plastic             # MATLAB 存的是初始间隙，塑性后加大间隙。
        step = e_star / self.row_bound
        for it in range(max_iter):
            new = simplex(y - step * (base + self.apply(y) / e_star), total)
            next_t = (1 + np.sqrt(1 + 4 * t * t)) / 2
            # 自适应重启防止动量在接触斑切换后持续振荡。
            if np.einsum("ij,ij->", y - new, new - p) > 0:
                next_t = 1.0
            y = new + (t - 1) / next_t * (new - p) if next_t > 1 else new
            p, t = new, next_t
            if it % 8 == 0 or it + 1 == max_iter:
                elastic = self.apply(p) / e_star
                z = base + elastic
                pressure_sum = float(np.sum(p, dtype=np.float64))
                gap = z - np.sum(p * z) / pressure_sum
                pn, gn = p / p0, gap / self.length
                fb = np.max(np.abs(np.hypot(pn, gn) - pn - gn))
                penetration = max(0.0, -float(gap.min()))
                # 以实际求出的压力积分检查力控投影和网格面积。
                computed_load = self.area * pressure_sum
                load_rel_error = abs(computed_load - target_load) / target_load
                if (fb <= self.fb_tol and penetration <= self.penetration_tol
                        and load_rel_error <= self.load_tol):
                    return Contact(p, elastic, gap, float(fb), penetration, it + 1,
                                   computed_load, load_rel_error)
        raise RuntimeError(
            f"弹性接触未通过验收：FB={fb:.3g}，穿透={penetration:.3g} mm，"
            f"计算载荷={computed_load:.8g} N，目标载荷={target_load:.8g} N，"
            f"载荷相对误差={load_rel_error:.3g}"
        )


def effective_modulus(row):
    return 1 / sum((1 - row[f"n{i}"] ** 2) / row[f"E{i}"] for i in (1, 2))


class BilinearPhysics:
    """由压力生成体内应力，保留逐深度的双线性硬化历史。"""

    def __init__(self, elastic, dx, dy, original_dz, depth_stride=4):
        self.elastic = elastic
        # 保留表面层。其他原始深度层分组，用中点代表，权重保留原深度积分。
        groups = [[0]] + [list(range(i, min(i + depth_stride, 51)))
                          for i in range(1, 51, depth_stride)]
        indices = [g[len(g) // 2] for g in groups]
        self.weights = np.asarray([len(g) * original_dz for g in groups], dtype=np.float32)
        n = elastic.n
        self.bases = np.empty((2, len(indices), 6, 2 * n, n + 1), dtype=np.complex64)
        for basis_index, nu in enumerate((0.0, 0.25)):
            for depth_index, z in enumerate(indices):
                self.bases[basis_index, depth_index] = stress_kernel(
                    elastic.h.shape, dx, dy, z * original_dz, nu
                )
        # 应力核对泊松比线性。保存差值，逐层组合以减少 512 网格的临时内存。
        self.bases[1] -= self.bases[0]

    def initial_state(self, row):
        shape = self.elastic.h.shape
        hardening = np.full((len(self.weights), *shape), row["Q1"], dtype=np.float32)
        return PlasticState(np.zeros_like(self.elastic.h), hardening)

    def constitutive_step(self, row, pressure, hardening):
        """对当前预测状态执行一次本构更新。返回位移增量和下一步硬化阈值。"""
        el = self.elastic
        e = effective_modulus(row)
        tangent_ratio = row["T1"] / e
        nu_factor = 4 * row["n1"]
        spectrum = rfft2(pressure.astype(np.float32), s=el.fft_shape, workers=el.workers)
        next_hardening = np.empty_like(hardening)
        integrated = np.zeros_like(hardening[0])
        for depth, weight in enumerate(self.weights):
            kernel = self.bases[0, depth] + nu_factor * self.bases[1, depth]
            stress = irfft2(kernel * spectrum, s=el.fft_shape, workers=el.workers)
            s11, s22, s33, s12, s13, s23 = stress[:, :el.n, :el.n]
            q = np.sqrt(3 * (s12**2 + s13**2 + s23**2)
                        + 0.5 * ((s11 - s22)**2 + (s11 - s33)**2 + (s22 - s33)**2))
            excess = np.maximum(q - hardening[depth], 0)
            next_hardening[depth] = hardening[depth] + tangent_ratio * excess
            integrated += weight * excess
        increment = (0.1 * (1 - tangent_ratio) / e) * integrated
        return increment, next_hardening

    def trajectory(self, row, tol=1e-6, steps=500):
        """独立的纯本构基线，供数值对照。"""
        state = self.initial_state(row)
        plastic, hardening = state.displacement, state.hardening
        initial = self.elastic.solve(row, plastic)
        contact = initial
        samples = [(plastic.copy(), initial)]
        for step in range(steps):
            increment, hardening = self.constitutive_step(row, contact.pressure, hardening)
            plastic += increment
            contact = self.elastic.solve(row, plastic, contact.pressure)
            if step in (7, 23, 63) or increment.max() <= tol:
                samples.append((plastic.copy(), contact))
            if increment.max() <= tol:
                return samples, step + 1
        raise RuntimeError(f"本构轨迹超过 {steps} 轮，最后更新={increment.max():.3g} mm")


def references(archive, size, rows, original_h):
    """仅在验收阶段调用。检查工况和形貌后把 512 网格参考降到同一分辨率。"""
    names = {"pressure": ("pressure_distribution", "P"),
             "elastic": ("equivalent_elastic_displacement", "u_elastic_equivalent"),
             "plastic": ("equivalent_plastic_displacement", "u_plastic_equivalent")}
    with ZipFile(archive) as zip_file:
        for number in range(26, 31):
            prefix = f"case_{number:06d}/"
            params = loadmat(BytesIO(zip_file.read(prefix + "parameters.mat")),
                             simplify_cells=True)["parameters"]
            if any(not np.isclose(params[key], value) for key, value in rows[number - 1].items()):
                raise ValueError(f"验证工况 {number} 的参数与输入表不符。")
            with h5py.File(BytesIO(zip_file.read(prefix + "equivalent_surface.mat"))) as mat:
                h_ref = np.asarray(mat["h_equivalent"]).T
                if not np.allclose(h_ref, original_h, rtol=0, atol=1e-12):
                    raise ValueError(f"验证工况 {number} 的原始形貌不一致。")
            fields = {}
            for name, (file, key) in names.items():
                with h5py.File(BytesIO(zip_file.read(prefix + file + ".mat"))) as mat:
                    fields[name] = cell_average(np.asarray(mat[key]).T, size)
            yield number, fields
