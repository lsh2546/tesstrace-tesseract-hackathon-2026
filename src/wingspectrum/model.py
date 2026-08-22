from __future__ import annotations

from dataclasses import dataclass

import numpy as np


WAVELENGTHS = np.arange(300.0, 1101.0, 10.0)
BASIS_CENTERS = np.array([330, 370, 410, 460, 520, 590, 680, 780, 900, 1020])
BASIS_WIDTHS = np.array([28, 30, 34, 42, 50, 60, 70, 85, 95, 105])
BASE_REFLECTANCE = 0.04
FIXED_ABSORPTANCE = 0.06
SPECIES_SUM_WEIGHT = 0.5


def _normalized_gaussian(center: float, width: float) -> np.ndarray:
    values = np.exp(-0.5 * ((WAVELENGTHS - center) / width) ** 2)
    return values / values.sum()


UVS_WEIGHT = 0.72 * _normalized_gaussian(365, 32) + 0.28 * _normalized_gaussian(445, 42)
VS_WEIGHT = 0.22 * _normalized_gaussian(405, 32) + 0.78 * _normalized_gaussian(505, 55)
HUMAN_WEIGHT = (
    0.25 * _normalized_gaussian(445, 38)
    + 0.55 * _normalized_gaussian(555, 48)
    + 0.20 * _normalized_gaussian(610, 52)
)
SOLAR_WEIGHT = (
    0.08 * _normalized_gaussian(365, 50)
    + 0.47 * _normalized_gaussian(545, 150)
    + 0.45 * _normalized_gaussian(850, 190)
)

BASIS = np.stack(
    [np.exp(-0.5 * ((WAVELENGTHS - c) / w) ** 2) for c, w in zip(BASIS_CENTERS, BASIS_WIDTHS)],
    axis=1,
)
BASIS /= np.maximum(BASIS.sum(axis=1, keepdims=True), 1.0)


@dataclass(frozen=True)
class Spectrum:
    reflectance: np.ndarray
    transmittance: np.ndarray
    jacobian_reflectance: np.ndarray


def optics(design: np.ndarray) -> Spectrum:
    """Map bounded coating controls to an energy-conserving smooth spectrum.

    This is a reduced-order spectral coating model, not a fabricated multilayer claim.
    The controls define overlapping manufacturable spectral bands; R + T + A = 1.
    """
    logits = BASIS @ np.asarray(design, dtype=float)
    activation = 1.0 / (1.0 + np.exp(-logits))
    reflectance = BASE_REFLECTANCE + 0.70 * activation
    transmittance = 1.0 - FIXED_ABSORPTANCE - reflectance
    jacobian = (0.70 * activation * (1.0 - activation))[:, None] * BASIS
    return Spectrum(reflectance, transmittance, jacobian)


def branch_metrics(spectrum: Spectrum) -> dict[str, float]:
    delta_r = spectrum.reflectance - BASE_REFLECTANCE
    visible_delta = HUMAN_WEIGHT * delta_r
    human_mean = float(np.sum(visible_delta))
    human_color_variation = float(np.sum(HUMAN_WEIGHT * (delta_r - human_mean) ** 2))
    return {
        "uvs_visibility": float(UVS_WEIGHT @ delta_r),
        "vs_visibility": float(VS_WEIGHT @ delta_r),
        "human_reflectance": human_mean,
        "human_color_variation": human_color_variation,
        "solar_transmittance": float(SOLAR_WEIGHT @ spectrum.transmittance),
    }


def loss_and_gradient(design: np.ndarray) -> tuple[float, np.ndarray, dict[str, float]]:
    spectrum = optics(design)
    metrics = branch_metrics(spectrum)
    uvs = metrics["uvs_visibility"]
    vs = metrics["vs_visibility"]

    # Smooth worst-species visibility: maximizing this prevents one vision class
    # from being sacrificed for the other.
    temperature = 0.035
    exp_uvs = np.exp(-uvs / temperature)
    exp_vs = np.exp(-vs / temperature)
    denominator = exp_uvs + exp_vs
    smooth_min = -temperature * np.log(denominator)
    dmin_duvs = exp_uvs / denominator
    dmin_dvs = exp_vs / denominator

    delta_r = spectrum.reflectance - BASE_REFLECTANCE
    human_mean = metrics["human_reflectance"]
    human_variation = metrics["human_color_variation"]
    solar_t = metrics["solar_transmittance"]
    regularization = float(np.mean(np.asarray(design) ** 2))

    loss = (
        -2.4 * smooth_min
        - SPECIES_SUM_WEIGHT * (uvs + vs)
        + 15.0 * human_mean**2
        + 1.5 * human_variation
        + 0.22 * solar_t
        + 0.003 * regularization
    )

    dloss_dr = -2.4 * (dmin_duvs * UVS_WEIGHT + dmin_dvs * VS_WEIGHT)
    dloss_dr -= SPECIES_SUM_WEIGHT * (UVS_WEIGHT + VS_WEIGHT)
    dloss_dr += 30.0 * human_mean * HUMAN_WEIGHT
    dloss_dr += 3.0 * HUMAN_WEIGHT * (delta_r - human_mean)
    dloss_dr -= 0.22 * SOLAR_WEIGHT
    gradient = spectrum.jacobian_reflectance.T @ dloss_dr
    gradient += 0.006 * np.asarray(design) / len(design)
    return float(loss), gradient, metrics


def optimize(steps: int = 180, learning_rate: float = 0.16) -> dict[str, object]:
    design = np.full(len(BASIS_CENTERS), -3.0)
    initial_design = design.copy()
    first_moment = np.zeros_like(design)
    second_moment = np.zeros_like(design)
    history = []
    for step in range(1, steps + 1):
        loss, gradient, metrics = loss_and_gradient(design)
        history.append({"step": step - 1, "loss": loss, **metrics})
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        corrected_first = first_moment / (1.0 - 0.9**step)
        corrected_second = second_moment / (1.0 - 0.999**step)
        design -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)

    final_loss, _, final_metrics = loss_and_gradient(design)
    history.append({"step": steps, "loss": final_loss, **final_metrics})
    return {
        "initial_design": initial_design.tolist(),
        "final_design": design.tolist(),
        "history": history,
        "initial_spectrum": summarize(initial_design),
        "final_spectrum": summarize(design),
    }


def summarize(design: np.ndarray) -> dict[str, object]:
    spectrum = optics(np.asarray(design))
    return {
        "wavelength_nm": WAVELENGTHS.tolist(),
        "reflectance": spectrum.reflectance.tolist(),
        "transmittance": spectrum.transmittance.tolist(),
        "metrics": branch_metrics(spectrum),
    }
