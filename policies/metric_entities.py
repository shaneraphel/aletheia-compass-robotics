# Copyright (C) 2026 Aletheia contributors
# SPDX-License-Identifier: GPL-3.0-only
"""Measured metric, spectral, secular, weld and compass entities.

Extracted mathematical operators only. No language model, model forward, model
weights, private runtime path, or mother-model code is included.
"""
from __future__ import annotations

import torch

def projector(vectors: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.numel() == 0:
        return torch.zeros((vectors.shape[0], vectors.shape[0]), dtype=torch.float64)
    basis = vectors[:, indices]
    return basis @ basis.T


def rank_one_secular_residuals(
    inertia_values: torch.Tensor,
    updated_values: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """Evaluate |D+sum(T_i)|/(1+|D|+sum|T_i|) without product overflow.

    D(mu)=prod(lambda_i-mu), T_i=z_i^2*prod_{j!=i}(lambda_j-mu). A common
    logarithmic scale and compensated summation handle the ordinary path.
    Near roots, severe cancellation is evaluated with 80-digit decimal arithmetic
    on the exact float64 inputs; this is not a clamp or a change of the formula.
    """
    import math
    from decimal import Decimal, localcontext

    inertia_values = inertia_values.detach().to(device="cpu", dtype=torch.float64)
    updated_values = updated_values.detach().to(device="cpu", dtype=torch.float64)
    z = z.detach().to(device="cpu", dtype=torch.float64)
    if inertia_values.ndim != 1 or z.shape != inertia_values.shape or updated_values.ndim != 1:
        raise ValueError("Base eigenvalues and z must align; evaluation roots must be one-dimensional")
    if inertia_values.numel() == 0:
        raise ValueError("Secular evaluation needs at least one eigenvalue")
    if not (torch.isfinite(inertia_values).all() and torch.isfinite(updated_values).all()
            and torch.isfinite(z).all()):
        raise ValueError("Secular inputs must be finite")
    base = inertia_values.tolist()
    coefficients = z.tolist()
    scale = max(
        float(inertia_values.abs().max()), float(updated_values.abs().max()),
        float(z.square().max()),
    )

    def high_precision(mu: float) -> float:
        with localcontext() as context:
            context.prec = 80
            root = Decimal.from_float(mu)
            differences = [Decimal.from_float(value) - root for value in base]
            direct = Decimal(1)
            for difference in differences:
                direct *= difference
            terms = []
            for index, coefficient in enumerate(coefficients):
                term = Decimal.from_float(coefficient) ** 2
                for other_index, difference in enumerate(differences):
                    if other_index != index:
                        term *= difference
                terms.append(term)
            numerator = abs(direct + sum(terms, Decimal(0)))
            denominator = Decimal(1) + abs(direct) + sum((abs(term) for term in terms), Decimal(0))
            return float(numerator / denominator)

    residuals = []
    for root in updated_values.tolist():
        if not math.isfinite(scale):
            residual = high_precision(root)
        elif scale == 0:
            residual = 0.0
        else:
            # Division before subtraction also avoids overflow in lambda-mu.
            scaled_base = [value / scale for value in base]
            scaled_root = root / scale
            differences = [(value - scaled_root) / 2 for value in scaled_base]
            scaled_z = [value / math.sqrt(scale) / math.sqrt(2) for value in coefficients]
            direct = math.prod(differences)
            terms = [
                coefficient * coefficient * math.prod(differences[:index] + differences[index + 1:])
                for index, coefficient in enumerate(scaled_z)
            ]
            numerator = abs(math.fsum([direct, *terms]))
            scaled_denominator = math.fsum([abs(direct), *(abs(term) for term in terms)])
            if scaled_denominator == 0:
                residual = 0.0
            elif numerator / scaled_denominator < 1e-7:
                # Log-scaling alone loses significant digits when huge terms cancel.
                residual = high_precision(root)
            else:
                # factor=(2*scale)^n. Select a branch that never forms it directly.
                log_factor = len(base) * (math.log(scale) + math.log(2))
                if log_factor >= 0:
                    inverse = math.exp(-log_factor) if log_factor < 745 else 0.0
                    residual = numerator / (inverse + scaled_denominator)
                else:
                    factor = math.exp(log_factor) if log_factor > -745 else 0.0
                    residual = numerator * factor / (1 + scaled_denominator * factor)
        if not math.isfinite(residual):
            raise RuntimeError("Scale-stable secular evaluation produced a non-finite residual")
        residuals.append(residual)
    return torch.tensor(residuals, dtype=torch.float64)


def metric_entity(x: torch.Tensor, previous: torch.Tensor | None) -> dict[str, torch.Tensor]:
    """Construct all entities in float64 on CPU from an observed [time, feature] tensor."""
    x = x.detach().to(device="cpu", dtype=torch.float64)
    n = x.shape[0]
    if n < 2:
        raise ValueError("a metric trajectory needs at least two observations")
    center = x.mean(dim=0, keepdim=True)
    xc = x - center
    _, singular, vh = torch.linalg.svd(xc, full_matrices=False)
    tolerance = float(singular.max()) * max(xc.shape) * torch.finfo(torch.float64).eps
    rank = max(1, int((singular > tolerance).sum()))
    basis = vh[:rank].T
    coordinates = xc @ basis
    gram_coordinates = xc @ xc.T
    diagonal = gram_coordinates.diag()
    distances_squared = (diagonal[:, None] + diagonal[None, :] - 2 * gram_coordinates).clamp_min(0)
    centering = torch.eye(n, dtype=torch.float64) - torch.ones((n, n), dtype=torch.float64) / n
    gram_distances = -0.5 * centering @ distances_squared @ centering
    gram_error = float((gram_coordinates - gram_distances).abs().max())

    positive_distances = distances_squared[distances_squared > 1e-20]
    scale = float(positive_distances.median()) if positive_distances.numel() else 1.0
    affinity = torch.exp(-distances_squared / max(scale, 1e-20))
    affinity.fill_diagonal_(0)
    laplacian = affinity.sum(dim=1).diag() - affinity
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    eta = 1.5 / max(float(eigenvalues[-1]), 1e-20)
    scaled = eta * eigenvalues
    kernel = torch.nonzero(eigenvalues.abs() <= 1e-9, as_tuple=False).flatten()
    trapped = torch.nonzero((scaled > 1e-9) & (scaled <= 0.05), as_tuple=False).flatten()
    decay = torch.nonzero((scaled > 0.05) & (scaled < 1.0), as_tuple=False).flatten()
    oscillatory = torch.nonzero((scaled >= 1.0) & (scaled < 2.0), as_tuple=False).flatten()
    unstable = torch.nonzero(scaled >= 2.0, as_tuple=False).flatten()
    projections = [projector(eigenvectors, part) for part in (kernel, trapped, decay, oscillatory, unstable)]
    direct_sum_error = float((sum(projections) - torch.eye(n, dtype=torch.float64)).abs().max())

    # The same point cloud admits a second spectral reading after a rank-one residual update.
    # Evaluate the non-singular polynomial form of the secular equation so that
    # repeated and zero eigenvalues remain well-defined.
    # Work in the numerically nonzero basis: Z.T @ Z has exactly the nonzero
    # spectrum of Xc.T @ Xc, without a costly hidden_dimension^3 decomposition.
    residual = coordinates[-1] - coordinates[-2]
    inertia = coordinates.T @ coordinates
    inertia_values, inertia_vectors = torch.linalg.eigh(inertia)
    updated = inertia + residual.outer(residual)
    updated_values = torch.linalg.eigvalsh(updated)
    z = inertia_vectors.T @ residual
    secular_residuals = rank_one_secular_residuals(inertia_values, updated_values, z)

    # Cross-trajectory coupling is intentionally not symmetrized.
    if previous is None:
        previous = torch.roll(x, shifts=1, dims=0)
        previous[0] = x[0]
    previous = previous.detach().to(device="cpu", dtype=torch.float64)
    previous_coordinates = (previous - previous.mean(dim=0, keepdim=True)) @ basis
    coupling = coordinates.T @ laplacian @ previous_coordinates
    skew = coupling - coupling.T
    # Keep the raw coupling for audit, but apply the finite exponential series
    # to a normalized operator.  This makes the truncation and inverse tests
    # meaningful instead of overflowing on activations with large magnitude.
    weld_normalization = torch.maximum(skew.norm(), torch.tensor(1.0, dtype=torch.float64))
    rho = skew / weld_normalization
    identity = torch.eye(rank, dtype=torch.float64)
    weld = identity.clone()
    power = identity.clone()
    powers, traces = [], []
    for order in range(1, 5):
        power = power @ rho
        powers.append(power.clone())
        traces.append(torch.trace(power) / order)
        weld = weld + power / __import__("math").factorial(order)
    t = 1e-3
    determinant_lhs = -torch.linalg.slogdet(identity - t * rho)[1]
    determinant_rhs = sum(trace * t**order for order, trace in enumerate(traces, start=1))
    weld_inverse = torch.linalg.solve(weld, identity)

    # Compass path: measured scale steps, residual vectors, unique bridge, and exact readback.
    step_lengths = torch.linalg.vector_norm(x[1:] - x[:-1], dim=1)
    length_scale = float(step_lengths.median().clamp_min(1e-12))
    scales = step_lengths / length_scale
    weights = torch.pow(torch.tensor(2.0, dtype=torch.float64), -scales)
    minimum = weights.min()
    bridge_candidates = torch.nonzero(weights == minimum, as_tuple=False).flatten()
    bridge = int(bridge_candidates[0]) if bridge_candidates.numel() == 1 else -1
    prefix_center = x.mean(dim=0, keepdim=True)
    compass_residual = x[-1] - prefix_center[0]
    readback = prefix_center[0] + compass_residual
    adjacent_cosine = torch.nn.functional.cosine_similarity(x[:-1], x[1:], dim=1).mean()

    return {
        "schema": torch.tensor([1]),
        "observations": x,
        "space_center": center,
        "space_basis_Q": basis,
        "space_coordinates_Z": coordinates,
        "space_singular_values": singular,
        "space_rank": torch.tensor(rank),
        "gram_coordinate_route": gram_coordinates,
        "gram_distance_route": gram_distances,
        "gram_dual_route_max_abs": torch.tensor(gram_error),
        "laplacian": laplacian,
        "laplacian_eigenvalues": eigenvalues,
        "laplacian_eigenvectors": eigenvectors,
        "spectral_eta": torch.tensor(eta),
        "spectral_kernel_indices": kernel,
        "spectral_trapped_indices": trapped,
        "spectral_decay_indices": decay,
        "spectral_oscillatory_indices": oscillatory,
        "spectral_unstable_indices": unstable,
        "spectral_kernel_projector": projections[0],
        "spectral_trapped_projector": projections[1],
        "spectral_decay_projector": projections[2],
        "spectral_oscillatory_projector": projections[3],
        "spectral_unstable_projector": projections[4],
        "spectral_direct_sum_max_abs": torch.tensor(direct_sum_error),
        "secular_base_eigenvalues": inertia_values,
        "secular_residual_u": residual,
        "secular_updated_eigenvalues": updated_values,
        "secular_root_residuals": secular_residuals,
        "secular_root_max_abs": secular_residuals.max(),
        "weld_cross_coupling_C": coupling,
        "weld_skew": skew,
        "weld_normalization": weld_normalization,
        "weld_rho": rho,
        "weld_matrix": weld,
        "weld_inverse": weld_inverse,
        "weld_powers": torch.stack(powers),
        "weld_trace_sequence": torch.stack(traces),
        "weld_inverse_max_abs": (weld @ weld_inverse - identity).abs().max(),
        "weld_det_trace_abs": (determinant_lhs - determinant_rhs).abs(),
        "compass_scales_s": scales,
        "compass_residual_u": compass_residual,
        "compass_center": prefix_center,
        "compass_path_weights": weights,
        "compass_shortcut_weight": minimum,
        "compass_excess_weight": weights.sum() - minimum,
        "compass_bridge_index": torch.tensor(bridge),
        "compass_roundtrip_max_abs": (readback - x[-1]).abs().max(),
        "adjacent_hidden_cosine": adjacent_cosine,
    }
