import torch
from utilities.utils import SPEED_SCALE
import sys
# ── 3. PHYSICS LOSS TERMS ──────────────────────────────────────────────────────


def jacobian_columns(model, x, col_indices):
    """
    Compute ∂C/∂x_i for selected feature columns via autograd.
    Returns tensor of shape (batch, len(col_indices)).
    """
    x = x.clone().requires_grad_(True)
    C = model(x)
    grads = torch.autograd.grad(C.sum(), x, create_graph=True)[0]
    return C, grads[:, col_indices]


def physics_losses(model, x_batch, feature_names, u_raw, v_raw):
    """
    Three physics-informed loss terms with safe autograd graph tracking and NaN assertions.
    """
    # ── 0. Sanity Check Inputs ───────────────────────────────────────────────
    if torch.isnan(x_batch).any():
        print("❌ CRITICAL: NaNs found in x_batch input tensor!", file=sys.stderr)
    if torch.isnan(u_raw).any() or torch.isnan(v_raw).any():
        print("❌ CRITICAL: NaNs found in u_raw or v_raw velocity tensors!", file=sys.stderr)

    idx = {name: feature_names.index(name) for name in feature_names}
    eps = 1e-6

    x_batch = x_batch.clone().requires_grad_(True)
    C = model(x_batch)

    if torch.isnan(C).any():
        print("❌ CRITICAL: Model output C contains NaNs!", file=sys.stderr)

    # First-order gradients
    grads = torch.autograd.grad(
        C.sum(), x_batch, create_graph=True, retain_graph=True
    )[0]

    if torch.isnan(grads).any():
        print("❌ CRITICAL: First-order gradients contain NaNs!", file=sys.stderr)

    dC_dsig = grads[:, idx["sigma0"]]
    dC_dspi = grads[:, idx["spice"]]
    dC_dlat = grads[:, idx["lat_norm"]]

    # Longitude derivative
    lon_sin = x_batch[:, idx["lon_sin"]]
    lon_cos = x_batch[:, idx["lon_cos"]]
    dC_dlon_sin = grads[:, idx["lon_sin"]]
    dC_dlon_cos = grads[:, idx["lon_cos"]]
    dC_dlon = (dC_dlon_sin * lon_cos) - (dC_dlon_cos * lon_sin)

    # ── L1: Diapycnal ─────────────────────────────────────────────────────────
    denom_sig = torch.clamp(dC_dsig.detach() ** 2, min=eps)
    L_diapycnal = (dC_dspi**2 / denom_sig).mean()

    # ── L2: Smoothness ────────────────────────────────────────────────────────
    d2C_dsig2 = torch.autograd.grad(
        dC_dsig.sum(), x_batch, create_graph=True, retain_graph=True
    )[0][:, idx["sigma0"]]

    if torch.isnan(d2C_dsig2).any():
        print("❌ CRITICAL: Second-order gradient d2C_dsig2 contains NaNs!", file=sys.stderr)

    L_smooth = (d2C_dsig2**2).mean()


    # ── L5: Advection ─────────────────────────────────────────────────────────
    # Replace raw u, v NaNs with 0.0 just in case
    u_clean = torch.nan_to_num(u_raw, nan=0.0)
    v_clean = torch.nan_to_num(v_raw, nan=0.0)

    speed_sq = torch.clamp(u_clean**2 + v_clean**2, min=1e-8)
    speed_raw = torch.sqrt(speed_sq)

    u_hat = u_clean / speed_raw
    v_hat = v_clean / speed_raw

    along_grad = u_hat * dC_dlon + v_hat * dC_dlat
    w_flow = speed_raw / (SPEED_SCALE + speed_raw)

    L_advect = (w_flow.detach() * along_grad**2).mean()

    # ── Debug Check Output Values ─────────────────────────────────────────────
    losses = {
        "L_diapycnal": L_diapycnal,
        "L_smooth": L_smooth,
        "L_advect": L_advect,
    }

    for name, val in losses.items():
        if torch.isnan(val):
            print(f"⚠️ NaN isolated in specific loss term: {name}", file=sys.stderr)

    return L_diapycnal, L_smooth, L_advect
