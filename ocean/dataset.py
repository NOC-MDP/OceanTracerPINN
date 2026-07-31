import torch
import numpy as np
import pandas as pd
import os
from utilities.utils import log_status

class TracerDataset(torch.utils.data.Dataset):
    """
    Wraps (X_scaled, y, u_raw, v_raw) so the DataLoader can serve velocity
    alongside features without shuffling them out of sync.
    """

    def __init__(self, X, y, u, v):
        # Convert inputs if they are numpy arrays or lists
        X_tensor = torch.tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X.to(torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32) if not isinstance(y, torch.Tensor) else y.to(torch.float32)
        u_tensor = torch.tensor(u, dtype=torch.float32) if not isinstance(u, torch.Tensor) else u.to(torch.float32)
        v_tensor = torch.tensor(v, dtype=torch.float32) if not isinstance(v, torch.Tensor) else v.to(torch.float32)

        self.X = torch.nan_to_num(X_tensor, nan=0.0)
        self.y = y_tensor  # Leave targets untouched or handle separately if NaNs exist

        # ── Fill NaN velocity values with 0.0 m/s ─────────────────────────────
        # For L_advect: speed=0 implies w_flow=0, naturally turning off advection
        # constraints for missing/unobserved velocity points.
        self.u = torch.nan_to_num(u_tensor, nan=0.0)
        self.v = torch.nan_to_num(v_tensor, nan=0.0)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.u[i], self.v[i]

# ── 4. TEMPORAL CROSS-VALIDATION ───────────────────────────────────────────────
#
# NEVER use random splits with oceanographic time series.
# Use temporal blocking: train on past, evaluate on withheld future periods.


def temporal_cv_splits(years, n_splits=5):
    """
    Expanding-window temporal CV splits.
    The first block (year_min to first boundary) is burn-in — training-only,
    never validated, since a fold needs some minimum training history.
    The remaining range is divided into n_splits equal validation blocks,
    so validation coverage spans the full range through year_max.
    """
    year_min, year_max = years.min(), years.max()
    block = (year_max - year_min) / (n_splits + 1)
    boundaries = [year_min + block * i for i in range(n_splits + 2)]  # n_splits+2 points

    splits = []
    for k in range(1, n_splits + 1):
        cut = boundaries[k]
        val_end = boundaries[k + 1]
        train_idx = np.where(years < cut)[0]
        if k == n_splits:
            # last fold: include year_max itself
            val_idx = np.where((years >= cut) & (years <= val_end))[0]
        else:
            val_idx = np.where((years >= cut) & (years < val_end))[0]
        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))
    return splits

def save_cv_results(cv_results, output_dir):
    df_cv = pd.DataFrame(cv_results)
    path = os.path.join(output_dir, "cv_results.csv")
    df_cv.to_csv(path, index=False)
    log_status(f"  Saved: {path}")

    log_status("── Temporal CV results ────────────────────────────────────")
    for row in cv_results:
        log_status(
            f"  Fold {row['fold']}  "
            f"train {int(row['train_year_min'])}-{int(row['train_year_max'])}  "
            f"val {int(row['val_year_min'])}-{int(row['val_year_max'])}  "
            f"R2={row['val_r2']:.3f}  RMSE={row['val_rmse']:.4f}  "
            f"N_val={int(row['n_val'])}"
        )
    r2s = [r["val_r2"] for r in cv_results]
    log_status(f"  Mean R2: {np.mean(r2s):.3f} +/- {np.std(r2s):.3f}")
    log_status("")
