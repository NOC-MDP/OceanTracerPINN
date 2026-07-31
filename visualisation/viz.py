import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def save_history_plot(history, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["epoch"], history["data_loss"], label="Data")
    axes[0].plot(history["epoch"], history["phys_loss"], label="Physics")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training losses")
    axes[0].legend()
    axes[0].set_yscale("log")

    axes[1].plot(history["epoch"], history["val_rmse"])
    axes[1].set(xlabel="Epoch", ylabel="RMSE", title="Validation RMSE")

    axes[2].plot(history["epoch"], history["val_r2"])
    axes[2].axhline(0, color="r", ls="--")
    axes[2].set(xlabel="Epoch", ylabel="R2", title="Validation R2")

    plt.tight_layout()
    path = os.path.join(output_dir, "training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_residuals_vs_speed(df_valid, y_pred_mn, y, output_dir):
    """
    Residual vs current speed diagnostic.
    If the advection loss is working, residuals should be flat across speed —
    i.e. the model should not be systematically worse in fast currents.
    """
    speed = np.sqrt(df_valid["u"].values ** 2 + df_valid["v"].values ** 2)
    resid = y_pred_mn - y

    # Bin by speed quantile for a clean summary line
    n_bins = 20
    bins = np.percentile(speed, np.linspace(0, 100, n_bins + 1))
    bin_idx = np.digitize(speed, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    bin_mid = 0.5 * (bins[:-1] + bins[1:])
    bin_rmse = np.array(
        [
            np.sqrt(np.mean(resid[bin_idx == i] ** 2))
            if (bin_idx == i).sum() > 0
            else np.nan
            for i in range(n_bins)
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(speed, resid, s=3, alpha=0.3, c="steelblue")
    axes[0].axhline(0, color="r", lw=1)
    axes[0].set(
        xlabel="Current speed (m/s)",
        ylabel="Residual",
        title="Residual vs speed\n(should be centred at all speeds)",
    )

    axes[1].plot(bin_mid, bin_rmse, "o-", color="steelblue", ms=5)
    axes[1].set(
        xlabel="Current speed (m/s)",
        ylabel="RMSE",
        title="RMSE by speed bin\n(flat = advection loss working)",
    )

    plt.tight_layout()
    path = os.path.join(output_dir, "residuals_vs_speed.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

# ── 9. DIAGNOSTICS ─────────────────────────────────────────────────────────────


def plot_ts_diagram(df, feat_names, y_pred):
    """T-S diagram coloured by tracer — key sanity check.
    Predictions should follow isopycnal contours for a conservative tracer."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sc1 = ax1.scatter(
        df["salinity"], df["temperature"], c=df["tracer"], cmap="plasma", s=5, alpha=0.6
    )
    ax1.set(
        xlabel="Salinity",
        ylabel="Temperature (°C)",
        title="Observed tracer on T-S diagram",
    )
    plt.colorbar(sc1, ax=ax1, label="Tracer")

    sc2 = ax2.scatter(
        df["salinity"], df["temperature"], c=y_pred, cmap="plasma", s=5, alpha=0.6
    )
    ax2.set(
        xlabel="Salinity",
        ylabel="Temperature (°C)",
        title="Predicted tracer on T-S diagram",
    )
    plt.colorbar(sc2, ax=ax2, label="Tracer")

    plt.tight_layout()
    plt.savefig("ts_diagram_check.png", dpi=150)


def plot_depth_residuals(y_true, y_pred, depths):
    """Residuals should be white noise vs depth — structure means missing physics."""
    resid = y_pred - y_true
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(resid, depths, s=3, alpha=0.3)
    axes[0].axvline(0, color="r")
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Depth (m)")
    axes[0].invert_yaxis()
    axes[0].set_title("Residual vs Depth — should be centred")

    axes[1].scatter(y_true, y_pred, s=3, alpha=0.3)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[1].plot(lims, lims, "r--")
    axes[1].set_xlabel("Observed")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title("1:1 plot")

    axes[2].hist(resid, bins=60, edgecolor="k")
    axes[2].set_xlabel("Residual")
    axes[2].set_title("Residual distribution")
    axes[2].set_xlabel("Residual")
    axes[2].set_title("Residual distribution")

    plt.tight_layout()
    plt.savefig("residual_diagnostics.png", dpi=150)
