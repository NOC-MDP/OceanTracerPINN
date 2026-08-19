from utilities.utils import (set_seed,
                                load_config,
                                load_observations,
                                summarise_dataset,
                                log_status)
from visualisation.viz import (save_history_plot,
                                    plot_residuals_vs_speed,
                                    plot_depth_residuals,
                                    plot_ts_diagram
)
from ocean.build_features import build_features
from ocean.dataset import temporal_cv_splits, save_cv_results
from models.architecture import TracerPINN
from models.monte_carlo import predict_with_uncertainty
from training.pinn import train_pinn
import torch
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main(cfg_path):
    log_status(f"Loading config from: {cfg_path}")
    # load yml config file and split
    cfg = load_config(cfg_path)
    project_cfg = cfg['project']
    training_cfg = cfg['training']
    model_cfg = cfg['model']
    validation_cfg = cfg['validation']
    dataset_cfg = cfg['dataset']
    loss_cfg = cfg['loss']

    set_seed(project_cfg['seed'])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(project_cfg['output_dir'], exist_ok=True)

    log_status(f"{'=' * 60}")
    log_status(" PINN Conservative Tracer Inference")
    log_status(f"{'=' * 60}")
    log_status(f"  Device       : {device}")
    log_status(f"  Obs file     : {dataset_cfg['parquet_file']}")
    log_status(f"  Tracer col   : {dataset_cfg['target']}")
    log_status(f"  Datetime col   : {dataset_cfg['inputs']['datetime']}")
    log_status(f"  Velocity cols: u='{dataset_cfg['inputs']['u']}', v='{dataset_cfg['inputs']['v']}'")
    log_status(f"  Epochs       : {training_cfg['epochs']}")
    log_status(f"  Output dir   : {project_cfg['output_dir']}")
    log_status(f"  Training Data Holdout   : {model_cfg['holdout_frac']}")
    log_status("Physics loss weights:")
    log_status(f"    lambda_diap   = {loss_cfg['lambda_diap']}")
    log_status(f"    lambda_smooth = {loss_cfg['lambda_smooth']}")
    log_status(f"    lambda_sec    = {loss_cfg['lambda_sec']}")
    log_status(f"    lambda_strat  = {loss_cfg['lambda_strat']}")
    log_status(f"    lambda_advect = {loss_cfg['lambda_advect']}")
    log_status("")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    log_status("── 1. Loading observations ─────────────────────────────────")
    df = load_observations(
        path=dataset_cfg['parquet_file'],
        tracer_col= dataset_cfg['target'],
        depth_col = dataset_cfg['inputs']['depth'],
        lon_col=dataset_cfg['inputs']['longitude'],
        lat_col=dataset_cfg['inputs']['latitude'],
        temp_col=dataset_cfg['inputs']['temperature'],
        sal_col=dataset_cfg['inputs']['salinity'],
        u_col=dataset_cfg['inputs']['u'],
        v_col=dataset_cfg['inputs']['v'],
        date_col=dataset_cfg['inputs']['datetime'],
        date_format=dataset_cfg['date_format']
    )
    summarise_dataset(df)

    # ── 2. Feature engineering ────────────────────────────────────────────────
    log_status("── 2. Engineering features (TEOS-10 + velocity) ────────────")
    X, y, feat_names, valid_mask = build_features(df)
    df_valid = df.iloc[valid_mask].reset_index(drop=True)
    years = df_valid["year"].values

    # Extract raw u, v aligned to valid observations for the physics loss.
    # If velocity was absent these are all-zero arrays and has_velocity=False,
    # so train_pinn will skip the advection loss regardless.
    u_valid = df_valid["u"].values.astype(np.float32)
    v_valid = df_valid["v"].values.astype(np.float32)

    log_status(f"  Features : {feat_names}")
    log_status(f"  Valid obs: {len(y):,}")

    # ── 2b. Carve out a true holdout test set (BEFORE CV / final training) ───
    log_status(f"── 2b. Carving out holdout test set ({cfg['model']['holdout_frac']:.0%}) ──")
    cutoff = np.percentile(years, 100 * (1 - cfg['model']['holdout_frac']))
    test_mask = years > cutoff
    train_mask = ~test_mask

    X_test, y_test = X[test_mask], y[test_mask]
    df_test = df_valid.iloc[test_mask].reset_index(drop=True)

    X, y = X[train_mask], y[train_mask]
    df_valid = df_valid.iloc[train_mask].reset_index(drop=True)
    years = years[train_mask]
    u_valid, v_valid = u_valid[train_mask], v_valid[train_mask]

    log_status(f"  Train+CV pool : {train_mask.sum():,} obs (years <= {cutoff:.0f})")
    log_status(f"  Held-out test : {test_mask.sum():,} obs (years >  {cutoff:.0f})")
    log_status("  This test set is not touched again until Step 5b.\n")

    # ── 3. Temporal cross-validation ──────────────────────────────────────────
    log_status(f"── 3. Temporal CV ({cfg['model']['cv_folds']} folds) ───────────────────────────")
    splits = temporal_cv_splits(years, n_splits=cfg['model']['cv_folds'])
    cv_results = []

    for fold, (tr_idx, val_idx) in enumerate(splits):
        log_status(
            f"  [Fold {fold + 1}/{len(splits)}]  "
            f"train n={len(tr_idx):,}  val n={len(val_idx):,}"
        )

        scaler_cv = StandardScaler()
        X_cv_scaled = scaler_cv.fit_transform(X)
        model_cv = TracerPINN(
            n_features=len(feat_names),
            hidden_dim=model_cfg['hidden_dim'],
            n_blocks=model_cfg['n_blocks'],
        ).to(device)

        model_cv, hist_cv = train_pinn(
            model_cv,
            X_cv_scaled[tr_idx],
            y[tr_idx],
            X_cv_scaled[val_idx],
            y[val_idx],
            feat_names,
            u_tr=u_valid[tr_idx],
            v_tr=v_valid[tr_idx],
            u_val=u_valid[val_idx],
            v_val=v_valid[val_idx],
            n_epochs=training_cfg['epochs'],
            eta_min=training_cfg['eta_min'],
            weight_decay=training_cfg['weight_decay'],
            batch_size=training_cfg['batch_size'],
            lr=training_cfg['learning_rate'],
            lambda_diap=loss_cfg['lambda_diap'],
            lambda_smooth=loss_cfg['lambda_smooth'],
            lambda_sec=loss_cfg['lambda_sec'],
            lambda_strat=loss_cfg['lambda_strat'],
            lambda_advect=loss_cfg['lambda_advect'],
            device=device,
        )

        cv_results.append(
            {
                "fold": fold + 1,
                "train_year_min": years[tr_idx].min(),
                "train_year_max": years[tr_idx].max(),
                "val_year_min": years[val_idx].min(),
                "val_year_max": years[val_idx].max(),
                "n_train": len(tr_idx),
                "n_val": len(val_idx),
                "val_r2": hist_cv["val_r2"].iloc[-1],
                "val_rmse": hist_cv["val_rmse"].iloc[-1],
            }
        )

    save_cv_results(cv_results, cfg['project']['output_dir'])

    # ── 4. Final model (train on all data) ────────────────────────────────────
    log_status("── 4. Training final model on all data ─────────────────────")
    # Fit scaler ONCE on unscaled training features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X)
    model_final = TracerPINN(
        n_features=len(feat_names),
        hidden_dim=model_cfg['hidden_dim'],
        n_blocks=model_cfg['n_blocks'],
    ).to(device)

    model_final, history = train_pinn(
        model_final,
        X_train_scaled,
        y,
        X_train_scaled,
        y,  # val == train for final loss monitoring only
        feat_names,
        u_tr=u_valid,
        v_tr=v_valid,
        u_val=u_valid,
        v_val=v_valid,
        n_epochs=training_cfg['epochs'],
        eta_min=training_cfg['eta_min'],
        weight_decay=training_cfg['weight_decay'],
        batch_size=training_cfg['batch_size'],
        lr=training_cfg['learning_rate'],
        lambda_diap=loss_cfg['lambda_diap'],
        lambda_smooth=loss_cfg['lambda_smooth'],
        lambda_sec=loss_cfg['lambda_sec'],
        lambda_strat=loss_cfg['lambda_strat'],
        lambda_advect=loss_cfg['lambda_advect'],
        device=device,
    )
    save_history_plot(history, project_cfg['output_dir'])

    # Save model + scaler for future inference
    model_path = os.path.join(project_cfg['output_dir'], f"pinn_{project_cfg['tracer']}.pt")
    torch.save(
        {
            "model_state": model_final.state_dict(),
            "feat_names": feat_names,
            "hidden_dim": model_cfg['hidden_dim'],
            "n_blocks": model_cfg['n_blocks'],
            "n_features": len(feat_names),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        },
        model_path,
    )
    log_status(f"  Saved model: {model_path}")

    # ── 5. Diagnostics ────────────────────────────────────────────────────────
    log_status("── 5. Diagnostics ──────────────────────────────────────────")
    X_sc = scaler.transform(X)
    y_pred_mn, y_pred_sd = predict_with_uncertainty(
        model_final,
        X_sc,
        n_samples=cfg['validation']['mc_samples'],
        device=device,
    )

    # Save predictions
    df_pred = df_valid.copy()
    df_pred["tracer_pred"] = y_pred_mn
    df_pred["tracer_uncert"] = y_pred_sd
    df_pred["residual"] = y_pred_mn - y
    pred_path = os.path.join(project_cfg['output_dir'], "predictions.csv")
    df_pred.to_csv(pred_path, index=False)
    log_status(f"  Saved: {pred_path}")

    overall_rmse = np.sqrt(np.mean((y_pred_mn - y) ** 2))
    overall_r2 = 1 - (np.sum((y_pred_mn - y) ** 2) / np.sum((y - y.mean()) ** 2))
    log_status(f"  In-sample RMSE       : {overall_rmse:.4f}")
    log_status(f"  In-sample R2         : {overall_r2:.3f}")
    log_status(f"  Mean uncertainty (1s): {y_pred_sd.mean():.4f}")

    # T-S diagram
    plot_ts_diagram(df_valid, feat_names, y_pred_mn)
    ts_path = os.path.join(project_cfg['output_dir'], "ts_diagram_check.png")
    if os.path.exists("ts_diagram_check.png"):
        os.replace("ts_diagram_check.png", ts_path)
    log_status(f"  Saved: {ts_path}")

    # Depth residuals
    plot_depth_residuals(y, y_pred_mn, df_valid["depth"].values)
    res_path = os.path.join(project_cfg['output_dir'], "residual_diagnostics.png")
    if os.path.exists("residual_diagnostics.png"):
        os.replace("residual_diagnostics.png", res_path)
    log_status(f"  Saved: {res_path}")

    # Uncertainty vs depth
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(
        y_pred_sd,
        df_valid["depth"].values,
        c=np.abs(y_pred_mn - y),
        cmap="hot_r",
        s=4,
        alpha=0.5,
    )
    ax.invert_yaxis()
    ax.set(
        xlabel="Prediction uncertainty (1s)",
        ylabel="Depth (m)",
        title="MC-Dropout uncertainty vs depth\n(colour = |residual|)",
    )
    plt.colorbar(sc, ax=ax, label="|residual|")
    plt.tight_layout()
    unc_path = os.path.join(project_cfg['output_dir'], "uncertainty_vs_depth.png")
    plt.savefig(unc_path, dpi=150)
    plt.close()
    log_status(f"  Saved: {unc_path}")

    # ── 5b. Held-out test evaluation (model has never seen this data) ────────
    log_status("\n── 5b. Held-out test evaluation ─────────────────────────────")
    X_test_sc = scaler.transform(X_test)  # scaler fit on train pool only
    y_test_pred_mn, y_test_pred_sd = predict_with_uncertainty(
        model_final, X_test_sc, n_samples=validation_cfg['mc_samples'], device=device,
    )

    test_rmse = np.sqrt(np.mean((y_test_pred_mn - y_test) ** 2))
    test_r2 = 1 - (
        np.sum((y_test_pred_mn - y_test) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
    )
    log_status(f"  Held-out RMSE : {test_rmse:.4f}")
    log_status(f"  Held-out R2   : {test_r2:.3f}")
    log_status(f"  Compare to in-sample R2 above ({overall_r2:.3f}) — a large gap "
            f"flags overfitting.")

    df_test_pred = df_test.copy()
    df_test_pred["tracer_pred"] = y_test_pred_mn
    df_test_pred["tracer_uncert"] = y_test_pred_sd
    df_test_pred["residual"] = y_test_pred_mn - y_test
    test_pred_path = os.path.join(project_cfg['output_dir'], "predictions_heldout_test.csv")
    df_test_pred.to_csv(test_pred_path, index=False)
    log_status(f"  Saved: {test_pred_path}")

    # Residuals vs current speed (only meaningful if velocity present)
    plot_residuals_vs_speed(df_valid, y_pred_mn, y, project_cfg['output_dir'])

    # ── Done ──────────────────────────────────────────────────────────────────
    log_status(f"{'=' * 60}")
    log_status(f"  All outputs written to: {project_cfg['output_dir']}/")
    log_status(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Physics-Informed Neural Network")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yml",
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    main(cfg_path=args.config)
