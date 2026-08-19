import torch
import pandas as pd
from ocean.dataset import TracerDataset
from models.loss import physics_losses

import torch.nn as nn

def train_pinn(
    model,
    X_tr,
    y_tr,
    X_val,
    y_val,
    feat_names,
    scaler,
    u_tr,
    v_tr,
    u_val,
    v_val,
    n_epochs,
    batch_size,
    eta_min,
    weight_decay,
    lr,
    device="cpu",
):
    """
    Training loop with 5-term physics loss.

    """
    X_tr_sc = scaler.transform(X_tr)
    X_val_sc = scaler.transform(X_val)

    # Use TracerDataset
    tr_ds = TracerDataset(X_tr_sc, y_tr, u_tr, v_tr)

    loader = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val_sc, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    model = model.to(device)
    # Initialize loss wrapper
    loss_scaler = AdaptivePINNLoss(num_losses=6).to(device)
    # Pass BOTH model and loss parameters to AdamW
    opt = torch.optim.AdamW(
        [
            {"params": model.parameters(), "weight_decay": weight_decay},
            {"params": loss_scaler.parameters(), "weight_decay": 0.0} # Do NOT weight decay loss parameters!
        ],
        lr=lr
    )
    # --- (CosineAnnealingLR) ---
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=n_epochs,  # Total epochs for a full cosine cycle
        eta_min=eta_min,    # Minimum learning rate target at the end of training
    )
    huber = nn.HuberLoss(delta=1.0)
    history = []

    for epoch in range(n_epochs):
        model.train()
        e_data = e_phys = 0.0
        e_total = 0.0

        for batch in loader:
            Xb, yb, ub, vb = batch
            ub = ub.to(device)
            vb = vb.to(device)


            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()

            # Data loss
            yp = model(Xb).squeeze(-1)
            L_dat = huber(yp, yb)

            # Physics losses
            physics_tuple = physics_losses(
                model,
                Xb,
                feat_names,
                u_raw=ub ,
                v_raw=vb,
            )

            # 3. Dynamic Adaptive Loss Weighting
            total_loss, loss_diag = loss_scaler(L_dat, physics_tuple)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping for stability
            opt.step()


            e_data += L_dat.item()
            # Sum raw physics losses for reporting
            L_phys_raw = torch.stack(physics_tuple).sum()
            e_phys += L_phys_raw.item()
            e_total += total_loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            yv_pred = model(X_val_t.to(device)).squeeze(-1).cpu()
            rmse = torch.sqrt(((yv_pred - y_val_t) ** 2).mean()).item()

            denom = ((y_val_t - y_val_t.mean()) ** 2).sum().item()
            r2 = 1.0 - (((yv_pred - y_val_t) ** 2).sum().item() / max(denom, 1e-8))
        # 3. Step the Cosine Scheduler (ONCE PER EPOCH)
        sched.step()

        # Optional: Log the current learning rate
        current_lr = opt.param_groups[0]["lr"]
        n_batches = len(loader)

        history.append(
            {
                "epoch": epoch,
                "total_loss": e_total / n_batches,
                "data_loss": e_data / n_batches,
                "phys_loss": e_phys / n_batches,
                "val_rmse": rmse,
                "val_r2": r2,
                "current_lr": current_lr
            }
        )

        if epoch % 50 == 0:
            print(
                f"  [{epoch:3d}] Total={e_total / n_batches:.4f}  "
                f"Data={e_data / n_batches:.4f}  "
                f"Phys(raw)={e_phys / n_batches:.5f}  "
                f"val RMSE={rmse:.4f}  R²={r2:.3f}  "
                f"LR={current_lr:.2e}"
            )

    return model, pd.DataFrame(history)
