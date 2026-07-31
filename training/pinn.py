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
    lambda_diap,
    lambda_smooth,
    lambda_sec,
    lambda_strat,
    lambda_advect,
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
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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

        for batch in loader:
            Xb, yb, ub, vb = batch
            ub = ub.to(device)
            vb = vb.to(device)


            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()

            # Data loss
            yp = model(Xb)
            L_dat = huber(yp, yb)

            # Physics losses
            Ld, Ls, Lt, Lstr, Ladv = physics_losses(
                model,
                Xb,
                feat_names,
                u_raw=ub ,
                v_raw=vb,
            )

            L_phys = (
                lambda_diap * Ld
                + lambda_smooth * Ls
                + lambda_sec * Lt
                + lambda_strat * Lstr
                + (lambda_advect * Ladv)
            )

            (L_dat + L_phys).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


            e_data += L_dat.item()
            e_phys += L_phys.item()

        # Validation
        model.eval()
        with torch.no_grad():
            yv_pred = model(X_val_t.to(device)).cpu()
            rmse = torch.sqrt(((yv_pred - y_val_t) ** 2).mean()).item()
            r2 = (
                1
                - ((yv_pred - y_val_t) ** 2).sum().item()
                / ((y_val_t - y_val_t.mean()) ** 2).sum().item()
            )
        # 3. Step the Cosine Scheduler (ONCE PER EPOCH)
        sched.step()

        # Optional: Log the current learning rate
        current_lr = opt.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "data_loss": e_data / len(loader),
                "phys_loss": e_phys / len(loader),
                "val_rmse": rmse,
                "val_r2": r2,
                "current_lr": current_lr
            }
        )

        if epoch % 50 == 0:
            print(
                f"  [{epoch:3d}] data={e_data / len(loader):.4f}  "
                f"phys={e_phys / len(loader):.5f}  "
                f"val RMSE={rmse:.4f}  R²={r2:.3f} "
                f"LR={current_lr:.2e} "
            )

    return model, pd.DataFrame(history)
