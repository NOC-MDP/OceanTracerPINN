import torch
import numpy as np

def predict_with_uncertainty(
    model, X_scaled, n_samples=100, batch_size=100_000, device="cpu"
):

    model.train()  # keep dropout ON

    n = X_scaled.shape[0]
    preds_mean = np.zeros(n, dtype=np.float32)
    preds_sq = np.zeros(n, dtype=np.float32)

    for i in range(0, n, batch_size):
        X_batch = torch.tensor(X_scaled[i : i + batch_size], dtype=torch.float32).to(
            device
        )

        batch_preds = []

        for _ in range(n_samples):
            with torch.no_grad():
                y = model(X_batch).cpu().numpy()
            batch_preds.append(y)

        batch_preds = np.stack(batch_preds)

        preds_mean[i : i + batch_size] = batch_preds.mean(axis=0)
        preds_sq[i : i + batch_size] = (batch_preds**2).mean(axis=0)

        del X_batch, batch_preds
        torch.cuda.empty_cache()

    preds_std = np.sqrt(preds_sq - preds_mean**2)
    return preds_mean, preds_std
