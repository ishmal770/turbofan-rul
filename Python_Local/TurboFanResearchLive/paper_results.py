import numpy as np
import matplotlib.pyplot as plt
import torch
import joblib
import os
from sklearn.metrics import r2_score
from utils.tshae_utils import load_tshae_model
from utils.metric_dataloader import MetricDataPreprocessor
from omegaconf import OmegaConf


def main():
    gpr_model_path = r"outputs\2026-07-23\20-16-00\gpr_best_model.pkl"
    tshae_model_path = r"best_models\FD003\tshae\best_rve_model.pt"
    config_path = r"best_models\FD003\tshae\.hydra\config.yaml"
    output_path = r"outputs\gpr_visualizations\paper_results.png"

    print("Loading models...")
    model_tshae = load_tshae_model(tshae_model_path)
    model_tshae.eval()
    gpr = joblib.load(gpr_model_path)

    config = OmegaConf.load(config_path)
    config.data_preprocessor.train_dl_num_workers = 0
    config.data_preprocessor.test_dl_num_workers = 0
    config.data_preprocessor.val_dl_num_workers = 0
    preproc = MetricDataPreprocessor(**config.data_preprocessor)
    train_loader, test_loader, val_loader = preproc.get_dataloaders()

    print("Extracting latent vectors...")
    device = "cpu"
    z_all, rul_all = [], []
    model_tshae.to(device)
    with torch.no_grad():
        for data in val_loader:
            pairs_mode = val_loader.dataset.return_pairs
            if pairs_mode:
                x, _, _, y, _, _ = data
            else:
                x, y = data
            z, *_ = model_tshae(x)
            z_all.append(z.numpy())
            rul_all.append(y.numpy())

    z_val = np.concatenate(z_all)
    rul_val = np.concatenate(rul_all).ravel()
    z_val = np.column_stack([z_val, z_val[:,0]**2, z_val[:,1]**2, z_val[:,0]*z_val[:,1]])
    rul_pred, rul_std = gpr.predict(z_val, return_std=True)

    # metrics
    rmse = np.sqrt(np.mean((rul_pred - rul_val) ** 2))
    mse = np.mean((rul_pred - rul_val) ** 2)
    r2 = r2_score(rul_val, rul_pred)
    mae = np.mean(np.abs(rul_pred - rul_val))

    print(f"RMSE: {rmse:.2f}")
    print(f"MSE:  {mse:.2f}")
    print(f"R2:   {r2:.4f}")
    print(f"MAE:  {mae:.2f}")

    # bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ['RMSE', 'MAE', 'R\u00b2']
    values = [rmse, mae, r2]
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    bars = ax.bar(metrics, values, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax.set_ylabel('Value', fontsize=13)
    ax.set_title('GPR + TSHAE Performance Metrics (FD003)', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(min(0, r2*1.15), max(rmse, mae)*1.3)

    # add info box
    textstr = f"Dataset: FD003\nVal samples: {len(rul_val)}\nMAE: {mae:.2f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved to {output_path}")
    plt.show()


if __name__ == '__main__':
    main()
