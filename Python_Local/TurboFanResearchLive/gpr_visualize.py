import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import argparse

from utils.tshae_utils import load_tshae_model
from utils.metric_dataloader import MetricDataPreprocessor
from models.gpr_model import GPRegressor


def get_z_and_rul(model, loader):
    model.eval()
    z_all, rul_all = [], []
    with torch.no_grad():
        for data in loader:
            pairs_mode = loader.dataset.return_pairs
            if pairs_mode:
                x, pos_x, neg_x, y, _, _ = data
            else:
                x, y = data
            z, mean, log_var, x_hat = model(x)
            z_all.append(z.numpy())
            rul_all.append(y.numpy())
    z_all = np.concatenate(z_all)
    rul_all = np.concatenate(rul_all).ravel()
    return z_all, rul_all


def augment_z(z):
    return np.column_stack([z, z[:, 0]**2, z[:, 1]**2, z[:, 0] * z[:, 1]])


def get_dataloaders():
    preproc = MetricDataPreprocessor(
        dataset_name="FD003",
        max_rul=125,
        window_size=32,
        sensors=['s_1','s_2','s_3','s_4','s_5','s_6','s_7','s_8','s_9','s_10',
                 's_11','s_12','s_13','s_14','s_15','s_16','s_17','s_18','s_19','s_20','s_21'],
        train_size=0.99,
        alpha=1,
        dir_path='./CMAPSSData/',
        fix_seed=True,
        downsample_healthy_train=True,
        downsample_healthy_validation=False,
        downsample_healthy_test=False,
        downsample_healthy_train_p=0.9,
        downsample_healthy_validation_p=0.8,
        downsample_healthy_test_p=0,
        downsample_rul_threshold_train=40,
        downsample_rul_threshold_validation=False,
        downsample_rul_threshold_test=False,
        train_ds_mode="train",
        train_ds_return_pairs=False,
        train_ds_eps=2,
        train_ds_max_eps=10,
        train_ds_triplet_healthy_rul=120,
        test_ds_mode="test",
        test_ds_return_pairs=False,
        test_ds_eps=2,
        test_ds_max_eps=10,
        test_ds_triplet_healthy_rul=120,
        val_ds_mode="train",
        val_ds_return_pairs=False,
        val_ds_eps=2,
        val_ds_max_eps=10,
        val_ds_triplet_healthy_rul=120,
        train_dl_batch_size=64,
        train_dl_shuffle=True,
        train_dl_num_workers=4,
        test_dl_batch_size=256,
        test_dl_shuffle=False,
        test_dl_num_workers=4,
        val_dl_batch_size=256,
        val_dl_shuffle=True,
        val_dl_num_workers=4,
    )
    return preproc.get_dataloaders()


def load_gpr_and_data(gpr_checkpoint, tshae_checkpoint):
    print(f"Loading TSHAE from {tshae_checkpoint}...")
    model_tshae = load_tshae_model(tshae_checkpoint)
    model_tshae.eval()
    for param in model_tshae.parameters():
        param.requires_grad = False

    print("Loading data...")
    train_loader, test_loader, val_loader = get_dataloaders()
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    print("Extracting latent vectors...")
    z_train, rul_train = get_z_and_rul(model_tshae, train_loader)
    z_val, rul_val = get_z_and_rul(model_tshae, val_loader)

    idx = np.random.choice(len(z_train), size=2500, replace=False)
    z_train, rul_train = z_train[idx], rul_train[idx]

    print(f"Loading GPR from {gpr_checkpoint}...")
    gpr = GPRegressor()
    gpr.load(gpr_checkpoint)

    return gpr, z_train, z_val, rul_train, rul_val, model_tshae, val_loader


def plot_gpr_results(gpr, z_train, z_val, rul_train, rul_val, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    z_val_aug = augment_z(z_val)
    rul_pred_val, rul_std_val = gpr.predict(z_val_aug)
    rmse = np.sqrt(np.mean((rul_pred_val - rul_val) ** 2))

    z1_min, z1_max = z_val[:, 0].min() - 0.5, z_val[:, 0].max() + 0.5
    z2_min, z2_max = z_val[:, 1].min() - 0.5, z_val[:, 1].max() + 0.5
    h = 0.05
    xx, yy = np.meshgrid(np.arange(z1_min, z1_max, h),
                         np.arange(z2_min, z2_max, h))
    z_mesh = np.c_[xx.ravel(), yy.ravel()]
    z_mesh_aug = augment_z(z_mesh)
    z_mesh_pred, z_mesh_std = gpr.predict(z_mesh_aug)
    z_mesh_pred = z_mesh_pred.reshape(xx.shape)
    z_mesh_std = z_mesh_std.reshape(xx.shape)
    levels = np.linspace(0, 125, 26)

    fig = plt.figure(figsize=(20, 16))

    # 1. Predicted RUL contour over latent space
    ax1 = fig.add_subplot(2, 3, 1)
    cf = ax1.contourf(xx, yy, z_mesh_pred, levels=levels, cmap='viridis', extend='both')
    ax1.scatter(z_val[:, 0], z_val[:, 1], c='w', s=2, alpha=0.3)
    cbar = plt.colorbar(cf, ax=ax1)
    cbar.set_label('Predicted RUL')
    ax1.set_xlabel('z - dim 1')
    ax1.set_ylabel('z - dim 2')
    ax1.set_title('GP Predicted RUL over Latent Space')

    # 2. GP Uncertainty
    ax2 = fig.add_subplot(2, 3, 2)
    cf2 = ax2.contourf(xx, yy, z_mesh_std, levels=20, cmap='Reds')
    ax2.scatter(z_val[:, 0], z_val[:, 1], c='k', s=2, alpha=0.2)
    cbar2 = plt.colorbar(cf2, ax=ax2)
    cbar2.set_label('Uncertainty (std)')
    ax2.set_xlabel('z - dim 1')
    ax2.set_ylabel('z - dim 2')
    ax2.set_title('GP Prediction Uncertainty')

    # 3. Actual vs Predicted RUL
    ax3 = fig.add_subplot(2, 3, 3)
    sc3 = ax3.scatter(rul_val, rul_pred_val, c=rul_std_val, s=8, cmap='Reds', alpha=0.6)
    cbar3 = plt.colorbar(sc3, ax=ax3)
    cbar3.set_label('Uncertainty (std)')
    lo = min(rul_val.min(), rul_pred_val.min())
    hi = max(rul_val.max(), rul_pred_val.max())
    ax3.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='Ideal')
    ax3.set_xlabel('True RUL')
    ax3.set_ylabel('Predicted RUL')
    ax3.set_title(f'Actual vs Predicted RUL  |  RMSE: {rmse:.2f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Latent space colored by true RUL
    ax4 = fig.add_subplot(2, 3, 4)
    sc4 = ax4.scatter(z_val[:, 0], z_val[:, 1], c=rul_val, s=5, cmap='viridis')
    cbar4 = plt.colorbar(sc4, ax=ax4)
    cbar4.set_label('True RUL')
    ax4.set_xlabel('z - dim 1')
    ax4.set_ylabel('z - dim 2')
    ax4.set_title('Latent Space Colored by True RUL')

    # 5. Residuals
    ax5 = fig.add_subplot(2, 3, 5)
    residuals = rul_pred_val - rul_val
    sc5 = ax5.scatter(rul_val, residuals, c=rul_std_val, s=8, cmap='Reds', alpha=0.6)
    ax5.axhline(y=0, color='k', linestyle='--', lw=1.5)
    cbar5 = plt.colorbar(sc5, ax=ax5)
    cbar5.set_label('Uncertainty (std)')
    ax5.set_xlabel('True RUL')
    ax5.set_ylabel('Residual (Predicted - True)')
    ax5.set_title(f'Residuals  |  Mean Error: {np.mean(residuals):.2f}')
    ax5.grid(True, alpha=0.3)

    # 6. Prediction error distribution
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(residuals, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax6.axvline(x=0, color='r', linestyle='--', lw=1.5)
    ax6.set_xlabel('Prediction Error (Predicted - True)')
    ax6.set_ylabel('Count')
    ax6.set_title(f'Error Distribution  |  Std: {np.std(residuals):.2f}')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'gpr_results_overview.png')
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close('all')


def plot_gpr_3d_surface(gpr, z_val, rul_val, save_dir):
    z1_min, z1_max = z_val[:, 0].min() - 0.5, z_val[:, 0].max() + 0.5
    z2_min, z2_max = z_val[:, 1].min() - 0.5, z_val[:, 1].max() + 0.5
    h = 0.05
    xx, yy = np.meshgrid(np.arange(z1_min, z1_max, h),
                         np.arange(z2_min, z2_max, h))
    z_mesh = np.c_[xx.ravel(), yy.ravel()]
    z_mesh_aug = augment_z(z_mesh)
    z_mesh_pred, z_mesh_std = gpr.predict(z_mesh_aug)
    z_mesh_pred = z_mesh_pred.reshape(xx.shape)
    z_mesh_std = z_mesh_std.reshape(xx.shape)

    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(xx, yy, z_mesh_pred, cmap='viridis', alpha=0.8,
                            linewidth=0, antialiased=True)
    ax1.scatter(z_val[:, 0], z_val[:, 1], rul_val, c='r', s=3, alpha=0.4)
    ax1.set_xlabel('z1'); ax1.set_ylabel('z2'); ax1.set_zlabel('Predicted RUL')
    ax1.set_title('GP RUL Surface (Mean)')
    cbar = fig.colorbar(surf, ax=ax1, shrink=0.7)
    cbar.set_label('RUL')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(xx, yy, z_mesh_std, cmap='Reds', alpha=0.8,
                             linewidth=0, antialiased=True)
    ax2.set_xlabel('z1'); ax2.set_ylabel('z2'); ax2.set_zlabel('Uncertainty (std)')
    ax2.set_title('GP Uncertainty Surface')
    cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.7)
    cbar2.set_label('Std')

    plt.tight_layout()
    path = os.path.join(save_dir, 'gpr_3d_surface.png')
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close('all')


def plot_engine_trajectory(gpr, z_val, rul_val, val_loader, model_tshae, engine_id, save_dir):
    engine_ids = list(val_loader.dataset.ids)
    if engine_id not in engine_ids:
        print(f"Engine {engine_id} not found. Available IDs: {sorted(engine_ids)[:10]}...")
        engine_id = int(np.random.choice(engine_ids))
        print(f"Using engine {engine_id} instead.")

    x, y_true = val_loader.dataset.get_run(engine_id)
    model_tshae.eval()
    with torch.no_grad():
        z_run, _, _, _ = model_tshae(x)

    z_run = z_run.numpy()
    z_run_aug = augment_z(z_run)
    rul_pred_run, rul_std_run = gpr.predict(z_run_aug)
    y_true = y_true.numpy().ravel()
    t = np.arange(len(y_true))

    z1_min, z1_max = z_val[:, 0].min() - 0.5, z_val[:, 0].max() + 0.5
    z2_min, z2_max = z_val[:, 1].min() - 0.5, z_val[:, 1].max() + 0.5
    h = 0.05
    xx, yy = np.meshgrid(np.arange(z1_min, z1_max, h),
                         np.arange(z2_min, z2_max, h))
    z_mesh = np.c_[xx.ravel(), yy.ravel()]
    z_mesh_aug = augment_z(z_mesh)
    z_mesh_pred, _ = gpr.predict(z_mesh_aug)
    z_mesh_pred = z_mesh_pred.reshape(xx.shape)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: RUL trajectory with confidence band
    ax = axes[0, 0]
    ax.plot(t, y_true, 'b-', lw=2, label='True RUL')
    ax.plot(t, rul_pred_run, 'r-', lw=2, label='GP Predicted RUL')
    ax.fill_between(t, rul_pred_run - 1.96 * rul_std_run,
                    rul_pred_run + 1.96 * rul_std_run,
                    color='r', alpha=0.15, label='95% CI')
    ax.set_xlabel('Time (Cycle)'); ax.set_ylabel('RUL')
    ax.set_title(f'Engine Unit #{engine_id}: RUL Trajectory')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Top-right: trajectory in latent space
    ax = axes[0, 1]
    sc = ax.scatter(z_val[:, 0], z_val[:, 1], c=rul_val, s=3, cmap='viridis', alpha=0.4)
    plt.colorbar(sc, ax=ax, label='True RUL')
    ax.plot(z_run[:, 0], z_run[:, 1], 'r-', lw=2, alpha=0.8)
    ax.scatter(z_run[0, 0], z_run[0, 1], c='g', s=80, marker='o', label='Start', zorder=5)
    ax.scatter(z_run[-1, 0], z_run[-1, 1], c='r', s=80, marker='x', label='End', zorder=5)
    ax.set_xlabel('z - dim 1'); ax.set_ylabel('z - dim 2')
    ax.set_title(f'Engine #{engine_id} Latent Trajectory')
    ax.legend()

    # Bottom-left: GP surface with trajectory overlaid
    ax = axes[1, 0]
    cf = ax.contourf(xx, yy, z_mesh_pred, levels=np.linspace(0, 125, 26),
                     cmap='viridis', extend='both')
    plt.colorbar(cf, ax=ax, label='GP Predicted RUL')
    ax.plot(z_run[:, 0], z_run[:, 1], 'w-', lw=2, alpha=0.9)
    ax.scatter(z_run[0, 0], z_run[0, 1], c='g', s=80, marker='o', zorder=5)
    ax.scatter(z_run[-1, 0], z_run[-1, 1], c='r', s=80, marker='x', zorder=5)
    ax.set_xlabel('z - dim 1'); ax.set_ylabel('z - dim 2')
    ax.set_title(f'GP Surface + Engine #{engine_id} Trajectory')

    # Bottom-right: uncertainty along trajectory
    ax = axes[1, 1]
    ax.plot(t, rul_std_run, 'purple', lw=2)
    ax.fill_between(t, 0, rul_std_run, color='purple', alpha=0.15)
    ax.set_xlabel('Time (Cycle)'); ax.set_ylabel('GP Uncertainty (std)')
    ax.set_title(f'Engine #{engine_id}: Prediction Uncertainty')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f'engine_trajectory_{engine_id}.png')
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close('all')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize GPR results')
    parser.add_argument('--gpr_checkpoint', type=str,
                        default='outputs/2026-07-17/06-10-35/gpr_best_model.pkl')
    parser.add_argument('--tshae_checkpoint', type=str,
                        default='./best_models/FD003/tshae/best_rve_model.pt')
    parser.add_argument('--save_dir', type=str,
                        default='./outputs/gpr_visualizations')
    parser.add_argument('--engine_id', type=int, default=None)
    args = parser.parse_args()

    gpr, z_train, z_val, rul_train, rul_val, model_tshae, val_loader = \
        load_gpr_and_data(args.gpr_checkpoint, args.tshae_checkpoint)

    z_val_aug = augment_z(z_val)
    rul_pred, rul_std = gpr.predict(z_val_aug)
    rmse = np.sqrt(np.mean((rul_pred - rul_val) ** 2))
    print(f"\nValidation RMSE: {rmse:.2f}")
    print(f"Mean uncertainty: {np.mean(rul_std):.2f}")

    plot_gpr_results(gpr, z_train, z_val, rul_train, rul_val, args.save_dir)
    plot_gpr_3d_surface(gpr, z_val, rul_val, args.save_dir)
    if args.engine_id is not None:
        plot_engine_trajectory(gpr, z_val, rul_val, val_loader, model_tshae, args.engine_id, args.save_dir)
    print("Done!")
