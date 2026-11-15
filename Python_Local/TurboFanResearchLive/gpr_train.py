from utils.tshae_utils import load_tshae_model      #import the load_tshae_model function from the tshae_utils.py file
from utils.metric_dataloader import MetricDataPreprocessor
from models.gpr_model import GPRegressor #impport the GPRegressor class from the gpr_model.py file
import numpy as np
import torch
import json
import os

import hydra


def get_z_and_rul(model, loader):

    """
    Runs the frozen TSHAE encoder over a dataloader and collects the latent
    vectors (z) and the true RUL values for every sample.

    NOTE: TSHAE no longer has a regressor, so model(x) now returns
    (z, mean, log_var, x_hat) instead of the old (rul_hat, z, mean, log_var, x_hat).
    There is no predicted RUL here at all, the true RUL (rul_all) comes
    straight from the dataloader's labels, not from the model. Predicted RUL
    only exists downstream, once z is passed into the GP.
    """
    
    model.eval()
    z_all, rul_all = [], []
    with torch.no_grad():
        for data in loader:
            pairs_mode = loader.dataset.return_pairs
            if pairs_mode:
                x, pos_x, neg_x, y, _, _ = data
            else:
                x, y = data

            z, mean, log_var, x_hat = model(x)  # matches current TSHAE.forward()
            z_all.append(z.numpy())
            rul_all.append(y.numpy())  # true RUL comes from the dataset, not the model

    z_all = np.concatenate(z_all)
    rul_all = np.concatenate(rul_all).ravel()
    return z_all, rul_all


@hydra.main(version_base=None, config_path="./configs", config_name="config.yaml")
def main(config):

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    print(f"output dir: {output_dir}")

   #load the pretrained tsahe model
    tshae_checkpoint_path = config.diffusion.checkpoint_tshae.path
    model_tshae = load_tshae_model(tshae_checkpoint_path)
    model_tshae.eval()
    for param in model_tshae.parameters():
        param.requires_grad = False

    # load the dataholders
    preproc = MetricDataPreprocessor(**config.data_preprocessor)
    train_loader, test_loader, val_loader = preproc.get_dataloaders()
    print(f"train set: {len(train_loader.dataset)} val set: {len(val_loader.dataset)}")

    # take the latent vectors and the true RUL values from training and validation sets
    # (rul_train / rul_val are  from the dataset, not model predictions)
    z_train, rul_train = get_z_and_rul(model_tshae, train_loader)
    z_val, rul_val = get_z_and_rul(model_tshae, val_loader)

   
    idx = np.random.choice(len(z_train), size=2900, replace=False)
    z_train, rul_train = z_train[idx], rul_train[idx]

    #sqaured features
    z_train = np.column_stack([z_train, z_train[:,0]**2, z_train[:,1]**2, z_train[:,0]*z_train[:,1]])
    z_val = np.column_stack([z_val, z_val[:,0]**2, z_val[:,1]**2, z_val[:,0]*z_val[:,1]])

    print(f"z_train shape: {z_train.shape}, rul_train shape: {rul_train.shape}")
    print(f"z_val shape: {z_val.shape}, rul_val shape: {rul_val.shape}")

    # fit the GP model on the training set
    gpr = GPRegressor(
        length_scale=1.0,
        nu=1.5,
        noise_level=1e-8,
        noise_level_bounds=(1e-10, 1),
        alpha=1e-7,
        n_restarts_optimizer=7
    )
    gpr.fit(z_train, rul_train)

    #evaluate the GP model on the validation set
    rul_pred_val, rul_std_val = gpr.predict(z_val)
    rmse_val = np.sqrt(np.mean((rul_pred_val - rul_val) ** 2))
    mean_uncertainty = np.mean(rul_std_val)

    print(f"Validation RMSE: {rmse_val:.4f}")
    print(f"Mean predicted uncertainty (std): {mean_uncertainty:.4f}")

    # save the model and results
    gpr.save(os.path.join(output_dir, "gpr_best_model.pkl"))

    results = {
        "val_rmse": float(rmse_val),
        "val_mean_uncertainty": float(mean_uncertainty)
    }
    with open(os.path.join(output_dir, "gpr_results.json"), 'w') as f:
        json.dump(results, f)

    print(f"Saved GP model and results at {output_dir}")


if __name__ == "__main__":
    main()