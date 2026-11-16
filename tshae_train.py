
from utils.metric_dataloader import MetricDataPreprocessor
from utils.loss import TotalLoss
from utils.metric import KNNRULmetric
import torch
from collections import defaultdict
import json
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import gc
from tshae_test import Tester

import hydra
from hydra.utils import instantiate

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


class Trainer:

    def __init__(self, model, optimizer, train_loader, val_loader, test_loader, n_epochs, total_loss, validate,
                 save_model, save_history, rul_threshold=125, n_neighbors=3,
                 add_noise_train=False, add_noise_val=False, noise_mean=0, noise_std=1,
                 scheduler=None, verbose=True, device=None):
        """
        Initializes the Trainer object.

        :param rul_threshold: threshold for the KNN latent-space metric (replaces old RUL RMSE/score)
        :param n_neighbors: neighbors for the KNN latent-space metric
        (other params unchanged from before)
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.n_epochs = n_epochs
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.total_loss = total_loss
        self.history = defaultdict(list)
        self.verbose = verbose
        self.validate = validate
        self.save_model = save_model
        self.save_history = save_history
        self.add_noise_train = add_noise_train
        self.add_noise_val = add_noise_val
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.metric = KNNRULmetric(rul_threshold=rul_threshold, n_neighbors=n_neighbors)
        self.log = logging.getLogger(__name__)

    def train_epoch(self):
        """
        Training loop. TSHAE now returns (z, mean, log_var, x_hat) — no y_hat.
        """
        epoch_loss = defaultdict(list)
        self.model.train()
        for batch_idx, data in enumerate(self.train_loader):
            batch_len = len(self.train_loader.dataset)
            pairs_mode = self.train_loader.dataset.return_pairs
            self.optimizer.zero_grad()

            if pairs_mode:
                x, pos_x, neg_x, true_rul, _, _ = data
                if self.add_noise_train:
                    x += torch.empty_like(x).normal_(mean=self.noise_mean, std=self.noise_std)
                    pos_x += torch.empty_like(pos_x).normal_(mean=self.noise_mean, std=self.noise_std)
                    neg_x += torch.empty_like(neg_x).normal_(mean=self.noise_mean, std=self.noise_std)
                x = x.to(self.device)
                pos_x = pos_x.to(self.device)
                neg_x = neg_x.to(self.device)

                z, mean, log_var, x_hat = self.model(x)
                z_pos, *_ = self.model(pos_x)
                z_neg, *_ = self.model(neg_x)

                loss_dict = self.total_loss(
                    mean=mean,
                    log_var=log_var,
                    x=x,
                    x_hat=x_hat,
                    z=z,
                    z_pos=z_pos,
                    z_neg=z_neg
                )
            else:
                x, true_rul = data
                if self.add_noise_train:
                    x += torch.empty_like(x).normal_(mean=self.noise_mean, std=self.noise_std)
                x = x.to(self.device)
                z, mean, log_var, x_hat = self.model(x)
                loss_dict = self.total_loss(
                    mean=mean,
                    log_var=log_var,
                    x=x,
                    x_hat=x_hat,
                    z=z
                )

            loss = loss_dict["TotalLoss"]
            loss.backward()
            self.optimizer.step()

            for key in loss_dict:
                epoch_loss[key].append(loss_dict[key].item() * len(x))

        if self.scheduler:
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]
            self.log.info(f"learning_rate: {lr}")
        for key in loss_dict:
            self.history["Train_" + key].append(sum(epoch_loss[key]) / batch_len)

    def valid_epoch(self):
        """
        Validation loop. Same fix: z, mean, log_var, x_hat = self.model(x), no y_hat.
        """
        epoch_loss = defaultdict(list)
        self.model.train()
        for batch_idx, data in enumerate(self.val_loader):
            batch_len = len(self.val_loader.dataset)
            pairs_mode = self.val_loader.dataset.return_pairs
            with torch.no_grad():

                if pairs_mode:
                    x, pos_x, neg_x, y, _, _ = data
                    if self.add_noise_val:
                        x += torch.empty_like(x).normal_(mean=self.noise_mean, std=self.noise_std)
                        pos_x += torch.empty_like(pos_x).normal_(mean=self.noise_mean, std=self.noise_std)
                        neg_x += torch.empty_like(neg_x).normal_(mean=self.noise_mean, std=self.noise_std)
                    x = x.to(self.device)
                    pos_x = pos_x.to(self.device)
                    neg_x = neg_x.to(self.device)

                    z, mean, log_var, x_hat = self.model(x)
                    z_pos, *_ = self.model(pos_x)
                    z_neg, *_ = self.model(neg_x)

                    loss_dict = self.total_loss(mean=mean, log_var=log_var, x=x, x_hat=x_hat, z=z,
                                                 z_pos=z_pos, z_neg=z_neg)
                else:
                    x, y = data
                    x = x.to(self.device)
                    if self.add_noise_val:
                        x += torch.empty_like(x).normal_(mean=self.noise_mean, std=self.noise_std)
                    z, mean, log_var, x_hat = self.model(x)
                    loss_dict = self.total_loss(mean=mean, log_var=log_var, x=x, x_hat=x_hat, z=z)

                for key in loss_dict:
                    epoch_loss[key].append(loss_dict[key].item() * len(y))
        for key in loss_dict:
            self.history["Val_" + key].append(sum(epoch_loss[key]) / batch_len)

    def get_latent_metric(self, datasetloader):
        """
        Replaces the old RUL RMSE/score. TSHAE no longer predicts RUL, so model
        quality is judged by how well true RUL is structured in the latent
        space z, using the same KNNRULmetric as tshae_test.py.

        NOTE: check utils/metric.py to see whether LOWER or HIGHER means
        "better" for this metric. This assumes lower is better (mirrors old
        RMSE convention) — flip the comparison in train() if that's wrong.
        """
        self.model.eval()
        z_all, true_rul_all = [], []
        pairs_mode = datasetloader.dataset.return_pairs
        with torch.no_grad():
            for data in datasetloader:
                if pairs_mode:
                    x, _, _, y, _, _ = data
                else:
                    x, y = data
                x = x.to(self.device)
                z, mean, log_var, x_hat = self.model(x)
                z_all.append(z.cpu().numpy())
                true_rul_all.append(y.numpy())
        z_all = np.concatenate(z_all)
        true_rul_all = np.concatenate(true_rul_all).ravel()
        return self.metric.fit_calculate(z=z_all, rul=true_rul_all)

    def train(self, n_epochs=None):
        """
        Calls train and validate epochs, tracks the latent-space metric instead
        of RUL RMSE/score, saves the best model based on that metric.
        """
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        output_dir = hydra_cfg['runtime']['output_dir']
        best_metric = float("inf")
        if n_epochs:
            self.n_epochs = n_epochs
        for epoch in range(self.n_epochs):
            self.train_epoch()
            if self.validate:
                self.valid_epoch()
            if self.verbose:
                for key in self.history:
                    self.log.info(f"Epoch:{epoch} {key}: {self.history[key][-1] :3.3f}")

            val_metric = self.get_latent_metric(self.val_loader)
            self.history["Validation_metric"].append(val_metric)
            writer.add_scalar("Metric/Val", val_metric, epoch)
            print(f"val latent metric: {val_metric :6.4f} best: {best_metric :6.4f}")

            if val_metric < best_metric:
                best_metric = val_metric
                if self.save_model:
                    torch.save(self.model, output_dir + "/best_rve_model.pt")
                    print(f"Saved best model, metric: {val_metric :6.4f} at {output_dir}/best_rve_model.pt")

        if self.save_model:
            torch.save(self.model, output_dir + "/rve_model.pt")
            print("Saved", output_dir + "/rve_model.pt")
        if self.save_history:
            with open(output_dir + "/history.json", 'w') as fp:
                json.dump(self.history, fp)
        self.plot_learning_curves(output_dir, show=False)
        writer.flush()
        writer.close()

    def plot_learning_curves(self, path, show=True, save=True):
        """
        Dropped RegLoss + Validation_score/RMSE panels (TSHAE no longer
        produces y_hat). Tracks ReconLoss, KLLoss, TripletLoss, and the
        latent-space KNN metric instead.
        """
        history = self.history
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(14, 6))

        ax[0][0].plot(history['Train_TotalLoss'], label='train loss')
        ax[0][0].plot(history['Val_TotalLoss'], label='val loss')

        ax[0][1].plot(history['Train_KLLoss'], label='train kl_loss')
        ax[0][1].plot(history['Val_KLLoss'], label='val kl_loss')

        ax[1][0].plot(history['Train_ReconLoss'], label='train reconstruction loss')
        ax[1][0].plot(history['Val_ReconLoss'], label='val reconstruction loss')

        ax[1][1].plot(history['Train_TripletLoss'], label='train triplet loss')
        ax[1][1].plot(history['Val_TripletLoss'], label='val triplet loss')

        for row in ax:
            for a in row:
                a.legend(loc='upper right')
                a.grid(True)
                a.set_yscale('log')

        if save:
            images_dir = os.path.join(path, "images")
            os.makedirs(images_dir, exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(images_dir, "Training Curves.png"))
            plt.cla()
            plt.clf()
            plt.close('all')
            gc.collect()
        if show:
            plt.show()

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(history['Validation_metric'], label='Validation latent metric')
        ax2.legend(loc='upper right')
        ax2.grid(True)
        if save:
            images_dir = os.path.join(path, "images")
            plt.tight_layout()
            plt.savefig(os.path.join(images_dir, "Validation Metric.png"))
            plt.close('all')
        if show:
            plt.show()


@hydra.main(version_base=None, config_path="./configs", config_name="config.yaml")
def main(config):
    """
    Main function for training and testing the TSHAE model.
    """

    if config.random_seed.fix == True:
        import random
        import os

        torch.manual_seed(config.random_seed.seed)
        torch.cuda.manual_seed_all(config.random_seed.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        np.random.seed(config.random_seed.seed)
        random.seed(config.random_seed.seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']

    preproc = MetricDataPreprocessor(**config.data_preprocessor)
    train_loader, test_loader, val_loader = preproc.get_dataloaders()

    encoder = instantiate(config.model.encoder)
    decoder = instantiate(config.model.decoder)
    model_rve = instantiate(config.model.rve, encoder=encoder, decoder=decoder)

    optimizer = instantiate(config.optimizer, params=model_rve.parameters())

    if config.scheduler.mode.enable:
        scheduler = instantiate(config.scheduler.self, optimizer=optimizer)
    else:
        scheduler = None

    total_loss = TotalLoss(config)

    trainer = Trainer(**config.trainer.trainer,
                       model=model_rve,
                       optimizer=optimizer,
                       train_loader=train_loader,
                       val_loader=val_loader,
                       test_loader=test_loader,
                       total_loss=total_loss,
                       scheduler=scheduler,
                       rul_threshold=config.knnmetric.rul_threshold,
                       n_neighbors=config.knnmetric.n_neighbors
                       )
    trainer.train()

    path = output_dir
    model = torch.load(path + "/best_rve_model.pt", weights_only=False)

    rul_threshold = config.knnmetric.rul_threshold
    n_neighbors = config.knnmetric.n_neighbors
    tester = Tester(
        **config.trainer.tester,
        path=path,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        rul_threshold=rul_threshold,
        n_neighbors=n_neighbors
        )
    tester.test()


if __name__ == "__main__":
    # If running in WSL, run in console first:
    # export QT_QPA_PLATFORM=offscreen
    main()