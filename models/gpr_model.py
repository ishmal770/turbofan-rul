import numpy as np
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


class GPRegressor:
    def __init__(self, length_scale=1.0, nu=2.5, noise_level=1e-7,
                 noise_level_bounds=(1e-10, 1), alpha=1e-7,
                 n_restarts_optimizer=5): #length scale is how far off the values have to be to be considered similar, nu is the smoothness of the function, noise level is how much noise is in the data, alpha is for numerical stability, n_restarts_optimizer is how many times to restart the optimizer    
        self.kernel = (
            1.0 * Matern(length_scale=length_scale, nu=nu)
            + WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level_bounds) #m(x) is already pretrainde in skilitlearn, mattern and white kernel are parts of the k(x,x) function
            #distance between both points, close points get high similaraity vise versa, white kernel is for noise in the data to prevent overfitting, alpha is for numerical stability
        )
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.model = None

    def fit(self, z_train, rul_train): #z train is the latent vecotrs, RUL train is the RUL for each mathcing vector
        # z_train: shape (n_samples, 2) — your latent vectors
        # rul_train: shape (n_samples,) — true RUL values
        self.model = GaussianProcessRegressor( #builds a skilit model with the kernel, alpha and n_restarts_optimizer
            kernel=self.kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=True
        )
        self.model.fit(z_train, rul_train) #training the model with the latent vectors and the RUL values, this is where the GP learns the mapping from latent space to RUL

    def predict(self, z):
        # returns (mean_prediction, std_uncertainty)
        mean, std = self.model.predict(z, return_std=True)
        return mean, std

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)