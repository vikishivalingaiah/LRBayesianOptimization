import numpy as np
from sklearn.gaussian_process.kernels import RBF as rbf_kernel
from scipy.optimize import minimize
import sys
import logging


class GaussianRegressor:
    def __init__(self, length_scale=1.0, noise=1e-2):
        """

        Args:
           length_scale (float, optional): initial length_scale for RBF kernel, default 1.0
           noise (float, optional): noise to be added to predictions, default 1e-2
        Attributes:
           length_scale : length_scale for RBF kernel
           noise : noise to be added to predictions

        """
        self.noise = noise
        self.train = False
        self.length_scale = length_scale
        self.kernel = rbf_kernel(self.length_scale)

    def fit(self, x, y):
        """

        Args:
            x (ndarray, required): Input features
            Y (ndarray,required): Target values
        Returns:
            None
        """
        self.x = x
        self.y = y
        self.train = True

    def predict(self, x_star, num_sample_curves):
        """

        Args:
            x_star (ndarray, required): input values to predict targets
            num_sample_curves (long, required): Number of curves to sample

        Returns:
            y_mean (ndarray): mean values of gaussian curves
            y_cov (ndarray): covariance matrix
            y_sample_curves (ndarray): targets for num_sample_curves, returns only is num_sample_curves is mentioned

        """

        if self.train is False:
            y_mean = np.zeros(x_star.shape[0])

            y_cov = self.kernel(x_star)
            y_samples = np.random.multivariate_normal(y_mean.ravel(), y_cov, 3)
        else:
            try:

                k = self.kernel(self.x, self.x) + self.noise * self.noise * np.eye(self.x.shape[0])

                k_star = self.kernel(self.x, x_star)
                k_double_star = self.kernel(x_star, x_star) + self.noise * np.eye(x_star.shape[0])
                k_inverse = np.linalg.inv(k)
                k_star_transpose = np.transpose(k_star)
                y_mean = np.dot(np.dot(k_star_transpose, k_inverse), self.y)
                y_cov = k_double_star - np.dot(np.dot(k_star_transpose, k_inverse), k_star)
                y_samples = np.random.multivariate_normal(y_mean.ravel(), y_cov, num_sample_curves)
            except np.linalg.LinAlgError as exc:
                logging.info("The kernel, is not returning a positive definite matrix.,exit")
                logging.info(exc.args)
                sys.exit(1)

        return y_mean, y_cov, y_samples

    def optimize_l(self):
        """
        Optimize the length_scale value for RBF Kernel
        Args:
            None
        Returns:
            None
        """
        # bounds = np.array([[0,10]])
        res = minimize(self._log_marginal_likelihood(self.x, self.y, self.noise), self.length_scale, method='L-BFGS-B')
        self.length_scale = res.x[0]
        self.kernel = rbf_kernel(self.length_scale)
        return self.length_scale

    def _log_marginal_likelihood(self, X, Y, noise=1e-2):
        """

        Args:
            X (ndarray, required): input features
            Y (ndarray,required): target features
            noise (float,optional): noise value, 1e-2

        Returns:
            None

        """

        def _log_marginal_likelihood_func(length_scale, X=X, Y=Y, noise=noise):
            Y = np.ravel(Y)
            kernel = rbf_kernel(length_scale)

            k = kernel(X)
            noise_mat = noise * noise * np.eye(X.shape[0])
            k_y = k + noise_mat
            k_y_inverse = np.linalg.inv(k_y)
            ml = -0.5 * np.dot(np.dot(np.transpose(Y), k_y_inverse), Y) - 0.5 * np.log(np.linalg.det(k_y)) - X.shape[
                0] * 0.5 * np.log(2 * np.pi)

            return -1 * ml

        return _log_marginal_likelihood_func
