

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import logging


class BayesianOptimizer:
    def __init__(self, bounds, c_lambda, gpr, lambda_0=None, c_lambda_0=None, num_function_evaluations=2, xi=0.01,
                 plotutil=None):
        """
        Bayesian oprimizer implementing bayesian optimization with gaussian process regressor for one dimensional input
        functions
        Args:
            bounds (ndarray, required): bound values for optimization of lambda
            c_lambda (function, required): Objective function
            gpr (GaussianRegressor, required): gaussian process regressor
             lambda_0 (ndarray, optional): inputs evaluated by c_lambda,
                                default None
            c_lambda_0 (ndarray, optional): observed function evaluations,default None
            num_function_evaluations (long, optional): Number of times to evaluated the function c_lambda, default 2

            xi (float, optional): Exploitation vs Exploration parameters, default 0.01
            plotutil (PlotUtil, optional): plotutil object to plot the results, default None

        Attributes:
            bounds (ndarray, required): bound values for optimization of lambda
            c_lambda (function, required): Objective function
            gpr (GaussianRegressor, required): gaussian process regressor
            lambda_0 (ndarray, optional): inputs evaluated by c_lambda
            c_lambda_0 (ndarray, optional): observed function evaluations,default None
            num_function_evaluations (long, optional): Number of times to evaluated the function c_lambda
            xi (float, optional): Exploitation vs Exploration parameters
            plotutil (PlotUtil, optional): plotutil object to plot the results
        """
        self.lambda_0 = lambda_0
        self.c_lambda_0 = c_lambda_0
        self.gpr = gpr
        self.c_lambda = c_lambda
        self.num_function_evaluations = num_function_evaluations
        self.bounds = bounds
        self.xi = xi
        self.x_star = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)
        self.num_sample_curves = 3
        self.plotutil = plotutil

    def expected_improvement(self, x_star):
        """
        Activation function for bayseian optimization expected improvement
        Args:
            x_star (ndarray, optional): input samples

        Returns:
            u_lambda (ndarray): Activation output

        """

        mu_lambda, cov_lambda, samples = self.gpr.predict(self.lambda_0, self.num_sample_curves)
        ci_best = np.min(mu_lambda)

        mu_lambda_hat, cov_lambda_hat, samples_hat = self.gpr.predict(x_star, self.num_sample_curves)

        std = np.sqrt(np.diag(cov_lambda_hat)).reshape(-1, 1)

        Z = (ci_best - mu_lambda_hat - self.xi) / std
        Z_cdf = norm.cdf(Z)

        u_lambda = std * Z * Z_cdf + std * norm.pdf(Z)
        u_lambda[std == 0.0] = 0.0

        return u_lambda

    def find_lambda_byEI(self):
        """
        Find the best possible lambda give EI activation function
        Args:
            None
        Returns:
            None

        """

        dim = self.lambda_0.shape[1]
        min_val = 1
        min_x = None

        def min_obj(X):
            # Minimization objective is the negative acquisition function
            return -1 * self.expected_improvement(X.reshape(-1, dim))[0]

        for x0 in np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(25, dim)):
            res = minimize(min_obj, x0=x0, bounds=self.bounds, method='L-BFGS-B')

            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x
                # print(min_x)
        return min_x.reshape(-1, 1)

    def optimize(self):
        """
        Optimize the objective function
        Args:
            None
        Returns:
            lambda (ndarray): Array of found lambdas during optimization
            c_lambda (ndarray): Array of best cost values for the objective function

        """
        logging.info("generating initial values for bayesian optimization")
        if self.lambda_0 is None or self.c_lambda_0 is None:
            self.generate_initial_samples()
        for i in range(self.num_function_evaluations):
            logging.info("=" * 20)
            logging.info("Evaluation iteration: " + str(i + 1))
            self.gpr.fit(self.lambda_0, self.c_lambda_0)
            self.gpr.optimize_l()
            mu_lambda_1, sigma_lambda_1, samples_1 = self.gpr.predict(self.x_star, self.num_sample_curves)
            next_lambda = self.find_lambda_byEI()
            Y_new = self.c_lambda(next_lambda)  # train_resnet(lr=next_lambda[0][0])
            if i % 4 == 0:
                plt.figure()
                plt.rc('font', size=8)
                plt.rc('legend', fontsize=8)
            logging.info("Lambda :" + str(next_lambda) + "C_lambda :" + str(Y_new))
            plt.subplot(4, 2, 2 * (i % 4) + 1)
            self.plotutil.plot_gp(mu_lambda_1, sigma_lambda_1, self.x_star, self.lambda_0, self.c_lambda_0, samples_1,
                                  show_legend=i == 0, savefig=False)
            plt.title(f'Iteration {i + 1}')

            plt.subplot(4, 2, 2 * (i % 4) + 2)
            self.plotutil.plot_acquisition(self.x_star, self.expected_improvement(self.x_star), next_lambda,
                                           show_legend=i == 0, savefig=False)

            if self.plotutil is not None:
                if (i + 1) % 4 == 0 or (i + 1) == self.num_function_evaluations:
                    self.plotutil.pp.savefig()

            self.lambda_0 = np.append(self.lambda_0, next_lambda)[np.newaxis][:].reshape(-1, 1)
            self.c_lambda_0 = np.append(self.c_lambda_0, Y_new)[np.newaxis][:].reshape(-1, 1)

        self.plotutil.plot_convergence(self.lambda_0, self.c_lambda_0, show_legend=True, savefig=False)
        if self.plotutil is not None:
            self.plotutil.pp.savefig()
        logging.info("Optimization complete")
        return self.lambda_0, self.c_lambda_0

    def generate_initial_samples(self):
        """
        Genereate the initial observation for starting bayesian optimization
        Args:
            None
        Returns:
            None
        """

        lambda_0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        lambda_0 = np.array([lambda_0]).reshape(-1, 1)
        final_loss = self.c_lambda(lambda_0)
        self.lambda_0 = lambda_0
        self.c_lambda_0 = np.array([final_loss]).reshape(-1, 1)
