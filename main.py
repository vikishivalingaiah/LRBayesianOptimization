from base.regressors import GaussianRegressor
from base.optimizers import BayesianOptimizer
import numpy as np
from base.utils import PlotUtils
import matplotlib.pyplot as plt
from NeuralNet.models import KMnistResNet
from NeuralNet.trainers import Trainer
from NeuralNet import dataloaders
import torch
import logging
from datetime import datetime


def test_GPR():
    path = 'results/'
    filename = "gpr_test.pdf"

    # Initialize plotutil
    plotutil = PlotUtils(path, filename)
    X = np.arange(-5, 5, 0.2).reshape(-1, 1)
    # X_new = np.array([2]).reshape(-1, 1)
    X_train = np.array([-4, -1.8, 1]).reshape(-1, 1)
    Y_train = np.sin(X_train)

    g = GaussianRegressor(1.0, 0.01)

    g.fit(X_train, Y_train)
    g.optimize_l()
    y_mean, y_cov, y_samples = g.predict(X, 3)
    plotutil.plot_gp(y_mean, y_cov, X, X_train, Y_train, y_samples, np.sin, True, True, True)
    plt.legend(loc='best')
    plt.show()
    plotutil.pp.close()


def test_bayesian_optimization():
    path = 'results/'
    filename = "bo_test.pdf"
    logging.info("Storing results in " + filename + path)
    # Initialize plotutil
    plotutil = PlotUtils(path, filename)
    noise = 1e-3
    function_evaluations = 10

    def objective_function(X, noise=noise):
        return -(-np.sin(3 * X) - X ** 2 + 0.7 * X + noise * np.random.randn(*X.shape))

    bounds = np.array([[-5, 5]])
    X_train = np.array([0.3]).reshape(-1, 1)
    Y_train = objective_function(X_train)

    gpr = GaussianRegressor(1.0, 0.01)
    plt.figure(figsize=(20, 20))
    b = BayesianOptimizer(bounds, objective_function, gpr, X_train, Y_train, function_evaluations, 0.5, plotutil)
    lambda_values, c_lambda_values = b.optimize()

    # Get Best learning rate and loss
    logging.info("=" * 48)
    logging.info("Best lambda")
    best_lambda = lambda_values[np.argmin(c_lambda_values)]
    logging.info(best_lambda)
    logging.info("best c_lambda")
    best_c_lambda = np.min(c_lambda_values)
    logging.info(best_c_lambda)
    logging.info("=" * 48)

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=[[i, j] for i, j in zip(list(lambda_values.ravel()), list(c_lambda_values.ravel()))],
             rowLabels=[i + 1 for i in range(function_evaluations+1)],
             colLabels=[r'$\lambda$', r'c($\lambda$)'],
             cellColours=[["salmon", "salmon"] if j == best_c_lambda and i == best_lambda else ["white", "white"]
                          for i, j in zip(list(lambda_values.ravel()), list(c_lambda_values.ravel()))], loc="center")
    fig.tight_layout()
    plotutil.pp.savefig()
    plt.show()
    plotutil.pp.close()


def optimize_learning_rate():
    def lr_objective(lr):
        kmnist_resnt = KMnistResNet()
        trainer = Trainer(kmnist_resnt, 1, torch.nn.CrossEntropyLoss(), torch.optim.SGD)
        final_accuracy, losses, batch_losses = trainer.train(train_data_loader, test_data_loader, lr.ravel()[0])
        return losses[-1]

    path = 'results/'
    i = 0
    max_accepted_loss_training = 100  # Loss to stop the learninig range training
    early_stopping = True  # early stopping for lrr test
    lr_multiplier = 1.08  # multiplier for intial lr to scal lr per batch
    lr_initial = 1e-4
    loss_bound = 2  # Time inital loss for lr_initial

    length_scale = 1  # initial l for rbf kernel in gaussian regressor
    function_evaluations = 10  # number of evaluations for bo
    noise = 0.01  # noise for gaussian regressor
    logging.info("*"*60)
    logging.info("Bayeisan optimization")
    logging.info("*" * 60)
    for xi in [0.001, 0.01, 0.05, 0.1, 1, 2]:
        logging.info("Running bo for xi=" + str(xi))
        # Set filename for saving results
        filename = "bo_result_%s.pdf" % i
        i = i + 1
        logging.info("Results stored in :" + path + filename)
        # Initialize plotutil
        plotutil = PlotUtils(path, filename)

        # Find learning rate range
        kmnist_resnet = KMnistResNet()
        trainer_batch = Trainer(kmnist_resnet, 1, torch.nn.CrossEntropyLoss(), torch.optim.SGD)
        train_data_loader, test_data_loader = dataloaders.load_kmnist()
        batch_losses, lr_list = trainer_batch.get_learning_rate_range(lr_initial, lr_multiplier, train_data_loader,
                                                                      early_stopping, max_accepted_loss_training)
        last_index = np.argmax(np.greater(batch_losses, batch_losses[0] * loss_bound))
        plotutil.plot_loss(lr_list[0:last_index], batch_losses[0:last_index], show_legend=True, savefig=True)
        #lr_bounds = (lr_list[0], lr_list[last_index])
        lr_bounds = (0,1)
        bounds = np.array(lr_bounds).reshape(1, -1)

        # Plot learning rate range and xi values in table
        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ax.table(cellText=[[str(lr_bounds)], [xi]],
                 rowLabels=["learning rate bounds", "xi"],
                 colLabels=["values"],
                 loc="center")
        fig.tight_layout()
        plotutil.pp.savefig()

        gpr = GaussianRegressor(length_scale, noise)
        b = BayesianOptimizer(bounds, lr_objective, gpr, num_function_evaluations=function_evaluations, xi=xi,
                              plotutil=plotutil)
        lambda_values, c_lambda_values = b.optimize()

        # Get Best learning rate and loss
        logging.info("=" * 48)
        logging.info("Best learning rate")
        best_lambda = lambda_values[np.argmin(c_lambda_values)]
        logging.info(best_lambda)
        logging.info("Lowest Loss")
        best_c_lambda = np.min(c_lambda_values)
        logging.info(best_c_lambda)
        logging.info("=" * 48)

        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ax.table(cellText=[[i, j] for i, j in zip(list(lambda_values.ravel()), list(c_lambda_values.ravel()))],
                 rowLabels=[i + 1 for i in range(function_evaluations + 1)],
                 colLabels=[r'$\lambda$', r'c($\lambda$)'],
                 cellColours=[["salmon", "salmon"] if j == best_c_lambda and i == best_lambda else ["white", "white"]
                              for i, j in
                              zip(list(lambda_values.ravel()), list(c_lambda_values.ravel()))],
                 loc="center")
        fig.tight_layout()
        plotutil.pp.savefig()

        plt.show()

        # close plotutil file
        plotutil.pp.close()
        logging.info("*"*20)
        logging.info("Bayesian optimization complete")
        logging.info("*" * 20)


if __name__ == '__main__':
    logfilename = datetime.now().strftime('log_%H_%M_%d_%m_%Y.log')
    logging.basicConfig(filename="./logs/" + logfilename,  level=logging.INFO)
    #test_GPR()
    #test_bayesian_optimization()
    optimize_learning_rate()
