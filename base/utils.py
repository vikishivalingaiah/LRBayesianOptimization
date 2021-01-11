import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class PlotUtils:
    def __init__(self, path, filename):
        """

        Args:
            path(str, required): Path to save the pdf file
            filename(str, required): name for pdf file
        Attributes:
            path: Path to save the pdf file
            filename: name for pdf file
            pp: Matplotlib PdfPages object

        """
        self.path = path
        self.filename = filename
        self.pp = PdfPages(path+filename)

    def plot_gp(self, y_mean, y_cov, x_star, x=None, y=None, sample_curves=[], f=None, show_legend=False, savefig=False,
                plot_samples=False):
        """

        Args:
            y_mean(1D ndarray, required): mean of the gaussian regressor output
            y_cov(1D ndarray, required): covariance of the gaussian regressor output
            x_star(1D ndarray, required): input to gauusian regressor
            x(1D ndarray, optional): input training samples, default None
            yx(1D ndarray, optional): output training samples, default None
            sample_curves(list of 1D ndarray, optional): sample curves from gaussian regressor
            f(function, optional): actual function f
            show_legend(bool, optional): show legend for graph,default False
            savefig(bool, optional): save figure for graph,default False
            plot_samples(bool, optional): plot sample curves

        Returns:
            None
        """
        x_star = x_star.ravel()
        y_mean = y_mean.ravel()
        uncertainty = 1.96 * np.sqrt(np.diag(y_cov))

        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'c($\lambda$)')
        plt.fill_between(x_star, y_mean + uncertainty, y_mean - uncertainty, alpha=0.1, label="uncertainty")
        plt.plot(x_star, y_mean, label='Mean')
        if plot_samples:
            for i, sample in enumerate(sample_curves):
                plt.plot(x_star, sample, lw=1, ls='--', label=f'Sample {i + 1}')
        if x is not None and y is not None:
            plt.plot(x, y, 'kx', markersize=10, label='Training_samples')
        if f is not None:
            plt.plot(x_star, f(x_star), "r*", markersize=5, label="f'actual")
        if show_legend:
            plt.legend()
        if savefig:
            self.pp.savefig()

    def plot_convergence(self, X_sample, Y_sample, n_init=2, show_legend=False, savefig=False):
        """
        Args:
            X_sample(1D ndarray, required): input to bayesian optimization
            Y_sample(1D ndarray, required): output of bayesian optimization
            n_init(long, optional): bayesian optimization  iteration from which to calculate convergence, default 2
            show_legend(bool, optional): show legend for graph,default False
            savefig(bool, optional): save figure for graph,default False

        Returns:
            None
        """
        plt.figure(figsize=(12, 3))

        x = X_sample[n_init:].ravel()
        y = Y_sample[n_init:].ravel()
        r = range(1, len(x) + 1)

        x_neighbor_dist = [np.abs(a - b) for a, b in zip(x, x[1:])]
        y_max_watermark = np.minimum.accumulate(y)

        plt.subplot(1, 2, 1)
        plt.plot(r[1:], x_neighbor_dist, 'bo-', label="distance")
        plt.xlabel('Iteration')
        plt.ylabel('Distance')
        plt.title('Distance between consecutive x\'s')

        plt.subplot(1, 2, 2)
        plt.plot(r, y_max_watermark, 'ro-', label="convergence")
        plt.xlabel('Iteration')
        plt.ylabel('Best Y')
        plt.title('Value of best selected sample')
        if show_legend:
            plt.legend()
        if savefig:
            self.pp.savefig()

    def plot_acquisition(self, X, Y, X_next, show_legend=False, savefig=False):
        """

        Args:
            X(1D ndarray, required): input to acquisition function
            Y(1D ndarray, required): outout of acquisition function
            X_next(ndarray, required): maxima of the acquisition function
            show_legend(bool, optional): show legend for graph,default False
            savefig(bool, optional): save figure for graph,default False

        Returns:
            None

        """
        plt.plot(X, Y, 'r-', lw=1, label='Acquisition function')
        plt.axvline(x=X_next, ls='--', c='k', lw=1, label='Next sampling location')
        if show_legend:
            plt.legend()
        if savefig:
            self.pp.savefig()

    def plot_loss(self, X, Y, show_legend=False, savefig=False):
        """

        Args:
            X(1D ndarray, required): batch numbers
            Y(1D ndarray, required): loss values
            show_legend(bool, optional): show legend for graph,default False
            savefig(bool, optional): save figure for graph,default False

        Returns:

        """
        plt.plot(X, Y, label="batch_loss")
        plt.xlabel("learning rate")
        plt.ylabel("batch loss")
        plt.title("Loss vs Learning Rate")
        if show_legend:
            plt.legend()
        if savefig:
            self.pp.savefig()
