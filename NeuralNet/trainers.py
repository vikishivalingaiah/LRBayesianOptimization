import torch
import time
from tqdm.autonotebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import logging


class Trainer:
    def __init__(self, model, epochs, loss_function, optimizer):
        """
        Trainer class provides functions for training
        Args:
            model (torch.nn.modules.module.Module, required): Pytorch model to train
            epochs (long, required): Number of epochs to train
            loss_function (torch.nn.modules.module.Module, required): loss function from torch.nn
            optimizer (torch.optim.SGD, required): optimizer from torch.optim
        Attributes:
            model: Pytorch model to train
            epochs: Number of epochs to train
            loss_function: loss function from torch.nn
            optimizer: optimizer from torch.optim

        """

        self.model = model
        self.epochs = epochs
        self.loss_function = loss_function
        self.optimizer = optimizer

    def train(self, train_data_loader, test_data_loader, lr=0.1):
        """
        Function to train the model
        Args:
            train_data_loader(torch.utils.data.DataLoader, required): data loader of training set
            test_data_loader(torch.utils.data.DataLoader, required): data loader of test set
            lr(float, optiona): learning rate for the optimizer

        Returns:
            final_accuracy(float): Final validation accuracy after training
            losses(list): list of loss per epoch
            batch_losses(list): list of loss per batch in last epoch

        """

        start_ts = time.time()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        optimizer = self.optimizer(self.model.parameters(), lr=lr)

        losses = []
        batches = len(train_data_loader)
        val_batches = len(test_data_loader)
        batch_losses = []

        for epoch in range(self.epochs):
            total_loss = 0
            lr_local = lr
            progress = tqdm(enumerate(train_data_loader), desc="Loss: ", total=batches)

            # ----------------- TRAINING  --------------------
            # set model to training
            self.model.train()
            previous_loss = 0.0
            val_loss_list_epoch = []
            for i, data in progress:
                X, y = data[0].to(device), data[1].to(device)

                self.model.zero_grad()
                outputs = self.model(X)
                loss = self.loss_function(outputs, y)
                loss.backward()
                optimizer.step()

                current_loss = loss.item()
                batch_losses.append(current_loss)
                total_loss += current_loss

                progress.set_description("Loss: {:.4f}".format(total_loss / (i + 1)))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ----------------- VALIDATION  -----------------
            val_losses = 0
            precision, recall, f1, accuracy = [], [], [], []

            # set model to evaluating (testing)

            self.model.eval()
            with torch.no_grad():
                for i, data in enumerate(test_data_loader):
                    X, y = data[0].to(device), data[1].to(device)

                    outputs = self.model(X)  # this get's the prediction from the network

                    val_losses += self.loss_function(outputs, y)

                    predicted_classes = torch.max(outputs, 1)[1]  # get class from network's prediction

                    # calculate P/R/F1/A metrics for batch
                    for acc, metric in zip((precision, recall, f1, accuracy),
                                           (precision_score, recall_score, f1_score, accuracy_score)):
                        acc.append(
                            self.calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                        )

            logging.info(f"Epoch {epoch + 1}/{self.epochs}, training loss: {total_loss / batches} \
                    , validation loss: {val_losses / val_batches}")
            self.print_scores(precision, recall, f1, accuracy, val_batches)
            losses.append(total_loss / batches)

        final_accuracy = sum(accuracy) / val_batches
        logging.info(f"Training time: {time.time() - start_ts}s")
        return final_accuracy, losses, batch_losses

    def get_learning_rate_range(self, lr, lr_multiplier, train_data_loader, early_stopping=True, max_loss=100):
        """
        Function to find the learning rate range by lrr test
        Args:
            lr(float, required): inital value of lr
            lr_multiplier(float, required): Multiplier that changes the learning rate per batch
             train_data_loader(torch.utils.data.DataLoader, required): data loader of training set
            early_stopping(bool, optional): if set stops the training early when loss reaches max_loss, default is True
            max_loss(float, optional): max_loss at which the training should stop, default 100

        Returns:
            batch_losses(list): list of loss per batch
            lr_list(list): list of lr values used during training

        """
        batches = len(train_data_loader)
        batch_losses = []

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        total_loss = 0
        progress = tqdm(enumerate(train_data_loader), desc="Loss: ", total=batches)

        # ----------------- TRAINING  --------------------
        # set model to training
        self.model.train()
        lr_list = []

        for i, data in progress:
            lr_list.append(lr)
            optimizer = self.optimizer(self.model.parameters(), lr=lr)
            X, y = data[0].to(device), data[1].to(device)

            self.model.zero_grad()
            outputs = self.model(X)
            loss = self.loss_function(outputs, y)
            loss.backward()
            optimizer.step()
            lr = lr * lr_multiplier

            current_loss = loss.item()
            batch_losses.append(current_loss)
            total_loss += current_loss

            progress.set_description("Loss: {:.4f}".format(total_loss / (i + 1)))
            if early_stopping and (total_loss / (i + 1)) > max_loss:
                break
        return batch_losses, lr_list

    def calculate_metric(self, metric_fn, true_y, pred_y):
        """
        Just a utility calculating metrics
        Args:
            metric_fn (function,required): Function to calculate the metric
            true_y (ndarray, required): Real target values
            pred_y (ndarray, required): predicted target values

        Returns:
            float: calculated metric
        """
        # multi class problems need to have averaging method
        if "average" in inspect.getfullargspec(metric_fn).args:
            return metric_fn(true_y, pred_y, average="macro")
        else:
            return metric_fn(true_y, pred_y)

    def print_scores(self, p, r, f1, a, batch_size):
        """
        Just a utility printing function
        Args:
            p(float, required): precision
            r(float, required): recall
            f1(float, required): f1 value
            a(float, required): accuracy
            batch_size(long, required): batchsize

        Returns:
            None
        """

        for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
            logging.info(f"\t{name.rjust(14, ' ')}: {sum(scores) / batch_size:.4f}")
