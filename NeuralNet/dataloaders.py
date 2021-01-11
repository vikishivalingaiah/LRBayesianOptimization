import torch
import torchvision


def load_kmnist(path="datasets"):
    """
    Function to load data KMNIST, if dataset is not present will download automatically
    Args:
        path(str, required): path to the kmnist dataset directory

    Returns:
        train_data_loader(torch.utils.data.DataLoader): data loader of training set
        test_data_loader(torch.utils.data.DataLoader): data loader of test set


    """
    kmnist_test = torchvision.datasets.KMNIST(root=path, download=True, train=False,
                                              transform=torchvision.transforms.Compose(
                                                  [torchvision.transforms.ToTensor()]))
    kmnist_train = torchvision.datasets.KMNIST(root=path, download=True, train=True,
                                               transform=torchvision.transforms.Compose(
                                                   [torchvision.transforms.ToTensor()]))
    test_data_loader = torch.utils.data.DataLoader(kmnist_test,
                                                   batch_size=32,
                                                   shuffle=True)

    train_data_loader = torch.utils.data.DataLoader(kmnist_train,
                                                    batch_size=32,
                                                    shuffle=True)
    return train_data_loader, test_data_loader
