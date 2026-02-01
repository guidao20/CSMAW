from torchvision import datasets
from utils.config import DATASET_CONFIG

def get_dataset(dataset_name, data_dir, download=True):

    config = DATASET_CONFIG[dataset_name]

    if dataset_name == 'mnist':
        train_dataset = datasets.MNIST(
            root=data_dir, train=True, download=download, transform=config["train_transform"]
        )
        test_dataset = datasets.MNIST(
            root=data_dir, train=False, download=download, transform=config["test_transform"]
        )

    elif dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True, download=download, transform=config["train_transform"]
        )
        test_dataset = datasets.CIFAR10(
            root=data_dir, train=False, download=download, transform=config["test_transform"]
        )

    elif dataset_name == 'gtsrb': 
        train_dataset = datasets.GTSRB(
            root=data_dir, split='train', download=download, transform=config["train_transform"]
        )
        test_dataset = datasets.GTSRB(
            root=data_dir, split='test', download=download, transform=config["test_transform"]
        )
        
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose from  ['mnist', 'cifar10', 'gtsrb']")

    return train_dataset, test_dataset, config["num_classes"]