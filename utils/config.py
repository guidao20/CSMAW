from torchvision import transforms

mnist_fc_size = 2048
cifar10_fc_size = 2048
gtsrb_fc_size = 2048

DATASET_CONFIG = {
    'mnist': {
        'in_channels': 3,
        'num_classes': 10,
        'fc_input_size': mnist_fc_size,
        'train_transform': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]),
        'test_transform': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
    },
    'cifar10': {
        'in_channels': 3,
        'num_classes': 10,
        'fc_input_size': cifar10_fc_size,
        'train_transform': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        'test_transform': transforms.Compose([
            transforms.ToTensor(),
        ])
    },
    'gtsrb': {
        'in_channels': 3,
        'num_classes': 43,
        'fc_input_size': gtsrb_fc_size,
        'train_transform': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
        ]),
        'test_transform': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
    }
}