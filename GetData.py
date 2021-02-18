from ImportModule import *


def tensor_transform(train_root, test_root, args):
    transform_train = transforms.Compose([#transforms.Resize((512, 512)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_test = transforms.Compose([#transforms.Resize((512, 512)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [40000, 10000])
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # train_dataset = torchvision.datasets.ImageFolder(root=train_root, transform=transform_train)
    # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [round(len(train_dataset)*0.8), round(len(train_dataset)*0.2)])
    # test_dataset = torchvision.datasets.ImageFolder(root=test_root, transform=transform_test)

    partition = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}

    train_loader = torch.utils.data.DataLoader(partition['train'], batch_size=args.train_batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(partition['val'], batch_size=args.test_batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(partition['test'], batch_size=100, shuffle=False)

    # print(train_dataset[15000])

    data_list = [train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader]

    return data_list
