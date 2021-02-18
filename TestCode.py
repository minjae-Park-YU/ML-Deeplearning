from ImportModule import *


def test(net, test_loader, args):

    net.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data

            if args.use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = net(inputs)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)  # batch_size

        test_acc = 100 * correct / total

    return test_acc
