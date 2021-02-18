from ImportModule import *


def validate(net, val_loader, criterion, args):

    net.eval()

    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data

            if args.use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = net(inputs)

            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
    return val_loss, val_acc
