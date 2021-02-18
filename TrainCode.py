from ImportModule import *


def train(net, train_loader, optimizer, criterion, args):

    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for data in train_loader:
        optimizer.zero_grad()
        inputs, labels = data

        if args.use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)  # 1. row 값 중에 최댓값 (_ 처리해서 무시)
                                       # 2. 최댓값의 column index  (predicted로 받아옴)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = train_loss / len(train_loader)
    train_acc = 100 * correct / total

    return net, train_loss, train_acc
