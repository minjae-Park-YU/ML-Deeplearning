import torch
from torchvision import datasets, transforms                                  # Data Processing Package
import torch.nn as nn

# ====== Data Processing ====== #
batch_size = 128
train_dataset = datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),                     # Data type을 Tensor 형태로 변경 -> rescaling
                                   transforms.Normalize((0.1307,), (0.3081,)) # Image Pixel 값을 변경(red, green, blue) -> centering(offset) and rescaling
                               ]))                                            # ToTensor()로 변환시킨 경우에는 (0~255 => 0~1)로 변한 상태에서 Normalize함
test_dataset = datasets.MNIST('./data', train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1037,), (0.3081,))
                              ]))
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000]) # 60000개 중 50000개는 train set, 10000개는 test set으로 랜덤으로 나눔
print(len(train_dataset), len(val_dataset), len(test_dataset))                            #사이즈 확인

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)   # Train set만 Shuffle 하는걸로 봐서 학습이 고르게 되기 위해 shuffle 하는듯
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)      # validaion, test는 평가를 하기 위한 것이기 때문에 굳이 shuffle 안해줘도 되는것같음
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# ====== Inspecting Dataset ====== # (없어도 되는 부분이긴 함)
examples = enumerate(train_loader)                          # 리스트 안에 데이터를 순번 : index 로 묶어서 할당해줌
batch_idx, (example_data, example_targets) = next(examples) # 아까 생성한 데이터 리스트에서 batch_idx, (data, label)? 이렇게 가져옴

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 9))
for i in range(6):
    plt.subplot(2, 3, i+1)                                             # 2행 3열로 그래프 배치할건데 i+1로 하나씩 지정해 주는 부분임
    plt.tight_layout()                                                 # 여백 지정. 아무것도 지정 안해줬으니 default 값
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')  # imshow => 이미지를 일단 칸에 넣어놓는다고 생각, cmap => 색깔, interpolation => 보간법 사용해서 이미지 자연스레 처리
    plt.title("Ground Truth:{}".format(example_targets[i]))
plt.show()                                                             # 이거 해야 볼 수 있음

# ====== Model Architecture ====== #
class LinearModel(nn.Module):                                                 # LinearModel을 상속받아서 input = 28 * 28 = 784, output = 10(0~9사이의 숫자)로 layer 하나 생성
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_features=784, out_features=10, bias=True)

    def forward(self, x):
        x = self.linear(x)                                                   # Linear first layer
        return x

# ====== Cost Function Define (Loss Function Define) ====== #
cls_loss = nn.CrossEntropyLoss()                                              # CrossEntropy를 Cost function으로 지정

# ====== Train & Evaluation ======= #
import torch.optim as optim
from sklearn.metrics import accuracy_score

# ------ Construct Model ------ #
model = LinearModel()
print('Number of {} parameters'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))   # update 해야하는 parameter의 개수를 출력

# ------ Construct Optimizer ------ #
lr = 0.005                                        # Learning Rate
optimizer = optim.SGD(model.parameters(), lr=lr)  # SGD(Stochastic gradient descent) : 경사하강법 사용

list_epoch = []
list_train_loss = []
list_val_loss = []
list_acc = []
list_acc_epoch = []

epoch = 30
for i in range(epoch):
    # ------ train ------ #
    train_loss = 0
    model.train()
    optimizer.zero_grad()

    for input_X, true_y in train_loader:
        input_X = input_X.squeeze()                    # squeeze() : 차원의 원소가 1인 차원을 없애줌.. 이해잘 안됨 왜 여기서 쓰지
        input_X = input_X.view(-1, 784)                # data의 차원을 (data 개수, 784)로 변경
        pred_y = model(input_X)

        loss = cls_loss(pred_y.squeeze(), true_y)      # loss 계산
        loss.backward()                                # 역으로 가면서 gradient 계산
        optimizer.step()                               # 계산된 gradient 바탕으로 weight update
        train_loss += loss.detach().numpy()            # 미분을 멈추고(detach()), numpy배열로 변환(numpy())
    train_loss = train_loss / len(train_loader)        # loss의 평균값을 구함
    list_train_loss.append(train_loss)                 # loss의 평균을 리스트에 추가
    list_epoch.append(i)                               # epoch 추가

    # ------ validation ------ #
    val_loss = 0
    model.eval()
    optimizer.zero_grad()

    for input_X, true_y in test_loader:
        input_X = input_X.squeeze()
        input_X = input_X.view(-1, 784)
        pred_y = model(input_X)

        loss = cls_loss(pred_y.squeeze(), true_y)
        val_loss += loss.detach().numpy()
    val_loss = val_loss / len(val_loader)
    list_val_loss.append(val_loss)

    # ------ Evaluation ------ #
    correct = 0
    model.eval()
    optimizer.zero_grad()

    for input_X, true_y in test_loader:
        input_X = input_X.squeeze()
        input_X = input_X.view(-1, 784)
        # 밑에 두 부분 이해 잘 안됨
        pred_y = model(input_X).max(1, keepdim=True)[1].squeeze()
        correct += pred_y.eq(true_y).sum()

    acc = correct.numpy() / len(test_loader.dataset)
    list_acc.append(acc)
    list_acc_epoch.append(i)

    print('Epoch: {}, Train Loss: {}, Val Loss: {}, Test Acc: {}%'.format(i, train_loss, val_loss, acc*100))

fig = plt.figure(figsize=(15, 5))
# ====== Lost Fluctuation ====== #
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(list_epoch, list_train_loss, label='train_loss')
ax1.plot(list_epoch, list_val_loss, '--', label='val_loss')
ax1.set_xlabel('epoch')
ax1.set_ylabel('Acc')
ax1.grid()
ax1.legend()
ax1.set_title('epoch vs loss')

# ====== Metric Fluctuation ====== #
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(list_acc_epoch, list_acc, marker='x', label='Accuracy metric')
ax2.set_xlabel('epoch')
ax2.set_ylabel('Acc')
ax2.grid()
ax2.legend()
ax2.set_title('epoch vs Accuracy')

plt.show()
