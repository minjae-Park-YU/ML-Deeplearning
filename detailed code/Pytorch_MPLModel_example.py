import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d #3차원 그래프를 그릴 수 있는 모듈

# ====== Generating Dataset ====== #
num_data = 2400
x1 = np.random.rand(num_data) * 10 #0~1 사이의 랜덤한 값 2400개 추출
x2 = np.random.rand(num_data) * 10 #0~1 사이의 랜덤한 값 2400개 추출
print(np.shape(x1)) #차원을 수시로 확인해 주어야 나중에 문제 생겨도 바로 찾을 수 있음
print(np.shape(x2)) #차원을 수시로 확인해 주어야 나중에 문제 생겨도 바로 찾을 수 있음
e = np.random.normal(0, 0.5, num_data) #정규분포 난수 생성 (평균, 표준편차, 사이즈)
X = np.array([x1, x2]).T #pytorch에서는 Data의 사이즈가 앞에 나와야 해서 전치행렬 함
y = 2*np.sin(x1) + np.log(0.5*x2**2) + e #임의의 2변수 함수 생성

# ====== Split Dataset in Train, Validation, Test ====== #
#데이터를 세트별로 나눔
train_X, train_y = X[:1600, :], y[:1600]
val_X, val_y = X[1600:2000, :], y[1600:2000]
test_X, test_y = X[2000:, :], y[2000:]

# ====== Visualize Each Dataset ====== #
fig = plt.figure(figsize=(12,5))
axl = fig.add_subplot(1, 3, 1, projection='3d') #projection = 3D로 그래프 그리고 싶을때
axl.scatter(train_X[:, 0], train_X[:, 1], train_y, c=train_y, cmap='jet')

axl.set_xlabel('x1')
axl.set_ylabel('x2')
axl.set_zlabel('y')
axl.set_title('Train Set Distribution')
axl.view_init(40, -60)
axl.invert_xaxis()

ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.scatter(val_X[:, 0], val_X[:, 1], val_y, c=val_y, cmap='jet')

ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('y')
ax2.set_title('Validation Set Distribution')
ax2.set_zlim(-10, 6)
ax2.view_init(40, -60)
ax2.invert_xaxis()

ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.scatter(test_X[:, 0], test_X[:, 1], test_y, c=test_y, cmap='jet')

ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.set_zlabel('y')
ax3.set_title('Test Set Distribution')
ax3.set_zlim(-10, 6)
ax3.view_init(40, -60)
ax3.invert_xaxis()

plt.show()

# ====== Basic nn Model & MLP Model ====== #
import torch
import torch.nn as nn

class LinearModel(nn.Module): #nn.Module에 있는걸 상속받아서 필요한 부분만 오버라이딩
    def __init__(self):
        super(LinearModel, self).__init__() #상속 받을 때 LinearModel을 알아서 찾아가서 받음
        self.linear = nn.Linear(in_features=2, out_features=1, bias=True) #Wx + b의 기능, x1, x2가 input이므로 in_features = 2, 원하는 output값이 1개이므로 out_features = 1

    def forward(self, x): #이름을 무조건 forward로 정해주어야함
        return self.linear(x) #X를 받아서 예측 값을 계산하도록 식 구현

class MLPModel(nn.Module):
    def __init__(self): #Linear 2개와 ReLU 함수를 정의함(생성자)
        super(MLPModel, self).__init__()
        self.linear1 = nn.Linear(in_features=2, out_features=200)
        self.linear2 = nn.Linear(in_features=200, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        #아래 두줄이 반복되는 모양이 신경망이 깊어지는 것임
        x = self.relu(x)
        x = self.linear2(x)
        return x

lm = LinearModel() #인스턴스 생성
print(lm.linear.weight) #W가 어떻게 할당되어 있는지 볼 수 있음
print(lm.linear.bias) #b가 어떻게 할당되어 있는지 볼 수 있음

# ====== cost function ====== #
reg_loss = nn.MSELoss() #MSE Loss
# cost function test #
test_pred_y = torch.Tensor([0, 0, 0, 0]) #원래는 list이지만, pytorch 연산에서는 torch.Tensor로 형변환해서 사용해줘야함
test_true_y = torch.Tensor([0, 1, 0, 1])

print(reg_loss(test_pred_y, test_true_y)) #MSE loss의 공식대로 계산하면 0.5가 나옴 => 제대로 작동하고 있구나
print(reg_loss(test_true_y, test_true_y)) #정확히 일치 => cost function의 loss 값이 0임

# ====== Train & Evaluation ====== #
import torch.optim as optim #optimizer 모듈
from sklearn.metrics import mean_absolute_error

# ====== Construct Model ====== #
# model = LinearModel()
model = MLPModel()
print('{} parameters'.format(sum(p.numel() for p in model.parameters() if p.requires_grad))) #학습을 당할 파라미터 수를 카운팅 해주는 코드

# ====== Construct Optimizer ====== #
lr = 0.005 #learning rate
optimizer = optim.SGD(model.parameters(), lr=lr) #optimizer 생성

# 매 학습 단계에서의 epoch 값과 그때의 값을 저장할 리스트를 만듬 => 제대로 최적화 되는지 그래프로 확인
list_epoch = []
list_train_loss = []
list_val_loss = []
list_mae = []
list_mae_epoch = []

epoch = 4000 #학습 횟수 지정
for i in range(epoch):

    # ====== Train ====== #
    #밑의 코드 두개는 시작전에 해주는게 좋음
    model.train() # model을 train mode로 설정함
    optimizer.zero_grad() #남아있을 수도 있는 잔여 그라디언트를 0으로 초기화

    input_x = torch.Tensor(train_X) #형변환
    true_y = torch.Tensor(train_y) #형변환
    pred_y = model(input_x)
    #print(input_x.shape, true_y.shape, pred_y.shape) #혹시 모르니 차원을 확인해봄

    loss = reg_loss(pred_y.squeeze(), true_y)
    loss.backward() #이 코드로 그라디언트를 모두 계산
    optimizer.step() #위에서 계산한 그라디언트를 바탕으로 파라미터를 업데이트
    list_epoch.append(i)
    list_train_loss.append(loss.detach().numpy())

    # ====== Validation ====== #
    model.eval() #evaluation mode로 바꿈
    optimizer.zero_grad() #마찬가지로 잔여 그래디언트 제거
    input_x = torch.Tensor(val_X)
    true_y = torch.Tensor(val_y)
    pred_y = model(input_x)
    loss = reg_loss(pred_y.squeeze(), true_y)
    list_val_loss.append(loss.detach().numpy())

    # ====== Evaulation ====== #
    if i % 200 == 0: #200번 학습 마다 실제 데이터 분포화 모델이 예측한 분포를 그려봄
        # ====== Calculate MAE ====== # MAE : 평균 절대 오차
        model.eval()
        optimizer.zero_grad()
        input_x = torch.Tensor(test_X)
        true_y = torch.Tensor(test_y)
        pred_y = model(input_x).detach().numpy()
        mae = mean_absolute_error(true_y, pred_y) #sklearn 함수들은 True값이 먼저, predict 값이 나중에 들어간다는거 주의
        list_mae.append(mae)
        list_mae_epoch.append(i)

        fig = plt.figure(figsize=(15,5))

        # ====== True Y Scattering ====== #
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1.scatter(test_X[:, 0], test_X[:, 1], test_y, c=test_y, cmap='jet')

        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_zlabel('y')
        ax1.set_zlim(-10, 6)
        ax1.view_init(40, -40)
        ax1.set_title('True test y')
        ax1.invert_xaxis()

        # ====== Predicted Y Scattering ====== #
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2.scatter(test_X[:, 0], test_X[:, 1], pred_y, c=pred_y[:, 0], cmap='jet')

        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_zlabel('y')
        ax2.set_zlim(-10, 6)
        ax2.view_init(40, -40)
        ax2.set_title('Predicted test y')
        ax2.invert_xaxis()

        # ====== Just for Visualizaing with High Resolution ====== #
        input_x = torch.Tensor(train_X)
        pred_y = model(input_x).detach().numpy()

        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        ax3.scatter(train_X[:, 0], train_X[:, 1], pred_y, c=pred_y[:, 0], cmap='jet')

        ax3.set_xlabel('x1')
        ax3.set_ylabel('x2')
        ax3.set_zlabel('y')
        ax3.set_zlim(-10, 6)
        ax3.view_init(40, -40)
        ax3.set_title('Predicted train y')
        ax3.invert_xaxis()

        plt.show()
        print(i, loss)

fig = plt.figure(figsize=(15,5))

# ====== Loss Fluctuation ====== #
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(list_epoch, list_train_loss, label='train_loss')
ax1.plot(list_epoch, list_val_loss, '--', label='val_loss')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax1.set_ylim(0, 5)
ax1.grid()
ax1.legend()
ax1.set_title('epoch vs loss')

# ====== Metric Fluctuation ====== #
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(list_mae_epoch, list_mae, marker='x', label='mae metric')

ax2.set_xlabel('epoch')
ax2.set_ylabel('mae')
ax2.grid()
ax2.legend()
ax2.set_title('epoch vs mae')


plt.show()


