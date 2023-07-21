#%%
## 基本目標
# 1. 了解torch.nn.Module的繼承
# 2. 了解torch.nn.Sequential、nn.Linear的使用
# 3. 熟悉model.train()、model.eval()、及torch.no_grad()的使用時機
# 4. 了解torch.optim中的optimizer
# 5. 了解torch.nn中的loss function
# 6. 了解torch.device的使用
# 7. 獨立設計並創建模型

#%%
from sklearn.datasets import make_classification
# Import Dataset
train_x, train_y = make_classification(n_samples = 100, n_features = 5)
print(train_x.shape, '\n', train_x[:3])
print(train_y.shape, '\n', train_y[:3])
print(set(train_y))
# %%
import torch 
from torch import nn 


#nn.sigmoid()跟nn.functional.sigmoid()分別是class跟function要搞清楚怎麼用



class ForwardModel(nn.Module):
    ''' 教學範例，請自行修改以下設神經網路架構 '''
    def __init__(self):
        #super這行一定要寫才會繼承
        super(ForwardModel, self).__init__()
        #super().__init__()也一樣
        #sequential 會自動包一條網路追裡面的weight，可以看一下squential做了什麼
        self.stage1 = nn.Sequential(
            nn.Linear(in_features = 5, out_features = 10, bias= True),
            nn.ReLU(),
        )
        self.stage2 = nn.Sequential(
            nn.Linear(10, 6),
            nn.ReLU(),
            nn.Linear(6, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )
        

    def forward(self, X):
        # nn.Module 內建__call__會導向forward
        out = self.stage1(X)
        out = self.stage2(out)
        out = nn.functional.sigmoid(out)
        return out

#%%

model = ForwardModel()
loss_func = torch.nn.BCELoss()
#model 會自動將weight存成parameters
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#%%
device = torch.device("mps")

X = torch.tensor(train_x,dtype=torch.float32).to(device)
Y = torch.tensor(train_y,dtype=torch.float32).to(device)
model.to(device)
Y
#%%
epoch = 1000
model.train()#加上標記model會自動儲存train的資料
for _ in range(epoch):
    y_hat = model(X)
    loss = loss_func(y_hat.squeeze(), Y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(round(loss.item(), 4))
# %%
model.eval()
print(X[0])
print(Y[0])
with torch.no_grad():
    pred = model(X)
    print("pred values:", pred)
# %%

## 實作目標: (週六前完成)
# 0. 以Credit Card Fraud Detection為資料集(Notion下載)研討會裡面
# 1. 能夠順利切換CPU/GPU跑訓練模型，並比較CPU/GPU速度
# 2. 紀錄loss數值的變化，並畫出loss變化圖(盡可能比較不同情境下的loss變化)
# 3. 調整網路架構、調整loss函數、調整學習器，來嘗試提升模型準確率(盡可能紀錄下來)
# 4. 儲存模型權重(torch.save)
# 5. 載入模型權重(torch.load)並切換CPU/GPU環境進行evaluation
