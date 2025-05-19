import torch
def get_rectangle():
    import random
    width = random.random()
    height = random.random()
    s = width * height
    return width, height, s

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        pass
    def __len__(self):
        return 500
    def __getitem__(self, i):
        width, height, s = get_rectangle()
        x = torch.FloatTensor([width, height])
        y = torch.FloatTensor([s])
        return x, y
dataset = Dataset()

loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=8,
                                     shuffle=True,
                                     drop_last=True)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=2, out_features=32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32, out_features=32),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=32, out_features=1),
        )
    def forward(self, x):
        return self.fc(x)
model = Model()
def train():
    #优化器,根据梯度调整模型参数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    #计算loss的函数
    loss_fun = torch.nn.MSELoss()

    #让model进入train模式,开启dropout等功能
    model.train()

    #全量数据遍历100轮
    for epoch in range(100):

        #按批次遍历loader中的数据
        for i, (x, y) in enumerate(loader):

            #模型计算
            out = model(x)

            #根据计算结果和y的差,计算loss,在计算结果完全正确的情况下,loss为0
            loss = loss_fun(out, y)

            #根据loss计算模型的梯度
            loss.backward()

            #根据梯度调整模型的参数
            optimizer.step()

            #梯度归零,准备下一轮的计算
            optimizer.zero_grad()

        if epoch % 20 == 0:
            print(epoch, loss.item()) 
train()