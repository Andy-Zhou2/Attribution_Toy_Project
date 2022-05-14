import torch as t
from dataset import RandomMatrixDataset
from torch.utils.data import DataLoader
from model import TraceNet
import torch.optim as optim
import time
import torch.nn as nn

MAX_SIZE = 2

GPU = t.device("cuda:0")
net = TraceNet(MAX_SIZE)
net = net.to(GPU)

train_dataset = RandomMatrixDataset(MAX_SIZE)
train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True, pin_memory=False, num_workers=0)

BCELoss = nn.BCELoss()

epoch = 0
for i in range(1000):
    # print(f'start epoch: {epoch}')
    epoch_start_time = time.time()
    loss_epoch = 0

    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    for _, data in enumerate(train_dataloader):
        optimizer.zero_grad()  # gradient reset
        matrix, trace_vec, trace = data
        matrix = matrix.to(GPU)
        trace_vec = trace_vec.to(GPU)

        pred_trace = net(matrix)
        pred_trace = t.softmax(pred_trace, dim=1)
        loss = BCELoss(pred_trace, trace_vec)

        loss_epoch += loss.item()
        loss.backward()

        optimizer.step()

    epoch += 1

    print('epoch, loss:', epoch, loss_epoch)
# epoch_time = time.time() - epoch_start_time
# print('epoch time:', epoch_time)
state = {"weight": net.state_dict()}
t.save(state, f'./nets/TraceNetClassification_size2_0-5_BCE_1e-3_1000.net')
