import torch as t
from dataset import RandomMatrixDataset
from torch.utils.data import DataLoader
from model import TraceNet
import torch.optim as optim
import time
import torch.nn as nn

MAX_SIZE = 10

GPU = t.device("cuda:0")
net = TraceNet(MAX_SIZE)
net = net.to(GPU)

train_dataset = RandomMatrixDataset(MAX_SIZE)
train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True, pin_memory=False, num_workers=0)

MSELoss = nn.MSELoss()

epoch = 0
while True:
    print(f'start epoch: {epoch}')
    epoch_start_time = time.time()
    loss_epoch = 0

    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    for i, data in enumerate(train_dataloader):
        optimizer.zero_grad()  # gradient reset
        matrix, correct_trace = data
        matrix = matrix.to(GPU)
        correct_trace = correct_trace.to(GPU)

        pred_trace = net(matrix)
        pred_trace = pred_trace.view(-1)
        loss = MSELoss(pred_trace, correct_trace)

        loss_epoch += loss.item()
        loss.backward()

        optimizer.step()

    epoch += 1

    if epoch % 1000 == 0:
        print('epoch, loss:', epoch, loss_epoch)
        epoch_time = time.time() - epoch_start_time
        print('epoch time:', epoch_time)
        state = {"weight": net.state_dict()}
        t.save(state, f'TraceNet_{epoch}.net')
