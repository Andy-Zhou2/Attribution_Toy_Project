import torch as t
from dataset import RandomMatrixDataset, fit_matrix_linear
from torch.utils.data import DataLoader
from model import TraceNet
import torch.optim as optim
import time
import torch.nn as nn

MAX_SIZE = 10

GPU = t.device("cuda:0")

train_dataset = RandomMatrixDataset(MAX_SIZE)
train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True, pin_memory=False, num_workers=0)

MSELoss = nn.MSELoss()

for i in range(10):
    for j in range(10):
        print(f'masking entry {(i, j)}')
        net = TraceNet(MAX_SIZE)  # restart training process
        net = net.to(GPU)

        matrix = t.tensor([[1, 2], [3, 4]], dtype=t.float)
        matrix *= 10000

        fitted_matrix = fit_matrix_linear(matrix, 2, 10)
        fitted_matrix = t.unsqueeze(fitted_matrix, 0).to(GPU)
        fitted_matrix.requires_grad = True

        pred_trace = net(fitted_matrix)
        print('initial prediction:', pred_trace)

        epoch = 0
        epoch_start_time = time.time()
        for epoch in range(300):
            # print(f'start epoch: {epoch}')
            loss_epoch = 0

            optimizer = optim.Adam(net.parameters(), lr=1e-3)

            for _, data in enumerate(train_dataloader):
                optimizer.zero_grad()  # gradient reset
                matrix, correct_trace = data
                matrix = matrix.to(GPU)
                matrix[:, i * 10 + j] = 0  # masking!
                correct_trace = correct_trace.to(GPU)

                pred_trace = net(matrix)
                pred_trace = pred_trace.view(-1)
                loss = MSELoss(pred_trace, correct_trace)

                loss_epoch += loss.item()
                loss.backward()

                optimizer.step()

            epoch += 1

        print('epoch, loss:', epoch, loss_epoch)
        epoch_time = time.time() - epoch_start_time
        print('epoch time:', epoch_time)
        state = {"weight": net.state_dict()}
        t.save(state, f'./nets/TraceNet_mask_{epoch}_{i}_{j}.net')
