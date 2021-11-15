import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch


def train_net(net,
              train_dataloader,
              device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              n_epoh=10):
    criterion = nn.MSELoss()
    net.to(device)
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    mse_list = []
    for epoch in range(n_epoh):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                mse_list.append(running_loss)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    print('Finished Training')
    return net, mse_list


def test_net(net,
             test_dataloader,
             device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    iter_i = 0

    with torch.no_grad():
        for data in test_dataloader:
            inputs, _ = data[0].to(device), data[1].to(device)
            outputs = net(inputs)

            if iter_i == 0:
                d = outputs
            else:
                d = torch.cat((d, outputs), 0)

            iter_i += 1

    return d.cpu().numpy()
