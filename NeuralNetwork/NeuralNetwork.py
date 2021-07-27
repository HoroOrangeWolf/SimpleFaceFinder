import torch
from torch import nn


class FaceRecognitionNeuralNetwork(nn.Module):
    def __init__(self, in_channels=1):
        super(FaceRecognitionNeuralNetwork, self).__init__()

        self.network = nn.Sequential(
            MultiBranchNetwork(in_channels=in_channels),
            nn.Flatten(),
            nn.Linear(in_features=1000, out_features=3096),
            nn.ReLU(),
            nn.Linear(in_features=3096, out_features=3096),
            nn.ReLU(),
            nn.Linear(in_features=3096, out_features=4)
        )

    def forward(self, x):
        return self.network(x)


def trainModel(model, data_loader, optimizer, loss_fn):
    avg = 0
    size = len(data_loader.dataset)
    batch_size = int(size / len(data_loader))
    for batch, (x, y) in enumerate(data_loader):
        prediction = model(x)
        loss = loss_fn(prediction, y)

        loss.zero_grad()
        optimizer.backward()
        loss.step()

        avg += loss.item()

        if batch != 0 and batch % 100 == 0:
            print(f'Avg loss: {avg/100:>5f}  [{batch*batch_size}/{size}')

def testModel(model, data_loader, loss_fn):

    avg = 0
    with torch.no_grad():
        for x,y in data_loader:
            prediction =

class MultiBranchNetwork(nn.Module):
    def __init__(self, in_channels=1):
        super(MultiBranchNetwork, self).__init__()

        self.branch_1_5x5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=9, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=9, out_channels=12, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

        self.branch_2_5x5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=9, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)))

        self.branch_3_10x10 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=(10, 10), stride=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

    def forward(self, x):
        x_1 = self.branch_1_5x5(x)

        x_2 = self.branch_2_5x5(x)

        x_3 = self.branch_3_10x10(x)

        return torch.cat((x_1, x_2, x_3), 0)
