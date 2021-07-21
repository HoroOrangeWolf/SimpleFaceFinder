import torch.optim
import torchvision.transforms
from torch.utils.data import DataLoader

from DataPreparer.DataPreparer import DataPreparer
from DatasetImplementation.DatasetImplementation import DatasetImplementation
from torch import nn

learning_rate = 1e-3
batch_size = 2

DataPreparer(element_count=50, data_path='DataForDataset', data_output_path='DatasetImage')

data_learn = DatasetImplementation(data_path='DatasetImage', picture_transformation=torchvision.transforms.ToTensor())

training_data_loader = DataLoader(data_learn, batch_size=batch_size, shuffle=True)

DataPreparer(element_count=50, data_path='DataForDataset', data_output_path='DataTest')

data_test = DatasetImplementation(data_path='DataTest', picture_transformation=torchvision.transforms.ToTensor())

test_data_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True)


class FaceRecognitionNetwork(nn.Module):
    def __init__(self):
        super(FaceRecognitionNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, stride=(1, 1), kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=12, out_channels=18, kernel_size=(5, 5), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.Linear(in_features=17298, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=4)
        )

    def forward(self, x):
        return self.network(x)


def trainNeuralNetwork(model1, loss_fn1, optimizer1, data_loader):
    avg = 0
    size = len(data_loader.dataset)
    for count, (X, y) in enumerate(data_loader):
        prediction = model1(X)
        loss = loss_fn1(prediction, y)
        avg += loss.item()

        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()

        if count % 100 == 0:
            print(f'loss avg: {avg/(count + 1):>5f} [{len(X)*count}/{size}]')
            avg = 0


model = FaceRecognitionNetwork()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.L1Loss()

for i in range(100):
    trainNeuralNetwork(model1=model, loss_fn1=loss_fn, optimizer1=optimizer, data_loader=training_data_loader)
