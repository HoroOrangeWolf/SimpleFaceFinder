import torch.optim
import torchvision.transforms
from torch.utils.data import DataLoader

from DataPreparer.DataPreparer import DataPreparer
from DatasetImplementation.DatasetImplementation import DatasetImplementation
from torch import nn

learning_rate = 1e-3
batch_size = 64
element_count = 1000

DataPreparer(element_count=element_count, data_path='DataForDataset', data_output_path='DatasetImage')

data_learn = DatasetImplementation(data_path='DatasetImage', picture_transformation=torchvision.transforms.ToTensor())

training_data_loader = DataLoader(data_learn, batch_size=batch_size, shuffle=True)

DataPreparer(element_count=element_count - 800, data_path='DataForDataset', data_output_path='DataTest')

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

        if count % 2 == 0:
            print(f'loss avg: {avg / 2 :>5f} [{len(X) * count}/{size}]')
            avg = 0


def testNeuralNetwork(model1, data_loader, loss_fn1):
    batch_s = data_loader.__len__() / len(data_loader)
    with torch.no_grad():
        avg = 0
        for batch, (X, y) in enumerate(data_loader):
            prediction = model1(X)
            avg += loss_fn1(prediction, y).item()
        print(f'Test error: \n avg loss: {avg / (batch_s * len(data_loader))}')


model = FaceRecognitionNetwork()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.L1Loss()

model.load_state_dict(torch.load('model_weights.pth'))

for i in range(30):
    print(f'Epos: {i + 1} \n ------------------------------------')
    trainNeuralNetwork(model1=model, loss_fn1=loss_fn, optimizer1=optimizer, data_loader=training_data_loader)
    testNeuralNetwork(model1=model, loss_fn1=loss_fn, data_loader=test_data_loader)

torch.save(model.state_dict(), 'model_weights.pth')
