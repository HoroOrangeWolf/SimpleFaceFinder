import torch.optim
import torchvision.transforms
from torch.utils.data import DataLoader
from NeuralNetwork.NeuralNetwork import FaceRecognitionNeuralNetwork as Face
from NeuralData.DataPreparer import DataPreparer
from NeuralData.DatasetImplementation import DatasetImplementation
from NeuralNetwork.NeuralNetwork import trainModel
from NeuralNetwork.NeuralNetwork import testModel

batch_size = 1
learning_rate = 1e-3
epochs = 10

# Creating test data
DataPreparer(element_count=100, data_path='DataForDataset', data_output_path='TestData')

test_data = DatasetImplementation(data_path='TestData',
                                  picture_transformation=torchvision.transforms.ToTensor())

test_data_loader = DataLoader(test_data,
                              batch_size=batch_size,
                              shuffle=True)

# Creating learning data
DataPreparer(element_count=800, data_path='DataForDataset', data_output_path='LearningData')

test_data = DatasetImplementation(data_path='LearningData',
                                  picture_transformation=torchvision.transforms.ToTensor())

learn_data_loader = DataLoader(test_data,
                               batch_size=batch_size,
                               shuffle=True)

# Init neural network
model = Face(in_channels=1)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.L1Loss()

for i in range(epochs):
    print(f'------Epochs{i + 1}-------')
    trainModel(model=model, optimizer=optimizer, loss_fn=loss_fn, data_loader=learn_data_loader)
    testModel(model=model, loss_fn=loss_fn, data_loader=test_data_loader)
