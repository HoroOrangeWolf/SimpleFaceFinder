import torch.optim
import torchvision.transforms
from torch.utils.data import DataLoader
from NeuralNetwork.NeuralNetwork import FaceRecognitionNeuralNetwork as Face
from NeuralData.DataPreparer import DataPreparer
from NeuralData.DatasetImplementation import DatasetImplementation

batch_size = 32
learning_rate = 1e-3

# Creating test data
DataPreparer(element_count=100, data_path='DataForDataset', data_output_path='TestData')

# Creating learning data
DataPreparer(element_count=800, data_path='DataForDataset', data_output_path='LearningData')

test_data = DatasetImplementation(data_path='TestData',
                                  picture_size=(1028, 1028),
                                  picture_transformation=torchvision.transforms.ToTensor())

test_data_loader = DataLoader(dataset=test_data,
                              batch_size=batch_size,
                              shuffle=True)

test_data = DatasetImplementation(data_path='LearningData',
                                  picture_size=(1028, 1028),
                                  picture_transformation=torchvision.transforms.ToTensor())

learn_data_loader = DataLoader(dataset=test_data,
                               batch_size=batch_size,
                               shuffle=True)

# Init neural network
model = Face(in_channels=1)

optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.L1Loss()
