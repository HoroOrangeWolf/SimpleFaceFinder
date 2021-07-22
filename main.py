import torch.optim
import torchvision.transforms
from torch.utils.data import DataLoader

from DataPreparer.DataPreparer import DataPreparer
from DatasetImplementation.DatasetImplementation import DatasetImplementation
from matplotlib import pyplot as plt
from matplotlib import patches
from NeuralNetwork.FaceRecognitionNetwork import FaceRecognitionNetwork

learning_rate = 1e-3
batch_size = 1
element_count = 1000

DataPreparer(element_count=element_count, data_path='DataForDataset', data_output_path='DatasetImage')

data_learn = DatasetImplementation(data_path='DatasetImage', picture_transformation=torchvision.transforms.ToTensor())

training_data_loader = DataLoader(data_learn, batch_size=batch_size, shuffle=True)

DataPreparer(element_count=element_count - 800, data_path='DataForDataset', data_output_path='DataTest')

data_test = DatasetImplementation(data_path='DataTest', picture_transformation=torchvision.transforms.ToTensor())

test_data_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True)


def trainNeuralNetwork(model1, loss_fn1, optimizer1, data_loader):
    avg = 0
    size = len(data_loader.dataset)
    for count, (X1, y1) in enumerate(data_loader):
        predict = model1(X1)
        loss = loss_fn1(predict, y1)
        avg += loss.item()

        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()

        if count % 2 == 0:
            print(f'loss avg: {avg / 2 :>5f} [{len(X1) * count}/{size}]')
            avg = 0


def testNeuralNetwork(model1, data_loader, loss_fn1):
    batch_s = data_loader.__len__() / len(data_loader)
    with torch.no_grad():
        avg = 0
        for batCH, (X1, y1) in enumerate(data_loader):
            predict = model1(X1)
            avg += loss_fn1(predict, y1).item()
        print(f'Test error: \n avg loss: {avg / (batch_s * len(data_loader))}')


model = FaceRecognitionNetwork()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.L1Loss()

model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

s = '''for i in range(100):
    print(f'Epos: {i + 1} \n ------------------------------------')
    trainNeuralNetwork(model1=model, loss_fn1=loss_fn, optimizer1=optimizer, data_loader=training_data_loader)
    testNeuralNetwork(model1=model, loss_fn1=loss_fn, data_loader=test_data_loader)
    if i % 10 == 0:
        print('Saving state dict!')
        torch.save(model.state_dict(), 'model_weights.pth')
        print('Saved state dict!')
'''

buff, ax = plt.subplots(5)

with torch.no_grad():
    for batch, (X, y) in enumerate(test_data_loader):

        prediction = model(X)
        x_pic = prediction[0][0] * 1028
        y_pic = prediction[0][1] * 1028
        width_pic = prediction[0][2] * 1028
        height_pic = prediction[0][3] * 1028

        image = torchvision.transforms.ToPILImage()(X[0])

        ax[batch].axis('off')
        ax[batch].imshow(image)

        patch = patches.Rectangle((x_pic, y_pic), width_pic, height_pic,
                                  edgecolor='r',
                                  facecolor='none',
                                  linewidth=1)

        ax[batch].add_patch(patch)
plt.show()
