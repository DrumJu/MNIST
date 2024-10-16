import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = torchvision.datasets.MNIST(root='dataset', train=True, transform=transforms.Compose([ transforms.Resize(size=32),transforms.ToTensor()]), download=True)
test_dataset=torchvision.datasets.MNIST(root='dataset', train=False, transform=transforms.Compose([transforms.Resize(size=32),transforms.ToTensor()]), download=True)
train_loader=DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader=DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)
train_data_size=len(train_dataset)
test_data_size=len(test_dataset)
class MyModel(torch.nn.Module):
    def __init__(self):
       super(MyModel, self).__init__()
       self.model=nn.Sequential(
       torch.nn.Conv2d(in_channels=1,out_channels= 6, kernel_size=5,stride=1, padding=0),
        torch.nn.MaxPool2d(kernel_size=(2,2), ),
        torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5,stride=1, padding=0),
        torch.nn.MaxPool2d(kernel_size=(2,2)),
       torch.nn.Flatten(),
        torch.nn.Linear(in_features=16 * 5 * 5, out_features=120),
       torch.nn.Linear(in_features=120,out_features=84),
         torch.nn.Linear(in_features=84, out_features=10),
        )
    def forward(self, x):
        x = self.model(x)
        return x
my_model=MyModel().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(my_model.parameters(), lr=0.0008)
Step=0
EPOCHS = 12

for epoch in range(EPOCHS):
    for data in train_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = my_model(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Step += 1
        if Step % 100 == 0:
            print(f'第{Step}次训练，loss={loss.item()}')
    accuracy=0
    total_accuracy=0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = my_model(images)
            loss = loss_fn(outputs, labels)
            accuracy = (outputs.argmax(axis=1) == labels).sum()
            total_accuracy +=accuracy
        print(f"第{epoch+1}轮训练结束，准确率为{total_accuracy/len(test_dataset)*100:.4f}%")
        torch.save(my_model, f'MNIST_{epoch}_acc{total_accuracy/test_data_size}.pth')









