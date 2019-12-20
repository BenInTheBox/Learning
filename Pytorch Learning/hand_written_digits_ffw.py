import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

''' ----- Data preparation ----- '''
# Data set
train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

# Reformat data set in batch with shuffle (recommended batch size = 8-60)
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

for data in trainset:
    print(data)
    break

x, y = data[0][0], data[1][0]
print(x, "    ", y)
print(x.shape)  # torch.Size([1, 28, 28]) 1x28x28 is the wrong size (pytorch format)
plt.imshow(x.view([28, 28]))  # reshape to 28x28
plt.show()

# Need balanced data set so need to check it
total = 0
counter_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
for data in trainset:
    Xs, ys = data
    for y in ys:
        counter_dict[int(y)] += 1
        total += 1

for i in counter_dict:
    print(f"{i}: {counter_dict[i]/total*100}")


''' ----- Neural network ----- '''


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        # Neural network initialization
        self.fc1 = nn.Linear(28*28, 64)  # NN regular layers (input, output)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)  # Output layer with 10 neurons because it is the output size

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # Can create NN with the first layers dedicated to processing and the next ones for computing

        return F.log_softmax(x, dim=1)  # Will receive batchs so dim(=axis) is needed to explain witch direction is the distribution


net = Net()
print(net)
X = torch.rand((28, 28))
X = X.view(-1, 28*28)
output = net(X)
print(output)
optimizer = optim.Adam(net.parameters(), lr=0.001)  # optimizer, can use decaying learning rate for better perf

EPOCHS = 3 # Number of time that the whole dataset is used to train

for epoch in range(EPOCHS):
    for data in trainset:
        # Data is a batch of feature sets and labels
        X, y = data
        net.zero_grad() # reset the gradient between batch
        output = net(X.view(-1, 28*28))
        loss = F.nll_loss(output, y)  # For scaler value use nll_loss, if vector mean squared errors
        loss.backward()
        optimizer.step()
    print(loss)

correct = 0
total = 0

with torch.no_grad():
    for data in trainset:
        X, y = data
        output = net(X.view(-1, 784))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

print("Accuracy: ", round(correct/total, 3))

plt.imshow(X[0].view(28, 28))
plt.show()
print(torch.argmax(net(X[0].view(-1, 784))[0]))