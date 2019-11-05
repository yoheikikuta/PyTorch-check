import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    net = Net()
    print(net)

    params = list(net.parameters())
    print(len(params))
    print(params[0].size())  # conv1's .weight

    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(out)

    net.zero_grad()
    out.backward(torch.randn(1, 10))
    print(out)  # same out as the previous one because of zero grad

    output = net(input)
    target = torch.randn(10)  # a dummy target, for example
    target = target.view(1, -1)  # make it the same shape as output
    criterion = nn.MSELoss()

    loss = criterion(output, target)
    print(loss)
    print(loss.grad_fn)  # type: ignore
    print(loss.grad_fn.next_functions[0][0])  # type: ignore # Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # type: ignore # ReLU

    net.zero_grad()     # zeroes the gradient buffers of all parameters
    print('conv1.bias.grad before backward: ', end="")
    print(net.conv1.bias.grad)
    loss.backward()
    print('conv1.bias.grad after backward: ', end="")
    print(net.conv1.bias.grad)

    # weight = weight - learning_rate * gradient
    learning_rate = 0.01
    for f in net.parameters():
        f.data.sub_(f.grad.data * learning_rate)  # type: ignore

    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    # in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()    # Does the update
