from __future__ import print_function

import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import app, flags, logging
from torchvision import datasets, transforms

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'file_url',
    default="https://docs.google.com/uc?export=download&id=1KVOHbHnjCd1L-ookcd7CxDqb7rb8-DSx",  # noqa: E501
    help="LeNet model file URL.")

flags.DEFINE_string(
    'data_dir',
    default=os.path.join(os.path.dirname(__file__), "..", "data"),
    help="Path of the data directory.")

flags.DEFINE_string(
    'model_name',
    default="lenet_mnist_model.pth",
    help="File name of the pretrained model.")


# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def test(model, device, test_loader, epsilon):
    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        # get the index of the max log-probability
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        # If the initial prediction is wrong,
        # don't bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        loss.backward()
        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        # get the index of the max log-probability
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")  # noqa: E501

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


def main(argv):
    # If there doesn't exist the model, donwload a pretrained model from
    # https://drive.google.com/drive/folders/1fn83DF14tWmit0RTKWRhPq5uVXt73e0h.
    if not os.path.isfile(os.path.join(FLAGS.data_dir, FLAGS.model_name)):
        logging.info('Model download')
        urllib.request.urlretrieve(FLAGS.file_url,
                                   os.path.join(FLAGS.data_dir, FLAGS.model_name))

    # MNIST Test dataset and dataloader declaration
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(FLAGS.data_dir, train=False, download=True,
                       transform=transforms.Compose([transforms.ToTensor(), ])),
        batch_size=1, shuffle=True)

    # Define what device we are using
    use_cuda = False
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # Initialize the network
    model = Net().to(device)
    # Load the pretrained model
    logging.info('Load the pretrained model')
    model.load_state_dict(torch.load(os.path.join(FLAGS.data_dir, FLAGS.model_name),
                                     map_location='cpu'))
    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()

    accuracies = []
    examples = []

    # Run test for each epsilon
    logging.info('Run test for each epsilon')
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    for eps in epsilons:
        acc, ex = test(model, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
