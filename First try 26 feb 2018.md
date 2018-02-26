# Predictive-coding
MBCS Internship 2018

##############################
# MNIST
##############################

# import libraries (torch and matplotlib)
import cv2
import torch
import torchvision
from torch.autograd import Variable

# import numpy as np
import random
# import matplotlib.pyplot as plt # only necessary if we want to plot anything that is now commented


# loading the mnist dataset training
from mnist import MNIST
mndata = MNIST('/home/quirine/python-mnist/data')
images, labels = mndata.load_training()

# show example of mnist picture
# a = np.array(images[0])    # make array/vector of image
# a = a.reshape([28, -1])    # reshape into 28x28 image
# print(a.shape)
# plt.imshow(a)
# plt.show()

# convert pixelvalue into 0-1 scale for all images
def image_convert(images):

    for img in range(len(images)):
        single_image = torch.FloatTensor([images[img]])
        converted_image = single_image / 255
        images[img] = converted_image
    return images


# now we define our network and the activation function in the forward(self) method
class Net (torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # dtype = torch.cuda.FloatTensor to run on GPU
        dtype = torch.FloatTensor
        # create random Tensors to hold representations  and wraps them in Variables
        self.neurons = Variable(torch.rand(60000, 100).type(dtype), requires_grad=True)
        # create random Tensors for weights, and wrap them in Variables --> from normal distribution.
        # So, it is also possible to have negative weights
        self.weights = Variable(torch.randn(100, 784).type(dtype), requires_grad=True)

    def forward(self, representation):
        # this is where the activation function is and Ihat will be calculated
        # input is self and one row of neurons_amount (=representation of one image)

        # 1. matrix duplication between the row of one representation and the whole matrix of weights
        # indexing in neurons via net.neurons[]
        # goals is to loop over the representations outside this function
        print("mm now")
        s_activation = torch.mm(self.neurons[representation, 0], self.weights)
        print(s_activation[0, 0])


        # 2. activation function in matrix duplication form and not a for loop
        # computes the activation with the activation function from Zambrano et al 2017
        # neurons = matrix with all the representations for all neurons
        # weights = the whole matrix, since that is necessary for a fully connected network


        # parameters:
        v0 = torch.FloatTensor([0.5])
        mf = v0
        tauy = torch.FloatTensor([15])
        taun = torch.FloatTensor([50])
        h = torch.FloatTensor([0.1])
        print(v0)
        # parameters transformed into constants
        v0 = mf
        c1 = 2 * mf * tauy ** 2
        c2 = 2 * v0 * taun * tauy
        c3 = tauy * (mf * tauy * 2 * (mf + 1) * taun)
        c4 = v0 * taun * (tauy + taun)
        c0 = h / (torch.exp(((c1 * v0) / (2 + c2)) / ((c3 * v0) / (2 + c4))) - 1)

        # actual activation_function
        ihat = h / (torch.exp((c1 * s_activation + c2) / (c3 * s_activation + c4) - 1) - c0 + (h / 2))



        # afunc = np.vectorize(activation_function)
        # ihat = afunc(s_activation)

        return ihat



# net = Net(100, images) # --> to create the weights and neuron matrices





if __name__ == "__main__":
    conv_images = image_convert(images)
    print("images converted")
    print(type(conv_images))
    # neurons, weights = neuron_layer(conv_images, 100)

    net = Net()
    print("test neurons", net.neurons[0, 0])
    print("test weights", net.weights[0, 0])

    input_test = torch.rand(1, 784)
    out = net(input_test, 0)
    print(out)







