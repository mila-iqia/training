import os
from torchvision import datasets, transforms

folder = os.environ['DATA_DIRECTORY']
os.makedirs(folder + '/mnist', exist_ok=True)

datasets.MNIST(folder + '/mnist', train=True, download=True, transform=transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.1307,), (0.3081,))
]))


datasets.MNIST(folder + '/mnist', train=False, download=True, transform=transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.1307,), (0.3081,))
]))

