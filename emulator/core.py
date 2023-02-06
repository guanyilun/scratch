import numpy as np
import torch
from functools import partial
from tqdm import tqdm
from collections import OrderedDict

from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from network import Net


class Emulator:
    def __init__(self, NNClass=Net, epoches=100, test_size=0.2, optimizer=None, criterion=None, precondition=None, **kwargs):
        self.NNClass = partial(NNClass, **kwargs)
        self.epoches = epoches
        self.test_size = test_size
        self.optimizer = optimizer if optimizer is not None else partial(optim.SGD, lr=0.001, momentum=0.9)
        self.criterion = criterion if criterion is not None else nn.MSELoss() 
        self.precondition = precondition  # need to have forward and backward methods for transf and inverse transf

    def get_samples(self, samples):
        # samples: dict[parameter_name] -> list of choices, assume all choices are float and have the same length
        return np.vstack([np.array(v) for _, v in samples.items()]).T

    def emulate(self, func, samples):
        features = self.get_samples(samples)

        # run all combinations and gather results
        res_list = []
        for i in range(features.shape[0]):  # samples has shape (n_samples, n_parameters)
            res = func(**{k:v for k, v in zip(samples.keys(), features[i,:])})
            # optionally apply a transformation
            if self.precondition is not None: res = self.precondition.forward(res)
            res_list.append(res)

        # we can train a model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        features = torch.as_tensor(features, dtype=torch.float)
        labels = torch.as_tensor(np.array(res_list), dtype=torch.float)
        if len(features.shape) == 1: features = features.reshape(-1, 1)
        if len(labels.shape) == 1: labels = labels.reshape(-1, 1)

        # split into train and test 
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)

        # convert the data to PyTorch tensors and send to GPU if available
        train_features = torch.as_tensor(train_features, dtype=torch.float32, device=device)
        train_labels = torch.as_tensor(train_labels, dtype=torch.float32, device=device)
        test_features = torch.as_tensor(test_features, dtype=torch.float32, device=device)
        test_labels = torch.as_tensor(test_labels, dtype=torch.float32, device=device)

        # create TensorDatasets and DataLoaders
        train_data = TensorDataset(train_features, train_labels)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        # build model, optimizer and loss function
        model = self.NNClass(features.shape[-1], labels.shape[-1]).to(device)
        optimizer = self.optimizer(model.parameters())
        criterion = self.criterion

        # train the model
        print("Training emulator...")
        pbar = tqdm(range(self.epoches))
        for epoch in pbar:
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
            pbar.set_postfix({'loss': f"{running_loss/(i+1):.3f}"})

        # Test the model
        with torch.no_grad():
            test_loss = 0.0
            for i, data in enumerate(zip(test_features, test_labels)):
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
            avg_test_loss = test_loss / (i + 1)
        print("Ave Test loss: {:.3f}".format(avg_test_loss))
        
        # return emulated function
        def emulated_func(**kwargs):
            values = [kwargs.get(k, None) for k in samples.keys()]
            if None in values: raise ValueError('Missing argument')
            values = torch.as_tensor(values, dtype=torch.float)
            res = model(values).detach().numpy()
            if self.precondition is not None: res = self.precondition.backward(res)
            if len(res.shape) == 1 and len(res) == 1: res = res[0]
            return res
        emulated_func.model = model if device=='cpu' else model.cpu()
        return emulated_func

# a decorator interface
class emulate:
    def __init__(self, samples=None, NNClass=Net, epoches=100, lr=0.01, momentum=0.9, weight_decay=0, nsamps=1000, **kwargs):
        optimizer = partial(optim.SGD, lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.em = Emulator(NNClass=NNClass, epoches=epoches, optimizer=optimizer, **kwargs)
        self.samples = samples
        self.nsamps = nsamps  # only used when samples is None
    def __call__(self, func):
        if self.samples is None:
            # get samples from function signature
            import inspect
            sig = inspect.signature(func)
            self.samples = OrderedDict({k: np.random.randn(self.nsamps) for k in sig.parameters.keys()})
        return self.em.emulate(func, self.samples)