#%%
import numpy as np
import torch
from network import Net
from functools import partial
from tqdm import tqdm

from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

class emulate:
    def __init__(self, args, NNClass=Net, lr=0.001, momentum=0.9, epoches=100, verbose=False, **kwargs):
        self.NNClass = partial(NNClass, **kwargs)
        self.lr = lr
        self.momentum = momentum
        self.epoches = epoches
        self.args = args
        self.verbose = verbose

    def __call__(self, func):
        # build all args combinations
        args_combo = get_args_combo(self.args)

        # run all combinations and gather results
        res_list = []
        for args_ in args_combo:
            res = func(**{k:v for k, v in zip(self.args.keys(), args_)})
            res_list.append(res)

        # now we have all data and results, we can train a model
        features = torch.as_tensor(args_combo, dtype=torch.float)
        labels = torch.as_tensor(res_list, dtype=torch.float)
        if len(features.shape) == 1: features = features.reshape(-1, 1)
        if len(labels.shape) == 1: labels = labels.reshape(-1, 1)

        # split into train and test 
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)

        # convert the data to PyTorch tensors and send to GPU if available
        train_features = torch.as_tensor(train_features, dtype=torch.float32)
        train_labels = torch.as_tensor(train_labels, dtype=torch.float32)
        test_features = torch.as_tensor(test_features, dtype=torch.float32)
        test_labels = torch.as_tensor(test_labels, dtype=torch.float32)

        # create TensorDatasets and DataLoaders
        train_data = TensorDataset(train_features, train_labels)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        # build model, optimizer and loss function
        model = self.NNClass(idim=features.shape[-1], odim=labels.shape[-1])
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)

        # train the model
        print("Training emulator...")
        for epoch in tqdm(range(self.epoches)):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
            if self.verbose: print('Epoch {} loss: {:.3f}'.format(epoch + 1, running_loss / (i + 1)))

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
        
        def func_(**kwargs):
            values = [kwargs.get(k, None) for k in self.args.keys()]
            if None in values: raise ValueError('Missing argument')
            values = torch.as_tensor(values, dtype=torch.float)
            res = model(values).detach().numpy()
            if len(res.shape) == 1: res = res[0]
            return res

        return func_


# utility functions
def get_args_combo(args):
    return np.array(np.meshgrid(*args.values())).T.reshape(-1, len(args))
