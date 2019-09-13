from lib.dataset import HouseNumberTrainDataset
from models.cnn import ConvNet
import torch
from lib.dataset import get_loader
from catalyst.dl import SupervisedRunner
from torch.utils.data import DataLoader
import collections
import os

model = ConvNet(rnn_hidden=96)
model = model.cuda()

# experiment setup
logdir = "./logdir"
num_epochs = 20


# data
loaders = collections.OrderedDict()
loaders["train"] = get_loader(os.path.join('data', 'train'),
                              os.path.join('data', 'train.mat'),
                              batch_size=64,
                              shuffle=True)
loaders["valid"] = get_loader(os.path.join('data', 'test'),
                              os.path.join('data', 'test.mat'),
                              batch_size=128,
                              shuffle=False)


criterion = torch.nn.BCEWithLogitsLoss()
# criterion = torch.nn.MSELoss()
# criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=8e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

# model runner
runner = SupervisedRunner()

# model training
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    logdir=logdir,
    num_epochs=num_epochs,
    verbose=True
)
