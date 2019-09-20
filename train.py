from models.cnn import ConvNet
import torch
from lib.dataset import get_loader
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AUCCallback, F1ScoreCallback
import collections
import os
from lib.loss import CustomLoss

model = ConvNet(rnn_hidden=32)
model = model.cuda()

# experiment setup
logdir = "./logdir"
num_epochs = 20


# data
loaders = collections.OrderedDict()
loaders["train"] = get_loader(os.path.join('data', 'train'),
                              os.path.join('data', 'train.mat'),
                              batch_size=32,
                              shuffle=True)
loaders["valid"] = get_loader(os.path.join('data', 'test'),
                              os.path.join('data', 'test.mat'),
                              batch_size=128,
                              shuffle=False)


criterion = CustomLoss()
# criterion = torch.nn.NLLLoss()
# criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

callbacks = None # [AUCCallback(), F1ScoreCallback()]

# model runner
# Trainer(model, loaders).run()
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
    callbacks=callbacks,
    verbose=True
)
