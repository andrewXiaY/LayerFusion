import torch
import os
import torch.nn as nn
import torch.nn.functional as TF
from workflows.train_loop import generic_train_loop
from workflows.eval_loop import generic_eval_loop
from models.alex_net import AlexNet
from datasets.dataset import DiskImageDataset
from configs import *

TASK = "ssl_noise_add"
checkpoints_path = "./checkpoints/" + TASK

if not os.path.exists(checkpoints_path):
    os.mkdir(checkpoints_path)

dataset_train = DiskImageDataset(TRAIN_DATAPATH, TASK, end=-1)
dataset_test = DiskImageDataset(TEST_DATAPATH, TASK, end=-1)
"""
we have to pass 5 parameters to train_loop
    1. dataloder
    2. model
    3. loss function
    4. optimizer
    5. epoch number
"""

# specify dataloader for train and test
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

# initialize model
model = AlexNet(OUTFEATURES[TASK])
model.cuda()

# initialize loss function
criterion = nn.CrossEntropyLoss()

# initialize optimizer
optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.00001)

print("Training model ======>")
for epoch in range(MAX_EPOCH):
    cur_loss = generic_train_loop(train_loader, model, criterion, optim, epoch + 1)
    torch.save(model.state_dict(), os.path.join(checkpoints_path, f"model_{epoch + 1}.pth"))
    val_loss = generic_eval_loop(test_loader, criterion, model)
    
