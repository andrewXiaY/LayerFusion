
import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable

def generic_train_loop(train_loader, model, criterion, optimizer, i_epoch, task="long"):
    model.train()
    loss_ = []
    for i_batch, batch in enumerate(train_loader):
        batch["data"] = Variable(batch["data"]).cuda()
        if task != "long":
            batch["label"] = Variable(batch["label"]).cuda()
        else:
            batch["label"] = Variable(batch["label"]).cuda().long()

        out = model(batch["data"])["output"]
        loss = criterion(out, batch["label"].data)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_.append(loss.item())

        if i_batch % 10 == 0 and i_batch != 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                i_epoch, i_batch * len(batch["data"]), len(train_loader.dataset),
                100. * i_batch / len(train_loader), loss.item()))
    return loss_