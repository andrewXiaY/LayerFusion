
import torch

def generic_eval_loop(val_loader, criterion, model, task="long"):
    model.eval()
    validation_loss, correct = 0, 0
    
    for ind, batch in enumerate(val_loader):
        batch["data"] = batch["data"].cuda()
        if task == "long":
            batch["label"] = batch["label"].cuda().long()
        else:
            batch["label"] = batch["label"].cuda()

        out = model(batch["data"])["output"]
        validation_loss += criterion(out, batch["label"].data).item()
        pred = out.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        if task == "long":
            correct += pred.eq(batch["label"].data.view_as(pred)).cpu().sum()
    
    validation_loss /= ind 
    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return validation_loss