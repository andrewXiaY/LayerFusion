from models.fusion_net_final  import FusionNet
import torch
import os
import torch.nn as nn
import torch.nn.functional as TF
from workflows.train_loop import generic_train_loop
from workflows.eval_loop import generic_eval_loop
from configs import * 
from datasets.dataset import DefaultDataset

TASK = "pretrained_classification_fusion_final_all"
MAX_EPOCH = 100
checkpoints_path = "./checkpoints/" + TASK

if not os.path.exists(checkpoints_path):
    os.mkdir(checkpoints_path)

parameters = [
    {"model_name": "alex_net", "path": "./checkpoints/ssl_rotate/model_20.pth", "out_features": 4},
    {"model_name": "alex_net", "path": "./checkpoints/ssl_jigsaw/model_20.pth", "out_features": 24},
    {"model_name": "alex_net", "path": "./checkpoints/ssl_box_blur/model_20.pth", "out_features": 10},
    #{"model_name": "alex_net", "path": "./checkpoints/ssl_noise_add/model_20.pth", "out_features": 4},
    {"model_name": "alex_net", "path": "./checkpoints/ssl_moveblur/model_20.pth", "out_features": 36}
]

print(f"Working Directory: {os.getcwd()}")
dataset_train = DefaultDataset(["./data/train_images_0.npy", "./data/train_labels_0.npy"])
dataset_test = DefaultDataset(["./data/test_images.npy", "./data/test_labels.npy"])

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

print("Initializing model")
model = FusionNet(parameters)
model.cuda()

# initialize loss function
criterion = nn.CrossEntropyLoss()

# initialize optimizer
#optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.00001)
optim = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training model ======>")
for epoch in range(MAX_EPOCH):
    cur_loss = generic_train_loop(train_loader, model, criterion, optim, epoch + 1)
    if((epoch + 1) % 20 == 0):
        torch.save(model.state_dict(), os.path.join(checkpoints_path, f"model_{epoch + 1}.pth"))
    val_loss = generic_eval_loop(test_loader, criterion, model)



