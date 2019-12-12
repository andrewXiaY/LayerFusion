import torch.nn as nn
import torch
from models.alex_net import AlexNet
from models.vgg_a import VGG_A

models = {'alex_net' : AlexNet,
          'vgg_a' : VGG_A
          }

def load_model(params):
    model_name, path, out_features = params["model_name"], params["path"], params["out_features"]
    model =  models[model_name](out_features)
    model.load_state_dict(torch.load(path))
    model.eval()
    model.cuda()
    return model

class FusionNet(nn.Module):
    def __init__(self, parameters, classes_num=10):
        super(FusionNet, self).__init__()
        self.frozen_models = []

        for params in parameters:
            model = load_model(params)

            # freeze parameters of pretrained model
            for ps in model.parameters():
                ps.requires_grad = False
            self.frozen_models.append(model)
        
        self.fc1 = nn.Sequential(
            nn.Linear(4096 * len(self.frozen_models), 4096),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True)
        )
	
        self.fc4 = nn.Sequential(nn.Linear(512, classes_num))

        self.fc_layers = [self.fc1, self.fc2, self.fc3, self.fc4]


    def forward(self, x):
        output = {}
        frozen_layer = []
        for m in self.frozen_models:
            frozen_layer.append(m(x)["flatten"])

        x = torch.cat(frozen_layer, dim=1)
        output["flatten"] = x

        for fc in self.fc_layers:
            x = fc(x)
        output["output"] = x

        return output

