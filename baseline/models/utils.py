
from models.deeplabv3.deeplabv3 import *
from models.deeplabv2 import *
import torchvision.models as models
import copy

def get_model(model_name:str, num_classes:int, output_dim:int):

    if model_name == 'deeplabv3p':
        model = DeepLabv3Plus(models.resnet101(pretrained=True), num_classes=num_classes, output_dim=output_dim)
    elif model_name == 'deeplabv2':
        model = DeepLabv2(models.resnet101(pretrained=True), num_classes=num_classes, output_dim=output_dim)
    return model

# --------------------------------------------------------------------------------
# Define EMA: Mean Teacher Framework
# --------------------------------------------------------------------------------
class EMA(object):
    def __init__(self, model, alpha):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha

    def update(self, model):
        decay = min(1 - 1 / (self.step + 1), self.alpha)
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        self.step += 1

if __name__ == '__main__':
    pass