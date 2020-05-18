import torch
import torchvision.models as models
import torchvision.transforms as transforms
from copy import deepcopy
import numpy as np


class DeepFeatureExtractor:
    def __init__(self, model=None, gpu=False):
        self.model = None
        self.layers = []
        self.model_type = model
        self.gpu = gpu

        self.resizer = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

        self.supported_models = ['vgg16', 'vgg19', 'resnet50', 'resnet101']

        self.layer_names = {}

        if model in self.supported_models:
            if model == 'vgg16':
                self.model = models.vgg16()
            if model == 'vgg19':
                self.model = models.vgg19()
            if model == 'resnet50':
                self.model = models.resnet50()
            if model == 'resnet101':
                self.model = models.resnet101()
        else:
            err = "Model {} not available. Available models: {}".format(model, self.supported_models)
            raise ValueError(err)

        if model == "vgg16" or model == "vgg19":
            for layer, idx in [("fc1", 0), ("fc2", 3)]:
                l = self.model._modules.get("classifier")[idx]
                self.layer_names[str(l)] = layer
                self.layers.append(l)

        if model == "resnet50" or model == "resnet101":
            l = self.model._modules.get("avgpool")
            self.layer_names[str(l)] = "avgpool"
            self.layers.append(l)

        self.model.eval()

        self.device = "cpu"
        if self.gpu:
            self.device = "cuda"
            self.model.cuda()

        # set the hooks
        self.outputs = {}

        def hook(module, input, output):

            if self.gpu:
                self.outputs[str(module)] = output.cpu().squeeze().detach().numpy()
            else:
                self.outputs[str(module)] = output.squeeze().detach().numpy()

        for layer in self.layers:
            layer.register_forward_hook(hook)

    def set_model(self, model):
        self.model_type = model

        if model in self.supported_models:
            if model == 'vgg16':
                self.model = models.vgg16()
            if model == 'vgg19':
                self.model = models.vgg19()
            if model == 'resnet50':
                self.model = models.resnet50()
            if model == 'resnet101':
                self.model = models.resnet101()
        else:
            err = "Model {} not available. Available models: {}".format(model, self.supported_models)
            raise ValueError(err)

        self.device = "cpu"
        if self.gpu:
            self.device = "cuda"
            self.model.cuda()

    def set_layers(self, layers):
        self.layers.clear()

        self.layer_names = {}

        for layer in layers:
            l = self.model._modules.get(layer)
            self.layer_names[str(l)] = layer
            if l is None:
                err = "Layer {} not available in model {}.".format(layer, self.model_type)
                raise ValueError(err)

            self.layers.append(l)

        # set the hooks
        self.outputs = {}

        def hook(module, input, output):
            if module not in self.outputs:
                self.outputs[module] = []

            self.outputs[module].append(output.squeeze().detach().numpy())

        for layer in self.layers:
            layer.register_forward_hook(hook)

    def get_features(self, images):

        t_imgs = []

        for img in images:
            t_img = self.normalize(self.to_tensor(self.resizer(img)))
            t_imgs.append(t_img.to(self.device))

        t_imgs = torch.stack(t_imgs)

        out = self.model(t_imgs)

        features = deepcopy(self.outputs)
        renamed_features = {}
        for k, v in features.items():
            renamed_features[self.layer_names[k]] = np.array(v)

        self.outputs = {}

        return renamed_features
