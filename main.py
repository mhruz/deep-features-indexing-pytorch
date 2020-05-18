import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

from networks import DeepFeatureExtractor

if __name__ == "__main__":

    dfe = DeepFeatureExtractor("resnet50", ["avgpool"], gpu=False)

    img = Image.open(r"e:\kreatura.png")
    img2 = Image.open(r"f:\mywork\- hostee -\P2280003.JPG")

    feat = dfe.get_features([img, img])
    feat2 = dfe.get_features([img2, img2])
