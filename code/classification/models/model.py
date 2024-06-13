"""
This file uses torchvision pretrained models, but modifies the final fc layer to the number of classes
"""
import torch
import torch.nn as nn 

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_32, ViT_B_32_Weights
from torchvision.models import vgg19_bn, VGG19_BN_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import swin_b, Swin_B_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models import maxvit_t, MaxVit_T_Weights
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

from transformers import CvtForImageClassification

class CvTForCustomClassification(nn.Module):
    def __init__(self, num_classes, model_name='microsoft/cvt-13'):
        super(CvTForCustomClassification, self).__init__()
        self.model = CvtForImageClassification.from_pretrained(model_name)
        self.model.classifier = torch.nn.Linear(in_features=384, out_features=num_classes, bias=True)

    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
        logits = outputs.logits
        return logits

from transformers import MobileViTForImageClassification
class MobileViTForCustomClassification(nn.Module):
    def __init__(self, num_classes, hf_model_name='apple/mobilevit-x-small'):
        super(MobileViTForCustomClassification, self).__init__()
        self.model = MobileViTForImageClassification.from_pretrained(hf_model_name)
        self.model.classifier = torch.nn.Linear(in_features=384, out_features=num_classes, bias=True)

    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
        logits = outputs.logits
        return logits

from transformers import MobileViTV2ForImageClassification
class MobileViTv2ForCustomClassification(nn.Module):
    def __init__(self, num_classes, hf_model_name='apple/mobilevitv2-1.0-imagenet1k-256'):
        super(MobileViTv2ForCustomClassification, self).__init__()
        self.model = MobileViTV2ForImageClassification.from_pretrained(hf_model_name)
        self.model.classifier = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
        logits = outputs.logits
        return logits

from transformers import RegNetForImageClassification
class RegNetForCustomClassification(nn.Module):
    def __init__(self, num_classes, hf_model_name='facebook/regnet-y-004'):
        super(RegNetForCustomClassification, self).__init__()
        self.model = RegNetForImageClassification.from_pretrained(hf_model_name)
        self.model.classifier[1] = torch.nn.Linear(in_features=440, out_features=num_classes, bias=True)

    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
        logits = outputs.logits
        return logits

from transformers import DeiTForImageClassification
class DeiTForCustomClassification(nn.Module):
    def __init__(self, num_classes, hf_model_name='facebook/deit-small-distilled-patch16-224'):
        super(DeiTForCustomClassification, self).__init__()
        self.model = DeiTForImageClassification.from_pretrained(hf_model_name)
        self.model.classifier = torch.nn.Linear(in_features=384, out_features=num_classes, bias=True)

    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
        logits = outputs.logits
        return logits

from transformers import PvtV2ForImageClassification
class PvtV2ForCustomClassification(nn.Module):
    def __init__(self, num_classes, hf_model_name='OpenGVLab/pvt_v2_b0'):
        super(PvtV2ForCustomClassification, self).__init__()
        self.model = PvtV2ForImageClassification.from_pretrained(hf_model_name)
        self.model.classifier = torch.nn.Linear(in_features=256, out_features=num_classes, bias=True)

    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
        logits = outputs.logits
        return logits

from transformers import SwinForImageClassification
class SwinForCustomClassification(nn.Module):
    def __init__(self, num_classes, hf_model_name='microsoft/swin-base-patch4-window7-224-in22k'):
        super(SwinForCustomClassification, self).__init__()
        self.model = SwinForImageClassification.from_pretrained(hf_model_name)
        self.model.classifier = torch.nn.Linear(in_features=1024, out_features=num_classes, bias=True)

    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
        logits = outputs.logits
        return logits

def get_custom_model(
    model_name:str,
    num_classes:int,
    pretrained:bool=True
):
    if model_name == 'resnet18':
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
            model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        else:
            model = resnet18()
            model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    elif model_name == 'resnet34':
        if pretrained:
                weights = ResNet34_Weights.DEFAULT
                model = resnet34(weights=weights)
                model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        else:
            model = resnet34()
            model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    elif model_name == "resnet50":
        if pretrained:
                weights = ResNet50_Weights.DEFAULT
                model = resnet50(weights=weights)
                model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        else:
            model = resnet50()
            model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    elif model_name == 'vit_b_32':
        if pretrained:
            weights = ViT_B_32_Weights.DEFAULT
            model = vit_b_32(weights=weights)
            model.heads = nn.Linear(in_features=768, out_features=num_classes, bias=True)
        else:
            model = vit_b_32()
            model.heads = nn.Linear(in_features=768, out_features=num_classes, bias=True)
    elif model_name == 'vit_b_16':
        if pretrained:
            weights = ViT_B_16_Weights.DEFAULT
            model = vit_b_16(weights=weights)
            model.heads = nn.Linear(in_features=768, out_features=num_classes, bias=True)
        else:
            model = vit_b_32()
            model.heads = nn.Linear(in_features=768, out_features=num_classes, bias=True)
    elif model_name == 'vgg19':
        if pretrained:
            weights = VGG19_BN_Weights.DEFAULT
            model = vgg19_bn(weights=weights)
            model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        else:
            model = vgg19_bn()
            model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
    elif model_name == 'swin_b':
        if pretrained:
            weights = Swin_B_Weights.DEFAULT
            model = swin_b(weights=weights)
            model.head = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        else:
            model = swin_b()
            model.head = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
    elif model_name == 'inception_v3':
        if pretrained:
            weights = Inception_V3_Weights.DEFAULT
            model = inception_v3(weights=weights)
            model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        else:
            model = inception_v3()
            model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    elif model_name == 'convnext_base':
        if pretrained:
            weights = ConvNeXt_Base_Weights.DEFAULT
            model = convnext_base(weights=weights)
            model.classifier[2] = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        else:
            model = convnext_base()
            model.classifier[2] = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
    elif model_name == 'efficientnet_v2_m':
        if pretrained:
            weights = EfficientNet_V2_M_Weights.DEFAULT
            model = efficientnet_v2_m(weights=weights)
            model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        else:
            model = efficientnet_v2_m()
            model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    elif model_name == 'mobilenet_v3_large':
        if pretrained:
            weights = MobileNet_V3_Large_Weights.DEFAULT
            model = mobilenet_v3_large(weights=weights)
            model.classifier[3] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        else:
            model = mobilenet_v3_large()
            model.classifier[3] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    elif model_name == 'maxvit_t':
        if pretrained: 
            weights = MaxVit_T_Weights.DEFAULT
            model = maxvit_t(weights=weights)
            model.classifier[5] = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        else:
            model = maxvit_t()
            model.classifier[5] = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    elif model_name == 'resnext50_32x4d':
        if pretrained:
            weights = ResNeXt50_32X4D_Weights.DEFAULT
            model = resnext50_32x4d(weights=weights)
            model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        else:
            model = resnext50_32x4d()
            model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    elif model_name == 'cvt_13':
        hf_model_name = 'microsoft/cvt-13'
        if pretrained:
            model = CvTForCustomClassification(num_classes, hf_model_name)
    elif model_name == 'mobile_vit_xs':
        hf_model_name = 'apple/mobilevit-x-small'
        if pretrained:
            model = MobileViTForCustomClassification(num_classes, hf_model_name)
    elif model_name == 'mobile_vit_v2':
        hf_model_name = 'apple/mobilevitv2-1.0-imagenet1k-256'
        if pretrained:
            model = MobileViTv2ForCustomClassification(num_classes, hf_model_name)
    elif model_name == 'regnet_y':
        hf_model_name = 'facebook/regnet-y-004'
        if pretrained:
            model = RegNetForCustomClassification(num_classes, hf_model_name)
    elif model_name == 'diet_distilled_s':
        hf_model_name = 'facebook/deit-small-distilled-patch16-224'
        if pretrained:
            model = DeiTForCustomClassification(num_classes, hf_model_name)
    elif model_name == 'pvt_v2':
        hf_model_name = 'OpenGVLab/pvt_v2_b0'
        if pretrained:
            model = PvtV2ForCustomClassification(num_classes, hf_model_name)
    elif model_name == 'swinb_22k':
        hf_model_name = 'microsoft/swin-base-patch4-window7-224-in22k'
        if pretrained:
            model = SwinForCustomClassification(num_classes, hf_model_name)
            

    return model 