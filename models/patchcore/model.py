import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class PatchCoreFeatureExtractor(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        backbone = models.wide_resnet50_2(weights=weights)

        self.layer0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        for param in self.parameters():
            param.requires_grad = False

        self.device = device
        self.to(device)
        self.eval()

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        feat2 = self.layer2(x)                                              # (B, 512,  H/8,  W/8)
        feat3 = self.layer3(feat2)                                          # (B, 1024, H/16, W/16)
        feat3 = F.interpolate(feat3, size=feat2.shape[-2:],
                              mode='bilinear', align_corners=False)         # (B, 1024, H/8,  W/8)
        features = torch.cat([feat2, feat3], dim=1)                        # (B, 1536, H/8,  W/8)
        features = F.avg_pool2d(features, kernel_size=3, stride=1, padding=1)  # locally aware
        return features

    def extract(self, x):
        """patch_features (B*H*W, C) 와 spatial shape (H, W) 반환"""
        x = x.to(self.device)
        with torch.no_grad():
            features = self.forward(x)                  # (B, 1536, H, W)
        B, C, H, W = features.shape
        patch_features = features.permute(0, 2, 3, 1).reshape(B * H * W, C)
        return patch_features.cpu().numpy(), (H, W)
