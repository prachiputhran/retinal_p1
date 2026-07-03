"""
model.py
Shared-backbone, multi-head architecture for joint DR + DME classification.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

import config

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class MultiLabelRetinalNet(nn.Module):
    """
    Shared CNN backbone with two heads:
      - dr_head:  DR grade (5 classes, ordinal 0-4)
      - dme_head: DME risk (3 classes, ordinal 0-2)

    Sharing the backbone forces a joint retinal feature representation
    instead of training two separate models — and mirrors the fact that
    DR and DME are clinically correlated conditions.
    """

    def __init__(self, backbone_name: str = config.BACKBONE, pretrained: bool = config.PRETRAINED,
                 num_dr_classes: int = config.NUM_DR_CLASSES, num_dme_classes: int = config.NUM_DME_CLASSES,
                 dropout: float = 0.3):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "resnet50":
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = tv_models.resnet50(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone

        elif backbone_name == "efficientnet_b4":
            if not TIMM_AVAILABLE:
                raise ImportError("pip install timm")
            backbone = timm.create_model("efficientnet_b4", pretrained=pretrained, num_classes=0)
            in_features = backbone.num_features
            self.backbone = backbone

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.dropout = nn.Dropout(dropout)
        self.dr_head = nn.Linear(in_features, num_dr_classes)
        self.dme_head = nn.Linear(in_features, num_dme_classes)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        features = self.dropout(features)
        return self.dr_head(features), self.dme_head(features)

    def get_target_layer(self):
        """Last conv layer, for Grad-CAM hook attachment."""
        if self.backbone_name == "resnet50":
            return self.backbone.layer4[-1]
        elif self.backbone_name == "efficientnet_b4":
            return self.backbone.conv_head
        raise ValueError(f"No target layer defined for {self.backbone_name}")


class MultiTaskLoss(nn.Module):
    """Weighted sum of DR and DME cross-entropy losses, with optional class weights."""

    def __init__(self, dr_weight: float = config.DR_LOSS_WEIGHT, dme_weight: float = config.DME_LOSS_WEIGHT,
                 dr_class_weights: torch.Tensor = None, dme_class_weights: torch.Tensor = None):
        super().__init__()
        self.dr_weight = dr_weight
        self.dme_weight = dme_weight
        self.dr_criterion = nn.CrossEntropyLoss(weight=dr_class_weights)
        self.dme_criterion = nn.CrossEntropyLoss(weight=dme_class_weights)

    def forward(self, dr_logits, dme_logits, dr_labels, dme_labels):
        dr_loss = self.dr_criterion(dr_logits, dr_labels)
        dme_loss = self.dme_criterion(dme_logits, dme_labels)
        total = self.dr_weight * dr_loss + self.dme_weight * dme_loss
        return total, dr_loss.item(), dme_loss.item()


if __name__ == "__main__":
    model = MultiLabelRetinalNet(backbone_name="resnet50", pretrained=False)
    dummy = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE)
    dr_out, dme_out = model(dummy)
    print(f"DR output: {dr_out.shape}, DME output: {dme_out.shape}")
    print(f"Target layer: {model.get_target_layer()}")