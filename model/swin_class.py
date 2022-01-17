import timm
import torch.nn as nn


class ClassSwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = self.load_backbone()
        self.head = self.load_head()

    def load_backbone(self):
        backbone = timm.create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=0)
        return backbone

    def load_head(self):
        num_features = self.backbone.num_features
        head = nn.Sequential(nn.Linear(num_features, 20))
        return head

    def foward(self, x):
        out = self.backbone(x)
        out = out.reshape(-1, self.backbone.num_features)
        out = self.head(out)
        return out
