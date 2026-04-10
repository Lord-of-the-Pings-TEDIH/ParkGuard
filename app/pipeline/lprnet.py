import torch
import torch.nn as nn

class LPRNet(nn.Module):
    """
    A lightweight Convolutional Recurrent Neural Network (CRNN) specialized for License Plates.
    This architecture is based on the famous LPRNet paper (2018).
    It completely bypasses character segmentation and reads the entire plate at once.
    """
    def __init__(self, num_classes: int, dropout_rate: float = 0.5):
        super(LPRNet, self).__init__()
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1), # Layer 1
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=(1, 1), padding=1),
            
            nn.Conv2d(64, 128, 3, stride=1, padding=1), # Layer 2
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=(2, 1), padding=1),
            
            nn.Conv2d(128, 256, 3, stride=1, padding=1), # Layer 3
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, 3, stride=1, padding=1), # Layer 4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=(2, 1), padding=1),
            nn.Dropout(dropout_rate),
            
            nn.Conv2d(256, 256, (3, 1), stride=1, padding=(1, 0)), # Layer 5
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.container = nn.Sequential(
            nn.Conv2d(256, num_classes, (1, 13), stride=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x is a batched image (B, 3, 24, 94)
        x = self.backbone(x)
        logits = self.container(x) # (B, num_classes, 1, 18)
        logits = torch.mean(logits, dim=2) # Collapse H dim: (B, num_classes, 18)
        return logits

def infer_lprnet_dummy(crop):
    """
    Dummy fallback function until LPRNet PyTorch weights are specifically trained and downloaded
    for Romanian font. In a fully trained environment, we load LPRNet.pth here.
    """
    return None, 0.0
