import torchvision
import torch.nn as nn

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator

class PreMaskRCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # load a Pre-trained Mask R-CNN model on  COCO dataset
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        
        # For replacing the classifier (last layer) => 2 (background + pedestrian)
        self.num_classes = num_classes

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # Replace the pre-trained head with new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256

        # and replace the mask predictor with a new one
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            self.num_classes
            )

    def forward(self, imgs, targets=None):
        if targets:
            return self.model(imgs, targets)
        else:
            return self.model(imgs)
    

class MaskRCNN_MobileNet_v2_bn(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.num_classes = num_classes
        # Load pretrained mobilenet_v2 and extract features
        backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
        backbone.out_channels = 1280

        # Make RPN generate 5 x 3
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        # Define the feature maps that we will use to perform the roi cropping
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )

        # Put the pieces together
        self.model = MaskRCNN(
            backbone,
            num_classes=self.num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

    def forward(self, imgs, targets=None):
        if targets:
            return self.model(imgs, targets)
        else:
            return self.model(imgs)