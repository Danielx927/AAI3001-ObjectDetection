import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_faster_rcnn_model(num_classes: int):
    # base model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    )

    # replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
