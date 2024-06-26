import os
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

# Check for CUDA availability
if torch.cuda.is_available():
    print("CUDA is available. Listing GPUs:")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available.")

# Define the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Define the custom dataset class
class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.classes = ['car', 'dog', 'person', 'bicycle']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "Annotations"))))
        self.valid_indices = self._filter_valid_indices()

    def _filter_valid_indices(self):
        valid_indices = []
        for idx in range(len(self.images)):
            annotation_path = os.path.join(self.root, "Annotations", self.annotations[idx])
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            boxes = []
            for obj in root.iter("object"):
                xmin = int(obj.find("bndbox").find("xmin").text)
                ymin = int(obj.find("bndbox").find("ymin").text)
                xmax = int(obj.find("bndbox").find("xmax").text)
                ymax = int(obj.find("bndbox").find("ymax").text)
                if xmin < xmax and ymin < ymax:
                    boxes.append([xmin, ymin, xmax, ymax])
            if len(boxes) > 0:
                valid_indices.append(idx)
        return valid_indices

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        img_path = os.path.join(self.root, "JPEGImages", self.images[actual_idx])
        annotation_path = os.path.join(self.root, "Annotations", self.annotations[actual_idx])

        img = Image.open(img_path).convert("RGB")
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []
        for obj in root.iter("object"):
            label = self.class_to_idx[obj.find("name").text]
            xmin = int(obj.find("bndbox").find("xmin").text)
            ymin = int(obj.find("bndbox").find("ymin").text)
            xmax = int(obj.find("bndbox").find("xmax").text)
            ymax = int(obj.find("bndbox").find("ymax").text)
            if xmin < xmax and ymin < ymax:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.valid_indices)

# Define the transforms
import torchvision.transforms as T

def get_transform(train):
    return T.ToTensor()

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None and b[1] is not None]
    if len(batch) == 0:
        return [], []
    return tuple(zip(*batch))

# Initialize the model
model = fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
num_classes = 4  # 4 classes including background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

# Load the dataset
dataset = VOCDataset(root='Flir_ADAS.v2i.voc/train', transforms=get_transform(train=True))
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)

# Define the optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     for images, targets in data_loader:
#         if len(images) == 0:  # Skip if no valid images in the batch
#             continue
#         images = [image.to(device) for image in images]
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#         loss_dict = model(images, targets)
#         losses = sum(loss for loss in loss_dict.values())
#         optimizer.zero_grad()
#         losses.backward()
#         optimizer.step()
#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {losses.item()}")
#     lr_scheduler.step()
#
# # Save the model
# torch.save(model.state_dict(), "faster_rcnn_model.pth")

# Load the saved model weights
model.load_state_dict(torch.load("faster_rcnn_model.pth", map_location=device))
model.to(device)

# Load the test dataset
test_dataset = VOCDataset(root='Flir_ADAS.v2i.voc/test', transforms=get_transform(train=False))
test_data_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)

# Evaluation functions
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)
    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_ - x1_ + 1) * (y2_ - y1_ + 1)
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area
    return iou

def calculate_ap_from_precision_recall(precision, recall):
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])
    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap

def calculate_map(true_boxes, pred_boxes, iou_threshold=0.5):
    average_precisions = []
    for label in range(len(true_boxes)):
        true_boxes_label = true_boxes[label]
        pred_boxes_label = pred_boxes[label]
        if len(pred_boxes_label) == 0:
            continue
        pred_boxes_label = np.array(sorted(pred_boxes_label, key=lambda x: x[4], reverse=True))
        tp = np.zeros(len(pred_boxes_label))
        fp = np.zeros(len(pred_boxes_label))
        num_true_boxes = len(true_boxes_label)
        for i, pred_box in enumerate(pred_boxes_label):
            if len(pred_box) < 5:
                continue
            pred_bbox = pred_box[:4]
            pred_score = pred_box[4]
            best_iou = 0.0
            best_match = -1
            for j, true_box in enumerate(true_boxes_label):
                iou = calculate_iou(pred_bbox, true_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match = j
            if best_iou >= iou_threshold and best_match != -1:
                if len(true_boxes_label) > best_match:
                    tp[i] = 1
                    true_boxes_label.pop(best_match)
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + np.finfo(np.float64).eps)
        recall = tp_cumsum / num_true_boxes
        ap = calculate_ap_from_precision_recall(precision, recall)
        average_precisions.append(ap)
    map_value = np.mean(average_precisions)
    return map_value

# Prepare for evaluation
model.eval()
true_boxes = [[] for _ in range(num_classes)]
pred_boxes = [[] for _ in range(num_classes)]

with torch.no_grad():
    for images, targets in test_data_loader:
        if len(images) == 0:
            continue
        images = [image.to(device) for image in images]
        outputs = model(images)
        for i, output in enumerate(outputs):
            target = targets[i]
            for label in range(num_classes):
                true_boxes_label = target['boxes'][target['labels'] == label].cpu().numpy().tolist()
                true_boxes[label].extend(true_boxes_label)
                pred_mask = output['labels'] == label
                pred_boxes_with_scores = [np.append(output['boxes'][i].detach().cpu().numpy(), output['scores'][i].detach().cpu().numpy()) for i in np.where(pred_mask.cpu())[0]]
                pred_boxes[label].extend(pred_boxes_with_scores)

# Calculate mAP
iou_threshold = 0.5
map_value = calculate_map(true_boxes, pred_boxes, iou_threshold)
print(f"mAP: {map_value:.4f}")

# Visualization
def visualize_image(image, boxes, labels):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=2)
        draw.text((xmin, ymin), str(label.item()), fill="red", font=font)

    return image

# Load a test image and visualize the predictions
test_image, _ = test_dataset[0]

model.eval()
with torch.no_grad():
    prediction = model([test_image.to(device)])[0]

# Convert the test image tensor to a PIL image
image_with_boxes = visualize_image(F.to_pil_image(test_image), prediction['boxes'].cpu(), prediction['labels'].cpu())

# Save the image with bounding boxes
output_path = 'test_image_with_boxes.jpg'
image_with_boxes.save(output_path)
print(f"Image saved at {output_path}")