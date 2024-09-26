import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms, ops
import torchmetrics
import pytorch_lightning as pl
from od_datasets import TomatoDataset

# hyperparams
in_channels = 3
num_classes = 4
learning_rate = 0.001
batch_size = 32
num_epochs = 10

image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])

def collate_fn(batch):
    return tuple(zip(*batch))

# train_dataset = datasets.ImageFolder("/Users/pepe/dev/upb/topics/datasets/cats/train", transform=image_transforms)
train_dataset = TomatoDataset(
    root="/home/pepe/dev/upb/topics/tomato/images/train",
    annotation="/home/pepe/dev/upb/topics/tomato/annotations/tomatOD_train.json",
    transforms=image_transforms,
    )
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=24,
    persistent_workers=True,
    collate_fn=collate_fn
)
# test_dataset = datasets.ImageFolder("/Users/pepe/dev/upb/topics/datasets/cats/test", transform=image_transforms)
test_dataset = TomatoDataset(
    root="/home/pepe/dev/upb/topics/tomato/images/val",
    annotation="/home/pepe/dev/upb/topics/tomato/annotations/tomatOD_test.json",
    transforms=image_transforms,
    )
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=24,
    persistent_workers=True,
    collate_fn=collate_fn
)


class MyRetinaNet(pl.LightningModule):
    def __init__(self, num_classes, freeze_backbone=False):
        super().__init__()
        self.weights = torchvision.models.detection.RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights=self.weights)
        in_features = self.model.backbone.out_channels
        num_anchors = self.model.head.classification_head.num_anchors
        self.model.head.classification_head.num_classes = num_classes
        cls_logits = torch.nn.Conv2d(in_features, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(cls_logits.weight, std=0.01)
        torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))
        self.model.head.classification_head.cls_logits = cls_logits
        # self.preprocess = self.weights.transforms()
        self.test_step_outputs = []

    def forward(self, x, target=None):
        return self.model(x, target)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        images = torch.stack(images).float()
        outputs = self.model(images, targets)
        # Calculate Total Loss
        # loss = self.compute_loss(outputs, targets)
        loss = sum(outputs.values())

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        images = torch.stack(images).float()
        outputs = self.model(images, targets)
        # Calculate Total Loss
        loss = self.compute_loss(outputs, targets)
        loss = torch.as_tensor(loss)
        logs = {"val_loss": loss}
        return {"val_loss": loss, "log": logs}

    def compute_loss(self, outputs, targets):
        total_loss = 0
        for out, target in zip(outputs, targets):
            # Extract predictions
            try:
                pred_boxes = out['boxes']
                pred_scores = out['scores']
                pred_labels = out['labels']
            except TypeError:
                print(out, target)
            # Extract ground truth
            true_boxes = target['boxes']
            # true_scores = target['scores']
            true_labels = target['labels'].float()

            # Use a matching algorithm to assign each ground truth box to the most appropriate predicted box
            iou_matrix = ops.box_iou(pred_boxes, true_boxes)
            # Check if there are any matched indices
            if iou_matrix.numel() == 0:
                # No matches found, create a background class for each ground truth box
                num_true_boxes = true_boxes.size(0)
                background_labels = torch.zeros(num_true_boxes, dtype=torch.long, device=pred_labels.device)

                # Assume that background class is index 0, adjust as needed
                matched_boxes = true_boxes
                matched_scores = torch.zeros(num_true_boxes, dtype=torch.float32, device=pred_scores.device)
                matched_labels = background_labels

                # Compute localization loss (e.g., smooth L1 loss) with background boxes
                localization_loss = F.smooth_l1_loss(pred_boxes, matched_boxes)

                # Compute classification loss (e.g., CrossEntropy loss) with background labels
                classification_loss = F.cross_entropy(pred_scores, matched_labels)

                # You can adjust the weights for the two components based on your needs
                total_loss = localization_loss + self.hyperparameters['alpha'] * classification_loss

                return total_loss
            # Use a matching algorithm to assign each ground truth box to the most appropriate predicted box
            matched_indices = iou_matrix.argmax(dim=0)

            # Matched predictions
            matched_boxes = pred_boxes[matched_indices]
            matched_scores = pred_scores[matched_indices]
            # matched_labels = pred_labels[matched_indices]

            # Compute localization loss (e.g., smooth L1 loss)
            localization_loss = F.smooth_l1_loss(matched_boxes, true_boxes)

            # Compute classification loss (e.g., CrossEntropy loss)
            classification_loss = F.cross_entropy(matched_scores, true_labels)

            # Combine classification and regression losses
            alpha = 0.5  # You may need to adjust this hyperparameter
            # You can adjust the weights for the two components based on your needs
            total_loss += localization_loss + alpha * classification_loss

        return total_loss

    def test_step(self, batch, batch_idx, *args, **kwargs):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        images = torch.stack(images).float()
        outputs = self.model(images, targets)
        # Calculate Total Loss
        loss = self.compute_loss(outputs, targets)
        loss = torch.as_tensor(loss)
        self.test_step_outputs.append(loss)
        return {"loss": loss}

    def on_test_epoch_end(self):
        epoch_average = torch.stack(self.test_step_outputs).mean()
        self.log("test_epoch_average", epoch_average)
        # self.test_evaluator.accumulate()
        # self.test_evaluator.summarize()
        # metric = self.test_evaluator.coco_eval["bbox"].stats[0]
        # metric = torch.as_tensor(metric)
        # logs = {"AP": metric}
        self.test_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-4)


model = MyRetinaNet(num_classes=num_classes)
x = torch.randn(1, 3, 224, 224)
model.eval()
print(model(x))

torch.set_float32_matmul_precision('high')
trainer = pl.Trainer(
    accelerator="gpu", # para GPUs Nvidia: "gpu"
    devices=1,
    min_epochs=1,
    max_epochs=num_epochs,
    precision="bf16-mixed"
)
trainer.fit(model, train_loader, test_loader)

trainer.test(model, test_loader)