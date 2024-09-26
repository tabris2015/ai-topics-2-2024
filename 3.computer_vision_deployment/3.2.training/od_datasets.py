import os
import torch
import torchvision.transforms
from torch.utils.data import Dataset

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from PIL import Image
from torchvision.transforms.v2 import functional as F

from pycocotools.coco import COCO


class TomatoDataset(Dataset):
    def __init__(self, annotation, root, transforms):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]["file_name"]

        img = Image.open(os.path.join(self.root, path))
        num_objs = len(coco_annotation)

        # bounding boxes
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        areas = []
        for i in range(num_objs):
            # bbox
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])
            # labels
            labels.append(coco_annotation[i]["category_id"])
            # areas
            areas.append(coco_annotation[i]["area"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)

        img_id = torch.tensor([img_id])

        is_crowd = torch.zeros((num_objs,), dtype=torch.int64)

        my_annotation = {
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id,
            "area": areas,
            "iscrouw": is_crowd,
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


def get_transform():
    custom_transforms = [torchvision.transforms.ToTensor()]
    return torchvision.transforms.Compose(custom_transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    train_data_dir = "/home/pepe/dev/upb/topics/tomato/images/train"
    train_coco = "/home/pepe/dev/upb/topics/tomato/annotations/tomatOD_train.json"

    my_dataset = TomatoDataset(root=train_data_dir, annotation=train_coco, transforms=get_transform())
    batch_size = 1

    train_loader = torch.utils.data.DataLoader(
        my_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for imgs, annotations in train_loader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        print(annotations)
