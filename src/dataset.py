import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset

class PotholeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, "images")
        self.ann_dir    = os.path.join(root_dir, "annotations")
        self.transform = transform

        # find all images
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith((".png"))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        img_path = os.path.join(self.images_dir, img_name)

        base = os.path.splitext(img_name)[0]
        ann_path = os.path.join(self.ann_dir, base + ".xml")

        # load image
        img = Image.open(img_path).convert("RGB")

        # read bboxes from VOC-XML
        boxes = []
        tree = ET.parse(ann_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            cls = obj.find("name").text
            bnd = obj.find("bndbox")

            xmin = int(bnd.find("xmin").text)
            ymin = int(bnd.find("ymin").text)
            xmax = int(bnd.find("xmax").text)
            ymax = int(bnd.find("ymax").text)

            boxes.append([cls, xmin, ymin, xmax, ymax])

        if self.transform:
            img, boxes = self.transform(img, boxes)

        return img, boxes
    
