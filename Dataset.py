import os
import torch

from torch.utils.data import Dataset
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.io import read_image

root = '../../Data/PennFudanPed'

class PennFudanDataset(Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        self.img_path = root+'/PNGImages'
        self.msk_path = root+'/PedMasks'
        self.img_list = sorted(os.listdir(self.img_path))
        self.msk_list = sorted(os.listdir(self.msk_path))
    
    def __getitem__(self, idx):
        img = read_image(os.path.join(self.img_path, self.img_list[idx]))
        msk = read_image(os.path.join(self.msk_path, self.msk_list[idx]))

        # Extract unique object IDs from mask image
        obj_ids = torch.unique(msk)

        # The 1st ID is background, so it is removed
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # Split the mask to binary masks according to id value
        msks = (msk == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # Get bounding box coordinate for each mask
        boxes = masks_to_boxes(msks)

        # Only one class (Pedestrian) is considered, with background already removed. 
        # All remaining masks belong to the Pedestrian class.
        lbls = torch.ones((num_objs,), dtype=torch.int64)

        img_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap the img and target with torchvision tensor
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY",
                                                   canvas_size = F.get_size(img))
        target["masks"] = tv_tensors.Mask(msks)
        target["labels"] = lbls
        target["image_id"] = img_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.img_list)

