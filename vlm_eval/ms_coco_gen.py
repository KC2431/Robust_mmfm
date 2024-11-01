import torchvision.datasets as dset
import torchvision.transforms as transforms
from coco_cf_loader import COCO_CF_dataset
from torch.utils.data import DataLoader

def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch

coco_2017 = dset.CocoCaptions(root='/home/htc/kchitranshi/SCRATCH/COCO_2017/val2017/',
                              annFile='/home/htc/kchitranshi/SCRATCH/COCO_2017/annotations/captions_val2017.json',
                              transforms=transforms.ToTensor())

coco_cf = COCO_CF_dataset(base_dir='/home/htc/kchitranshi/SCRATCH/COCO_CF/')
dl_coco_cf = DataLoader(coco_cf, batch_size=100,collate_fn=custom_collate_fn)


# Collect both captions from each batch in one step
coco_cf_captions = []

for batch in dl_coco_cf:
    # Extend the list with both captions at once without list comprehension
    coco_cf_captions.extend(batch['caption_0'])
    coco_cf_captions.extend(batch['caption_1'])
