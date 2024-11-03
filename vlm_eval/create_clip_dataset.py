from torch.utils.data import Dataset
from torch.utils.data import Subset
import torch
from coco_cf_loader import COCO_CF_dataset, MS_COCO_dataset





def main():

    torch.manual_seed(42)
    # Intialising seeds for data
    data_seeds = [i for i in range(107,122)]

    ms_coco_base_dir = "/home/htc/kchitranshi/SCRATCH/MS_COCO"
    apgd_attacks_base_dir = "/home/htc/kchitranshi/SCRATCH/APGD_SAMPLES"
    
