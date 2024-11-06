# Code adapted from https://github.com/openai/CLIP/blob/main/
from transformers import CLIPProcessor, CLIPModel
import argparse



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, choices=['MS_COCO','medium','base','all'], help='Data on which clip was fine-tuned')
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100", "ImageNet", "Caltech101", "Caltech256", "Food101"])
    args = parser.parse_args()
    


if __name__ == "__main__":
    main()