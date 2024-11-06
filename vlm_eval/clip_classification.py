# Code adapted from https://github.com/openai/CLIP/blob/main/
from transformers import CLIPProcessor, CLIPModel
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def zeroshot_classifier(classnames, templates, processor, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to('cuda')
            class_embeddings = model.get_text_features(text_inputs['input_ids']) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def classification_collate_fn(batch):
    images, labels = zip(*batch)
    labels = torch.tensor(labels)
    return images, labels

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, choices=['MS_COCO','medium','base','all'], help='Data on which clip was fine-tuned')
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100", "ImageNet", "Caltech101", "Caltech256", "Food101"])
    args = parser.parse_args()
    
    if args.dataset == "CIFAR10":
        from datasets_classes_templates import CIFAR10_CLASSES_TEMPLATES as classes_templates
        from torchvision.datasets import CIFAR10
        data = CIFAR10(root='/software/ais2t/pytorch_datasets/cifar10/', train=False, download=False)
    elif args.dataset == "CIFAR100":
        from datasets_classes_templates import CIFAR100_CLASSES_TEMPLATES as classes_templates
        from torchvision.datasets import CIFAR100
        data = CIFAR100(root='/software/ais2t/pytorch_datasets/cifar100/', train=False, download=False)
    elif args.dataset == "ImageNet":
        from datasets_classes_templates import ImageNet_CLASSES_TEMPLATES as classes_templates
        from torchvision.datasets import ImageNet
        data = ImageNet(root='/software/ais2t/pytorch_datasets/imagenet/', split='val')
    elif args.dataset == "Caltech101":
        from datasets_classes_templates import Caltech101_CLASSES_TEMPLATES as classes_templates
        from torchvision.datasets import Caltech101
        data = Caltech101(root='/home/htc/kchitranshi/SCRATCH/', download=False)
    elif args.dataset == "Caltech256":
        from datasets_classes_templates import Caltech256_CLASSES_TEMPLATES as classes_templates
        from torchvision.datasets import Caltech256
        data = Caltech256(root='/home/htc/kchitranshi/SCRATCH/', download=False)
    elif args.dataset == "Food101":
        from datasets_classes_templates import Food101_CLASSES_TEMPLATES as classes_templates
        from torchvision.datasets import Food101
        data = Food101(root='/home/htc/kchitranshi/SCRATCH/', download=False, split='test')

    print(f'Conducting zero-shot image classification on {args.dataset}')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    data_loader = DataLoader(data, batch_size=128, collate_fn=classification_collate_fn)

    zeroshot_weights = zeroshot_classifier(classes_templates['classes'], 
                                           classes_templates['templates'], 
                                           processor, 
                                           model
    )

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for i, (images, target) in enumerate(tqdm(data_loader)):
            target = target.to(device)
            images = list(images)

            images = processor(images=images, return_tensors="pt").to(device)

            # predict
            image_features = model.get_image_features(images['pixel_values']).to(device)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += image_features.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100 

    print(f"Top-1 accuracy: {top1:.2f}")
    print(f"Top-5 accuracy: {top5:.2f}")



if __name__ == "__main__":
    main()