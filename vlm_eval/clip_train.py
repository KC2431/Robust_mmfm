from random import seed, shuffle
from typing import Callable
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel




class ModelTrainer:

    def __init__(self, 
                 model: Callable, 
                 processor: Callable,
                 data_name: str,
                 train_data_loader: torch.utils.data.DataLoader, 
                 val_data_loader: torch.utils.data.DataLoader,
                 num_epochs: int, 
                 learning_rate: float = 5e-7, 
                 weight_decay: float = 1e-3,
                 device: str = "cuda:0",
                 save_model: bool = False,
                 save_model_path: str = "/home/htc/kchitranshi/SCRATCH/",
                 data_seed: int = 42
    ) -> None:

        self.model = model
        self.processor = processor
        self.data_name = data_name
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.save_model = save_model
        self.save_model_path = save_model_path
        self.data_seed = data_seed

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    
    def train(self):
        self.model.train()
        progress_bar = tqdm(range(self.num_epochs))
        for epoch in progress_bar:
            for batch_idx, batch in enumerate(self.train_data_loader):
                self.optimizer.zero_grad()
                processed_input = self.processor(text=batch["caption"], 
                                                 images=batch["image"], 
                                                 return_tensors="pt", 
                                                 padding=True, 
                                                 max_length=77, 
                                                 truncation=True
                )
                outputs = self.model(input_ids=processed_input['input_ids'].squeeze().to(self.device),
                                     pixel_values=processed_input['pixel_values'].squeeze().to(self.device),
                                     attention_mask=processed_input['attention_mask'].squeeze().to(self.device),
                                     return_loss=True
                )
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch+1}/{self.num_epochs} Loss: {loss.item()}")
            progress_bar.set_postfix(
                epoch="{}/{}".format(epoch+1,self.num_epochs),
                loss=loss.item(),
                lr=self.learning_rate
            )

        if self.save_model:
            torch.save(self.model.state_dict(), self.save_model_path + f'clip_model_dataset_{self.data_name}_num_epochs_{self.num_epochs}_data_{self.data_name}_data_seed_{self.data_seed}.pt')
            print(f"Saving fine-tuned model as clip_model_dataset_{self.data_name}_num_epochs_{self.num_epochs}_data_{self.data_name}_data_seed_{self.data_seed}.pt")
    
    def eval(self):
        self.model.eval()
        nb_batches = len(self.val_data_loader)
        tqdm_object = tqdm(self.val_data_loader, total=len(self.val_data_loader))
        epoch_loss = 0.0   
        for i, batch in enumerate(tqdm_object):
            processed_input = self.processor(text=batch["caption"], 
                                                 images=batch["image"], 
                                                 return_tensors="pt", 
                                                 padding=True, 
                                                 max_length=77, 
                                                 truncation=True
                )
            outputs = self.model(
                input_ids=processed_input['input_ids'].squeeze(),
                attention_mask=processed_input['attention_mask'].squeeze(),
                pixel_values=processed_input['pixel_values'].squeeze(),
                return_loss=True)
            loss, logits_per_image = outputs.loss, outputs.logits_per_image 
            epoch_loss += loss.item()
            tqdm_object.set_postfix(
                batch="{}/{}".format(i+1,nb_batches),
                dev_loss=loss.item(),
                )
        epoch_loss = epoch_loss / nb_batches
        print(epoch_loss)

def main():
    import os
    os.environ['HF_HOME'] = '/home/htc/kchitranshi/SCRATCH/'    
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--data_seed', type=int, default=42)
    parser.add_argument('--data_name', type=str, default="MS_COCO", choices=["MS_COCO","base","medium","all"])
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--save_model_path', type=str, default="/home/htc/kchitranshi/SCRATCH/")
    parser.add_argument(
        "--data_seeds",
        nargs="+",
        type=int,
        default=[107],
        help="Seeds to use for each trial for picking demonstrations and eval sets",
    )
    args = parser.parse_args()

    from torch.utils.data import DataLoader
    from coco_cf_loader import MS_COCO_dataset, custom_collate_fn

    torch.manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


    for data_seed in args.data_seeds:
        print(f"Data Seed: {data_seed} Data Name: {args.data_name}")
        dataset = MS_COCO_dataset(base_dir='/home/htc/kchitranshi/SCRATCH/Datasets/MS_COCO_APGD_4', annotation_file=f'/json_files/data_name_{args.data_name}_data_seed_{data_seed}.json')

        train_size = int(0.8 * len(dataset))  # 80% for training
        val_size = len(dataset) - train_size   # 20% for validation

        # Randomly split into training and validation datasets
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Optional: Create DataLoaders for each subset
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=custom_collate_fn)

        trainer = ModelTrainer(model=model, 
                            processor=processor, 
                            data_name=args.data_name, 
                            train_data_loader=train_loader,
                            val_data_loader=val_loader, 
                            num_epochs=args.num_epochs, 
                            learning_rate=5e-7, 
                            weight_decay=1e-3,
                            device=device,
                            data_seed=args.data_seed,
                            save_model=args.save_model
        )

        trainer.train()


if __name__ == "__main__":
    main()