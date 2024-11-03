from random import seed, shuffle
from typing import Callable
import transformers
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel




class ModelTrainer:

    def __init__(self, 
                 model: Callable, 
                 processor: Callable,
                 data_name: str,
                 data_loader: torch.utils.data.DataLoader, 
                 num_epochs: int, 
                 learning_rate: float = 5e-7, 
                 weight_decay: float = 1e-3,
                 device: str = "cuda:0",
                 save_model: bool = False,
                 data_seed: int = 42
    ) -> None:

        self.model = model
        self.processor = processor
        self.data_name = data_name
        self.data_loader = data_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.save_model = save_model
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
            for batch_idx, batch in enumerate(self.data_loader):
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
            torch.save(self.model.state_dict(), f'clip_model_dataset_{self.data_name}_num_epochs_{self.num_epochs}_data_{self.data_name}_data_seed_{self.data_seed}.pt')


def main():
    import os
    os.environ['HF_HOME'] = '/home/htc/kchitranshi/SCRATCH/'    
    
    from torch.utils.data import DataLoader
    from coco_cf_loader import MS_COCO_dataset, custom_collate_fn

    torch.manual_seed(43)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    data = MS_COCO_dataset(base_dir='/home/htc/kchitranshi/SCRATCH/MS_COCO/')
    data_loader = DataLoader(data, 
                             batch_size=128,
                             collate_fn=custom_collate_fn,
                             shuffle=True,
    )
    
    trainer = ModelTrainer(model=model, 
                           processor=processor, 
                           data_name="MS_COCO", 
                           data_loader=data_loader, 
                           num_epochs=20, 
                           learning_rate=5e-7, 
                           weight_decay=1e-3,
                           device=device,
                           data_seed=42
    )

    trainer.train()


if __name__ == "__main__":
    main()