import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class SarcasmDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.inputs = list(data.inputs) 
        self.targets = list(data.targets)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_text = self.inputs[index]
        target_text = self.targets[index]
        return self.tokenizer(input_text, text_target=target_text, max_length=128, truncation=True) #TODO make max_length part of config dict
    
class SarcasmDataModule(pl.LightningDataModule):
    def __init__(self, data, tokenizer, collator):
        super().__init__()
        
        self.collator = collator
        
        self.train_df, val_df = train_test_split(data, shuffle=False, test_size=0.4) #TODO define test_size in config dict
        self.val_df, self.test_df = train_test_split(val_df, shuffle=False, test_size=0.5)
        self.train_dataset = SarcasmDataset(self.train_df, tokenizer)
        self.test_dataset = SarcasmDataset(self.test_df, tokenizer)
        self.val_dataset = SarcasmDataset(self.val_df, tokenizer)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=self.collator,
            batch_size=32,
            shuffle=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.collator,
            batch_size=16,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.collator,
            batch_size=4,
        )
