import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
from torch.optim import AdamW
import evaluate

from datamodule import SarcasmDataModule
 
class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super(Model, self).__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.lr = lr        
        
    def forward(self, batch):
        output = self.model(**batch)
        return output
    
    def generate(self, input_ids, attention_mask, max_length=128):
        return self.model.generate(input_ids, attention_mask=attention_mask, max_length=max_length)
    
    def _step(self, batch, idx):
        output = self(batch)
        loss = output.loss
        return loss
    
    def training_step(self, batch, idx):
        loss = self._step(batch, idx)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, idx):
        loss = self._step(batch, idx)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, idx):
        loss = self._step(batch, idx)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)
