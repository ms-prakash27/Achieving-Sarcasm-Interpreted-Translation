import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import T5Tokenizer, DataCollatorForSeq2Seq, AutoTokenizer
import os

from datamodule import SarcasmDataModule
from models import Model
import data_utils
import metrics

torch.set_float32_matmul_precision('medium')

pl.seed_everything(42, workers=True)

data = pd.read_excel('/blue/cai6307/n.kolla/data/Translation.xlsx', names=['sarcasm', 'interpretation', 'translation'])

translator_data = pd.DataFrame()

translator_input_prefix = 'interpret sarcasm and translate English to Telugu: ' # Needed as the models are pretrained for multiple tasks
translator_data['inputs'] = data.sarcasm.apply(lambda text: data_utils.preprocess(text, translator_input_prefix))
translator_data['targets'] = data.translation.apply(data_utils.preprocess)

translator_model_name = 'facebook/mbart-large-50-one-to-many-mmt'
evaluate = True

translator_tokenizer = AutoTokenizer.from_pretrained(translator_model_name, model_max_length=512)

if 'mbart' in translator_model_name:
    translator_tokenizer.src_lang = 'en_XX'
    translator_tokenizer.tgt_lang = 'te_IN'

translator_collator = DataCollatorForSeq2Seq(tokenizer=translator_tokenizer, model=translator_model_name)

translator_datamodule = SarcasmDataModule(translator_data, translator_tokenizer, translator_collator)

if not evaluate:
    translator_model = Model(translator_model_name, lr=1e-5)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )

    translator_checkpoint = ModelCheckpoint(
        dirpath='/blue/cai6307/n.kolla/finetune_ckpts/translatorsv2',
        filename='{val_loss:.3f}_{epoch}_{step}_model=' + translator_model_name.split("/")[-1],
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_weights_only=False,
        every_n_epochs=1,
        verbose=False,
        save_last=False,
        enable_version_counter=True
    )

    translator_trainer = pl.Trainer(
        max_epochs=50, 
        accelerator='gpu', 
        devices=-1, 
        callbacks=[translator_checkpoint, early_stopping],
        default_root_dir="/blue/cai6307/n.kolla/logs",
        deterministic=True, # To ensure reproducability
    )

    translator_trainer.fit(translator_model, datamodule=translator_datamodule)
else:
    ckpt_dir_path = '/blue/cai6307/n.kolla/finetune_ckpts/translatorsv2'
    ckpts = os.listdir(ckpt_dir_path)
    ckpt_path = ''
    if ckpts:
        model_losses = [float(ckpt.split('_')[1].split('=')[-1]) if translator_model_name.split("/")[-1] in ckpt else np.inf for ckpt in ckpts]
        if np.min(model_losses) != np.inf:
            ckpt = ckpts[np.argmin(model_losses)]
            ckpt_path = os.path.join(ckpt_dir_path, ckpt)
    if not ckpt_path:
        print('Model checkpoint is not available, re-train')
    else:
        translator_model = Model.load_from_checkpoint(ckpt_path, model_name=translator_model_name, lr=1e-5)
        translator_model.eval()
        device = torch.device('cuda')
        translator_model.to(device)
        
        inputs, targets = data_utils.separate_inputs_targets(translator_datamodule.test_df)
        preds = []
        for input_ in inputs:
            with torch.no_grad():
                output_ids = translator_model.generate(**translator_tokenizer([input_], return_tensors='pt').to(device))
            preds.extend(translator_tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        
        save_model_name = translator_model_name.split("/")[-1] + "v2"
        data_utils.save_tests(inputs, preds, save_model_name)
        
        # Use pretrained tokenizer for telugu evaluation
        preds = [data_utils.custom_tokenize(pred, translator_tokenizer) for pred in preds]
        targets = [[data_utils.custom_tokenize(target_text, translator_tokenizer) for target_text in target] for target in targets]
        
        print('bleu:', metrics.bleu_score(preds, targets))
        print('rouge:', metrics.rouge_scores(preds, targets))
