import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import T5Tokenizer, DataCollatorForSeq2Seq, AutoTokenizer

from datamodule import SarcasmDataModule
from models import Model
import data_utils
import metrics
import os

torch.set_float32_matmul_precision('medium')

pl.seed_everything(42, workers=True)

data = pd.read_excel('/blue/cai6307/n.kolla/data/Translation.xlsx', names=['sarcasm', 'interpretation', 'translation'])

interpreter_model_name = 'facebook/bart-large'
ckpt_dir_path = '/blue/cai6307/n.kolla/finetune_ckpts/interpreters'
evaluate = True

interpreter_data = pd.DataFrame()

interpreter_input_prefix = 'interpret sarcasm: ' # Needed as the models are pretrained for multiple tasks
interpreter_data['inputs'] = data.sarcasm.apply(lambda text: data_utils.preprocess(text, interpreter_input_prefix))
interpreter_data['targets'] = data.interpretation.apply(data_utils.preprocess)

interpreter_tokenizer = AutoTokenizer.from_pretrained(interpreter_model_name, model_max_length=512)

interpreter_collator = DataCollatorForSeq2Seq(tokenizer=interpreter_tokenizer, model=interpreter_model_name)

interpreter_datamodule = SarcasmDataModule(interpreter_data, interpreter_tokenizer, interpreter_collator)

if not evaluate:
    interpreter_model = Model(interpreter_model_name, lr=1e-5)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )

    interpreter_checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir_path,
        filename='{val_loss:.3f}_{epoch}_{step}_model=' + interpreter_model_name.split("/")[-1],
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_weights_only=False,
        every_n_epochs=1,
        verbose=False,
        save_last=False,
        enable_version_counter=True
    )

    interpreter_trainer = pl.Trainer(
        max_epochs=30, 
        accelerator='gpu', 
        devices=-1, 
        callbacks=[interpreter_checkpoint, early_stopping],
        default_root_dir="/blue/cai6307/n.kolla/logs",
        deterministic=True, # To ensure reproducability
    )

    interpreter_trainer.fit(interpreter_model, datamodule=interpreter_datamodule)
else:
    ckpts = os.listdir(ckpt_dir_path)
    ckpt_path = ''
    if ckpts:
        model_losses = [float(ckpt.split('_')[1].split('=')[-1]) if interpreter_model_name.split("/")[-1] in ckpt else np.inf for ckpt in ckpts]
        if np.min(model_losses) != np.inf:
            ckpt = ckpts[np.argmin(model_losses)]
            ckpt_path = os.path.join(ckpt_dir_path, ckpt)
    if not ckpt_path:
        print('Model checkpoint is not available, re-train')
    else:
        interpreter_model = Model.load_from_checkpoint(ckpt_path, model_name=interpreter_model_name, lr=1e-5)
        interpreter_model.eval()
        device = torch.device('cuda')
        interpreter_model.to(device)

        inputs, targets = data_utils.separate_inputs_targets(interpreter_datamodule.test_df)
        preds = []
        for input_ in inputs:
            with torch.no_grad():
                output_ids = interpreter_model.generate(**interpreter_tokenizer([input_], return_tensors='pt').to(device))
            preds.extend(interpreter_tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))

        data_utils.save_tests(inputs, preds, interpreter_model_name.split("/")[-1])

        print('bleu:', metrics.bleu_score(preds, targets))
        print('rouge:', metrics.rouge_scores(preds, targets))
        
        ref_inputs = [input_.replace(interpreter_input_prefix, '') for input_ in inputs]
        print('pinc:', metrics.pinc_score(preds, ref_inputs))
