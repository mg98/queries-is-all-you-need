import wandb
import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration
from .util import *

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = T5ForConditionalGeneration.from_pretrained(cfg.model).to(cfg.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate)
        self.es = EarlyStopping(cfg.es_patience, cfg.es_delta)
    
    def _decode(self, x):
        return self.cfg.tokenizer.decode(x, skip_special_tokens=True)

    def _validate(self, model, val_dataloader) -> float:
        model.eval()
        matches = 0
        total = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(self.cfg.device, non_blocking=True)
                outputs = model.generate(input_ids, max_length=self.cfg.output_max_length)
                decoded_outputs = [self._decode(output) for output in outputs]
                decoded_labels = [self._decode(label) for label in batch['labels']]
                matches += sum(1 for out, lbl in zip(decoded_outputs, decoded_labels) if out == lbl)
                total += len(input_ids)

        model.train()
        return matches / total

    def train(self, train_df: OrcasDataset, val_df: OrcasDataset):
        self.train_dataloader = DataLoader(train_df, 
                                  batch_size=self.cfg.batch_size, 
                                  num_workers=self.cfg.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
        self.val_dataloader = DataLoader(val_df, 
                                batch_size=self.cfg.batch_size*4,
                                num_workers=self.cfg.num_workers,
                                pin_memory=True)
        
        self.model.train()
        epoch = 0

        while True:
            epoch += 1
            losses = []
            for batch in self.train_dataloader:
                input_ids = batch['input_ids'].to(self.cfg.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.cfg.device, non_blocking=True)
                labels = batch['labels'].to(self.cfg.device, non_blocking=True)
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                losses.append(output.loss.item())
                output.loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            val_acc = self._validate(self.model, self.val_dataloader)
            wandb.log({'acc': val_acc, 'loss': np.mean(losses)})

            self.es(val_acc, epoch, self.model)
            wandb.run.summary['es_epoch'] = self.es.best_epoch
            wandb.run.summary['es_acc'] = self.es.best_score
            if self.es.early_stop:
                self.model = self.es.best_model
                break

    def test(self, test_df: OrcasDataset):
        data = []
        self.model.eval()
        with torch.no_grad():
            for batch in DataLoader(test_df, 
                                    batch_size=int(self.cfg.batch_size/3), 
                                    num_workers=self.cfg.num_workers, 
                                    pin_memory=True):
                
                input_ids = batch['input_ids'].to(self.cfg.device)
                output = self.model.generate(input_ids, 
                                        do_sample=False, 
                                        return_dict_in_generate=True, 
                                        output_scores=True,
                                        max_length=self.cfg.output_max_length,
                                        num_beams=self.cfg.beams, 
                                        num_return_sequences=self.cfg.num_results)
                
                docids = [self._decode(output_id) for output_id in output.sequences]
                scores = output.sequences_scores.cpu().detach().numpy()
                
                for i, (docid, score) in enumerate(zip(docids, scores)):
                    index = batch['index'].numpy()[math.floor(i / self.cfg.num_results)].item()
                    data.append([test_df.df['query'].iloc[index], test_df.df['docid'].iloc[index], docid, score])
        
        return data