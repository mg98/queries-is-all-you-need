import os
import pandas as pd
import torch
import wandb
from transformers import T5Tokenizer
from .trainer import Trainer
from .util import *

TRAIN_SPLIT_SIZE = 3
VAL_SPLIT_SIZE = 1
TEST_SPLIT_SIZE = 1

class Retrieval:

  def __init__(self, cfg):
    self.cfg = cfg
    wandb.init(
      project=f"{self.cfg.model}-{self.cfg.n_docs}-{self.cfg.docid_length}",
      name=f"{self.cfg.n_queries}",
      config={
        "learning_rate": self.cfg.learning_rate,
        "batch_size": self.cfg.batch_size,
        "model": self.cfg.model,
        "number_of_docs": self.cfg.n_docs,
        "number_of_queries": self.cfg.n_queries,
        "docid_length": self.cfg.docid_length,
        "docid_repr": self.cfg.docid_repr,
        "es_patience": self.cfg.es_patience,
        "es_delta": self.cfg.es_delta,
        "input_token_max_length": self.cfg.query_token_length,
        "output_token_max_length": self.cfg.docid_token_length,
        "output_max_length": self.cfg.output_max_length,
        "num_workers": self.cfg.num_workers,
        "train_split_size": TRAIN_SPLIT_SIZE,
        "val_split_size": VAL_SPLIT_SIZE,
        "test_split_size": TEST_SPLIT_SIZE,
      })
    wandb.Table.MAX_ARTIFACTS_ROWS = 2**31
    wandb.define_metric("acc", summary="max")
    wandb.define_metric("loss", summary="min")
  
  def __del__(self):
    wandb.finish()

  def _get_datasets(self, df):
    train_base_df, val_df, test_df = train_test_val_split(
      df,
      TRAIN_SPLIT_SIZE * self.cfg.max_queries * self.cfg.n_docs,
      VAL_SPLIT_SIZE * self.cfg.max_queries * self.cfg.n_docs,
      TEST_SPLIT_SIZE * self.cfg.max_queries * self.cfg.n_docs
    )

    # get a sample of n queries of each document for the train set
    train_df = pd.concat([group.head(self.cfg.n_queries) for _, group in train_base_df.groupby('docid')])
    train_df = train_df.sample(frac=1).reset_index(drop=True)

    train_dataset = OrcasDataset(self.cfg, train_df)
    val_dataset = OrcasDataset(self.cfg, val_df)
    test_dataset = OrcasDataset(self.cfg, test_df)
    
    return train_dataset, val_dataset, test_dataset
  
  def _log_test_results(self, test_results):
    table_name = f'retrieval_test_{self.cfg.n_docs}_{self.cfg.n_queries}'
    artifact = wandb.Artifact(name=table_name, type='results')
    artifact.add(
      wandb.Table(columns=['test_input', 'test_output', 'output', 'score'], data=test_results),
      table_name
    )
    wandb.run.log_artifact(artifact)

  def _log_summary(self, test_results):
    df = pd.DataFrame(test_results, columns=['test_input', 'test_output', 'output', 'score'])
    test_cases = df.groupby(['test_input', 'test_output'])
    
    for k in [1, 3, 5]:
        acc = test_cases.apply(lambda group: (group['test_output'].iloc[0] == group.head(k)['output']).any()).sum()
        wandb.run.summary[f'top{k}_acc'] = acc / float(test_cases.ngroups) if test_cases.ngroups > 0 else None

    try:
        artifact = wandb.use_artifact(f'retrieval_summary:latest')
        summary_table = artifact.get('retrieval_summary')
    except wandb.errors.CommError:
        summary_table = wandb.Table(columns=[
          'n_docids', 'n_queries', 
          'top1_acc', 'top3_acc', 'top5_acc'
          ])

    summary_table.add_data(
      self.cfg.n_docs, 
      self.cfg.n_queries, 
      wandb.run.summary[f'top1_acc'], 
      wandb.run.summary[f'top3_acc'], 
      wandb.run.summary[f'top5_acc']
    )

    artifact = wandb.Artifact('retrieval_summary', type='results')
    artifact.add(summary_table, 'retrieval_summary')
    wandb.log_artifact(artifact)

  def _log_model(self, model):
    try:
      model_path = f'model_{self.cfg.n_docs}_{self.cfg.n_queries}.pth'
      torch.save(model.state_dict(), model_path)
      artifact = wandb.Artifact(f'model_{self.cfg.n_docs}_{self.cfg.n_queries}', type='model')
      artifact.add_file(model_path)
      wandb.run.log_artifact(artifact)
    finally:
      if os.path.exists(model_path):
        os.remove(model_path)
  
  def run(self):
    df = get_data(
      n_docs=self.cfg.n_docs, 
      n_queries_per_docs=(TRAIN_SPLIT_SIZE+VAL_SPLIT_SIZE+TEST_SPLIT_SIZE)*self.cfg.max_queries, 
      filter_from=self.cfg.min_pop, 
      filter_to=self.cfg.max_pop
      )
    df['docid'] = df['docid'].apply(lambda docid: transform_docid(docid, self.cfg.docid_length, self.cfg.docid_repr))
    
    train_df, val_df, test_df = self._get_datasets(df)
    
    trainer = Trainer(self.cfg)
    trainer.train(train_df, val_df)
    del train_df, val_df

    test_results = trainer.test(test_df)
    del test_df

    self._log_test_results(test_results)
    self._log_summary(test_results)
    self._log_model(trainer.model)
