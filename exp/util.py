from transformers import T5ForConditionalGeneration, T5Tokenizer
import random
import pandas as pd
import numpy as np
import torch
from copy import deepcopy

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)

MODEL_NAME = 't5-small'
LEARNING_RATE = 1e-3
BATCH_SIZE  = 32

# Early stopping
ES_PATIENCE  = 20
ES_DELTA     = 0.01

class EarlyStopping:
  def __init__(self, patience=ES_PATIENCE, min_delta=ES_DELTA):
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.best_score = 0
    self.best_model = None
    self.best_epoch = 0
    self.early_stop = False

  def __str__(self):
   return f'patience={self.patience},min_delta={self.min_delta},counter={self.counter},best_score={self.best_score},early_stop={self.early_stop}'
  
  def __call__(self, score, epoch, model):
    if score < self.best_score + self.min_delta:
      self.counter += 1
      if self.counter >= self.patience:
        self.early_stop = True
    else:
      self.best_score = score
      self.best_model = deepcopy(model)
      self.best_epoch = epoch
      self.counter = 0


model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to('cuda')
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

def get_input_and_attention_masks(d, train_or_labels = 'train'):
  tokenized = tokenizer(d.astype(str).tolist(), padding='max_length', return_tensors="pt", truncation=True,
                        max_length=128 if train_or_labels == 'train' else 8)
  input_ids = tokenized['input_ids'].to('cuda')
  attention_mask = tokenized['attention_mask'].to('cuda')
  return input_ids, attention_mask
  
def decode(x):
 return tokenizer.decode(x, skip_special_tokens=True)


# Precompute dataframes
DF = pd.read_csv('orcas.tsv', sep='\t', header=None,
        names=['query', 'docid'])
DOCID_COUNTS = DF.groupby('docid').count()
 
def get_data(n_docs, n_queries_per_docs, filter_from=None, filter_to=None):
  """
  Get data (query-docid pairs) about `n_docs` documents and `n_queries_per_docs` associated queries per document.
  Only draw from documents that generally have between `filter_from` and `filter_to` queries linked to them.
  """
  if filter_from is None:
    filter_from = n_queries_per_docs
  elif filter_from < n_queries_per_docs:
    raise Exception('filter_from cannot be smaller than n_queries_per_docs')

  condition = (DOCID_COUNTS['query'] >= filter_from)
  if filter_to is not None:
    condition &= (DOCID_COUNTS['query'] < filter_to)

  eligible_docids = DOCID_COUNTS[condition].index
  df = DF[DF['docid'].isin(eligible_docids)]
  sampled_docids = df['docid'].drop_duplicates().sample(n=n_docs)

  base_df = pd.DataFrame()
  for docid in sampled_docids:
    docid_df = df[df['docid'] == docid].sample(n=n_queries_per_docs)
    base_df = pd.concat([base_df, docid_df])
  base_df.reset_index(drop=True, inplace=True)

  return base_df