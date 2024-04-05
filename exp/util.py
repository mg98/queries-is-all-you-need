from transformers import T5ForConditionalGeneration, T5Tokenizer
import random
import pandas as pd
import numpy as np
import torch
from copy import deepcopy
from sklearn.model_selection import train_test_split

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
      if self.counter >= self.patience and self.best_model is not None:
        self.early_stop = True
    else:
      self.best_score = score
      self.best_model = deepcopy(model)
      self.best_epoch = epoch
      self.counter = 0


def get_input_and_attention_masks(d, is_label = False):
  tokenized = tokenizer(d.astype(str).tolist(), padding='max_length', return_tensors="pt", truncation=True,
                        max_length=8 if is_label else 20)
  input_ids = tokenized['input_ids'].to('cuda')
  attention_mask = tokenized['attention_mask'].to('cuda')
  return input_ids, attention_mask
  
def decode(x):
 return tokenizer.decode(x, skip_special_tokens=True)

# Precompute dataframes
DF = pd.read_csv('orcas.tsv', sep='\t', header=None,
        names=['query', 'docid'])
DOCID_COUNTS = DF.groupby('docid').count()

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=True)

tokenized_docids = DF['docid'].apply(tokenizer.tokenize)
unique_tokens = set([token for sublist in tokenized_docids for token in sublist])
INT_TOKEN_IDS = [id for token, id in tokenizer.get_vocab().items() if token in unique_tokens]
INT_TOKEN_IDS.append(tokenizer.eos_token_id)
MAX_NEW_TOKENS = max(len(tokens) for tokens in unique_tokens)
del tokenized_docids, unique_tokens

def restrict_decode_vocab(batch_idx, prefix_beam):
    return INT_TOKEN_IDS

model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to('cuda')

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

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

def train_test_val_split(df, train_size, val_size, test_size):
  train_df, val_and_test_df = train_test_split(df,
    train_size=train_size,
    test_size=val_size+test_size, 
    stratify=df['docid']
  )
  val_df, test_df = train_test_split(val_and_test_df, 
    train_size=val_size,
    test_size=test_size, 
    stratify=val_and_test_df['docid']
  )
  return train_df, val_df, test_df