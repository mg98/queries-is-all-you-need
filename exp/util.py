import pandas as pd
import hashlib
import base64
from copy import deepcopy
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class EarlyStopping:
  def __init__(self, patience, min_delta):
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
      print('New best score:', score, 'at epoch', epoch)
      self.best_model = deepcopy(model)
      self.best_epoch = epoch
      self.counter = 0

class OrcasDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tokenized_query = self.cfg.tokenizer(self.df['query'].iloc[idx], 
                                         padding='max_length', 
                                         truncation=True, 
                                         max_length=self.cfg.query_token_length, 
                                         return_tensors="pt")
        tokenized_docid = self.cfg.tokenizer(self.df['docid'].iloc[idx], 
                                        padding='max_length', 
                                        truncation=True, 
                                        max_length=self.cfg.docid_token_length,
                                        return_tensors="pt")
        return {
            'index': idx,
            'input_ids': tokenized_query['input_ids'].squeeze(0),
            'attention_mask': tokenized_query['attention_mask'].squeeze(0),
            'labels': tokenized_docid['input_ids'].squeeze(0)
        }
  
# INT_TOKEN_IDS = None
# def restrict_decode_vocab(batch_idx, prefix_beam):
#   return INT_TOKEN_IDS

def transform_docid(docid, length: int, repr: str):
  """
  Transforms a document ID based on the specified length and representation.

  Args:
    docid (str): The document ID to transform.
    length (int): The desired length of the transformed ID. If set to 0, original ID is returned.
    repr (str): The representation type of the transformed ID. Valid options are 'hex' and 'num'.

  Returns:
    str: The transformed document ID.

  Raises:
    ValueError: If an invalid representation type is provided.
  """
  if length is None:
    return docid
  
  h = hashlib.sha1(docid.encode())

  if repr == 'hex':
    return h.hexdigest()[-length:]
  elif repr == 'num':
    n = int(h.hexdigest()[-length:], 16) % 10**length
    return str(n).zfill(length)
  elif repr == 'base64':
    return base64.b64encode(h.digest()).decode('ascii')[-length:]
  elif repr == 'bin':
    return bin(int(h.hexdigest()[-length:], 16))[2:]
  else:
    raise ValueError('Invalid representation')


def get_data(n_docs, n_queries_per_docs, filter_from=None, filter_to=None):
  """
  Get data (query-docid pairs) about `n_docs` documents and `n_queries_per_docs` associated queries per document.
  Only draw from documents that generally have between `filter_from` and `filter_to` queries linked to them.
  """
  DF = pd.read_csv('orcas.tsv', sep='\t', header=None, names=['query', 'docid'])
  docid_counts = DF.groupby('docid').count()

  if filter_from is None:
    filter_from = n_queries_per_docs
  elif filter_from < n_queries_per_docs:
    raise Exception('filter_from cannot be smaller than n_queries_per_docs')

  condition = (docid_counts['query'] >= filter_from)
  if filter_to is not None:
    condition &= (docid_counts['query'] < filter_to)

  eligible_docids = docid_counts[condition].index
  df = DF[DF['docid'].isin(eligible_docids)]
  sampled_docids = df['docid'].drop_duplicates().sample(n=n_docs)

  base_df = pd.DataFrame()
  for docid in sampled_docids:
    docid_df = df[df['docid'] == docid].sample(n=n_queries_per_docs)
    base_df = pd.concat([base_df, docid_df])
  base_df.reset_index(drop=True, inplace=True)

  # if DOCID_CHAR_LENGTH > 0:
  #   base_df['docid'] = base_df['docid'].apply(transform_docid)

  # tokenized_docids = base_df['docid'].drop_duplicates().apply(tokenizer.tokenize)
  # unique_tokens = set([token for sublist in tokenized_docids for token in sublist])
  # INT_TOKEN_IDS = [id for token, id in tokenizer.get_vocab().items() if token in unique_tokens]
  # INT_TOKEN_IDS.append(tokenizer.eos_token_id)
  # MAX_NEW_TOKENS = max(len(tokens) for tokens in unique_tokens)
  
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

def batched(l, chunk_size):
    """Splits a list into chunks of a specified size. In Python 3.12, this could be replaced by `itertools.batched`."""
    l = list(l)
    return [l[i:i + chunk_size] for i in range(0, len(l), chunk_size)]
