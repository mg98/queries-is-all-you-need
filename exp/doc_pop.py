import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import wandb
from util import *

"""## Configuration"""
N_DOCIDS = 100
N_QUERIES = 1
TRAIN_SPLIT_SIZE = 3
VAL_SPLIT_SIZE = 1
TEST_SPLIT_SIZE = 1

X = int(sys.argv[1])

wandb.init(
  project=f"doc-pop-{N_DOCIDS}-{N_QUERIES}",
  name=f"{X}",
  config={
   "learning_rate": LEARNING_RATE,
   "batch_size": BATCH_SIZE,
   "model": MODEL_NAME,
   "number_of_docs": N_DOCIDS,
   "number_of_queries": N_QUERIES,
   "es_patience": ES_PATIENCE,
   "es_delta": ES_DELTA,
   "train_split_size": TRAIN_SPLIT_SIZE,
   "val_split_size": VAL_SPLIT_SIZE,
   "test_split_size": TEST_SPLIT_SIZE,
  }
)
wandb.define_metric("acc", summary="max")
wandb.define_metric("loss", summary="min")

"""## Load Dataset"""
DF = pd.read_csv('orcas.tsv', sep='\t', header=None, names=['query', 'docid']).drop_duplicates()

"""## Prepare Dataset"""

df = get_data(N_DOCIDS, N_QUERIES, 
        (TRAIN_SPLIT_SIZE+VAL_SPLIT_SIZE+TEST_SPLIT_SIZE)*X,
        (TRAIN_SPLIT_SIZE+VAL_SPLIT_SIZE+TEST_SPLIT_SIZE)*(X+1)
      )

train_base_df, val_df, test_df = train_test_val_split(
  TRAIN_SPLIT_SIZE*N_DOCIDS,
  VAL_SPLIT_SIZE*N_DOCIDS,
  TEST_SPLIT_SIZE*N_DOCIDS
)

"""## Split & Tokenization"""

val_inputs, val_att = get_input_and_attention_masks(val_df['query'])
test_inputs, _ = get_input_and_attention_masks(test_df['query'])

n = N_QUERIES

# get a sample of n queries of each document for the train set
train_df = pd.concat([group.head(n) for _, group in train_base_df.groupby('docid')])
train_df = train_df.sample(frac=1).reset_index(drop=True)
train_inputs, train_att = get_input_and_attention_masks(train_df['query'])
train_labels, _ = get_input_and_attention_masks(train_df['docid'], True)

"""## Training"""
early_stopping = EarlyStopping()
model.train()

for epoch in range(1, 1000):

  # training iteration
  for i in range(0, len(train_inputs), BATCH_SIZE):
    output = model(
      input_ids=train_inputs[i:i+BATCH_SIZE],
      attention_mask=train_att[i:i+BATCH_SIZE],
      labels=train_labels[i:i+BATCH_SIZE]
    )
    wandb.log({"loss": output.loss.item()})
    output.loss.backward()
    optimizer.step()
    optimizer.zero_grad()

  """### Accuracy Check"""
  model.eval()
  matches = 0
  limit = len(val_df)
  
  with torch.no_grad():
    for i in range(limit):
      output = model.generate(val_inputs[i].unsqueeze(0), max_length=8)
      if decode(output[0]) == val_df['docid'].iloc[i]:
        matches += 1
  
  val_acc = matches / float(limit)
  wandb.log({"acc": val_acc})

  early_stopping(val_acc, epoch, model)
  if early_stopping.early_stop:
    model = early_stopping.best_model
    break
  model.train()

wandb.run.summary["best_epoch"] = early_stopping.best_epoch
wandb.run.summary["best_val_acc"] = early_stopping.best_score

"""## Accuracy on Unseen Queries"""
data = []

model.eval()
with torch.no_grad():

  for i in range(len(test_df)):
    output = model.generate(test_inputs[i].unsqueeze(0),
          do_sample=False, return_dict_in_generate=True, output_scores=True,
          max_length=8, num_beams=10, num_return_sequences=10)

    docids = [decode(output_id) for output_id in output.sequences]
    scores = output.sequences_scores.cpu().detach().numpy()

    for docid, score in zip(docids, scores):
      data.append([test_df['query'].iloc[i], test_df['docid'].iloc[i], docid, score])


# Create results as table artifact
artifact = wandb.Artifact(name='test_results', type='results')
artifact.add(wandb.Table(
    columns=['test_input', 'test_output', 'output', 'score'], 
    data=data
  ), 
  f'test_results_{N_DOCIDS}_{N_QUERIES}'
)
wandb.run.log_artifact(artifact)

# Compute top-k accuracies
df = pd.DataFrame(data, columns=['test_input', 'test_output', 'output', 'score'])
test_cases = df.groupby(['test_input', 'test_output'])

for k in [1, 3, 5, 10]:
    acc = test_cases.apply(lambda group: group['test_output'].iloc[0] == group.head(k)['output']).sum()
    wandb.run.summary[f'top{k}_acc'] = acc/float(test_cases.ngroups)

wandb.finish()
