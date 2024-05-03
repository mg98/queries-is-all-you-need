import torch
import numpy as np
import random
import argparse
from transformers import T5Tokenizer
from .retrieval import Retrieval
from .util import *

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('program', 
                      help='Program to execute', 
                      choices=['retrieval', 'query_similarity', 'doc_pop'])
  parser.add_argument('n_docs', help='Number of unique documents', type=int)
  parser.add_argument('n_queries', help='Number of queries per document', type=int)
  parser.add_argument('--max-queries', help='The maximum number of n_queries that will be run in a set of experiments', type=int, default=20)
  parser.add_argument('--model', help='Model name', type=str, default='t5-small')
  parser.add_argument('-l', '--docid-length', help='Character length of target docids (if set to 0, will use originals from ORCAS)', type=int, default=0)
  parser.add_argument('--docid-repr', help='Representation of docids (if docid-length != 0)', type=str, choices=['hex', 'num', 'base64', 'bin'], default='hex')
  parser.add_argument('--min-pop', help='Document filter for minimum number of referenced queries', type=int)
  parser.add_argument('--max-pop', help='Document filter for maximum number of referenced queries', type=int)
  parser.add_argument('-b', '--batch-size', help='Batch size to be used in training', type=int, default=32)
  parser.add_argument('-w', '--num-workers', help='Number of parallel CPU worker units for PyTorch\'s DataLoader', type=int, default=5)
  parser.add_argument('-lr', '--learning-rate', help='Learning rate', type=float, default=1e-3)
  parser.add_argument('-p', '--es-patience', help='Early stopping patience', type=int, default=80)
  parser.add_argument('-d', '--es-delta', help='Early stopping minimum delta', type=int, default=1e-2)
  parser.add_argument('--beams', help='Number of beams for beam search', type=int, default=5)
  parser.add_argument('-k', '--num-results', help='Number of results to generate', type=int, default=5)
  parser.add_argument('-qtl', '--query-token-length', help='Max length for tokenization of queries', type=int, default=128)
  parser.add_argument('-dtl', '--docid-token-length', help='Max length for tokenization of docids (default: 2x docid-length)', type=int)
  parser.add_argument('--output-max-length', help='Generation output max length (including input and output)', type=int)
  parser.add_argument('--device', help='Device to use for PyTorch computations', type=str)

  args = parser.parse_args()
  if args.docid_token_length is None:
    args.docid_token_length = args.docid_length*2 if args.docid_length > 0 else 16
  if args.output_max_length is None:
    args.output_max_length = args.query_token_length + args.docid_token_length
  if args.n_docs >= 10000 or args.docid_length >= 20:
    args.es_patience *= 2
  if args.n_docs >= 10000:
    args.es_delta *= 5
  if args.device is None:
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
  
  return args

if __name__ == '__main__':
  cfg = parse_args()
  print(cfg.__dict__)

  if cfg.program == 'retrieval':
    cfg.tokenizer = T5Tokenizer.from_pretrained(cfg.model, legacy=True)
    Retrieval(cfg).run()