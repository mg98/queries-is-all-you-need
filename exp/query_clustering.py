import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from util import *

N_DOCIDS = 100

DF = pd.read_csv('orcas.tsv', sep='\t', header=None, names=['query', 'docid']).drop_duplicates()

for X in range(1, 11):

    docid_counts = DF.groupby('docid').count()
    eligible_docids = docid_counts[
        (docid_counts['query'] >= X*10) & (docid_counts['query'] < (X+1)*10)
        ].index
    df = DF[DF['docid'].isin(eligible_docids)]
    sampled_docids = df['docid'].drop_duplicates().sample(n=N_DOCIDS)

    similarity_values = []

    for docid in sampled_docids:
        queries = df[df['docid'] == docid].sample(n=10)['query'].tolist()
        inputs = [tokenizer(query, return_tensors='pt').input_ids.to('cuda') for query in queries]

        with torch.no_grad():
            embeddings = torch.stack([model.encoder(input_ids=input_ids).last_hidden_state.mean(dim=1) for input_ids in inputs])

        embeddings = embeddings.squeeze(1).cpu().numpy()
        cosine_sim_matrix = cosine_similarity(embeddings)
        
        # Since the cosine similarity of an embedding with itself is always 1, we need to exclude these values.
        # Also, we only need to consider the upper triangle of the matrix excluding the diagonal, since the matrix is symmetric.
        upper_triangle_indices = np.triu_indices_from(cosine_sim_matrix, k=1)
        mean_cosine_similarity = cosine_sim_matrix[upper_triangle_indices].mean()

        similarity_values.append(mean_cosine_similarity)

    mean_similarity = np.mean(similarity_values)
    print(f'{X*10}-{(X+1)*10}', mean_similarity)
    