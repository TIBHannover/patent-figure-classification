import re
import os
import json
import random

import torch
import torch.nn as nn
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords

from sentence_transformers import SentenceTransformer, models

nltk.download('stopwords', quiet=True)
stopwords = set(stopwords.words('english'))

device = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_value = 1337
batch_size = 512
set_seed(seed_value)

## Load PatentBERT model ##
bert_embedding_model = models.Transformer('anferico/bert-for-patents', max_seq_length=128)
pooling_model = models.Pooling(bert_embedding_model.get_word_embedding_dimension())
dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                            out_features=256, activation_function=nn.Tanh())
model = SentenceTransformer(modules=[bert_embedding_model, pooling_model, dense_model])
model.to(device)
model.eval()

workers = 32

def preprocess_concept(concept):
    """Preprocess concept label"""
    concept = concept.encode('utf-8').decode('unicode_escape')
    concept = concept.lower()
    concept = re.sub(r'\d+', '', concept)
    concept = re.sub(r'\s+', ' ', concept)
    concept = re.sub(r'[^\w\s]', '', concept).strip()
    return concept

def get_patent_bert_embeddings(concepts):
    """Get PatentBERT embeddings for the list of concepts"""
    with torch.inference_mode():
        features = model.encode(concepts, batch_size=512, convert_to_numpy=True,
                                show_progress_bar=False, device=device)
        return features

def remove_cluster_outliers(embeddings, clusters):
    """Remove outliers in a cluster using Inter-Quartile filtering"""
    filtered_concepts = []

    for cluster_id in np.unique(clusters):
        
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_data = embeddings[cluster_indices]
        
        Q1 = np.percentile(cluster_data, 25, axis=0)
        Q3 = np.percentile(cluster_data, 75, axis=0)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        non_outlier_mask = np.all((cluster_data >= lower_bound) & (cluster_data <= upper_bound), axis=1)
        
        if np.any(non_outlier_mask):
            filtered_concepts.append(cluster_indices[non_outlier_mask])  # Keep the original indices
        else:
            filtered_concepts.append(cluster_indices)

    filtered_concepts = np.concatenate(filtered_concepts)

    return filtered_concepts

def cluster_concepts(embeddings, eps=0.5, min_samples=1):
    """Run DBSCAN clustering on embeddings"""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=workers)
    clusters = dbscan.fit_predict(embeddings)

    return clusters

def find_representative_concept(cluster_dict):
    """Find the cluster closest to the center of the cluster"""
    new_dict = {}

    for _, data in cluster_dict.items():
        
        members = data['members']
        embeddings = np.array(data['embeddings'])

        centroid = np.mean(embeddings, axis=0).reshape(1, -1)
        distances = cosine_similarity(embeddings, centroid).flatten()
        
        representative_idx = np.argmax(distances)
        representative_concept = members[representative_idx]
        
        new_dict[representative_concept] = members

    return new_dict

def main():

    eps = 0.09  

    with open("object_concepts.json", "r") as rf:
        object_concepts = json.load(rf)["concepts"]
        concepts = list(set([preprocess_concept(concept) for concept in object_concepts]))

    print(f"Found {len(concepts)} concepts")

    embeddings = get_patent_bert_embeddings(concepts)
    
    clusters = cluster_concepts(embeddings, eps)

    filtered_concepts = remove_cluster_outliers(embeddings, clusters)

    cluster_dict = {}
    for cluster_id in np.unique(clusters):
        if cluster_id != -1:  # Exclude noise points, which are labeled -1
            
            member_indices = np.where(clusters == cluster_id)[0]
            member_indices = list(set(member_indices).intersection(set(filtered_concepts)))
            
            cluster_dict[cluster_id] = {
                "members": [concepts[i] for i in member_indices],
                "embeddings": [embeddings[i] for i in member_indices]
            }

    cluster_dict = find_representative_concept(cluster_dict)

    print(f"Total clusters: {len(cluster_dict)}")

    cluster_dict = {k: v for k, v in sorted(cluster_dict.items(), key=lambda x: len(x[1]))}

    with open(f'clusters/clustered_concepts_{eps}.json', 'w') as f:
        json.dump(cluster_dict, f, indent=2)

if __name__ == "__main__":
    main()
