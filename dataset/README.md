# Preparing the data for PatFigCLS and PatFigVQA

For training and evaluation, the datasets can either be prepared or downloaded directly.

## Preparing datasets

First download the Extended CLEF-IP 2011 and DeepPatent2 dataset to ROOT_DIR.
Then set the ROOT_DIR in 'config.yaml' file.

Run the following code in sequence to prepare the PatFigCLS and PatFigVQA datasets.

1. Get the list of Object and Projection labels from DeepPatent2 dataset.

```
python3 fetch_object_and_projection_concepts.py
```

2. Map Projection labels to standardized Projection labels.

```
python3 concepts/cluster_projection_concepts.py
```

3. Cluster Object concept labels.

```
python3 concepts/cluster_object_concepts.py
```

4. Prepare raw datasets.

```
python3 create_raw_datasets.py
```

5. Preprare PatFigCLS dataset.

```
python3 create_classification_splits.py
python3 create_shards_cls.py
```

6. Prepare PatFigVQA dataset.

```
python3 create_few_shot_vqa_dataset.py
python3 create_shards_vqa.py
```

## Download datasets

Download the dataset directly from Zenodo.org

1. [PatFigVQA Dataset](https://doi.org/10.5281/zenodo.14907472)
2. [PatFigCLS Dataset](https://doi.org/10.5281/zenodo.14905550)
