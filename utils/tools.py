import pandas as pd 
import os

def reader(partition, data_path):
    data = []
    for file_name in os.listdir(os.path.join(data_path, partition)):
        with open(os.path.join(data_path, partition, file_name)) as file:
            data.append(pd.read_csv(file, index_col=None, usecols=["sequence", "family_accession"]))

    all_data = pd.concat(data)        

    return all_data["sequence"], all_data["family_accession"]

def build_vocab(data):
    # Build the vocabulary
    voc = set()
    rare_AAs = {'X', 'U', 'B', 'O', 'Z'}
    for sequence in data:
        voc.update(sequence)

    unique_AAs = sorted(voc - rare_AAs)
    
    # Build the mapping
    word2id = {w: i for i, w in enumerate(unique_AAs, start=2)}
    word2id['<pad>'] = 0
    word2id['<unk>'] = 1
    
    return word2id
    
def build_labels(targets):
    unique_targets = targets.unique()
    fam2label = {target: i for i, target in enumerate(unique_targets, start=1)}
    fam2label['<unk>'] = 0
    
    print(f"There are {len(fam2label)} labels.")
        
    return fam2label