import os

import pandas as pd


def read_sharded_partition(partition, data_path):
    return pd.concat(
        [
            pd.read_csv(os.path.join(data_path, partition, f), index_col=None)
            for f in os.listdir(os.path.join(data_path, partition))
        ]
    )


def reader(partition, data_path):
    data = []
    for file_name in os.listdir(os.path.join(data_path, partition)):
        with open(os.path.join(data_path, partition, file_name)) as file:
            data.append(
                pd.read_csv(
                    file, index_col=None, usecols=["sequence", "family_accession"]
                )
            )

    all_data = pd.concat(data)

    return all_data["sequence"], all_data["family_accession"]


# remove labels not shown
def reader_slim(partition, data_path):
    data = []
    for file_name in os.listdir(os.path.join(data_path, partition)):
        with open(os.path.join(data_path, partition, file_name)) as file:
            data.append(
                pd.read_csv(
                    file, index_col=None, usecols=["sequence", "family_accession"]
                )
            )

    all_data = pd.concat(data)

    partition_dirs = ["train", "dev", "test"]
    partition_frames = [read_sharded_partition(d, data_path) for d in partition_dirs]
    df_trn, df_vld, _ = partition_frames

    trn_fams = set(df_trn.family_accession.unique())
    vld_fams = set(df_vld.family_accession.unique())
    trn_fams_exc = trn_fams - vld_fams
    all_data = all_data[~all_data.family_accession.isin(trn_fams_exc)]

    return all_data["sequence"], all_data["family_accession"]


def build_vocab(data):
    # Build the vocabulary
    voc = set()
    rare_AAs = {"X", "U", "B", "O", "Z"}
    for sequence in data:
        voc.update(sequence)

    unique_AAs = sorted(voc - rare_AAs)

    # Build the mapping
    word2id = {w: i for i, w in enumerate(unique_AAs, start=2)}
    word2id["<pad>"] = 0
    word2id["<unk>"] = 1

    return word2id


def build_labels(targets):
    unique_targets = targets.unique()
    fam2label = {target: i for i, target in enumerate(unique_targets, start=1)}
    fam2label["<unk>"] = 0

    print(f"There are {len(fam2label)} labels.")

    return fam2label
