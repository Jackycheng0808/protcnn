import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def read_sharded_partition(partition, data_path):
    return pd.concat(
        [
            pd.read_csv(os.path.join(data_path, partition, f), index_col=None)
            for f in os.listdir(os.path.join(data_path, partition))
        ]
    )


def get_amino_acid_frequencies(data):
    aa_counter = Counter()

    for sequence in data:
        aa_counter.update(sequence)

    return pd.DataFrame(
        {"AA": list(aa_counter.keys()), "Frequency": list(aa_counter.values())}
    )


if __name__ == "__main__":
    DATA_PATH = "./random_split"
    partition_dirs = ["train", "dev", "test"]

    partition_frames = [read_sharded_partition(d, DATA_PATH) for d in partition_dirs]
    df_trn, df_vld, df_tst = partition_frames

    print("Training dataset")
    print("-" * 60)
    print(df_trn.head())

    print("All columns")
    print("-" * 60)
    print(df_trn.columns)

    train_data, train_targets = df_trn["sequence"], df_trn["family_accession"]

    analysis_folder = "./assets"
    if not os.path.exists(
        analysis_folder
    ):  # add folder if output directory doesn't exist
        os.makedirs(analysis_folder)

    # Plot the distribution of family sizes

    f, ax = plt.subplots(figsize=(8, 5))
    sorted_targets = (
        train_targets.groupby(train_targets).size().sort_values(ascending=False)
    )

    sns.histplot(sorted_targets.values, kde=True, log_scale=True, ax=ax)
    plt.title("Distribution of family sizes for the 'train' split")
    plt.xlabel("Family size (log scale)")
    plt.ylabel("# Families")
    plt.savefig(f"{analysis_folder}/family_distribution.png")
    plt.show()

    # Plot the distribution of sequences' lengths
    f, ax = plt.subplots(figsize=(8, 5))
    sequence_lengths = train_data.str.len()
    median = sequence_lengths.median()
    mean = sequence_lengths.mean()

    sns.histplot(sequence_lengths.values, kde=True, log_scale=True, bins=60, ax=ax)

    ax.axvline(mean, color="r", linestyle="-", label=f"Mean = {mean:.1f}")
    ax.axvline(median, color="g", linestyle="-", label=f"Median = {median:.1f}")

    plt.title("Distribution of sequence lengths")
    plt.xlabel("Sequence' length (log scale)")
    plt.ylabel("# Sequences")
    plt.legend(loc="best")
    plt.savefig(f"{analysis_folder}/sequence_lengths_distribution.png")
    plt.show()

    # Plot the distribution of AA frequencies

    f, ax = plt.subplots(figsize=(8, 5))

    amino_acid_counter = get_amino_acid_frequencies(train_data)

    sns.barplot(
        x="AA",
        y="Frequency",
        data=amino_acid_counter.sort_values(by=["Frequency"], ascending=False),
        ax=ax,
    )

    plt.title("Distribution of AAs' frequencies in the 'train' split")
    plt.xlabel("Amino acid codes")
    plt.ylabel("Frequency (log scale)")
    plt.yscale("log")
    plt.savefig(f"{analysis_folder}/frequency_trainsplit.png")
    plt.show()
