import os

import numpy as np
import pandas as pd
import torch

from utils import tools


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, word2id, fam2label, max_len, data_path, split):
        self.word2id = word2id
        self.fam2label = fam2label
        self.max_len = max_len

        self.data, self.label = tools.reader(split, data_path)
        # self.data, self.label = tools.reader_slim(split, data_path) # use fine-chosen dataset for training and testing

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.preprocess(self.data.iloc[index])
        label = self.fam2label.get(self.label.iloc[index], self.fam2label["<unk>"])

        return {"sequence": seq, "target": label}

    def preprocess(self, text):
        seq = []

        # Encode into IDs
        for word in text[: self.max_len]:
            seq.append(self.word2id.get(word, self.word2id["<unk>"]))

        # Pad to maximal length
        if len(seq) < self.max_len:
            seq += [self.word2id["<pad>"] for _ in range(self.max_len - len(seq))]

        # Convert list into tensor
        seq = torch.from_numpy(np.array(seq))

        # One-hot encode
        one_hot_seq = torch.nn.functional.one_hot(
            seq,
            num_classes=len(self.word2id),
        )

        # Permute channel (one-hot) dim first
        one_hot_seq = one_hot_seq.permute(1, 0)

        return one_hot_seq
