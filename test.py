import argparse
import os

import pytorch_lightning as pl
import torch
import torchmetrics
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from utils import tools
from utils.dataloader import SequenceDataset
from utils.model import ProtCNN


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description="gene sequence classifier with CNN model evaluation"
    )
    parser.add_argument(
        "--model",
        dest="model",
        help="Path of model for evaluation.",
        required=True,
        default="",
        type=str,
    )
    parser.add_argument(
        "--backbone", dest="backbone", help="Model backbone", default="", type=str
    )
    parser.add_argument(
        "--save_result", dest="save_result", help="String appended", type=str
    )
    parser.add_argument(
        "--data_dir",
        dest="data_dir",
        help="Directory path for data.",
        default="./random_split",
        type=str,
    )
    parser.add_argument(
        "--train_dir",
        dest="train_dir",
        help="Directory path for training data.",
        default="./random_split/train",
        type=str,
    )
    parser.add_argument(
        "--test_dir",
        dest="test_dir",
        help="Directory path for training data.",
        default="./random_split/test",
        type=str,
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", help="Batch size.", default=1, type=int
    )
    parser.add_argument(
        "--seq_max_len",
        help="maximum length of sequence including for training",
        type=int,
        default=120,
    )
    parser.add_argument("--use_gpu", type=int, default=0)
    parser.add_argument("--gpu", default="", type=str)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    ### initial arguments
    args = parse_args()
    if not os.path.exists(args.model):
        print("file no exist")
        exit()
    data_dir = args.data_dir
    train_dir = args.train_dir
    num_workers = args.num_workers
    batch_size = args.batch_size
    seed = args.seed

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    ### data preprocessing
    train_data, train_targets = tools.reader(train_dir, data_dir)

    word2id = tools.build_vocab(train_data)
    fam2label = tools.build_labels(train_targets)
    num_classes = len(fam2label)

    ### data loading
    test_dataset = SequenceDataset(
        word2id, fam2label, args.seq_max_len, data_dir, "test"
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    ### model loading
    model = ProtCNN(num_classes, backbone=args.backbone, seq_max_len=args.seq_max_len)

    saved_state_dict = torch.load(args.model)
    if "model_state_dict" in saved_state_dict:
        saved_state_dict = saved_state_dict["model_state_dict"]
    model.load_state_dict(saved_state_dict)
    model.to(device)

    model.eval()

    top1_acc = 0
    top5_acc = 0
    count = 0
    with torch.no_grad():
        for order, batch in enumerate(test_loader):
            x, y = batch["sequence"], batch["target"]
            x, y = x.to(device), y.to(device)
            y_hat = model(x)

            for i in range(len(y)):
                y_index, y_hat_index = y[i], y_hat[i]
                pred = torch.topk(y_hat_index, 5).indices

                if y_index == pred[0]:
                    top1_acc += 1
                    top5_acc += 1

                elif y_index in pred:
                    top5_acc += 1
            count += batch_size
            if order % 10 == 0:
                print(
                    f"Current Iter-{order}: top1_acc - {top1_acc/count:.6f} / top5_acc - {top5_acc/count:.6f}"
                )

    top1_acc = top1_acc / count
    top5_acc = top5_acc / count
    print(f"top1_acc:{top1_acc:.6f}")
    print(f"top5_acc:{top5_acc:.6f}")
