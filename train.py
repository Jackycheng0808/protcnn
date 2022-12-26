from utils.dataloader import SequenceDataset
from utils import tools
from utils.model import ProtCNN
import pytorch_lightning as pl
import torch
import os
import logging
import torch.nn.functional as F
import torchmetrics
from tensorboardX import SummaryWriter
from utils.optimizer import Ranger, RAdam
from torchinfo import summary

import argparse


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description="gene sequence classifier with CNN model"
    )
    parser.add_argument(
        "--output_version", dest="output_version", help="String appended", type=str
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
        "--val_dir",
        dest="val_dir",
        help="Directory path for validation data.",
        default="./random_split/val",
        type=str,
    )
    parser.add_argument(
        "--backbone", dest="backbone", help="Model backbone", default="", type=str
    )
    parser.add_argument(
        "--snapshot",
        dest="snapshot",
        help="Path of model snapshot.",
        default="",
        type=str,
    )
    parser.add_argument(
        "--num_epochs",
        dest="num_epochs",
        help="Maximum number of training epochs.",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", help="Batch size.", default=1, type=int
    )
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument(
        "--lr", dest="lr", help="Base learning rate.", default=0.0001, type=float
    )
    parser.add_argument("--early_stop", type=bool, default=False)
    parser.add_argument(
        "--scheduler", default=False, type=lambda x: (str(x).lower() == "true")
    )
    parser.add_argument(
        "--steps_per_shot",
        dest="steps_per_shot",
        help="save checkpoints every given steps",
        default=500,
        type=int,
    )
    parser.add_argument(
        "--save_checkpoint",
        dest="save_checkpoint",
        help="Directory path to save checkpoint.",
        default="./output",
        type=str,
    )
    parser.add_argument(
        "--seq_max_len",
        help="maximum length of sequence including for training",
        type=int,
        default=120,
    )
    parser.add_argument("--weight_decay", default=5e-6, type=float)
    parser.add_argument("--loss_function", type=str, default="categorical")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    return args


def train(model, data_loader, crit, optimizer, scheduler, epoch, args, device):
    logger.info("Starting training...")

    model.train()

    for iter, batch in enumerate(data_loader["train"]):
        # Data preprocess
        x, y = batch["sequence"], batch["target"]
        x, y = x.to(device), y.to(device)
        # Prediction
        y_hat = model(x)
        # Calculate loss
        loss = crit(y_hat, y)
        # Calculate multi-class accuracy
        pred = torch.argmax(y_hat, dim=1)
        train_acc = torchmetrics.Accuracy().to(device)
        acc = train_acc(pred, y)

        # Backward propagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        # Save checkpoint every 200 iterations
        if (iter + 1) % args.steps_per_shot == 0:
            logger.info(
                f"Epoch {epoch+1}/{args.num_epochs}, Iter {iter+1} Train_loss: {loss:.6f} Train_acc: {acc:.6f}"
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                },
                args.snapshot_folder
                + "/"
                + "epoch_"
                + str(epoch + 1)
                + "iter_"
                + f"{iter+1}"
                + ".pth",
            )

            # Save to tensorboard
            args.writer.add_scalar(
                "train_loss_step", loss, epoch * len(data_loader["train"]) + iter
            )
            args.writer.add_scalar(
                "train_acc_step", acc, epoch * len(data_loader["train"]) + iter
            )

            # Step val_loss
            for _, batch in enumerate(data_loader["dev"]):
                x_val, y_val = batch["sequence"], batch["target"]
                x_val, y_val = x_val.to(device), y_val.to(device)
                y_hat_val = model(x_val)
                loss_val = crit(y_hat_val, y_val)

                pred_val = torch.argmax(y_hat_val, dim=1)
                val_acc = torchmetrics.Accuracy().to(device)
                acc_val = val_acc(pred_val, y_val)
                break

            # Save to tensorboard
            args.writer.add_scalar(
                "val_loss_step", loss_val, epoch * len(data_loader["train"]) + iter
            )
            args.writer.add_scalar(
                "val_acc_step", acc_val, epoch * len(data_loader["train"]) + iter
            )

    if args.scheduler:
        scheduler.step()

    # Endding word for CI/CD
    logger.info("Finished training...")


@torch.no_grad()
def validate(model, data_loader, crit, epoch, args, device):
    logger.info("Starting validation...")

    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0

    for iter, batch in enumerate(data_loader):
        x, y = batch["sequence"], batch["target"]
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = crit(y_hat, y)
        loss_sum += loss

        pred = torch.argmax(y_hat, dim=1)
        valid_acc = torchmetrics.Accuracy().to(device)
        acc = valid_acc(pred, y)
        acc_sum += acc.item()

    # Calculate valid loss / acc per batch
    valid_acc = acc_sum / (iter)

    logger.info(f"Val_acc: {acc:.6f}")
    # Save to tensorboard
    args.writer.add_scalar("val_acc_epoch", valid_acc, epoch + 1)

    # Endding word for CI/CD
    logger.info("Finished Validation...")


def separate_irse_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if "model" in str(layer.__class__):
            continue
        if "container" in str(layer.__class__):
            continue
        else:
            if "BatchNorm1d" in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn


if __name__ == "__main__":

    ###  initial arguments
    args = parse_args()

    data_dir = args.data_dir
    train_dir = args.train_dir
    val_dir = args.val_dir
    num_workers = args.num_workers
    batch_size = args.batch_size
    epochs = args.num_epochs
    seed = args.seed
    # seq_max_len = 120

    # Initialize saving folder
    writer_folder = "output/writer/{}".format(args.output_version)
    snapshot_folder = "output/snapshots/{}".format(args.output_version)
    writer = SummaryWriter(writer_folder)
    args.writer = writer
    args.snapshot_folder = snapshot_folder
    args.writer_folder = writer_folder

    if not os.path.exists(
        snapshot_folder
    ):  # add folder if output directory doesn't exist
        os.makedirs(snapshot_folder)
        with open(os.path.join(snapshot_folder, "log.txt"), "w") as fp:
            pass

    if not os.path.exists(writer_folder):
        os.makedirs(writer_folder)

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_file = f"{snapshot_folder}/log.txt"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print paths
    logger.info("Log file is %s" % log_file)
    logger.info("Data path is %s" % data_dir)
    logger.info("Export path is %s" % args.save_checkpoint)

    pl.seed_everything(seed)  # set seed for reproducity

    ###  data preprocessing
    train_data, train_targets = tools.reader(train_dir, data_dir)

    word2id = tools.build_vocab(train_data)
    fam2label = tools.build_labels(train_targets)
    num_classes = len(fam2label)

    ###  data loading
    train_dataset = SequenceDataset(
        word2id, fam2label, args.seq_max_len, data_dir, "train"
    )
    dev_dataset = SequenceDataset(word2id, fam2label, args.seq_max_len, data_dir, "dev")
    test_dataset = SequenceDataset(
        word2id, fam2label, args.seq_max_len, data_dir, "test"
    )

    dataloaders = {}
    dataloaders["train"] = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    dataloaders["dev"] = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    dataloaders["test"] = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    ### model loading
    model = ProtCNN(num_classes, backbone=args.backbone, seq_max_len=args.seq_max_len)
    print(summary(model, input_size=(batch_size, 22, args.seq_max_len)))
    logger.info("Model backbone is %s" % args.backbone)

    if not args.snapshot == "":
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict["state_dict"])
        logger.info("Loaded model from %s" % args.snapshot)

    ### model training
    # Log training details
    logger.info("Training optimizer: %s" % args.optimizer)
    logger.info("Training learning rate: %g" % args.lr)
    logger.info("Training loss function: %s" % args.loss_function)
    logger.info("Training epochs: %d" % epochs)
    logger.info("Training batch size: %d" % batch_size)
    logger.info("Training weight decay: %g" % args.weight_decay)

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Choose loss function
    assert args.loss_function in ["categorical"]
    if args.loss_function == "categorical":
        crit = F.cross_entropy

    # backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(model)

    # Choose optimizer and scheduler
    # Separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
    optimizer_list = {
        "sgd": torch.optim.SGD(
            params=model.parameters(),
            weight_decay=args.weight_decay,
            lr=args.lr,
            momentum=0.9,
        ),
        "adam": torch.optim.Adam(
            params=model.parameters(),
            weight_decay=args.weight_decay,
            lr=args.lr,
        ),
        "Ranger": Ranger(
            params=model.parameters(),
            weight_decay=args.weight_decay,
            lr=args.lr,
        ),
        "RAdam": RAdam(
            params=model.parameters(),
            weight_decay=args.weight_decay,
            lr=args.lr,
        ),
    }

    optimizer = optimizer_list[args.optimizer]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[1, 3, 5], gamma=0.5
    )

    # Train model
    for epoch in range(args.num_epochs):
        train(model, dataloaders, crit, optimizer, scheduler, epoch, args, device)
        validate(model, dataloaders["dev"], crit, epoch, args, device)
