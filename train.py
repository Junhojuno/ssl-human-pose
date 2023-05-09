import time
from pathlib import Path
import argparse
import logging
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from src.dataset import (
    load_dataset,
    load_mixed_dataset
)
from src.model.pose_hrnet import HRNetW32, HRNetW48
from src.model.pose_cotrain import PoseCoTrain
from src.losses import pose_dual_loss_fn
from src.metrics import AverageMeter, accuracy
from src.utils import parse_yaml


logger = logging.getLogger(__name__)


@tf.function
def train(train_loader, model, optimizer):
    # train_loss = AverageMeter()
    # heavy_acc = AverageMeter()
    # lite_acc = AverageMeter()
    h_avg_acc, h_valid_cnt = 0, 0
    l_avg_acc, l_valid_cnt = 0, 0
    for ((sup_images, sup_hms), unsup_images) in train_loader:
        with tf.GradientTape() as tape:
            outputs = model(sup_images, unsup_images, training=True)
            loss = pose_dual_loss_fn(sup_hms, outputs)
            loss = tf.math.reduce_mean(loss)
            loss += sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables)
        )
        h_avg_acc, h_valid_cnt = accuracy(
            sup_hms, outputs[0], tf.constant(0.5, tf.float32)
        )
        l_avg_acc, l_valid_cnt = accuracy(
            sup_hms, outputs[1], tf.constant(0.5, tf.float32)
        )
        # heavy_acc.update(h_avg_acc, h_valid_cnt)
        # lite_acc.update(l_avg_acc, l_valid_cnt)

        # train_loss.update(loss, tf.shape(sup_images)[0])
    return h_avg_acc, l_avg_acc


@tf.function
def validation():
    pass


def define_argparser():
    parser = argparse.ArgumentParser('human pose regression parser')
    parser.add_argument(
        '--config', '-c',
        dest='config',
        required=True,
        help='yaml file path'
    )
    return parser.parse_args()


def main():
    tf.random.set_seed(0)
    np.random.seed(0)

    cfg = define_argparser()
    args = parse_yaml(cfg.config)

    # cwd = Path('.').resolve()  # current working dir

    cwd = Path('/data/Data').resolve()  # current working dir

    train_ds = load_mixed_dataset(
        args,
        cwd / 'datasets/basic_sources' / args.DATASET.NAME
    )

    # train_ds = load_mixed_dataset(
    #     args,
    #     str(cwd.parent / 'datasets' / args.DATASET.NAME)
    # )
    # val_ds = load_dataset(
    #     args,
    #     str(
    #         cwd.parent
    #         / 'datasets'
    #         / args.DATASET.NAME
    #         / args.DATASET.VAL.PATTERN
    #     ),
    #     'val',
    #     args.VAL.BATCH_SIZE,
    #     use_aug=False
    # )
    heavy_model = HRNetW48()
    lite_model = HRNetW32()
    _ = heavy_model(
        tf.random.normal(
            [1, *args.DATASET.COMMON.INPUT_SHAPE],
            # mean=args.DATASET.COMMON.MEANS,
            # stddev=args.DATASET.COMMON.STDS
        ),
        training=False
    )
    _ = lite_model(
        tf.random.normal(
            [1, *args.DATASET.COMMON.INPUT_SHAPE],
            # mean=args.DATASET.COMMON.MEANS,
            # stddev=args.DATASET.COMMON.STDS
        ),
        training=False
    )

    model = PoseCoTrain(
        heavy_model=heavy_model,
        lite_model=lite_model,
        scale_factor=args.AUG.SCALE_FACTOR,
        rotation_factor=args.AUG.ROT_FACTOR,
        n_joints_mask=args.AUG.N_JOINTS_MASK
    )
    # model.build([None, *args.DATASET.COMMON.INPUT_SHAPE])
    _ = model(
        tf.random.normal(
            [1, *args.DATASET.COMMON.INPUT_SHAPE],
            # mean=args.DATASET.COMMON.MEANS,
            # stddev=args.DATASET.COMMON.STDS
        ),
        tf.random.normal(
            [1, *args.DATASET.COMMON.INPUT_SHAPE],
            # mean=args.DATASET.COMMON.MEANS,
            # stddev=args.DATASET.COMMON.STDS
        ),
        training=False
    )

    optimizer = Adam(args.TRAIN.LR)
    h_acc, l_acc = train(train_ds, model, optimizer)
    print(h_acc.numpy(), l_acc.numpy())
    # criterion = pose_dual_loss_fn

    # checkpoint_prefix = os.path.join(
    #     args.OUTPUT.CKPT, "best_model.tf"
    # )

    # for epoch in tf.range(args.TRAIN.EPOCHS, dtype=tf.int64):
    #     continue


if __name__ == '__main__':
    main()
