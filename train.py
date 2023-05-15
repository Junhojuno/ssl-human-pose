import time
from pathlib import Path
import argparse
from numpy.random import seed

import tensorflow as tf
import wandb

from src.dataset import (
    load_dataset,
    load_mixed_dataset
)
from src.model.pose_hrnet import HRNetW32, HRNetW48
from src.model.pose_cotrain import PoseCoTrain
from src.function import (
    train_step_dual,
    val_step_dual,
    validate
)
from src.losses import pose_dual_loss_fn, keypoint_loss
from src.metrics import AverageMeter
from src.scheduler import MultiStepLR
from src.utils import parse_yaml, get_logger
from src.eval import load_eval_dataset


def load_metrics():
    train_loss = AverageMeter()
    train_h_acc = AverageMeter()
    train_l_acc = AverageMeter()

    val_h_loss = AverageMeter()
    val_l_loss = AverageMeter()
    val_h_acc = AverageMeter()
    val_l_acc = AverageMeter()
    return (
        train_loss, train_h_acc, train_l_acc,
        val_h_loss, val_l_loss, val_h_acc, val_l_acc
    )


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
    seed(0)  # numpy random seed

    cfg = define_argparser()
    args = parse_yaml(cfg.config)

    # cwd = Path('/data/Data').resolve()  # current working dir

    # train_ds = load_mixed_dataset(
    #     args,
    #     cwd / 'datasets/basic_sources' / args.DATASET.NAME
    # )

    cwd = Path('.').resolve()  # current working dir
    args.WANDB.NAME = \
        '{dataset}/{exp_title}/{model}/'\
        'bs{bs}_lr{lr}_s{sigma}_sf_{sf}_r{rot}'\
        .format(
            dataset=args.DATASET.NAME,
            exp_title=args.WANDB.SUBTITLE,
            model=args.MODEL.NAME,
            bs=args.TRAIN.BATCH_SIZE,
            lr=args.TRAIN.LR,
            sigma=args.DATASET.COMMON.SIGMA,
            sf=args.AUG.SCALE_FACTOR,
            rot=args.AUG.ROT_FACTOR
        )
    args.OUTPUT.DIR = output_dir = cwd / f'results/{args.WANDB.NAME}'

    # set save_ckpt dir
    heavy_ckpt_dir = output_dir / 'ckpt' / 'heavy'
    lite_ckpt_dir = output_dir / 'ckpt' / 'lite'
    heavy_ckpt_dir.mkdir(parents=True, exist_ok=True)
    lite_ckpt_dir.mkdir(parents=True, exist_ok=True)

    heavy_ckpt_prefix = str(heavy_ckpt_dir / 'best_model.tf')
    lite_ckpt_prefix = str(lite_ckpt_dir / 'best_model.tf')

    logger = get_logger(
        str(output_dir / 'work.log')
    )

    train_ds = load_mixed_dataset(
        args,
        cwd.parent / 'datasets' / args.DATASET.NAME
    )
    val_ds = load_dataset(
        args,
        str(
            cwd.parent
            / 'datasets'
            / args.DATASET.NAME
            / args.DATASET.VAL.PATTERN
        ),
        'val',
        args.VAL.BATCH_SIZE,
        use_aug=False
    )
    eval_ds = None
    if args.VAL.DO_EVAL:
        eval_ds = load_eval_dataset(
            str(
                cwd.parent
                / 'datasets'
                / args.DATASET.NAME
                / args.DATASET.VAL.PATTERN,
            ),
            args.VAL.BATCH_SIZE,
            args.DATASET.COMMON.K,
            args.DATASET.COMMON.INPUT_SHAPE
        )

    # initialize wandb
    run = None
    if args.WANDB.USE:
        run = wandb.init(
            project=args.WANDB.PROJECT,
            config=args,
            name=args.WANDB.NAME,
            resume=False,
            id=None,
            dir=args.OUTPUT.DIR  # should be generated right before
        )
        # define our custom x axis metric
        run.define_metric("epoch")
        run.define_metric("eval/*")
        # define which metrics will be plotted against it
        run.define_metric("loss/*", step_metric="epoch")
        run.define_metric("acc/*", step_metric="epoch")
        run.define_metric("lr", step_metric="epoch")

    model = PoseCoTrain(
        heavy_model=HRNetW48(),
        lite_model=HRNetW32(),
        scale_factor=args.AUG.SCALE_FACTOR,
        rotation_factor=args.AUG.ROT_FACTOR,
        n_joints_mask=args.AUG.N_JOINTS_MASK,
        name=args.MODEL.NAME
    )
    model.build([None, *args.DATASET.COMMON.INPUT_SHAPE])

    # set scheduler
    n_train_steps = int(
        args.DATASET.TRAIN.EXAMPLES // args.TRAIN.BATCH_SIZE
    )
    lr_scheduler = MultiStepLR(
        args.TRAIN.LR,
        lr_steps=[
            n_train_steps * epoch
            for epoch in args.TRAIN.LR_EPOCHS
        ],
        lr_rate=args.TRAIN.LR_FACTOR
    )
    optimizer = tf.keras.optimizers.Adam(lr_scheduler)

    lowest_heavy_loss = 1e+10
    lowest_lite_loss = 1e+10
    for epoch in range(args.TRAIN.EPOCHS):
        start_time = time.time()
        train_loss, train_h_acc, train_l_acc, \
            val_h_loss, val_l_loss, val_h_acc, val_l_acc = load_metrics()

        for inputs in train_ds:
            t_outputs = train_step_dual(
                inputs, model, optimizer, pose_dual_loss_fn)
            train_loss.update(t_outputs[0].numpy(), args.TRAIN.BATCH_SIZE)
            train_h_acc.update(t_outputs[1].numpy(), t_outputs[3].numpy())
            train_l_acc.update(t_outputs[2].numpy(), t_outputs[4].numpy())
        train_time = time.time() - start_time

        for inputs in val_ds:
            v_outputs = val_step_dual(inputs, model, keypoint_loss)
            val_h_loss.update(v_outputs[0].numpy(), args.VAL.BATCH_SIZE)
            val_l_loss.update(v_outputs[1].numpy(), args.VAL.BATCH_SIZE)
            val_h_acc.update(v_outputs[2].numpy(), v_outputs[4].numpy())
            val_l_acc.update(v_outputs[3].numpy(), v_outputs[5].numpy())

        current_lr = optimizer.lr(optimizer.iterations).numpy()

        total_time = time.time() - start_time

        # log results
        logger.info(
            f'Epoch[{epoch + 1:03d}/{args.TRAIN.EPOCHS}] '
            f'| TA: {int(train_time)}s/{int(total_time)}s'
        )
        logger.info(
            '\t[Train] '
            f'Loss: {float(train_loss.avg):.4f} '
            f'| H-Acc: {float(train_h_acc.avg):.4f} '
            f'| L-Acc: {float(train_l_acc.avg):.4f}'
        )
        logger.info(
            '\t[Val] '
            f'H-Loss: {float(val_h_loss.avg):.4f} '
            f'| L-Loss: {float(val_l_loss.avg):.4f} '
            f'| H-Acc: {float(val_h_acc.avg):.4f} '
            f'| L-Acc: {float(val_l_acc.avg):.4f} '
            f'[LR]: {float(current_lr)}'
        )
        if run:  # write on wandb server
            run.log(
                {
                    'loss/train': float(train_loss.avg),
                    'loss/val-h': float(val_h_loss.avg),
                    'loss/val-l': float(val_l_loss.avg),
                    'acc/train-h': float(train_h_acc.avg),
                    'acc/train-l': float(train_l_acc.avg),
                    'acc/val-h': float(val_h_acc.avg),
                    'acc/val-l': float(val_l_acc.avg),
                    'lr': float(current_lr),
                    'epoch': int(epoch + 1)
                }
            )

        # Terminate when NaN loss
        if tf.math.is_nan(train_loss.avg):
            logger.info('Training is Terminated because of NaN Loss.')
            raise ValueError('NaN Loss has coming up.')

        # save model newest weights
        model.heavy_model.save_weights(
            heavy_ckpt_prefix.replace('best', 'newest')
        )
        model.lite_model.save_weights(
            lite_ckpt_prefix.replace('best', 'newest')
        )
        if val_h_loss.avg < lowest_heavy_loss:
            lowest_heavy_loss = val_h_loss.avg
            model.heavy_model.save_weights(heavy_ckpt_prefix)
        if val_l_loss.avg < lowest_lite_loss:
            lowest_lite_loss = val_l_loss.avg
            model.lite_model.save_weights(lite_ckpt_prefix)

    # final evaluation with best model
    if eval_ds is not None:
        heavy_model = model.heavy_model
        heavy_model.load_weights(heavy_ckpt_prefix)

        lite_model = model.lite_model
        lite_model.load_weights(lite_ckpt_prefix)

        h_ap, h_details = validate(
            heavy_model,
            eval_ds,
            args.DATASET.COMMON.INPUT_SHAPE,
            args.VAL.COCO_JSON,
            'heavy_model',
            logger,
            use_flip=False
        )
        l_ap, l_details = validate(
            lite_model,
            eval_ds,
            args.DATASET.COMMON.INPUT_SHAPE,
            args.VAL.COCO_JSON,
            'lite_model',
            logger,
            use_flip=False
        )
        if args.WANDB.USE:
            h_eval_table = wandb.Table(
                data=[list(h_details.values())],
                columns=list(h_details.keys())
            )
            run.log({'eval/heavy': h_eval_table})

            l_eval_table = wandb.Table(
                data=[list(l_details.values())],
                columns=list(l_details.keys())
            )
            run.log({'eval/lite': l_eval_table})


if __name__ == '__main__':
    main()
