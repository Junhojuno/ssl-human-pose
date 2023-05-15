import time
from pathlib import Path
import argparse
from numpy.random import seed

import tensorflow as tf
import wandb

from src.dataset import load_dataset
from src.model.pose_hrnet import HRNetW32, HRNetW48
from src.model.pose_hrnet import HRNetW48, HRNetW32
from src.function import train_step, val_step, validate
from src.losses import keypoint_loss
from src.metrics import AverageMeter
from src.scheduler import MultiStepLR
from src.utils import parse_yaml, get_logger
from src.eval import load_eval_dataset


def load_metrics():
    train_loss = AverageMeter()
    train_acc = AverageMeter()

    val_loss = AverageMeter()
    val_acc = AverageMeter()
    return (
        train_loss, train_acc,
        val_loss, val_acc
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
    output_dir = cwd / f'results/{args.WANDB.NAME}'
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

    train_ds = load_dataset(
        args,
        str(
            cwd.parent
            / 'datasets'
            / args.DATASET.NAME
            / args.DATASET.TRAIN.PATTERN
        ),
        'train',
        args.TRAIN.BATCH_SIZE,
        use_aug=True
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

    if args.MODEL.NAME.endswith('w48'):
        model = HRNetW48()
    elif args.MODEL.NAME.endswith('w32'):
        model = HRNetW32()
    else:
        raise ValueError(
            "wrong args.MODEL.NAME. please check config file"
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

    lowest_loss = 1e+10
    for epoch in range(args.TRAIN.EPOCHS):
        start_time = time.time()
        train_loss, train_acc, val_loss, val_acc = load_metrics()

        for inputs in train_ds:
            t_outputs = train_step(inputs, model, optimizer, keypoint_loss)
            train_loss.update(t_outputs[0].numpy(), args.TRAIN.BATCH_SIZE)
            train_acc.update(t_outputs[1].numpy(), t_outputs[2].numpy())
        train_time = time.time() - start_time

        for inputs in val_ds:
            v_outputs = val_step(inputs, model, keypoint_loss)
            val_loss.update(v_outputs[0].numpy(), args.VAL.BATCH_SIZE)
            val_acc.update(v_outputs[1].numpy(), v_outputs[2].numpy())

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
            f'| Acc: {float(train_acc.avg):.4f} '
        )
        logger.info(
            '\t[Val] '
            f'Loss: {float(val_loss.avg):.4f} '
            f'| Acc: {float(val_acc.avg):.4f} '
            f'[LR]: {float(current_lr)}'
        )
        if run:  # write on wandb server
            run.log(
                {
                    'loss/train': float(train_loss.avg),
                    'loss/val': float(val_loss.avg),
                    'acc/train': float(train_acc.avg),
                    'acc/val': float(val_acc.avg),
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
        if val_loss.avg < lowest_loss:
            lowest_heavy_loss = val_loss.avg
            model.save_weights(heavy_ckpt_prefix)

    # final evaluation with best model
    if eval_ds is not None:
        model.load_weights(heavy_ckpt_prefix)

        _, details = validate(
            model,
            eval_ds,
            args.DATASET.COMMON.INPUT_SHAPE,
            args.VAL.COCO_JSON,
            'heavy_model',
            logger,
            use_flip=False
        )
        if args.WANDB.USE:
            eval_table = wandb.Table(
                data=[list(details.values())],
                columns=list(details.keys())
            )
            run.log({'eval': eval_table})

if __name__ == '__main__':
    main()
