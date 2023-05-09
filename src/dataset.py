from pathlib import Path
from typing import Dict

import tensorflow as tf

# from coco import MixCOCO
from src.transforms import (
    parse_example,
    parse_example_unlabeled,
    preprocess,
    preprocess_unlabeled
)
from src.heatmap import batch_gen_heatmap


def load_mixed_dataset(
    args: Dict,
    data_dir: Path
):
    labeled_ds = load_dataset(
        args,
        str(data_dir / args.DATASET.TRAIN.PATTERN),
        'train',
        args.TRAIN.BATCH_SIZE,
        use_aug=True

    )
    unlabeled_ds = load_unlabeled_dataset(
        args,
        str(data_dir / args.DATASET.UNLABELED.PATTERN),
        args.TRAIN.BATCH_SIZE,
        use_aug=True
    )
    ds = tf.data.Dataset.zip(
        (labeled_ds, unlabeled_ds),
        name='ssl_dataset'
    )
    return ds


def load_dataset(
    args: Dict,
    file_pattern: str,
    mode: str,
    batch_size: int,
    use_aug: bool = False
):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    ds = ds.interleave(tf.data.TFRecordDataset,
                       cycle_length=12,
                       block_length=48,
                       num_parallel_calls=AUTOTUNE)
    if mode == 'train':
        ds = ds.shuffle(
            buffer_size=5, reshuffle_each_iteration=True
        )
    ds = ds.map(
        lambda record: parse_example(record, args.DATASET.COMMON.K),
        num_parallel_calls=AUTOTUNE
    )
    ds = ds.map(
        lambda image, bbox, keypoints: preprocess(
            image, bbox, keypoints,
            use_image_norm=args.DATASET.COMMON.IMAGE_NORM,
            means=args.DATASET.COMMON.MEANS,
            stds=args.DATASET.COMMON.STDS,
            scale_factor=args.AUG.SCALE_FACTOR,
            rotation_prob=args.AUG.ROT_PROB,
            rotation_factor=args.AUG.ROT_FACTOR,
            flip_prob=args.AUG.FLIP_PROB,
            flip_kp_indices=args.AUG.KP_FLIP,
            half_body_prob=args.AUG.HALF_BODY_PROB,
            half_body_min_kp=args.AUG.HALF_BODY_MIN_KP,
            kpt_upper=args.AUG.KP_UPPER,
            input_shape=args.DATASET.COMMON.INPUT_SHAPE,
            use_aug=use_aug,
            use_album_aug=args.AUG.ALBUM,
        ),
        num_parallel_calls=AUTOTUNE
    )
    ds = ds.batch(batch_size, drop_remainder=True,
                  num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    output_shape = [
        args.DATASET.COMMON.INPUT_SHAPE[0] / 4,
        args.DATASET.COMMON.INPUT_SHAPE[1] / 4
    ]
    ds = ds.map(
        lambda images, keypoints: (
            images, batch_gen_heatmap(
                keypoints,
                output_shape,
                args.DATASET.COMMON.K,
                args.DATASET.COMMON.SIGMA
            )
        ),
        num_parallel_calls=AUTOTUNE
    )
    return ds


def load_unlabeled_dataset(
    args: Dict,
    unlabeled_pattern: str,
    batch_size: int,
    use_aug: bool = False
):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.list_files(unlabeled_pattern, shuffle=True)
    ds = ds.interleave(tf.data.TFRecordDataset,
                       cycle_length=12,
                       block_length=48,
                       num_parallel_calls=AUTOTUNE)

    ds = ds.shuffle(
        buffer_size=5, reshuffle_each_iteration=True
    )

    ds = ds.map(
        parse_example_unlabeled,
        num_parallel_calls=AUTOTUNE
    )
    ds = ds.take(args.DATASET.TRAIN.EXAMPLES)

    ds = ds.map(
        lambda image, bbox: preprocess_unlabeled(
            image, bbox,
            use_image_norm=args.DATASET.COMMON.IMAGE_NORM,
            means=args.DATASET.COMMON.MEANS,
            stds=args.DATASET.COMMON.STDS,
            scale_factor=args.AUG.SCALE_FACTOR,
            rotation_prob=args.AUG.ROT_PROB,
            rotation_factor=args.AUG.ROT_FACTOR,
            flip_prob=args.AUG.FLIP_PROB,
            flip_kp_indices=args.AUG.KP_FLIP,
            input_shape=args.DATASET.COMMON.INPUT_SHAPE,
            use_aug=use_aug,
            use_album_aug=args.AUG.ALBUM,
        ),
        num_parallel_calls=AUTOTUNE
    )
    ds = ds.batch(batch_size, drop_remainder=True,
                  num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return ds


if __name__ == '__main__':
    import yaml
    from easydict import EasyDict

    cwd = Path('.').resolve()

    yaml_file = cwd / 'config/test.yaml'
    data = yaml.load(
        open(yaml_file, 'r'), Loader=yaml.Loader
    )
    args = EasyDict(data)

    train_ds = load_dataset(
        args,
        str(
            cwd.parent
            / 'datasets'
            / args.DATASET.NAME
            / args.DATASET.TRAIN.PATTERN
        ),
        'train',
        1,
        True
    )
    # for inputs in train_ds.take(10):
    #     for inp in inputs:
    #         print(inp.shape, end='  ')
    #     print()
    # (1, 256, 192, 3)  (1, 64, 48, 17)  (1, 64, 48, 17, 2)  (1, 64, 48, 17, 1)

    unlabeled_ds = load_unlabeled_dataset(
        args,
        str(
            cwd.parent
            / 'datasets'
            / args.DATASET.NAME
            / args.DATASET.UNLABELED.PATTERN
        ),
        1,
        True
    )
    # for inputs in unlabeled_ds.take(10):
    #     for inp in inputs:
    #         print(inp.shape, end='  ')
    #     print()
    # (256, 192, 3)

    mix_ds = load_mixed_dataset(
        (train_ds, unlabeled_ds)
    )
    for inputs in mix_ds.take(10):
        for inp in inputs:
            for i in inp:
                print(i.shape, end='  ')
        print(len(inputs))
    # (1, 256, 192, 3)  (1, 64, 48, 17)  (1, 64, 48, 17, 2)  (1, 64, 48, 17, 1)  (256, 192, 3)  2
