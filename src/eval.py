import os
import math
import numpy as np
import sys
from contextlib import contextmanager
from typing import List

import cv2
import tensorflow as tf

from src.transforms import affine_transform, normalize_image


STATS_NAMES = [
    "AP",
    "Ap .5",
    "AP .75",
    "AP (M)",
    "AP (L)",
    "AR",
    "AR .5",
    "AR .75",
    "AR (M)",
    "AR (L)",
]


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def print_coco_eval(
    name_value, full_arch_name='ResNet50_rle', print_fn=print
):
    """print out markdown format performance table
    Args:
    name_value (dict): dictionary of metric name and value
    full_arch_name (str): name of the architecture
    print_fn (print, optional): function to print results
    """
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)

    print_fn(
        "| Arch "
        + " ".join(["| {}".format(name) for name in names])
        + " |"
    )
    print_fn("|-------" * (num_values + 1) + "|")

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + "..."
    print_fn(
        "| "
        + full_arch_name
        + " "
        + " ".join(["| {:.3f}".format(value) for value in values])
        + " |"
    )


def load_eval_dataset(
    file_pattern: str,
    batch_size: int,
    num_keypoints: int = 17,
    input_shape: List = [256, 192, 3]
):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    ds = ds.interleave(tf.data.TFRecordDataset,
                       cycle_length=12,
                       block_length=48,
                       num_parallel_calls=AUTOTUNE)

    ds = ds.map(
        lambda record: parse_example_for_cocoeval(record, num_keypoints),
        num_parallel_calls=AUTOTUNE
    )

    ds = ds.map(
        lambda img_id, image, bbox, keypoints: preprocess_for_cocoeval(
            img_id,
            image,
            bbox,
            keypoints,
            input_shape=input_shape),
        num_parallel_calls=AUTOTUNE
    )
    ds = ds.batch(batch_size, drop_remainder=True,
                  num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return ds


def parse_example_for_cocoeval(record, num_keypoints):
    feature_description = {
        'img_id': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'bbox': tf.io.FixedLenFeature([4, ], tf.float32),
        'keypoints': tf.io.FixedLenFeature([num_keypoints * 3, ], tf.float32),
    }
    example = tf.io.parse_example(record, features=feature_description)
    image = tf.io.decode_jpeg(example['image_raw'], channels=3)
    bbox = example['bbox']
    keypoints = tf.reshape(example['keypoints'], (-1, 3))
    img_id = example['img_id']
    return img_id, image, bbox, keypoints


def preprocess_for_cocoeval(img_id, img, bbox, kp, input_shape):
    kp = tf.cast(kp, tf.float32)

    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    bbox_center = tf.cast([x + w / 2., y + h / 2.], tf.float32)

    aspect_ratio = input_shape[1] / input_shape[0]
    h = tf.cond(
        tf.math.greater(w, aspect_ratio * h),
        lambda: w * 1.0 / aspect_ratio,
        lambda: h
    )
    scale = (h * 1.25) / input_shape[0]  # scale with bbox

    # transform to the object's center
    img, M = affine_transform(img, bbox_center, 0., scale, input_shape[:2])
    img = tf.cast(img, tf.float32)

    use_image_norm = True
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    img = tf.cond(
        tf.math.equal(use_image_norm, True),
        lambda: normalize_image(img, means, stds),
        lambda: img
    )
    return img_id, img, M


def flip(image):
    return tf.image.flip_left_right(image)


def flip_outputs(
    outputs: tf.Tensor,
    joint_pairs: List = [
        [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]
    ]
):
    outputs = outputs[:, :, ::-1, :].numpy()
    for pair in joint_pairs:
        tmp = outputs[:, :, :, pair[0]].copy()
        outputs[:, :, :, pair[0]] = outputs[:, :, :, pair[1]]
        outputs[:, :, :, pair[1]] = tmp

    return outputs


def get_preds(hms, Ms, input_shape, output_shape):
    preds = np.zeros((hms.shape[0], output_shape[-1], 3))
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            hm = hms[i, :, :, j]
            idx = hm.argmax()
            y, x = np.unravel_index(idx, hm.shape)
            px = int(math.floor(x + 0.5))
            py = int(math.floor(y + 0.5))
            if 1 < px < output_shape[1] - 1 and 1 < py < output_shape[0] - 1:
                diff = np.array([hm[py][px + 1] - hm[py][px - 1],
                                 hm[py + 1][px] - hm[py - 1][px]])
                diff = np.sign(diff)
                x += diff[0] * 0.25
                y += diff[1] * 0.25
            preds[i, j, :2] = [x * input_shape[1] / output_shape[1],
                               y * input_shape[0] / output_shape[0]]
            preds[i, j, -1] = hm.max() / 255

    # use inverse transform to map kp back to original image
    for j in range(preds.shape[0]):
        M_inv = cv2.invertAffineTransform(Ms[j])
        preds[j, :, :2] = np.matmul(
            M_inv[:, :2], preds[j, :, :2].T).T + M_inv[:, 2].T
    return preds


def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(
        batch_heatmaps, np.ndarray
    ), "batch_heatmaps should be numpy.ndarray"
    assert batch_heatmaps.ndim == 4, "batch_images should be 4-ndim"

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals
