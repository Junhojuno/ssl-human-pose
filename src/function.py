from typing import List, Callable
import numpy as np
from collections import OrderedDict
import json
from tempfile import NamedTemporaryFile

from tqdm import tqdm
import cv2
import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.metrics import accuracy
from src.eval import (
    flip,
    flip_outputs,
    suppress_stdout,
    print_coco_eval,
    get_max_preds,
    STATS_NAMES
)


@tf.function
def train_step(inputs, model, optimizer, criterion):
    sup_images, sup_hms = inputs
    with tf.GradientTape() as tape:
        outputs = model(sup_images, training=True)
        loss = criterion(sup_hms, outputs)
        loss = tf.math.reduce_mean(loss)
        loss += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables)
    )
    avg_acc, valid_cnt = accuracy(
        sup_hms, outputs, tf.constant(0.5, tf.float32)
    )

    return (
        loss,
        avg_acc,
        valid_cnt,
    )


@tf.function
def val_step(inputs, model, criterion):
    sup_images, sup_hms = inputs
    outputs = model(sup_images, training=False)
    loss = criterion(sup_hms, outputs)
    loss = tf.math.reduce_mean(loss)
    avg_acc, valid_cnt = accuracy(
        sup_hms, outputs, tf.constant(0.5, tf.float32)
    )
    return (
        loss,
        avg_acc,
        valid_cnt,
    )

@tf.function
def train_step_dual(inputs, model, optimizer, criterion_dual):
    (sup_images, sup_hms), unsup_images = inputs
    with tf.GradientTape() as tape:
        outputs = model(sup_images, unsup_images, training=True)
        loss = criterion_dual(sup_hms, outputs)
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
    return (
        loss,
        h_avg_acc, l_avg_acc,
        h_valid_cnt, l_valid_cnt
    )


@tf.function
def val_step_dual(inputs, model, criterion):
    sup_images, sup_hms = inputs
    outputs = model(
        sup_images,
        tf.zeros_like(sup_images),
        training=False
    )
    heavy_loss = criterion(sup_hms, outputs[0])
    lite_loss = criterion(sup_hms, outputs[1])

    heavy_loss = tf.math.reduce_mean(heavy_loss)
    lite_loss = tf.math.reduce_mean(lite_loss)

    h_avg_acc, h_valid_cnt = accuracy(
        sup_hms, outputs[0], tf.constant(0.5, tf.float32)
    )
    l_avg_acc, l_valid_cnt = accuracy(
        sup_hms, outputs[1], tf.constant(0.5, tf.float32)
    )
    return (
        heavy_loss, lite_loss,
        h_avg_acc, l_avg_acc,
        h_valid_cnt, l_valid_cnt
    )


def validate(
    model,
    eval_ds,
    input_shape: List,
    coco_path: str,
    print_name: str,
    print_func: Callable,
    use_flip: bool = True,
):
    """calculate AP"""
    with suppress_stdout():
        coco = COCO(coco_path)

    results = []
    for img_ids, imgs, Ms in tqdm(eval_ds, '[evaluation]'):
        img_ids = img_ids.numpy()
        Ms = Ms.numpy()

        pred_hms = model(imgs, training=False)
        pred_hms = pred_hms.numpy()

        if use_flip:
            imgs_fliped = flip(imgs)
            pred_hms_flipped = model(imgs_fliped, training=False)
            pred_hms_flipped = flip_outputs(pred_hms_flipped)

            pred_hms = (pred_hms + pred_hms_flipped) * 0.5

        xy, scores = get_max_preds(pred_hms)
        xy /= [input_shape[1] / 4, input_shape[0] / 4]
        xy *= [input_shape[1], input_shape[0]]
        pred_kpts = np.concatenate([xy, scores], axis=-1)

        kp_scores = scores[..., 0].copy()

        rescored_score = np.zeros((imgs.numpy().shape[0],))
        for i in range(imgs.numpy().shape[0]):
            M_inv = cv2.invertAffineTransform(Ms[i])
            pred_kpts[i, :, :2] = \
                np.matmul(M_inv[:, :2], pred_kpts[i, :, :2].T).T + \
                M_inv[:, 2].T

            # rescore
            score_mask = kp_scores[i] > 0.2  # score threshold in validation
            if np.sum(score_mask) > 0:
                rescored_score[i] = np.mean(kp_scores[i][score_mask])

            results.append(dict(image_id=int(img_ids[i]),
                                category_id=1,
                                keypoints=pred_kpts[i].reshape(-1).tolist(),
                                score=float(rescored_score[i])))

    with NamedTemporaryFile('w', suffix='.json') as fp:
        json.dump(results, fp)
        fp.flush()

        with suppress_stdout():
            result = coco.loadRes(fp.name)
            cocoEval = COCOeval(coco, result, iouType='keypoints')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

    # result_path = '{}/{}_{}.json'.format('models', 'ssl-pose', 'coco-val')

    # os.makedirs('./models', exist_ok=True)
    # with open(result_path, 'w') as f:
    #     json.dump(results, f)

    # with suppress_stdout():
    #     result = coco.loadRes(result_path)
    #     cocoEval = COCOeval(coco, result, iouType='keypoints')
    #     cocoEval.evaluate()
    #     cocoEval.accumulate()
    #     cocoEval.summarize()

    info_str = []
    for i, name in enumerate(STATS_NAMES):
        info_str.append((name, cocoEval.stats[i]))

    results = OrderedDict(info_str)
    print_coco_eval(results, print_name, print_func)
    return results['AP'], results  # return AP & all
