from typing import List, Callable
import numpy as np
from collections import OrderedDict
import json
from tempfile import NamedTemporaryFile

from tqdm import tqdm
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.eval import (
    flip,
    flip_outputs,
    suppress_stdout,
    print_coco_eval,
    get_max_preds,
    STATS_NAMES
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
