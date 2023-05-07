import os
import json
from typing import List, Dict
from pathlib import Path
import numpy as np

import tensorflow as tf
from pycocotools.coco import COCO


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte (list)."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double (list)."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint (list)."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(annot: Dict):
    feature = dict()
    image_raw = open(annot['image_file'], 'rb').read()
    feature['image_id'] = _int64_feature([annot['image_id']])
    feature['image_raw'] = _bytes_feature([image_raw])
    feature['bbox'] = _float_feature(annot['bbox'])
    return tf.train.Example(features=tf.train.Features(feature=feature))


def load_mscoco_unlabeled(
    coco_dir: Path,
    image_thr: float = 0.0
) -> List[Dict]:
    path = coco_dir / 'annotations/image_info_unlabeled2017.json'
    unlabeled_coco = COCO(path)

    bbox_file = os.path.join(
        coco_dir,
        'annotations',
        'unlabeled2017_detections_person_faster_rcnn.json'
    )
    all_boxes = None
    with open(bbox_file, 'r') as f:
        all_boxes = json.load(f)

    if not all_boxes:
        print('=> Load %s fail!' % bbox_file)
        return None

    print(f'=> Total boxes: {len(all_boxes)}')
    annots = []
    num_boxes = 0
    for det_res in all_boxes:
        if det_res['category_id'] != 1:
            continue
        image_id = det_res['image_id']
        image_file = coco_dir / f'images/unlabeled2017/{image_id:012d}.jpg'
        bbox = det_res['bbox']
        score = det_res['score']

        if score < image_thr:
            continue

        image_info = unlabeled_coco.loadImgs([image_id])[0]
        image_height = image_info['height']
        image_width = image_info['width']
        x1, y1, w, h = bbox
        x1 = np.max((0, x1))
        y1 = np.max((0, y1))
        x2 = np.min((image_width - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((image_height - 1, y1 + np.max((0, h - 1))))

        if x2 > x1 or y2 > y1:

            num_boxes = num_boxes + 1
            annots.append(
                {
                    'image_id': image_id,
                    'image_file': image_file,
                    'bbox': bbox,
                }
            )
    print(f'=> Total boxes after fliter low score@{image_thr}: {num_boxes}')
    return annots


def convert_to_tfrecord(
    coco_dir: Path,
    mode: str,
    save_dir: str,
    shard_size: int = 1024
) -> None:
    save_dir = os.path.join(save_dir, mode)
    os.makedirs(save_dir, exist_ok=True)

    coco_annots = load_mscoco_unlabeled(coco_dir)
    N = len(coco_annots)
    print(f'========== Total objects: {N} ==========')

    i = 0
    shard_count = 0
    while i < len(coco_annots):
        record_path = os.path.join(
            save_dir, f'{mode}_{N}_{shard_count:04d}.tfrecord'
        )
        with tf.io.TFRecordWriter(record_path) as writer:
            for j in range(shard_size):
                example = serialize_example(coco_annots[i])
                writer.write(example.SerializeToString())
                i += 1
                if i == len(coco_annots):
                    break
            if i >= len(coco_annots):
                break
        print('Finished writing', record_path)
        shard_count += 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cwd', '-d',
        default='./',
        required=False,
        help='current working directory'
    )
    args = parser.parse_args()

    cwd = Path(args.cwd).resolve()
    coco_dir = cwd.parent / 'datasets' / 'mscoco'
    save_dir = coco_dir / 'tfrecords'

    convert_to_tfrecord(
        coco_dir,
        mode='unlabeled',
        save_dir=save_dir,
        shard_size=1024
    )
