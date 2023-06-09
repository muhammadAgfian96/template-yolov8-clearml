import os
from src.data.downloader.method.cvat import (
    CVATSDKDownloader
)
from src.schema.coco import Coco as CocoSchema
from src.utils.general import read_json
import src.env as env

def test_cvat_sdk_downloader():

    cvat_sdk = CVATSDKDownloader()

    ls_path_dirs = cvat_sdk.get_local_dataset_coco(
        task_ids=[33, 34, 35],
        annotations_only=False
    )

    ann_cls = os.path.join(ls_path_dirs[0], "annotations", "instances_default.json")
    ann_det = os.path.join(ls_path_dirs[1], "annotations", "instances_default.json")
    ann_seg = os.path.join(ls_path_dirs[2], "annotations", "instances_default.json")

    d_ann_cls = read_json(ann_cls)

    task_cls = CocoSchema(**d_ann_cls).checking_task()
    assert task_cls == []

    d_ann_detec = read_json(ann_det)
    task_detect = CocoSchema(**d_ann_detec).checking_task()
    assert len(task_detect) == 1, "len(task_detect) != 1"
    assert task_detect == ['detection'], "task_detect != ['detection']"

    d_ann_seg = read_json(ann_seg)
    task_seg = CocoSchema(**d_ann_seg).checking_task()

    assert len(task_seg) == 2, "len(task_seg) != 2"
    assert 'detection' in task_seg, "detection not in task_seg"
    assert 'segmentation' in task_seg, "segmentation not in task_seg"