import os
import shutil
from src.data.converter.coco2yolo import Coco2Yolo
from src.data.downloader.method.cvat import (
    CVATHTTPDownloader, 
    CVATSDKDownloader
)
from src.schema.coco import Coco as CocoSchema
import pytest
from src.data.setup import setup_dataset
from src.utils.general import read_json
import src.env


def test_integration_cvat_downloader_to_coco_test_dedicated():

    print("Downloading")

    dataset_dir = os.path.join(os.getcwd(), "dataset-yolov8")
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    dataset_test_dir = f"{dataset_dir}-test"
    if os.path.exists(dataset_test_dir):
        shutil.rmtree(dataset_test_dir)

    cvat_http = CVATHTTPDownloader()
    cvat_sdk = CVATSDKDownloader()

    ls_path_dir_train_val = cvat_http.get_local_dataset_coco(
        task_ids=[34],
        annotations_only=False
    )

    ls_path_dir_test = cvat_sdk.get_local_dataset_coco(
        task_ids=[35],
        annotations_only=False
    )

    dir_train_val = ls_path_dir_train_val[0]
    dir_test = ls_path_dir_test[0]

    ann_train_val = os.path.join(dir_train_val, "annotations", "instances_default.json")
    d_ann_train_val_detec = read_json(ann_train_val)
    task_train_val = CocoSchema(**d_ann_train_val_detec).checking_task()
    assert len(task_train_val) == 1, "len(task_train_val) != 1"
    assert task_train_val[0] == 'detection', "task_train_val != ['detection']"
    
    ann_test = os.path.join(dir_test, "annotations", "instances_default.json")
    d_ann_test_detec = read_json(ann_test)
    task_test = CocoSchema(**d_ann_test_detec).checking_task()
    assert len(task_test) == 2, "len(task_test) != 2"
    assert 'detection' in task_test, "'detection' NOT IN task_test"

    # doing convert
    use_segments = True if 'segmentation' in task_train_val else False

    converter_train = Coco2Yolo(src_dir=dir_train_val, output_dir=dataset_dir)
    output_train, label_names = converter_train.convert(use_segments=use_segments)
    assert os.path.exists(output_train), f"dir {output_train} not exists"

    converter_test = Coco2Yolo(src_dir=dir_test, output_dir=dataset_test_dir)
    output_test, _ = converter_test.convert(use_segments=use_segments)
    assert os.path.exists(output_test), f"dir {output_test} not exists"

    # doing split
    setup_dataset(
        dataset_dir=output_train, 
        dataset_test=output_test, 
        label_names=label_names
    )

@pytest.mark.parametrize(
    "train_ratio, val_ratio, test_ratio", [
        (0.5, 0.25, 0.25),
        (0.5, 0.25, None),
    ]
)
def test_integration_cvat_downloader_to_coco_auto_train_val_test(
        train_ratio, val_ratio, test_ratio
    ):
    dataset_dir = os.path.join(os.getcwd(), "dataset-yolov8-train-val")
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    print("Downloading")

    cvat_http = CVATHTTPDownloader()

    ls_path_dir_train_val = cvat_http.get_local_dataset_coco(
        task_ids=[34],
        annotations_only=False
    )

    dir_train_val = ls_path_dir_train_val[0]

    ann_train_val = os.path.join(dir_train_val, "annotations", "instances_default.json")
    d_ann_train_val_detec = read_json(ann_train_val)
    task_train_val = CocoSchema(**d_ann_train_val_detec).checking_task()
    assert len(task_train_val) == 1, "len(task_train_val) != 1"
    assert task_train_val[0] == 'detection', "task_train_val != ['detection']"
    
    # doing convert
    use_segments = True if 'segmentation' in task_train_val else False

    converter_train = Coco2Yolo(src_dir=dir_train_val, output_dir=dataset_dir)
    output_train, label_names = converter_train.convert(use_segments=use_segments)
    assert os.path.exists(output_train), f"dir {output_train} not exists"


    # doing split
    setup_dataset(
        dataset_dir=output_train, 
        label_names=label_names,
        train_ratio=train_ratio,
        valid_ratio=val_ratio,
        test_ratio=test_ratio
    )

    ls_dir = ["train", "valid", "test"]
    for dir in ls_dir:
        img_p = os.path.join(dataset_dir, dir, "images")
        lbl_p = os.path.join(dataset_dir, dir, "labels")
        
        if test_ratio is None:
            continue

        assert os.path.exists(img_p), f"dir {img_p} not exists"
        assert os.path.exists(lbl_p), f"dir {lbl_p} not exists"
        
        len_img = len(os.listdir(img_p))
        len_lbl = len(os.listdir(lbl_p))

        assert len_img == len_lbl, f"len_img != len_lbl in {dir}"
        assert len_img > 0, f"len_img == 0 in {dir}"
        assert len_lbl > 0, f"len_lbl == 0 in {dir}"