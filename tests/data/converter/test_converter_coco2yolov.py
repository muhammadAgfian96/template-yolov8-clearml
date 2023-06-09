import os
from src.data.converter.coco2yolo import Coco2Yolo
from src.schema.coco import Coco as CocoSchema
from src.utils.general import read_json

def test_converter_coco2yolo():
    root = os.getcwd()
    src_dir_detection = os.path.join(root, "tmp_dir_cvat/Satelit Pond/batch-1")
    output_dir = os.path.join(root, "tmp_dir_cvat/Satelit Pond/batch-1-dst")

    converter = Coco2Yolo(src_dir=src_dir_detection, output_dir=output_dir)
    use_segments = True
    output_dir_convert, ls_names = converter.convert(use_segments=use_segments)

    assert os.path.exists(output_dir), "output_dir not exists"
    assert output_dir_convert == output_dir, "output_dir_convert not equal to output_dir"

    out_img_dir = os.path.join(output_dir, "images")
    out_lbl_dir = os.path.join(output_dir, "labels")
    ls_file_img = sorted(os.listdir(out_img_dir))
    ls_file_lbl = sorted(os.listdir(out_lbl_dir))

    for filename_img, filename_lbl in zip(ls_file_img, ls_file_lbl):
        name_img, ext_img = os.path.splitext(filename_img)
        name_lbl, ext_lbl = os.path.splitext(filename_lbl)
        assert name_img == name_lbl, "name_img not equal to name_lbl"
        assert ext_lbl == ".txt", "ext_lbl not equal to .txt"

        with open(os.path.join(out_lbl_dir, filename_lbl), "r") as f:
            lines = f.readlines()
            assert len(lines) > 0, "len(lines) not greater than 0"
            for line in lines:
                ls_line = line.replace("\n", "").split(" ")

                assert len(ls_line[0].split(".")) == 1, "cls_id not integer"
                assert len(ls_line[0]) < 2, f"cls_id not integer {ls_line[0]}"
                if not use_segments:
                    assert len(ls_line) == 5, f"len(ls_line) not equal to 5 {ls_line[0]}"

    # check the output files
    coco_path = os.path.join(src_dir_detection, "annotations", "instances_default.json")
    coco_json = read_json(coco_path)
    coco = CocoSchema(**coco_json)
    ls_task = coco.checking_task()
    assert len(ls_task) > 0, "coco checking task failed"
    assert "detection" in ls_task, "detection not in ls_task"
    assert "segmentation" not in ls_task, "segmentation not in ls_task"

    