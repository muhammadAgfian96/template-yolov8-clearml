import json
import os
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
from src.schema.coco import Coco as CocoSchema
from glob import glob
from uuid import uuid4

class Coco2Yolo:
    def __init__(self, src_dir, output_dir='./yolov8-dataset'):
        self.src_dir = src_dir
        self.output_dir = output_dir

        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

        self.src_img_dir = os.path.join(self.src_dir, 'images')
        self.src_lbl_filepath = os.path.join(self.src_dir, 'annotations', 'instances_default.json')

    @staticmethod
    def __min_index(arr1, arr2):
        """Find a pair of indexes with the shortest distance. 
        Args:
            arr1: (N, 2).
            arr2: (M, 2).
        Return:
            a pair of indexes(tuple).
        """
        dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
        return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

    @staticmethod
    def __merge_multi_segment(segments):
        """Merge multi segments to one list.
        Find the coordinates with min distance between each segment,
        then connect these coordinates with one thin line to merge all 
        segments into one.
        Args:
            segments(List(List)): original segmentations in coco's json file.
                like [segmentation1, segmentation2,...], 
                each segmentation is a list of coordinates.
        """
        s = []
        segments = [np.array(i).reshape(-1, 2) for i in segments]
        idx_list = [[] for _ in range(len(segments))]

        # record the indexes with min distance between each segment
        for i in range(1, len(segments)):
            idx1, idx2 = Coco2Yolo.__min_index(segments[i - 1], segments[i])
            idx_list[i - 1].append(idx1)
            idx_list[i].append(idx2)

        # use two round to connect all the segments
        for k in range(2):
            # forward connection
            if k == 0:
                for i, idx in enumerate(idx_list):
                    # middle segments have two indexes
                    # reverse the index of middle segments
                    if len(idx) == 2 and idx[0] > idx[1]:
                        idx = idx[::-1]
                        segments[i] = segments[i][::-1, :]

                    segments[i] = np.roll(segments[i], -idx[0], axis=0)
                    segments[i] = np.concatenate([segments[i], segments[i][:1]])
                    # deal with the first segment and the last one
                    if i in [0, len(idx_list) - 1]:
                        s.append(segments[i])
                    else:
                        idx = [0, idx[1] - idx[0]]
                        s.append(segments[i][idx[0]:idx[1] + 1])

            else:
                for i in range(len(idx_list) - 1, -1, -1):
                    if i not in [0, len(idx_list) - 1]:
                        idx = idx_list[i]
                        nidx = abs(idx[1] - idx[0])
                        s.append(segments[i][nidx:])
        return s
    
    
    @staticmethod
    def convert_coco_to_yolo(json_path, use_segments=False, output_dir='./labels'):
        """
        Convert coco format to yolo format.
        Args:
            json_path: (str) path to coco json file.
            use_segments: (bool) whether to use segments to represent bbox.
            output_dir: (str) path to save yolo format labels.
        Return:
            list of labels.
        """
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        with open(json_path) as f:
            coco_d = json.load(f)

        coco = CocoSchema(**coco_d)

        images = coco.get_imageid_to_image()
        imgToAnns = coco.get_imageid_to_annotations()
        
        # write labels in txt file
        cat_id2name =coco.get_categoryid_to_namecat()

        for image_id, img_annotatins in tqdm(imgToAnns.items(), desc="Converting COCO to YOLO"):
            img = images[image_id]
            h, w, fp_image =img.height, img.width, img.file_name

            bboxes = []
            segments = []

            for ann in img_annotatins:
                if ann.iscrowd:
                    continue

                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann.bbox, dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue
                
                cls = ann.category_id - 1 # here because category_id starts from 1
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                
                if use_segments:
                    if len(ann.segmentation) > 1:
                        s = Coco2Yolo.__merge_multi_segment(ann.segmentation)
                        s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                    else:
                        s = [j for i in ann.segmentation for j in i]  # all segments concatenated
                        s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                    
                    s = [cls] + s
                    if s not in segments:
                        segments.append(s)

            # Write for each images
            filename = fp_image.split('/')[-1]
            ext_file = filename.split('.')[-1]
            txt_file = os.path.join(output_dir, filename.replace(ext_file, 'txt'))
            with open(txt_file, 'a') as file:
                for i in range(len(bboxes)):
                    line = *(segments[i] if use_segments else bboxes[i]),  # cls, box or segments
                    file.write(('%g ' * len(line)).rstrip() % line + '\n')

        return list(cat_id2name.values())
    
    def __setup_directory(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

        # create new output directry
        self.out_img_dir = os.path.join(self.output_dir, 'images')
        self.out_lbl_dir = os.path.join(self.output_dir, 'labels')
        Path(self.out_img_dir).mkdir(parents=True, exist_ok=True)
        Path(self.out_lbl_dir).mkdir(parents=True, exist_ok=True)

        # copy to new output directory with uuid name
        for root, dirs, files in os.walk(self.src_img_dir):
            for filename in files:
                
                filename_only, ext = os.path.splitext(filename)
                new_filename_wo_ext = filename_only+'_' + str(uuid4().hex)
                new_filename_img = new_filename_wo_ext + ext
                new_filename_lbl = new_filename_wo_ext + '.txt'

                # images
                src_img_file = os.path.join(root, filename)
                dest_img_file = os.path.join(self.out_img_dir, new_filename_img)
                shutil.copy2(src_img_file, dest_img_file)

                # labels
                src_lbl_file = os.path.join(self.src_lbl_yolo, filename_only+'.txt')
                dest_lbl_file = os.path.join(self.out_lbl_dir, new_filename_lbl)
                shutil.copy2(src_lbl_file, dest_lbl_file)

    def convert(self, use_segments:bool):
        print("Start Converting COCO to YOLO")
        self.src_lbl_yolo = os.path.join(self.src_dir, "labels")
        list_categories = Coco2Yolo.convert_coco_to_yolo(
            json_path=self.src_lbl_filepath, 
            use_segments=use_segments,
            output_dir=self.src_lbl_yolo
        )
        print("Setup Directory")
        self.__setup_directory()
        print("Done")
        return self.output_dir, list_categories