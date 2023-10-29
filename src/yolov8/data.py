from src.utils.clearml_utils import init_clearml
import os
import shutil
from src.data.converter.coco2yolo import Coco2Yolo
from src.data.downloader.method.cvat import CVATHTTPDownloaderV2, CVATHTTPDownloaderV1
from src.schema.coco import Coco as CocoSchema
from src.data.setup import setup_dataset
from src.utils.general import read_json
from src.data.converter.coco2yolo import count_files_in_directory, image_extensions
import plotly.graph_objects as go

class DataHandler:
    def __init__(self, args_data, task_model=None):
        self.config = args_data
        self.source_type = self.__check_source()
        self.dataset_dir = os.path.join(os.getcwd(), "dataset-yolov8")
        self.dataset_test_dir = f"{self.dataset_dir}-test"
        self.exclude_cls = self.config.get("class_exclude", [])
        self.attributes_exclude = self.config.get("attributes_exclude", None)
        self.area_segment_min = self.config.get("area_segment_min", None)
        self.task_model = task_model

    def __check_source(self):
        source_type = set()
        for source, d in self.config.items():
            print(source, d)
            # if source not in ["params", "class_exclude", "attributes_exclude", "area_segment_min", "timeout"]:
            if source not in ["cvat", "label_studio", "s3"]:
                print(f"avoid {source}")
                continue


            for k, v in d.items():
                if v is None or v == "" or v == []:
                    continue
                source_type.add(source)
                print("add source", source_type)
        if len(source_type) == 1:
            return list(source_type)[0]
        else:
            raise ValueError("source must be just 1")


    def cvat_handler(self, task_model):

        # clean up when do new training 
        if os.path.exists(self.dataset_dir):
            shutil.rmtree(self.dataset_dir)
        if os.path.exists(self.dataset_test_dir):
            shutil.rmtree(self.dataset_test_dir)
            
        total_count_files = 0

        task_id_train = self.config["cvat"]["task_ids_train"]
        task_id_test = self.config["cvat"]["task_ids_test"]

        # label_names = [label_name for label_name in label_names if label_name not in self.exclude_cls]

        is_server1, about_cvat1 = CVATHTTPDownloaderV1(self.config).get_about_server()
        is_server2, about_cvat2 = CVATHTTPDownloaderV2(self.config).get_about_server()
        if is_server1:
            print("CVAT Server V1")
            print("about_cvat", about_cvat1)
            cvat_http = CVATHTTPDownloaderV1(self.config)
        elif is_server2:
            print("CVAT Server V2")
            print("about_cvat", about_cvat2)
            cvat_http = CVATHTTPDownloaderV2(self.config)
        else:
            raise ValueError("CVAT Server not found")
            
        
        ls_path_dir_projects = cvat_http.get_local_dataset_coco(
            task_ids=task_id_train,
            annotations_only=False
        )

        for project_dir in ls_path_dir_projects:
            print("\n📁 Dataset ", project_dir, " 📁")
            
            # get annotations and check task by annotations
            ann_train_val = os.path.join(project_dir, "annotations", "instances_default.json")
            d_anns = read_json(ann_train_val)
            coco = CocoSchema(**d_anns)
            # coco.filter_category(exclude_class=self.exclude_cls)
            # print("category", coco.get_categoryid_to_namecat())
            annotation_type = coco.checking_task()
            print("annotation_type", annotation_type, "task_model", self.task_model)
            use_segments = True if 'segmentation' in annotation_type else False
            if self.task_model == "detect":
                use_segments = False

            # converting raw coco -> yolo
            converter = Coco2Yolo(src_dir=project_dir, output_dir=self.dataset_dir)
            output_train, label_names, countfiles = converter.convert(
                use_segments=use_segments, 
                exclude_class=self.exclude_cls, 
                attributes_excluded=self.attributes_exclude,
                area_segment_min=self.area_segment_min
            )
            total_count_files += countfiles


        if task_id_test == [] or task_id_test is None:
            self.dataset_test_dir = None
        else:
            ls_path_dir_projects_test = cvat_http.get_local_dataset_coco(
                task_ids=task_id_test,
                annotations_only=False,
            )
            for project_dir in ls_path_dir_projects_test:
                ann_train_val = os.path.join(project_dir, "annotations", "instances_default.json")
                d_anns = read_json(ann_train_val)
                coco = CocoSchema(**d_anns)
                annotation_type = coco.checking_task()
                use_segments = True if 'segmentation' in annotation_type else False
                converter = Coco2Yolo(src_dir=project_dir, output_dir=self.dataset_test_dir)
                output_train, label_names, countfiles = converter.convert(
                    use_segments=use_segments, 
                    exclude_class=self.exclude_cls, 
                    attributes_excluded=self.attributes_exclude
                )   
                total_count_files += countfiles

        print("\n🧮 TOTAL COUNT IMAGES", total_count_files)
        print("label_names:", label_names)
        setup_dataset(
            dataset_dir=self.dataset_dir,
            dataset_test=self.dataset_test_dir,
            label_names=label_names,
            train_ratio=self.config["params"]["train_ratio"],
            valid_ratio=self.config["params"]["val_ratio"],
            test_ratio=self.config["params"]["test_ratio"],
            class_exclude=self.exclude_cls,
        )
        count_imgs_train = count_files_in_directory(os.path.join(self.dataset_dir, "train"), extensions=image_extensions)
        count_lbls_train = count_files_in_directory(os.path.join(self.dataset_dir, "train"), extensions=["txt"])
        count_imgs_val = count_files_in_directory(os.path.join(self.dataset_dir, "valid"), extensions=image_extensions)
        count_lbls_val = count_files_in_directory(os.path.join(self.dataset_dir, "valid"), extensions=["txt"])
        count_imgs_test = count_files_in_directory(os.path.join(self.dataset_dir, "test"), extensions=image_extensions)
        count_lbls_test = count_files_in_directory(os.path.join(self.dataset_dir, "test"), extensions=["txt"])
        print("🧮 [Train] TOTAL COUNT IMAGES", count_imgs_train, "| TOTAL COUNT LABELS", count_lbls_train)
        print("🧮 [Val] TOTAL COUNT IMAGES", count_imgs_val, "| TOTAL COUNT LABELS", count_lbls_val)
        print("🧮 [Test] TOTAL COUNT IMAGES", count_imgs_test, "| TOTAL COUNT LABELS", count_lbls_test)

        data_count = {
            "train": {
                "count_imgs": count_imgs_train,
                "count_lbls": count_lbls_train
            },
            "val": {
                "count_imgs": count_imgs_val,
                "count_lbls": count_lbls_val
            },
            "test": {
                "count_imgs": count_imgs_test,
                "count_lbls": count_lbls_test
            }
        }
        fig = self.visualize_data(data_count)
        return data_count, fig


    def export(self, task_model):
        figure = None
        if self.source_type == "s3":
            pass
        elif self.source_type == "cvat":
            result, figure = self.cvat_handler(task_model=task_model)

        elif self.source_type == "label_studio":
            pass
        else:
            raise ValueError("Cek config datanya pak. source must be s3, cvat or label_studio")
        
        return self.dataset_dir, figure
    
    def visualize_data(self, data):
        datasets = list(data.keys())

        # Mengumpulkan data count_imgs dan count_lbls untuk setiap dataset
        count_imgs = [data[dataset]['count_imgs'] for dataset in datasets]
        count_lbls = [data[dataset]['count_lbls'] for dataset in datasets]

        # Membuat bar chart
        fig = go.Figure(data=[
            go.Bar(name='Images', x=datasets, y=count_imgs),
            go.Bar(name='Labels', x=datasets, y=count_lbls)
        ])

        # Mengubah tampilan diagram
        fig.update_layout(
            title='Comparison of Image and Label Counts',
            xaxis_title='Dataset',
            yaxis_title='Count',
            barmode='group'  # Ini akan menampilkan diagram batang berdampingan
        )

        # Menampilkan diagram
        return fig
    
if __name__ == "__main__":
    from src.utils.clearml_utils import init_clearml
    task = init_clearml()
    
    from src.config import (
        args_augment, args_export, args_logging,
        args_task, args_data, args_train, args_val
    )
    
    from src.utils.general import get_task_yolo_name, yaml_loader, model_name_handler

    task_yolo = get_task_yolo_name(args_task["model_name"])

    handler = DataHandler(args_data=args_data, task_model=task_yolo)
    dataset_folder = handler.export(task_model=task_yolo)
    