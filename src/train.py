import ultralytics
from ultralytics import settings as settings_ultralytics
settings_ultralytics["clearml"] = False

from utils.clearml_utils import task_clearml as task
import os

from clearml import Task
from rich import print
from ultralytics import YOLO

from src.data.setup import cleanup_cache
from src.utils.clearml_utils import config_clearml
from src.utils.general import (get_task_yolo_name, model_name_handler,
                               yaml_loader)
from src.yolov8.callbacks import callbacks
from src.yolov8.data import DataHandler
from src.yolov8.exporter import export_handler

task = Task.current_task()
args_task, args_data, args_augment, args_train, args_val, args_export = config_clearml()

print("ultralytics: version", ultralytics.__version__)
task.add_tags(f"yv8-{ultralytics.__version__}")
# task.execute_remotely()

task_yolo = get_task_yolo_name(args_task["model_name"])
if not args_train["resume"]:
    model_name = model_name_handler(args_task["model_name"])
else:
    task.add_tags("resume")
    model_name = args_task["model_name"]
print("TASK_YOLO", task_yolo)

# Download Data
print("\n[Downloading Data]")
handler = DataHandler(args_data=args_data, task_model=task_yolo)
dataset_folder, figure_data_dist = handler.export(task_model=task_yolo)
task.get_logger().report_plotly(
    title="Data Distribution",
    series="Data Distribution",
    iteration=0,
    figure=figure_data_dist,
)
# dataset_folder = "dataset-yolov8"
data_yaml_file = os.path.join(dataset_folder, "data.yaml")
print("data_yaml_file", data_yaml_file)
if task_yolo == "classify":
    data_yaml_file = dataset_folder
if task_yolo == "segment":
    args_train["augment"] = False
datadotyaml = yaml_loader(data_yaml_file)


# Tagging
task.add_tags(task_yolo)
task.add_tags(os.path.basename(model_name).replace(".pt", ""))
task.add_tags(handler.source_type.upper())


# Utils
task.set_model_label_enumeration(
    {cls_name: idx for idx, cls_name in enumerate(datadotyaml["names"])}
)
print("datadotyaml", datadotyaml)

print("\n[Training]")
print("LOAD MODEL", model_name)
model_yolo = YOLO(model=model_name)


print("Override Callbacks")
for event, func in callbacks.items():
    model_yolo.clear_callback(event)
    model_yolo.add_callback(event, func)

args_val["imgsz"] = args_train["imgsz"]
if args_train["resume"]:
    print("RESUME TRAINING")
    model_yolo.resume = True
    model_yolo.train(
        data=data_yaml_file,
        epochs=args_train["epochs"],
        batch=args_train["batch"],
        patience=args_train["patience"],
        save_period=args_train["save_period"],
    )
else:
    model_yolo.train(data=data_yaml_file, **args_train)

cleanup_cache(dataset_folder)
if datadotyaml.get("test"):
    args_val["split"] = "test"
try:
    model_yolo.val(data=data_yaml_file, **args_val)
except Exception as e:
    print("Error Validation", e)

export_handler(
    yolo=model_yolo,
    task_yolo=task_yolo,
    dataset_folder=dataset_folder,
    args_export=args_export,
    args_training=args_train,
    args_task=args_task,
)
