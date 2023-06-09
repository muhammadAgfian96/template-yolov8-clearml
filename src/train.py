import json
import os
import shutil
import ultralytics
from ultralytics import YOLO
from clearml import Task
from rich import print
from src.data.setup import cleanup_cache
from src.yolov8.exporter import export_handler

from src.utils.general import get_task_yolo_name, yaml_loader, model_name_handler
from src.config import (
    args_augment,
    args_export,
    args_logging,
    args_task,
    args_data,
    args_train,
    args_val
)
from src.yolov8.callbacks import callbacks
from src.yolov8.data import DataHandler

# Task.force_requirements_env_freeze(False, os.path.join(os.getcwd(), "requirements.txt"))
Task.add_requirements(os.path.join(os.getcwd(), "requirements.txt"))
task = Task.init(
    project_name="Debugging/Yolov8",
    task_name="yolov8-train-debugging",
    reuse_last_task_id=False,
    # tags=['template-v2.0', 'debug'],
    auto_connect_frameworks={"pytorch": False, "matplotlib": False},
)
Task.current_task().set_script(
    repository="",
    branch="feat/datahandler",
    working_dir=".",
    entry_point="src/training.py",
)

Task.current_task().set_base_docker(
    docker_image="pytorch/pytorch:latest",
    docker_arguments=["--ipc=host", "--gpus all"],
    docker_setup_bash_script=[
        "apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg"
        " ffmpeg libsm6 libxext6"
    ],
)
print("ultralytics: version", ultralytics.__version__)
Task.current_task().connect(args_task, name="1. Task")
Task.current_task().connect(args_data, name="2. Data")
Task.current_task().connect(args_augment, name="3. Augment")
Task.current_task().connect(args_train, name="4. Training")
Task.current_task().connect(args_val, name="5. Testing")
Task.current_task().connect(args_export, name="6. Export")

# Merge all args
args_train.update(args_logging)
args_train.update(args_augment)

task_yolo = get_task_yolo_name(args_task["model_name"])
model_name = model_name_handler(args_task["model_name"])

# Download Data
print("\n[Downloading Data]")
handler = DataHandler(args_data=args_data)
dataset_folder = handler.export(task_model=task_yolo)

data_yaml_file = os.path.join(dataset_folder, "data.yaml")
if task_yolo == "classify":
    data_yaml_file = dataset_folder
datadotyaml = yaml_loader(data_yaml_file)

# Tagging
tags = ['template-v2.0', 'debug']
Task.current_task().set_tags(tags)
Task.current_task().add_tags(task_yolo)
Task.current_task().add_tags(os.path.basename(model_name))
Task.current_task().add_tags(handler.source_type.upper())

# Utils
Task.current_task().set_model_label_enumeration(
    {cls_name: idx for idx, cls_name in enumerate(datadotyaml["names"])}
)
print("datadotyaml", datadotyaml)


model_yolo = YOLO(model=model_name)

print("Override Callbacks")
for event, func in callbacks.items():
    print(event, func)
    model_yolo.clear_callback(event)
    model_yolo.add_callback(event, func)

args_val["imgsz"] = args_train["imgsz"]
model_yolo.train(data=data_yaml_file, **args_train)
# cleanup_cache(dataset_folder)
model_yolo.val()

export_handler(
    yolo=model_yolo,
    task_yolo=task_yolo, 
    dataset_folder=dataset_folder,
    args_export=args_export,
    args_training=args_train,
    args_task=args_task
)