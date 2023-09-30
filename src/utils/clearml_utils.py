import os
from clearml import Task, InputModel
from src.config import (
    args_augment,
    args_export,
    args_logging,
    args_task,
    args_data,
    args_train,
    args_val
)

def init_clearml():
    print("init clearml, Task.current_task=", Task.current_task())
    if Task.current_task() is None:
        Task.add_requirements("/workspace/requirements.txt")
        task = Task.init(
            project_name="Template/Yolov8",
            task_name="yolov8-train-new-template-v2.6",
            reuse_last_task_id=False,
            auto_connect_frameworks={"pytorch": False, "matplotlib": False},
        )

        task.set_script(
            repository="https://github.com/muhammadAgfian96/template-yolov8-clearml.git",
            # branch="public",
            working_dir=".",
            entry_point="src/train.py",
        )

        task.set_base_docker(
            docker_image="yolov8-custom:gpu-py3.10.11",
            docker_arguments=["-e PYTHONPATH=/workspace", "--gpus all", "--shm-size=24g"],
        )
        tags = ['🏷️ v2.6', '🐞 debug']
        task.set_tags(tags)
    return Task.current_task()

def config_clearml():
    Task.current_task().connect(args_task, name="1_Task")
    Task.current_task().connect(args_data, name="2_Data")
    Task.current_task().connect(args_augment, name="3_Augment")
    Task.current_task().connect(args_train, name="4_Training")
    Task.current_task().connect(args_val, name="5_Testing")
    Task.current_task().connect(args_export, name="6_Export")

    exclude_data = args_data.get("class_exclude", "")
    if exclude_data is None:
        exclude_data = ""
    ls_exclude = exclude_data.replace(", ", ",").replace(" ,", ",").split(",")

    if args_task["model_latest_id"] != "":
        print("Downloading latest model")
        latest_model = InputModel(model_id=args_task["model_latest_id"])
        path_latest_model = latest_model.get_weights()
        args_train["resume"] = True
        args_task["model_name"] = path_latest_model
        print("▶️ Resume training from", latest_model)
        

    args_train.update(args_logging)
    args_train.update(args_augment)
    args_data.update({"class_exclude":ls_exclude})
    args_data.update({"attributes_exclude":args_data.get("attributes_exclude", {})})
    args_data.update({"area_segment_min":args_data.get("area_segment_min", None)})
    
    return  args_task, args_data, args_augment, args_train, args_val, args_export