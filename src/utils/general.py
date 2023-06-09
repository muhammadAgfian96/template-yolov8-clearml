import yaml
from yaml.loader import SafeLoader
import json
import os
import shutil
from clearml import Task

def get_task_yolo_name(arg_model_name):
    task_yolo = "None"
    arg_model_name = arg_model_name.replace(".pt", "")
    if "-seg" in arg_model_name:
        task_yolo = "segment"
    if "-cls" in arg_model_name:
        task_yolo = "classify"
    if "-cls" not in arg_model_name and "-seg" not in arg_model_name:
        task_yolo = "detect"
    return task_yolo


def model_name_handler(arg_model_name):
    """
    return str: model_pt or path_model_yml
    """
    default_path_model_yaml = os.path.join(os.getcwd(), 'src/yolov8/yolov8.yaml')
    default_path_model_yaml = Task.current_task().connect_configuration(default_path_model_yaml, name="Model YAML")
    
    if 'yaml' in arg_model_name:
        # rename
        new_path_model_yaml = os.path.join(os.getcwd(), f'src/yolov8/{arg_model_name}')
        shutil.copy(default_path_model_yaml, new_path_model_yaml)
        
        print("config_file", default_path_model_yaml, new_path_model_yaml)
        return new_path_model_yaml
    else:
        new_model_name = arg_model_name+".pt"
        return new_model_name



def yaml_loader(filepath):
    with open(filepath, "r") as file_descriptor:
        data = yaml.load(file_descriptor, Loader=SafeLoader)
    return data


def read_json(ann_path):
    with open(ann_path, 'r') as f:
        ann_d = json.load(f)
    return ann_d