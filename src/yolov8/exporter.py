import os
import yaml
from yaml.loader import SafeLoader
from clearml import Task, OutputModel
from rich import print


def export_handler(yolo, task_yolo, dataset_folder, args_export, args_training, args_task):
    print(f'\n[Export Model]')

    with open(os.path.join(dataset_folder, "data.yaml"), "r") as f:
        data_yaml = yaml.load(f, Loader=SafeLoader)

    for format, is_use in args_export.items():
        try:
            if not is_use:
                continue
            
            if format == "engine":
                import torch

                print("torch.cuda.is_available():", torch.cuda.is_available())
                path_model = yolo.export(format=format, device=0)
            else:
                path_model = yolo.export(format=format)
            print(path_model)
            output_model_last = OutputModel(
                task=Task.current_task(),
                name=format + "-" + args_task["model_name"],
                comment=str(data_yaml["names"]),
            )
            output_model_last.update_weights(
                weights_filename=path_model, 
                target_filename=args_task["model_name"]+"."+format, 
                auto_delete_file=False
            )
            output_model_last.update_labels(
                {lbl: idx for idx, lbl in enumerate(data_yaml["names"])}
            )

            output_model_last.update_design(
                config_dict={
                    "net": args_task["model_name"].replace(".pt", ""),
                    "imgsz": args_training["imgsz"],
                    "task": task_yolo,
                }
            )
            output_model_last.set_metadata('imgsz', args_training["imgsz"], "int")
            output_model_last.set_metadata('task', task_yolo, "str")

        except Exception as e:
            print(e)
