IMAGE_NAME=yolov8:gpu-py3.10.11
IMAGE_NAME_CUSTOM=yolov8-custom:gpu-py3.10.11


run:
	docker run \
	--ipc=host \
	-it --rm --gpus all \
	-v ${PWD}:/workspace \
	-u $(id -u):$(id -g) \
	-e ENABLE_DEFAULT_CLEARML_CALLBACKS=false \
	-e PYTHONPATH=/workspace \
	-w /workspace \
	-v /home/agfian/clearml.conf:/root/clearml.conf \
	$(IMAGE_NAME) \
	bash

run-custom:
	docker run \
	--ipc=host \
	-it --rm --gpus all \
	-v ${PWD}:/workspace \
	-u $(id -u):$(id -g) \
	-e ENABLE_DEFAULT_CLEARML_CALLBACKS=false \
	-e PYTHONPATH=/workspace \
	-w /workspace \
	-v /home/agfian/clearml.conf:/root/clearml.conf \
	$(IMAGE_NAME_CUSTOM) \
	bash

build:
	docker build -t $(IMAGE_NAME) .

build-custom:
	docker build -t $(IMAGE_NAME_CUSTOM) .

test_code:
	pytest tests -v 