# Triton
Hơw to run: 
- Add torchscript model:
    - Unet34: add "model.pt" to folder "model_repository/unet34/1"
    - Resnet34: add "model.pt" to folder "model_repository/resnet34/1"
- Run Triton: docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:22.12-py3 tritonserver --model-repository=/models
- Run:
    - Unet34: python client.py
    - Resnet34: python client_resnet.py
