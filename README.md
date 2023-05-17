# TASK 1: Hero Name Recognition

## Enviroment:

- Ubuntu: 18.04
- CUDA Version: 12.0 
- Python 3.10.11

## Dependencies:

- torch==2.0.0+cu118
- torchvision==0.15.1+cu118
- albumentations==1.2.1
- opencv-python
- pandas
- numpy
- Pillow==9.5.0
- NVIDIA GPU and CUDA

## Pretrained Weight & Dataset to oneshot

1. Download [Detection pretrained weight]
2. Download [CheckNet weight]

## Test

For inference, run following commands. Check specific details in main.py 

'''
python main.py --test_dir 'test_images' 
    --text_output_file 'text.txt' 
    --check_point1 'checkpoints/Faster.pth' 
    --check_point2 'checkpoints/check.pt' 
    --oneshotcheck_path 'data/oneshot_data'
'''

## Training

Training process of two architectures in 2 files (Faster.ipynb and Fewshot.ipynb)

```
@Information{Task:1,
    author    = {Nguyen Duc Toan},
    title     = {Faster RCNN + Few-shot learning with Siamese Network},
    month     = {may},
    year      = {2023},
    Birthday  = {31/08/2000}
}
```