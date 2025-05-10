# ED-Mamba
A PET Image Reconstruction Network Based on Edge Perception and Mamba Guidance

## Model Architecture

Here is the architecture of the ED-Mamba model:

![Model Architecture](https://raw.githubusercontent.com/Ethevliu/ED-Mamba/main/assets/model.svg)



# Dataset
We used the processed version of the brain PET dataset from the Ultra-Low Dose PET Imaging Challenge 2022:

GitHub Repository: [Show-han/PET-Reconstruction (Processed Dataset)](https://github.com/Show-han/PET-Reconstruction)
After you’ve prepared your 2D-slice datasets, you need to change the datasets config to your data path.

```json

"datasets": {
    "train": {
        "name": "FFHQ",
        "mode": "HR",  
        "dataroot": "C:\\Users\\Admin\\data\\FFHQ"
    }
  }
```


# Code Structure


The code structure is as follows:
```
  ├── assets
  │   └── model.svg           # model structure
  ├── config
  │   └── sr_sr3_16_128.json  # config file for training and inference
  ├── core
  │   ├── logger.py           # logger
  │   ├── metrics.py          # function for evaluation
  │   └── wandb_logger.py     # wandb logger
  │   
  ├── datasets
  │   ├── __init__.py         # dataloader
  │   ├── LRHR_dataset.py     # dataset
  │   └── util.py             # dataset utils
  ├── model
  │   ├── sr3_modules         # main_model
  │   │   ├── guide.py        # guide net
  │   ├── __init__.py         # init
  │   ├── base_model.py       # function for model
  │   └── model.py            # The overall model
  │    
  ├── inference.py            # inference
  ├── train.py                # train
  └── paper_metric.py         # Calculation index
```


# Environment

This experiment uses the Ubuntu 20.04 operating system, with Python 3.8 and PyTorch 2.0.0 installed, and is configured with CUDA 11.8 to support GPU acceleration. A single NVIDIA RTX 3090 graphics card is utilized in the experiment.



# Training and Inference

To train the model, you can run the following command:

```bash
python train.py 
```

To inference the model, you can run the following command:

```bash
python inference.py 
```

All the experiment results and checkpoints will be saved to the 'experiments/' directory:


# Acknowledgement

We would like to thank the authors of previous related projects for generously sharing their code and insights:

- [Contrastive Diffusion Model with Auxiliary Guidance](https://github.com/Show-han/PET-Reconstruction)
- [Edge enhancement-based Densely Connected Network(EDCNN)](https://github.com/workingcoder/EDCNN)
