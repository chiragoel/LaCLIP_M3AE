# CLIP-MA-MMSD
This is the codebase for using a multi-view CLIP and ViT based architecture for MultiModal Sarcasm Detection on the MMSD2.0 dataset. This has been motivated from the [paper](https://github.com/JoeYing1019/MMSD2.0/tree/main) "MMSD2.0: Towards a Reliable Multi-modal Sarcasm Detection System". 

In the following, we will guide you how to use this repository step by step.

## Requirements

We recommend to use a python enviroment with python `3.9` for these experiments.

 ```angular2html
    python -m venv /path/to/new/virtual/environment
    pip install -r requirements.txt
    source /path/to/new/virtual/environment/bin/activate
    cd src  
  ```

## Dataset

Please download the dataset from the readme [here](https://github.com/JoeYing1019/MMSD2.0/tree/main) and put it in the `dataset` folder. The text json files are already present. Please modify the `WORKING_DIR` path in the `data_set.py` and `data_set_original.py` file according to where the `dataset` folder is present.

## Usage

You can tun the code in this repo using the following command:

```angular2html
    python3 main.py --model=<model_name> --text_name=<name of text json file> --aug=<Whether to use text based augs or not> --replicate_mmae=<Do you want the ViT to replicate the M3AE model> --num_train_epochs=10 --layers=3 ----output_dir=</path/to/output dir> 
  ```
An example command can be found in `train.sh`

We offer the following model implementations in the given repo:

- `model_name`: `MV_CLIP` - CLIP (LAION pre-trained) + ViT (BERT)
- `model_name`: `MV_LaCLIP` - LaCLIP (LAION pre-trained) + ViT (BERT)
- `model_name`: `MV_CLIP_MMAE` - CLIP (LAION pre-trained) + M3AE
- `model_name`: `MV_LaCLIP_MMAE` - LaCLIP (LAION pre-trained) + M3AE
where the CLIP based model is the backbone and the ViT/M3AE models are the classifier models

You can also set different flags like:
- `augs=True/False`: whether to add text and image augs or not
- `replicate_mmae=True/False`: whether to match the M3AE model configuration or not (use num heads as 12 and CLIP projection dim as 768 or 8 and 512)
- `layers`: Number of transformer blocks to use for the classifier model
- `max_length`: The maximum number of text tokens

 Feel free to play around with other flags.
