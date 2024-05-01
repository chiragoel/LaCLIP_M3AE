# LaCLIP_M3AE
This is the codebase for using a multi-view CLIP and ViT based architecture for MultiModal Sarcasm Detection on the MMSD2.0 dataset. This has been motivated from the [paper](https://github.com/JoeYing1019/MMSD2.0/tree/main) "MMSD2.0: Towards a Reliable Multi-modal Sarcasm Detection System". 

In the following, we will guide you how to use this repository step by step.

## Requirements

We recommend to use a python enviroment with python `3.9` for these experiments.

 ```angular2html
    python -m venv /path/to/new/virtual/environment
    source /path/to/new/virtual/environment/bin/activate
    pip install -r requirements.txt
    cd src  
  ```

## Dataset

Please download the dataset from the readme [here](https://github.com/JoeYing1019/MMSD2.0/tree/main) and put it in the `dataset` folder. The text json files are already present. Please modify the `WORKING_DIR` path in the `data_set.py` and `data_set_original.py` file according to where the `dataset` folder is present.

## Data Augmentations

Open-ai GPT3.5 api has been used for creating our text augmented dataset. If you plan to use this be mindful that this would incur costs according to OpenAI's pricing model. This is an offline step acan be run in the following manner:

### Requirements

```angular2html
    pip install openai 
  ```

### Usage
```
  python augment_data.py --path_api_key=</path/to/api/key> --path_training_dataset=</path/to/training/dataset/> --save_path=</path/to/save/augmented/data>
  ```

For more details see [here](https://github.com/chiragoel/LaCLIP_M3AE/blob/main-chirag-final/data_augmentations/README.md)

## Usage

Plase download the pre-trained model weights for LaCLIP and M3AE from [here](https://drive.google.com/file/d/1-4f9bDb-0S-Ei7_Tf7_rcUrR5KmUbFht/view?usp=sharing)

You can tun the code in this repo using the following command:

```angular2html
    python3 main.py --model=<model_name> --text_name=<name of text json file> --aug=<Whether to use text based augs or not> --replicate_mmae=<Do you want the ViT to replicate the M3AE model> --num_train_epochs=10 --layers=3 ----output_dir=</path/to/output dir> 
  ```
An example command can be found in `train.sh`

```angular2html
    python3 main.py --model=MV_LaCLIP_MMAE --text_name=text_json_final --augs=True --replicate_mmae=True --weight_decay=0.05 --train_batch_size=16 --dev_batch_size=16 --learning_rate=5e-4 --clip_learning_rate=1e-6 --num_train_epochs=10 --layers=3 --max_grad_norm=5 --dropout_rate=0.1 --warmup_proportion=0.2 --device=0 --output_dir=../output_dir
  ```

We offer the following model implementations in the given repo:

- `model_name`: `MV_CLIP_original` - CLIP (open-ai pre-trained) + ViT (BERT) [implementation in the original paper]
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

 Feel free to play around with other flags and hyperparameters
