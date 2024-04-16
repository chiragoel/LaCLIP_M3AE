# SIglip + MMSD

Libraries Used 
1. transformers==4.39.1
2. torch==1.13.1
3. torchvision==0.14.1


Had to make changes in transformers/trainer_pt_utils.py to make LRScheduler work --> "from torch.optim.lr_scheduler import _LRScheduler as LRScheduler"


