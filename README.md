# Laclip--MMSD2.0 code

Install instructions
Downloaded and install the scipy-stack libraries instead of loading from cluster.
Used "pip install --no-deps timm" when other libraries where already installed.

pretrained_models folder required for storing laion-400mn based laclip, clip models outside of src code
![image](https://github.com/chiragoel/CLIP-MA-MMSD/assets/36845045/4adb2c44-0480-4bee-8bf0-05342ce98695)

To create env do 
1. python3.9 -m venv {env_name}
2. source {env_name}/bin/activate
3. pip install -r laclip_env_requirements.txt

Steps in cluster
1. run "module load python/3.9 cuda cudnn"
2. source {env_name}/bin/activate
3. (example) python3 main.py --model LACLIP --text_name text_json_final --weight_decay 0.05 --train_batch_size 32 --dev_batch_size 32 --learning_rate 5e-4 --clip_learning_rate 1e-6 --num_train_epochs 10 --layers 3 --max_grad_norm 5 --dropout_rate 0.1 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --tag "laion_clip_model_test_run" --device 0 > laion_clip_test_run.log
