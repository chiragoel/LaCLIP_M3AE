echo 'Running exp 1'

python3 main_mmae_clip.py  --text_name text_json_final --weight_decay 0.05 --train_batch_size 8 --dev_batch_size 8 --learning_rate 1e-6  --num_train_epochs 10 --max_grad_norm 5 --dropout_rate 0.2 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0 --model_type base --layers 3 --is_aug True --output_dir ./MMAECLIP_exps/exp7> ./logs/MMAECLIP_base_3_exp7.log 2>&1

echo 'Running exp 2'

python3 main_mmae_clip.py  --text_name text_json_final --weight_decay 0.05 --train_batch_size 8 --dev_batch_size 8 --learning_rate 1e-6  --num_train_epochs 10 --max_grad_norm 5 --dropout_rate 0.2 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0 --model_type base --layers 12 --is_aug False --output_dir ./MMAECLIP_exps/exp8> ./logs/MMAECLIP_base_12_exp8.log 2>&1

echo 'Running exp 3'

python3 main_mmae_clip.py  --text_name text_json_final --weight_decay 0.05 --train_batch_size 8 --dev_batch_size 8 --learning_rate 1e-6  --num_train_epochs 10 --max_grad_norm 5 --dropout_rate 0.2 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0 --model_type base --layers 3 --is_aug False --output_dir ./MMAECLIP_exps/exp9> ./logs/MMAECLIP_base_3_exp9.log 2>&1

echo 'Running exp 4'

python3 main_mmae_clip.py  --text_name text_json_final --weight_decay 0.05 --train_batch_size 8 --dev_batch_size 8 --learning_rate 1e-6  --num_train_epochs 10 --max_grad_norm 5 --dropout_rate 0.2 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0 --model_type base --layers 12 --is_aug True --output_dir ./MMAECLIP_exps/exp10> ./logs/MMAECLIP_base_12_exp10.log 2>&1

echo 'Running exp 5'

python3 main_mmae_clip.py  --text_name text_json_final --weight_decay 0.05 --train_batch_size 8 --dev_batch_size 8 --learning_rate 1e-6  --num_train_epochs 10 --max_grad_norm 5 --dropout_rate 0.3 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0 --model_type base --layers 3 --is_aug True --output_dir ./MMAECLIP_exps/ex11> ./logs/MMAECLIP_base_3_exp11.log 2>&1

echo 'Running exp 6'

python3 main_mmae_clip.py  --text_name text_json_final --weight_decay 0.05 --train_batch_size 8 --dev_batch_size 8 --learning_rate 1e-6  --num_train_epochs 10 --max_grad_norm 5 --dropout_rate 0.3 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0 --model_type base --layers 3 --is_aug False --output_dir ./MMAECLIP_exps/exp12> ./logs/MMAECLIP_base_3_exp12.log 2>&1

echo 'Running exp 7' 

python3 main_mmae_clip.py  --text_name text_json_final --weight_decay 0.05 --train_batch_size 8 --dev_batch_size 8 --learning_rate 1e-6 --clip_learning_rate 5e-7 --num_train_epochs 10 --max_grad_norm 5 --dropout_rate 0.2 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0 --model_type base --layers 3 --is_aug False --output_dir ./MMAECLIP_exps/exp13> ./logs/MMAECLIP_base_3_exp13.log 2>&1

echo 'Running exp 8' 

python3 main_mmae_clip.py  --text_name text_json_final --weight_decay 0.05 --train_batch_size 8 --dev_batch_size 8 --learning_rate 1e-6 --clip_learning_rate 5e-7 --num_train_epochs 10 --max_grad_norm 5 --dropout_rate 0.2 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0 --model_type base --layers 12 --is_aug False --output_dir ./MMAECLIP_exps/exp14> ./logs/MMAECLIP_base_12_exp14.log 2>&1

echo 'Running exp 9' 

python3 main_mmae_clip.py  --text_name text_json_final --weight_decay 0.05 --train_batch_size 8 --dev_batch_size 8 --learning_rate 1e-6 --clip_learning_rate 5e-7 --num_train_epochs 10 --max_grad_norm 5 --dropout_rate 0.2 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0 --model_type base --layers 3 --is_aug True --output_dir ./MMAECLIP_exps/exp15> ./logs/MMAECLIP_base_3_exp15.log 2>&1

echo 'Running exp 10' 

python3 main_mmae_clip.py  --text_name text_json_final --weight_decay 0.05 --train_batch_size 8 --dev_batch_size 8 --learning_rate 1e-6 --clip_learning_rate 5e-7 --num_train_epochs 10 --max_grad_norm 5 --dropout_rate 0.2 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0 --model_type base --layers 12 --is_aug True --output_dir ./MMAECLIP_exps/exp16> ./logs/MMAECLIP_base_12_exp16.log 2>&1

echo 'Running exp 11'

python3 main_mmae_clip.py  --text_name text_json_final --weight_decay 0.05 --train_batch_size 8 --dev_batch_size 8 --learning_rate 1e-6 --clip_learning_rate 5e-7 --num_train_epochs 10 --max_grad_norm 5 --dropout_rate 0.3 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0 --model_type base --layers 3 --is_aug True --output_dir ./MMAECLIP_exps/ex17> ./logs/MMAECLIP_base_3_exp17.log 2>&1

echo 'Running exp 12' 

python3 main_mmae_clip.py  --text_name text_json_final --weight_decay 0.05 --train_batch_size 8 --dev_batch_size 8 --learning_rate 1e-6 --clip_learning_rate 5e-7 --num_train_epochs 10 --max_grad_norm 5 --dropout_rate 0.3 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0 --model_type base --layers 3 --is_aug False --output_dir ./MMAECLIP_exps/exp18> ./logs/MMAECLIP_base_3_exp18.log 2>&1