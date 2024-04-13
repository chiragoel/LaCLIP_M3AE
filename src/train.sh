# MMSD2.0
python3 main.py --model MV_CLIP --text_name text_json_final --weight_decay 0.05 --train_batch_size 32 --dev_batch_size 32 --learning_rate 5e-4 --clip_learning_rate 1e-6 --num_train_epochs 10 --layers 3 --max_grad_norm 5 --dropout_rate 0.1 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0 > MV_CLIP_MMSD2.log 2>&1 &
# MMSD2.0-LACLIP - vanilla version
python3 main.py --model LACLIP --text_name text_json_final --weight_decay 0.05 --train_batch_size 32 --dev_batch_size 32 --learning_rate 5e-4 --clip_learning_rate 1e-6 --num_train_epochs 1 --layers 3 --max_grad_norm 5 --dropout_rate 0.1 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0 > LACLIP_MMSD2.log 2>&1 &
# MMSD2.0-LACLIP - laclip_lr_rate_wt_decay
python3 main.py --model LACLIP --text_name text_json_final --weight_decay 0.2 --train_batch_size 32 --dev_batch_size 32 --learning_rate 5e-4 --clip_learning_rate 5e-4 --num_train_epochs 10 --layers 3 --max_grad_norm 5 --dropout_rate 0.1 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0 > LACLIP_MMSD2.log 2>&1 &



# MMSD
#python3 main.py --model MV_CLIP --text_name text_json_clean --weight_decay 0.005 --train_batch_size 32 --dev_batch_size 32 --learning_rate 3e-4 --clip_learning_rate 3e-7 --num_train_epochs 10 --layers 5 --max_grad_norm 6 --dropout_rate 0.3 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0 > MV_CLIP_MMSD.log 2>&1 &