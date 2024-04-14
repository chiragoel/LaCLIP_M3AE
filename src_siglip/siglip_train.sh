# siglip + MMSD2 experiments

echo "experiment 4 - dropout to 0.3"

python3 main.py --model SIGLIP --text_name text_json_final --weight_decay 0.05 --train_batch_size 8 --dev_batch_size 8 --learning_rate 5e-4 --clip_learning_rate 1e-6 --num_train_epochs 10 --layers 3 --max_grad_norm 5 --dropout_rate 0.1 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --tag "siglip_exp_run" --max_len 64 --modelname "siglip_test_exp_4" --device 0 > ../output_dir/siglip_test_exp_4.log 2>&1

echo "experiment 5 - dropout to 0.5"

python3 main.py --model SIGLIP --text_name text_json_final --weight_decay 0.05 --train_batch_size 8 --dev_batch_size 8 --learning_rate 5e-4 --clip_learning_rate 1e-6 --num_train_epochs 10 --layers 3 --max_grad_norm 5 --dropout_rate 0.5 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --tag "siglip_exp_run" --max_len 64 --modelname "siglip_test_exp_5" --device 0 > ../output_dir/siglip_test_exp_5.log 2>&1

echo "experiment 6 - dropout to 0.7"

python3 main.py --model SIGLIP --text_name text_json_final --weight_decay 0.05 --train_batch_size 8 --dev_batch_size 8 --learning_rate 5e-4 --clip_learning_rate 1e-6 --num_train_epochs 10 --layers 3 --max_grad_norm 5 --dropout_rate 0.7 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --tag "siglip_exp_run" --max_len 64 --modelname "siglip_test_exp_6" --device 0 > ../output_dir/siglip_test_exp_6.log 2>&1
