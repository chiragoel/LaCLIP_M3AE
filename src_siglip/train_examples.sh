# MMSD2.0
python3 main.py --model MV_CLIP --text_name text_json_final --weight_decay 0.05 --train_batch_size 32 --dev_batch_size 32 --learning_rate 5e-4 --clip_learning_rate 1e-6 --num_train_epochs 10 --layers 3 --max_grad_norm 5 --dropout_rate 0.1 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0 > MV_CLIP_MMSD2.log 2>&1 &
# MMSD2.0-LACLIP - vanilla version
python3 main.py --model LACLIP --text_name text_json_final --weight_decay 0.05 --train_batch_size 32 --dev_batch_size 32 --learning_rate 5e-4 --clip_learning_rate 1e-6 --num_train_epochs 1 --layers 3 --max_grad_norm 5 --dropout_rate 0.1 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0 > LACLIP_MMSD2.log 2>&1 &
# MMSD2.0-LACLIP - laclip_lr_rate_wt_decay
python3 main.py --model LACLIP --text_name text_json_final --weight_decay 0.2 --train_batch_size 32 --dev_batch_size 32 --learning_rate 5e-4 --clip_learning_rate 5e-4 --num_train_epochs 10 --layers 3 --max_grad_norm 5 --dropout_rate 0.1 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0 > LACLIP_MMSD2.log 2>&1 &

# siglip + MMSD2 --vanilla
python3 main.py --model SIGLIP --text_name text_json_final --weight_decay 0.05 --train_batch_size 8 --dev_batch_size 8 --learning_rate 5e-4 --clip_learning_rate 1e-6 --num_train_epochs 10 --layers 3 --max_grad_norm 5 --dropout_rate 0.1 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --tag "siglip_test_run" --max_len 64 --modelname "siglip_vanilla" --device 0 > ../output_dir/siglip_vanilla.log 2>&1


# MMSD2 with new args
python3 main.py --model MV_CLIP --text_name text_json_final --weight_decay 0.05 --train_batch_size 32 --dev_batch_size 32 --learning_rate 5e-4 --clip_learning_rate 1e-6 --num_train_epochs 10 --layers 3 --max_grad_norm 5 --dropout_rate 0.1 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0 --tag "mmsd2_vanilla_run" --modelname "mmsd_vanilla" > ../output_dir/MV_CLIP_MMSD2_vanilla.log 2>&1 
#MMSD with data_aug_total
python3 main.py --model MV_CLIP --text_name text_json_final --weight_decay 0.05 --train_batch_size 32 --dev_batch_size 32 --learning_rate 5e-4 --clip_learning_rate 1e-6 --num_train_epochs 10 --layers 3 --max_grad_norm 5 --dropout_rate 0.1 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0 --tag "mmsd2_data_aug_run" --modelname "mmsd_data_aug_total" --data_aug "total" > ../output_dir/MV_CLIP_MMSD2_data_aug_total.log 2>&1
# MMSD scaling text projection
python3 main.py --model MV_CLIP --text_name text_json_final --weight_decay 0.05 --train_batch_size 32 --dev_batch_size 32 --learning_rate 5e-4 --clip_learning_rate 1e-6 --num_train_epochs 10 --layers 3 --max_grad_norm 5 --dropout_rate 0.1 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0 --tag "mmsd2_scale_text_proj_run" --modelname "mmsd2_scale_text_proj" --scale_text_proj True > ../output_dir/MV_CLIP_MMSD2_scale_text_proj.log 2>&1 

#MMSD2 predict on baseline trained model
python3 predict.py --model MV_CLIP --text_name text_json_final --model_path "../../scratch/model_checkpoints/MV_CLIP/mmsd_vanilla.pt" --text_size 512 --image_size 768 --save_file "../output_dir/mmsd_vanilla_predict_result.json" --save_csv_file "../output_dir/mmsd_vanilla_predict_result.csv" --save_metrics_csv_file ""../output_dir/mmsd_vanilla_predict_metrics.csv"" --test_batch_size 32 --device 0
python3 predict.py --model MV_CLIP --text_name text_json_final --mode "aug_test" --model_path "../../scratch/model_checkpoints/MV_CLIP/mmsd_vanilla.pt" --text_size 512 --image_size 768 --save_file "../output_dir/mmsd_vanilla_predict_result_aug_test.json" --save_csv_file "../output_dir/mmsd_vanilla_predict_result_aug_test.csv" --save_metrics_csv_file ""../output_dir/mmsd_vanilla_predict_metrics_aug_test.csv"" --test_batch_size 32 --device 0

# MMSD
#python3 main.py --model MV_CLIP --text_name text_json_clean --weight_decay 0.005 --train_batch_size 32 --dev_batch_size 32 --learning_rate 3e-4 --clip_learning_rate 3e-7 --num_train_epochs 10 --layers 5 --max_grad_norm 6 --dropout_rate 0.3 --optimizer_name adam --text_size 512 --image_size 768 --warmup_proportion 0.2 --device 0 > MV_CLIP_MMSD.log 2>&1 &