echo "Exps with augs for LAION pretrained models replicating MMAE"

echo "Exp 5: MV_CLIP"
python3 main.py --model=MV_CLIP --text_name=text_json_final --augs=True --replicate_mmae=True --weight_decay=0.05 --train_batch_size=16 --dev_batch_size=16 --learning_rate=5e-4 --clip_learning_rate=1e-6 --num_train_epochs=10 --layers=3 --max_grad_norm=5 --dropout_rate=0.1 --warmup_proportion=0.2 --device=0 --output_dir=../models/final_clip_augs > ../models/logs/final_clip_augs.txt 

echo "Exp 6: MV_LaCLIP"
python3 main.py --model=MV_LaCLIP --text_name=text_json_final --augs=True --replicate_mmae=True --weight_decay=0.05 --train_batch_size=16 --dev_batch_size=16 --learning_rate=5e-4 --clip_learning_rate=1e-6 --num_train_epochs=10 --layers=3 --max_grad_norm=5 --dropout_rate=0.1 --warmup_proportion=0.2 --device=0 --output_dir=../models/final_laclip_augs > ../models/logs/final_laclip_augs.txt

echo "Exp 7: MV_CLIP_MMAE"
python3 main.py --model=MV_CLIP_MMAE --text_name=text_json_final --augs=True --replicate_mmae=True --weight_decay=0.05 --train_batch_size=16 --dev_batch_size=16 --learning_rate=5e-4 --clip_learning_rate=1e-6 --num_train_epochs=10 --layers=3 --max_grad_norm=5 --dropout_rate=0.1 --warmup_proportion=0.2 --device=0 --output_dir=../models/final_clip_mmae_augs >  ../models/logs/final_laclip_augs.txt 

echo "Exp 8: MV_LaCLIP_MMAE"
python3 main.py --model=MV_LaCLIP_MMAE --text_name=text_json_final --augs=True --replicate_mmae=True --weight_decay=0.05 --train_batch_size=16 --dev_batch_size=16 --learning_rate=5e-4 --clip_learning_rate=1e-6 --num_train_epochs=10 --layers=3 --max_grad_norm=5 --dropout_rate=0.1 --warmup_proportion=0.2 --device=0 --output_dir=../models/final_laclip_mmae_augs > ../models/logs/final_laclip_augs.txt  

echo "Exp 11: MV_CLIP_original MMAE"
python3 main.py --model=MV_CLIP_original --text_name=text_json_final --replicate_mmae=True --weight_decay=0.05 --train_batch_size=16 --dev_batch_size=16 --learning_rate=5e-4 --clip_learning_rate=1e-6 --num_train_epochs=10 --layers=3 --max_grad_norm=5 --dropout_rate=0.1 --warmup_proportion=0.2 --device=0 --output_dir=../models/final_clip_original_mmae > ../models/logs/final_clip_original_mmae.txt 