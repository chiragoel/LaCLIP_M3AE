echo "Exps without augs for LAION pretrained models replicating MMAE"

echo "Exp 1: MV_CLIP"
python3 main.py --model=MV_CLIP --text_name=text_json_final --replicate_mmae=True --weight_decay=0.05 --train_batch_size=32 --dev_batch_size=8 --learning_rate=5e-4 --clip_learning_rate=1e-6 --num_train_epochs=1 --layers=3 --max_grad_norm=5 --dropout_rate=0.1 --warmup_proportion=0.2 --device=0 --output_dir=../models/final_clip_noaugs > ../models/final_clip_noaugs/log.txt 

echo "Exp 2: MV_LaCLIP"
python3 main.py --model=MV_LaCLIP --text_name=text_json_final --replicate_mmae=True --weight_decay=0.05 --train_batch_size=32 --dev_batch_size=8 --learning_rate=5e-4 --clip_learning_rate=1e-6 --num_train_epochs=1 --layers=3 --max_grad_norm=5 --dropout_rate=0.1 --warmup_proportion=0.2 --device=0 --output_dir=../models/final_laclip_noaugs > ../models/final_laclip_noaugs/logs.txt 

echo "Exp 3: MV_CLIP_MMAE"
python3 main.py --model=MV_CLIP_MMAE --text_name=text_json_final --replicate_mmae=True --weight_decay=0.05 --train_batch_size=32 --dev_batch_size=8 --learning_rate=5e-4 --clip_learning_rate=1e-6 --num_train_epochs=1 --layers=3 --max_grad_norm=5 --dropout_rate=0.1 --warmup_proportion=0.2 --device=0 --output_dir=../models/final_clip_mmae_noaugs >  ../models/final_laclip_noaugs/logs.txt 

echo "Exp 4: MV_LaCLIP_MMAE"
python3 main.py --model=MV_LaCLIP_MMAE --text_name=text_json_final --replicate_mmae=True --weight_decay=0.05 --train_batch_size=32 --dev_batch_size=8 --learning_rate=5e-4 --clip_learning_rate=1e-6 --num_train_epochs=1 --layers=3 --max_grad_norm=5 --dropout_rate=0.1 --warmup_proportion=0.2 --device=0 --output_dir=../models/final_laclip_mmae_noaugs > ../models/final_laclip_noaugs/logs.txt

