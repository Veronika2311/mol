#!/bin/bash
#SBATCH --job-name=t5        # Название задачи
#SBATCH --error=/home/vaganeeva/chem_augm_finetune/logs/train_t5_augm_molt5-base-smiles2caption_lr1e-4.err        # Файл для вывода ошибок
#SBATCH --output=/home/vaganeeva/chem_augm_finetune/logs/train_t5_augm_molt5-base-smiles2caption_lr1e-4.log       # Файл для вывода результатов
#SBATCH --time=32:59:00         # Максимальное время выполнения
#SBATCH --gpus=1                # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2         # Требуемое кол-во CPU

python /home/vaganeeva/chem_augm_finetune/mol/code/trainscript.py --df '/home/vaganeeva/chem_augm_finetune/data/augm_train_full.txt' \
 --model_name '/home/vaganeeva/hugging_face/laituan245/molt5-base-smiles2caption' \
 --max_epochs 20 \
 --batch_size 16 \
 --learning_rate 1e-4 \
 --max_length_molecule 256 \
 --max_length_text 256 \
 --gradient_accumulation_steps 2 \
 --output_dir "/home/vaganeeva/chem_augm_finetune/trained_models/molt5-base-smiles2caption_lr1e-4"
