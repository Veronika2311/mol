#!/bin/bash
#SBATCH --job-name=t5        # Название задачи
#SBATCH --error=/home/vaganeeva/chem_augm_finetune/logs/train_augm_chem_t5_lr1e-4.err        # Файл для вывода ошибок
#SBATCH --output=/home/vaganeeva/chem_augm_finetune/logs/train_augm_chem_t5_lr1e-4.log       # Файл для вывода результатов
#SBATCH --time=32:59:00         # Максимальное время выполнения
#SBATCH --gpus=1                # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2         # Требуемое кол-во CPU

python /home/vaganeeva/chem_augm_finetune/mol/code/trainscript.py --df '/home/vaganeeva/chem_augm_finetune/data/augm_train_full.txt' \
 --model_name '/home/vaganeeva/hugging_face/GT4SD/multitask-text-and-chemistry-t5-base-standard' \
 --max_epochs 20 \
 --batch_size 16 \
 --learning_rate 1e-4 \
 --gradient_accumulation_steps 2 \
 --output_dir "/home/vaganeeva/chem_augm_finetune/trained_models/multitask-text-and-chemistry-t5-base-standard_lr1e-4"
