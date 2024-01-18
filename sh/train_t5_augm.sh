#!/bin/bash
#SBATCH --job-name=t5        # Название задачи
#SBATCH --error=/home/vaganeeva/chem_augm_finetune/logs/train_augm_chem_t5.err        # Файл для вывода ошибок
#SBATCH --output=/home/vaganeeva/chem_augm_finetune/logs/train_augm_chem_t5.log       # Файл для вывода результатов
#SBATCH --time=32:59:00         # Максимальное время выполнения
#SBATCH --gpus=1                # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2         # Требуемое кол-во CPU

python /home/vaganeeva/chem_augm_finetune/mol/code/trainscript.py --df '/home/vaganeeva/chem_augm_finetune/data/augm_train_full.txt' \
 --model_name '/home/vaganeeva/hugging_face/GT4SD/multitask-text-and-chemistry-t5-base-standard' \
 --max_epochs 20 \
 --batch_size 16 \
 --learning_rate 3e-5 \
 --output_dir "/home/vaganeeva/chem_augm_finetune/trained_models/multitask-text-and-chemistry-t5-base-standard"
