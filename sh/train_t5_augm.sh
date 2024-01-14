#!/bin/bash
SBATCH --job-name=t5        # Название задачи
SBATCH --error=t5-%j.err        # Файл для вывода ошибок
SBATCH --output=t5-%j.log       # Файл для вывода результатов
SBATCH --time=23:59:00         # Максимальное время выполнения
SBATCH --ntasks=16             # Количество MPI процессов
SBATCH --gpus=1                # Требуемое кол-во GPU
SBATCH --cpus-per-task=2         # Требуемое кол-во CPU

python /home/vaganeeva/chem_augm_finetune/mol/code/transcript.py --df '/home/vaganeeva/chem_augm_finetune/data/augm_train_full.txt' \
 --model_name '/home/vaganeeva/hugging_face/GT4SD/multitask-text-and-chemistry-t5-base-standard' \
 --max_epochs 20 \
 --batch_size 32
