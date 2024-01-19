#!/bin/bash
#SBATCH --job-name=t5        # Название задачи
#SBATCH --error=/home/vaganeeva/chem_augm_finetune/logs/eval_finetuned_multitask-text-and-chemistry-t5-base-standard_augm_test_from_test.err        # Файл для вывода ошибок
#SBATCH --output=/home/vaganeeva/chem_augm_finetune/logs/eval_finetuned_multitask-text-and-chemistry-t5-base-standard_augm_test_from_test.log       # Файл для вывода результатов
#SBATCH --time=32:59:00         # Максимальное время выполнения
#SBATCH --gpus=1                # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2         # Требуемое кол-во CPU



for i in {0..19};
do

mkdir -p /home/vaganeeva/chem_augm_finetune/mol/evaluation_results/augm_multitask-text-and-chemistry-t5-base-standard}/${i}
python /home/vaganeeva/chem_augm_finetune/mol/code/eval_t5_generation.py \
--test_path "/home/vaganeeva/chem_augm_finetune/mol/augm_test_from_test.txt" \
--checkpoint_path "/home/vaganeeva/chem_augm_finetune/trained_models/multitask-text-and-chemistry-t5-base-standard/e_${i}" \
--base_model "/home/vaganeeva/hugging_face/GT4SD/multitask-text-and-chemistry-t5-base-standard" \
--max_length 512 \
--batch_size 32 \
--output_dir "/home/vaganeeva/chem_augm_finetune/mol/evaluation_results/augm_multitask-text-and-chemistry-t5-base-standard_augm_test/${i}";

done