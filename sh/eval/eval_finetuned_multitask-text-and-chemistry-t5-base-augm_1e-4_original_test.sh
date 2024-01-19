#!/bin/bash
#SBATCH --job-name=eval_t5        # Название задачи
#SBATCH --error=/home/vaganeeva/chem_augm_finetune/logs/eval_finetuned_multitask-text-and-chemistry-t5-base-augm_1e-4_original_test.err        # Файл для вывода ошибок
#SBATCH --output=/home/vaganeeva/chem_augm_finetune/logs/eval_finetuned_multitask-text-and-chemistry-t5-base-augm_1e-4_original_test.log       # Файл для вывода результатов
#SBATCH --time=32:59:00         # Максимальное время выполнения
#SBATCH --gpus=1                # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2         # Требуемое кол-во CPU



for i in {0..19};
do

echo "Evaluating epoch ${i}"
mkdir -p /home/vaganeeva/chem_augm_finetune/mol/evaluation_results/augm_multitask-text-and-chemistry-t5-base-augm_lr1e-4/${i}
python /home/vaganeeva/chem_augm_finetune/mol/code/eval_t5_generation.py \
--test_path "/home/vaganeeva/chem_augm_finetune/mol/original_test.txt" \
--checkpoint_path "/home/vaganeeva/chem_augm_finetune/trained_models/multitask-text-and-chemistry-t5-base-augm_lr1e-4/e_${i}" \
--base_model "/home/vaganeeva/hugging_face/GT4SD/multitask-text-and-chemistry-t5-base-augm" \
--max_length 512 \
--batch_size 16 \
--output_dir "/home/vaganeeva/chem_augm_finetune/mol/evaluation_results/augm_multitask-text-and-chemistry-t5-base-augm_lr1e-4_original_test/${i}" \
--output_pred_file "/home/vaganeeva/chem_augm_finetune/mol/evaluation_results/augm_multitask-text-and-chemistry-t5-base-augm_lr1e-4_original_test/epoch_${i}_pred.tsv";
done
