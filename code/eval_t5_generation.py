import argparse
import logging
import os.path

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModel
import pandas as pd
from tqdm import tqdm
from luna.calculate import Calculator
from luna.ngram import BLEUMetrics, ROUGEMetrics, METEORMetrics
import numpy as np


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='t5 train')

    parser.add_argument('--test_path', type=str, required=True,
                        help='train df', )
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='model name', )
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--max_length', type=int, default=512, required=False)
    parser.add_argument('--batch_size', type=int, default=32, required=False)
    parser.add_argument('--output_dir', required=True)

    args = parser.parse_args()
    return args


def main(args):
    base_model = args.base_model
    test_path = args.test_path
    checkpoint_path = args.checkpoint_path
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer_chemt5 = AutoTokenizer.from_pretrained(base_model)
    model_chem_t5 = AutoModel.from_pretrained(checkpoint_path).eval().to(device)

    test_df = pd.read_csv(test_path, sep='\t')
    max_length = args.max_length
    batch_size = args.batch_size
    num_beams = 10
    test_fname = os.path.basename(test_path)
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        res = []
        for i in tqdm(range(0, len(test_df['SMILES']), batch_size)):
            encoded_dict_t5 = tokenizer_chemt5(list(test_df['SMILES'][i:i + batch_size]), return_tensors="pt",
                                               padding=True, truncation=True)

            outputs = model_chem_t5.generate(encoded_dict_t5["input_ids"].to(device), num_beams=num_beams,
                                             max_length=max_length)
            res.extend(tokenizer_chemt5.batch_decode(outputs, skip_special_tokens=True))

    assert len(res) == test_df.shape[0]
    res_df = pd.DataFrame()
    res_df["prediction"] = res
    output_pred_path = os.path.join(output_dir, f"pref_{test_fname}")
    res_df.to_csv(output_pred_path, sep='\t')

    canonical = res
    original = test_df["description"].tolist()

    metrics = [ROUGEMetrics('1')]
    calculator = Calculator(execute_parallel=True)
    metrics_dict = calculator.calculate(metrics=metrics, hyps=canonical, refs=original)
    for metric_name, metric_list in metrics_dict():
        print("ROUGE-1")
        print(np.mean(metric_list))
        print("----------------------")

    metrics = [ROUGEMetrics('2')]
    calculator = Calculator(execute_parallel=True)
    metrics_dict = calculator.calculate(metrics=metrics, hyps=canonical, refs=original)
    for metric_name, metric_list in metrics_dict():
        print("ROUGE-2")
        print(np.mean(metric_list))
        print("----------------------")

    metrics = [ROUGEMetrics('L')]
    calculator = Calculator(execute_parallel=True)
    metrics_dict = calculator.calculate(metrics=metrics, hyps=canonical, refs=original)
    for metric_name, metric_list in metrics_dict():
        print("ROUGE-L")
        print(np.mean(metric_list))
        print("----------------------")

    metrics = [BLEUMetrics(), METEORMetrics()]
    calculator = Calculator(execute_parallel=True)
    metrics_dict = calculator.calculate(metrics=metrics, hyps=canonical, refs=original)
    for metric_name, metric_list in metrics_dict():
        print(metric_name)
        print(np.mean(metric_list))
        print("----------------------")


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    main(args)
