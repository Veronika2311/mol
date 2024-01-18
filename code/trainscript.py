import os.path

import pandas as pd
from transformers import T5ForConditionalGeneration, AutoTokenizer
import argparse
import gc
import logging
from typing import List, Dict, Union

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer


class PairsDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        assert idx < len(self.x['input_ids'])
        item = {key: val[idx] for key, val in self.x.items()}
        item['decoder_attention_mask'] = self.y['attention_mask'][idx]
        item['labels'] = self.y['input_ids'][idx]
        return item

    @property
    def n(self):
        return len(self.x['input_ids'])

    def __len__(self):
        return self.n  # * 2


class DataCollatorWithPadding:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        max_len_input_ids = max(len(d["input_ids"]) for d in features)
        max_len_labels = max(len(d["labels"]) for d in features)
        batch = self.tokenizer.pad(
            features,
            padding=True,
            #padding='max_length',
            #truncation=True,
            #max_length=max_len_input_ids

        )
        ybatch = self.tokenizer.pad(
            {'input_ids': batch['labels'], 'attention_mask': batch['decoder_attention_mask']},
            #padding='max_length',
            padding=True
            #truncation=True,
            #max_length=max_len_labels
        )
        batch['labels'] = ybatch['input_ids']
        # TODO: note that I comment this
        batch['decoder_attention_mask'] = ybatch['attention_mask']

        return {k: torch.tensor(v) for k, v in batch.items()}


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def evaluate_model(model, test_dataloader):
    num = 0
    den = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            loss = model(**{k: v.to(model.device) for k, v in batch.items()}).loss
            num += len(batch) * loss.item()
            den += len(batch)
    val_loss = num / den
    return val_loss


def train_loop(
        model, train_dataloader, val_dataloader, output_dir,
        max_epochs=30,
        max_steps=1_000,
        lr=3e-5,
        gradient_accumulation_steps=4,
        cleanup_step=100,
        report_step=300,
        window=100,
):
    cleanup()
    optimizer = torch.optim.Adam(params=[p for p in model.parameters() if p.requires_grad], lr=lr)

    # ewm_loss = 0
    step = 0

    model.train()

    for epoch in range(max_epochs):
        epoch_train_loss_sum = 0.
        num_epoch_steps = 0
        logging.info(f"Epoch {epoch} / {max_epochs}")
        if step >= max_steps:
            break
        tq = tqdm(train_dataloader)
        model.train()
        for i, batch in enumerate(tq):

            batch['labels'][batch['labels'] == 0] = -100
            loss = model(**{k: v.to(model.device) for k, v in batch.items()}).loss
            epoch_train_loss_sum += float(loss)
            num_epoch_steps += 1
            loss.backward()

            if i and i % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if step >= max_steps:
                    break

            if i % cleanup_step == 0:
                cleanup()

            # w = 1 / min(i+1, window)
            # ewm_loss = ewm_loss * (1-w) + loss.item() * w
            # tq.set_description(f'loss: {ewm_loss:4.4f}')
            tq.set_description(f'\nloss: {loss:4.4f}')

            # if (i and i % report_step == 0 or i == len(train_dataloader)-1)  and val_dataloader is not None:
            #    model.eval()
            #    eval_loss = evaluate_model(model, val_dataloader)
            #    model.train()
            #    print(f'epoch {epoch}, step {i}/{step}: train loss: {loss:4.4f}  val loss: {eval_loss:4.4f}')
        del batch
        optimizer.zero_grad()
        model.eval()
        train_loss = epoch_train_loss_sum / num_epoch_steps
        val_loss = evaluate_model(model, val_dataloader)
        logging.info(f'epoch {epoch}. train loss: {train_loss:4.4f}  val loss: {val_loss:4.4f}')

        # if step % 1000 == 0:
        # model.save_pretrained(f't5_base_{dname}')
        if epoch % 1 == 0:
            checkpoint_dir = os.path.join(output_dir, f"e_{epoch}")
            model.save_pretrained(checkpoint_dir)
            # model.save_pretrained(f'/home/vaganeeva/chem_augm_finetune/trained_models/t5_base_{epoch}')

    cleanup()


def train_model(x, y, model_name, max_length_molecule, max_length_text, test_size=0.1, batch_size=32, **kwargs):
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    x1, x2, y1, y2 = train_test_split(x, y, test_size=test_size, random_state=42)
    train_dataset = PairsDataset(tokenizer(x1, padding='max_length', truncation=True, max_length=max_length_molecule),
                                 tokenizer(y1, padding='max_length', truncation=True, max_length=max_length_text))
    test_dataset = PairsDataset(tokenizer(x2, padding='max_length', truncation=True, max_length=max_length_molecule),
                                tokenizer(y2))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True,
                                  collate_fn=data_collator)
    val_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False, shuffle=True,
                                collate_fn=data_collator)

    train_loop(model, train_dataloader, val_dataloader, **kwargs)
    return model


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='t5 train')

    parser.add_argument('--df', type=str, required=True,
                        help='train df', )
    parser.add_argument('--model_name', type=str, required=True,
                        help='model name', )
    parser.add_argument('--max_epochs', type=int, required=True,
                        help='max_epochs', )
    parser.add_argument('--max_length_molecule', type=int, default=512, required=False)
    parser.add_argument('--max_length_text', type=int, default=512, required=False)
    parser.add_argument('--batch_size', type=int, required=True, help='batch_size', )
    parser.add_argument('--learning_rate', type=float, required=False, help='learning_rate',
                        default=3e-5)
    parser.add_argument('--output_dir', required=True)

    # parser.add_argument('--df', type=str, required=False,
    #                     help='train df', default='C:/Users/veron/notebooks/biochemnlp/augm_train_full.txt')
    # parser.add_argument('--model_name', type=str, required=False,
    #                     help='model name', default='C:/Users/veron/notebooks/biochemnlp/model')
    # parser.add_argument('--max_epochs', type=int, required=False,
    #                     help='max_epochs', default=3)
    # parser.add_argument('--max_length_molecule', type=int, default=512)
    # parser.add_argument('--max_length_text', type=int, default=512)
    # parser.add_argument('--batch_size', type=int, required=False,
    #                     help='batch_size', default=2)
    #
    # parser.add_argument('--output_dir', default="delete/", required=False)

    args = parser.parse_args()
    return args


def main(args):
    logging.info("Training with parameters:")
    for k, v in vars(args).items():
        logging.info(f"{k} : {v}")
    cleanup()
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    max_length_molecule = args.max_length_molecule
    max_length_text = args.max_length_text
    learning_rate = args.learning_rate
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # model_name = 'C:/Users/veron/notebooks/biochemnlp/model'
    # df = pd.read_csv('C:/Users/veron/notebooks/biochemnlp/augm_train_full.txt', sep='\t')

    datasets = {
        'train': pd.read_csv(args.df, sep='\t')
    }
    steps = 1_000_000
    for dname, d in datasets.items():
        print(f'\n\n\n  {dname}\n=====================\n\n')
        model = train_model(d['SMILES'].tolist(), d['description'].tolist(), model_name=args.model_name,
                            batch_size=args.batch_size, max_epochs=args.max_epochs, max_steps=steps,
                            max_length_text=max_length_text, max_length_molecule=max_length_molecule,
                            lr=learning_rate, output_dir=output_dir)
        model.save_pretrained(os.path.join(output_dir, "final_checkpoint/"))


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    main(args)
