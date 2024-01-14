import os
import pandas as pd
from sklearn.utils import shuffle
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from transformers.file_utils import cached_property
from typing import Tuple
from sklearn.model_selection import train_test_split
import gc
from tqdm.auto import tqdm, trange
from typing import List, Dict, Union
import argparse
import logging

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
        return self.n # * 2
    
class DataCollatorWithPadding:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=True,
        )
        ybatch = self.tokenizer.pad(
            {'input_ids': batch['labels'], 'attention_mask': batch['decoder_attention_mask']},
            padding=True,
        ) 
        batch['labels'] = ybatch['input_ids']
        batch['decoder_attention_mask'] = ybatch['attention_mask']
        
        return {k: torch.tensor(v) for k, v in batch.items()}
    
def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def evaluate_model(model, test_dataloader):
    num = 0
    den = 0

    for batch in test_dataloader:
        with torch.no_grad():
            loss = model(**{k: v.to(model.device) for k, v in batch.items()}).loss
            num += len(batch) * loss.item()
            den += len(batch)
    val_loss = num / den
    return val_loss

def train_loop(
    model, train_dataloader, val_dataloader, 
    max_epochs=30, 
    max_steps=1_000, 
    lr=3e-5,
    gradient_accumulation_steps=1, 
    cleanup_step=100,
    report_step=300,
    window=100,
):
    cleanup()
    optimizer = torch.optim.Adam(params = [p for p in model.parameters() if p.requires_grad], lr=lr)

    #ewm_loss = 0
    step = 0
    model.train()

    for epoch in trange(max_epochs):
        print(step, max_steps)
        if step >= max_steps:
            break
        tq = tqdm(train_dataloader)
        for i, batch in enumerate(tq):
            try:
                batch['labels'][batch['labels']==0] = -100
                loss = model(**{k: v.to(model.device) for k, v in batch.items()}).loss
                loss.backward()
            except Exception as e:
                print('error on step', i, e)
                loss = None
                cleanup()
                continue
            if i and i % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if step >= max_steps:
                    break

            if i % cleanup_step == 0:
                cleanup()

            #w = 1 / min(i+1, window)
            #ewm_loss = ewm_loss * (1-w) + loss.item() * w
            #tq.set_description(f'loss: {ewm_loss:4.4f}')
            tq.set_description(f'loss: {loss:4.4f}')

            if (i and i % report_step == 0 or i == len(train_dataloader)-1)  and val_dataloader is not None:
                model.eval()
                eval_loss = evaluate_model(model, val_dataloader)
                model.train()
                print(f'epoch {epoch}, step {i}/{step}: train loss: {loss:4.4f}  val loss: {eval_loss:4.4f}')
                
        #if step % 1000 == 0:
            #model.save_pretrained(f't5_base_{dname}')
        if epoch % 1 == 0:
            model.save_pretrained(f'/home/vaganeeva/chem_augm_finetune/trained_models/t5_base_{epoch}')
        
    cleanup()


def train_model(x, y, model_name, test_size=0.1, batch_size=32, **kwargs):
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    x1, x2, y1, y2 = train_test_split(x, y, test_size=test_size, random_state=42)
    train_dataset = PairsDataset(tokenizer(x1), tokenizer(y1))
    test_dataset = PairsDataset(tokenizer(x2), tokenizer(y2))
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True, collate_fn=data_collator)
    val_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False, shuffle=True, collate_fn=data_collator)

    train_loop(model, train_dataloader, val_dataloader, **kwargs)
    return model

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='t5 train')

    # parser.add_argument('--df', type=str, required=True,
    #                     help='train df', )
    # parser.add_argument('--model_name', type=str, required=True,
    #                     help='model name',)
    # parser.add_argument('--max_epochs', type=int, required=True,
    #                     help='max_epochs',)
    # parser.add_argument('--batch_size', type=int, required=True,
    #                     help='batch_size', )

    parser.add_argument('--df', type=str, required=False,
                        help='train df', default='C:/Users/veron/notebooks/biochemnlp/augm_train_full.txt')
    parser.add_argument('--model_name', type=str, required=False,
                        help='model name', default='C:/Users/veron/notebooks/biochemnlp/model')
    parser.add_argument('--max_epochs', type=int, required=False,
                        help='max_epochs', default=3)
    parser.add_argument('--batch_size', type=int, required=False,
                        help='batch_size', default=2)    

    args = parser.parse_args()
    return args


def main(args):
    cleanup()
    #device = "cuda:0" if torch.cuda.is_available() else "cpu"

    #model_name = 'C:/Users/veron/notebooks/biochemnlp/model'
    #df = pd.read_csv('C:/Users/veron/notebooks/biochemnlp/augm_train_full.txt', sep='\t')

    datasets = {
        'train': pd.read_csv(args.df, sep='\t')
        }
    steps = 1_000_000
    for dname, d in datasets.items():
        print(f'\n\n\n  {dname}\n=====================\n\n')
        model = train_model(d['SMILES'].tolist(), d['description'].tolist(), model_name=args.model_name, 
                            batch_size=args.batch_size, max_epochs=args.max_epochs, max_steps=steps)
        model.save_pretrained(f'/home/vaganeeva/chem_augm_finetune/trained_models/t5_base_{dname}')


if __name__ == '__main__':

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    main(args)
