import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from datasets import load_dataset
from tokenizers import Tokenizer # The main class for handling tokenization processes. It is the interface to all tokenization operations.
from tokenizers.models import WorldLevel # A model that uses a simple word-based tokenization approach.
from tokenizers.trainers import WordLevelTrainer # A trainer class specifically designed for training word-level tokenizers.
from tokenizers.pre_tokenizers import Whitespace # A pre-tokenizer that splits the input on whitespace.

from pathlib import Path


def get_all_sentences(ds, lang):
  for item in ds: 
    '''
    yield is a memory-efficient way to return values from a function, especially when the output is large or when the output is generated in a loop.
      The function does not load the entire dataset into memory at once. Instead, it processes and yields one batch at a time.
    '''
    yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
  tokenizer_path = Path(config['tokenizer_file'].format(lang))
  if not Path.exists(tokenizer_path):
    tokenizer = Tokenizer(WorldLevel(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[SOS]", "[ESO]"], min_frequency=2) # EOS is end of sentence, SOS is start of sentence
    tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
    tokenizer.save(str(tokenizer_path))
  else: 
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
  return tokenizer


def get_ds(config): 
  ds_raw = load_dataset('opus_books', f'{config['lang_src']}-{config['lang_tgt']}', split='train')
  
  # build tokenizer for source and target languages
  tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
  tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
  
  # 90-10 train-val split
  train_ds_size = int(len(ds_raw) * 0.9)
  val_ds_size = len(ds_raw) - train_ds_size
  train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
  
  
