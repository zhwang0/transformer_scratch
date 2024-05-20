import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from dataset import BilinguaDataset, causal_mask
from model import build_transformer
from config import get_config, get_weights_file_path

from datasets import load_dataset
from tokenizers import Tokenizer # The main class for handling tokenization processes. It is the interface to all tokenization operations.
from tokenizers.models import WorldLevel # A model that uses a simple word-based tokenization approach.
from tokenizers.trainers import WordLevelTrainer # A trainer class specifically designed for training word-level tokenizers.
from tokenizers.pre_tokenizers import Whitespace # A pre-tokenizer that splits the input on whitespace.

from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
import tqdm 
import warnings


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
  
  train_ds = BilinguaDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
  val_ds = BilinguaDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
  
  max_len_src = 0
  max_len_tgt = 0
  
  for item in ds_raw: 
    src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
    tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
    max_len_src = max(max_len_src, len(src_ids))
    max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
  print(f'Max length for source language: {max_len_src}')
  print(f'Max length for target language: {max_len_tgt}')
  
  train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
  val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False) # 1 batch size to evaluate one sentence by one
  
  return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocal_src_len, vocal_tgt_len):
  model = build_transformer(vocal_src_len, vocal_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
  

def train_model(config): 
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'Using device: {device}')
  
  Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
  
  train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
  model = get_model(config, len(tokenizer_src.get_vocab()), len(tokenizer_tgt.get_vocab())).to(device)
  writer = SummaryWriter(config['experiment_name'])
  
  optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
  
  initial_epoch = 0
  global_step = 0 # used for tensorboard to log the loss
  if config['preload'] is not None: 
    model_filename = get_weights_file_path(config, config['preload'])
    print(f'Loading model from {model_filename}')
    
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    initial_epoch = state['epoch'] + 1
    global_step = state['global_step'] 
    

  '''
  label_smoothing=0.1 is a regularization technique to prevent the model from becoming too confident 
    about its predictions. 
  Formular: 
    [0,1,0]→[ϵ,1−ϵ,ϵ] where ϵ is a small positive number.
  Benefits: 
    - Prevents the model from becoming too confident about its predictions
    - Encourages the model to learn more generalizable features
    - Helps to prevent overfitting
  '''
  loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
  
  
  for epoch in range(initial_epoch, config['num_epochs']):
    model.train()
    batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch}', total=len(train_dataloader))
    for batch in batch_iterator: 
      
      encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_len)
      decoder_input = batch['decoder_input'].to(device) # (batch_size, seq_len)
      encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1, seq_len)
      decoder_mask = batch['decoder_mask'].to(device) # (batch_size, 1, seq_len, seq_len)
      label = batch['label'].to(device) # (batch_size, seq_len), the label is the position of the vocab in the target language
      
      encoder_output = model.encoder(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
      decoder_output = model.decoder(decoder_input, encoder_output, encoder_mask, decoder_mask) # (batch_size, seq_len, d_model)
      proj_output = model.projection(decoder_output) # (batch_size, seq_len, tgt_vocal_size)
        
      # (batch_size, seq_len, tgt_vocal_size) --> (batch_size * seq_len, tgt_vocal_size)
      loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
      batch_iterator.set_postfix({'loss': f'{loss.item():6.3f}'})
      
      # Log the loss
      writer.add_scalar('loss', loss.item(), global_step)
      writer.flush()
      
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      
      global_step +=1 
  
  model_filename = get_weights_file_path(config, str(epoch))
  torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'global_step': global_step
  }, model_filename)
  
      
if __name__ == '__main__':
  warnings.filterwarnings('ignore')
  config = get_config()
  train_model(config)
  
  
