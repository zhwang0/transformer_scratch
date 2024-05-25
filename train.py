import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from dataset import BilinguaDataset, causal_mask
from model import build_transformer
from config import get_config, get_weights_file_path

from datasets import load_dataset
from tokenizers import Tokenizer # The main class for handling tokenization processes. It is the interface to all tokenization operations.
from tokenizers.models import WordLevel # A model that uses a simple word-based tokenization approach.
from tokenizers.trainers import WordLevelTrainer # A trainer class specifically designed for training word-level tokenizers.
from tokenizers.pre_tokenizers import Whitespace # A pre-tokenizer that splits the input on whitespace.

from torch.utils.tensorboard import SummaryWriter
import torchmetrics

import os
import warnings
from tqdm import tqdm 
from pathlib import Path


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device): 
  sos_idx = tokenizer_tgt.token_to_id('[SOS]')
  eos_idx = tokenizer_tgt.token_to_id('[ESO]')
  
  # precompute the encoder output and reuse itfor every token we get from the decoder
  encoder_output = model.encode(source, source_mask)

  # initialize the decoder input with the SOS token
  decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
  while True: 
    if decoder_input.size(1) == max_len: 
      break
    
    # build mask for target (decoder input)
    decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
    
    # calculate the output of the decoder
    out = model.decode(decoder_input, encoder_output, source_mask, decoder_mask)
    
    # get the next token 
    prob = model.project(out[:,-1])
    
    # select the token with the max probability (greedy decoding)
    _, next_word = torch.max(prob, dim=1)
    # decoder_input = torch.cat([decoder_input, next_word.unsqueeze(-1)], dim=-1) # more efficient for batch operations 
    decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=-1) # more suitable for single-item prediction
    
    if next_word == eos_idx: 
      break
  
  return decoder_input.squeeze(0) # remove the batch dimension


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_state, writer, num_examples=2): 
  model.eval()
  count = 0 
  
  source_texts = []
  expected = []
  predicted = []
  
  # sizze of the control window (just use a default value)
  try: 
    if os.name != 'nt':
      with os.popen('stty size', 'r') as console_size: 
        console_width = int(console_size.read().split()[1])
    else: 
      # Windows
      console_width = os.get_terminal_size().columns
  except:
    console_width = 80
    
  
  with torch.no_grad():
    for batch in validation_ds: 
      count += 1
      encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
      encoder_mask = batch['encoder_mask'].to(device) # (b, 1, 1, seq_len)
      decoder_input = batch['decoder_input'].to(device)
      decoder_mask = batch['decoder_mask'].to(device)
      
      assert encoder_input.shape[0] == 1, 'Batch size for validation should be 1'
      
      model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
      
      source_text = batch['src_text'][0]
      target_text = batch['tgt_text'][0]
      model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
      
      source_texts.append(source_text)
      expected.append(target_text)
      predicted.append(model_out_text)
      
      print_msg('-'*console_width) # tqdm is not suitable for print()
      print_msg(f'Source: {source_text}')
      print_msg(f'Target: {target_text}')
      print_msg(f'Predicted: {model_out_text}')
      
      if count == num_examples: 
        break
      
  if writer: 
    # evaluate the cahracter error rate 
    # compute the char error rate
    metric = torchmetrics.CharErrorRate()
    cer = metric(expected, predicted)
    writer.add_scalar('Validation cer', cer, global_state)
    writer.flush()
    
    # compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(expected, predicted)
    writer.add_scalar('Validation wer', wer, global_state)
    writer.flush()
    
    # compute the BLEU score
    metric = torchmetrics.BLEUScore()
    bleu = metric(expected, predicted)
    writer.add_scalar('Validation bleu', bleu, global_state)
    writer.flush()



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
    tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[SOS]", "[ESO]"], min_frequency=2) # EOS is end of sentence, SOS is start of sentence
    tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
    tokenizer.save(str(tokenizer_path))
  else: 
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
  return tokenizer


def get_ds(config): 
  print(config['lang_src'])
  ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')
  
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
  return model
  

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
    
    batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch}', total=len(train_dataloader))
    for batch in batch_iterator: 
      model.train()
      
      encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_len)
      decoder_input = batch['decoder_input'].to(device) # (batch_size, seq_len)
      encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1, seq_len)
      decoder_mask = batch['decoder_mask'].to(device) # (batch_size, 1, seq_len, seq_len)
      label = batch['label'].to(device) # (batch_size, seq_len), the label is the position of the vocab in the target language
      
      encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
      decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) # (batch_size, seq_len, d_model)
      proj_output = model.project(decoder_output) # (batch_size, seq_len, tgt_vocal_size)
        
      # (batch_size, seq_len, tgt_vocal_size) --> (batch_size * seq_len, tgt_vocal_size)
      loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
      batch_iterator.set_postfix({'loss': f'{loss.item():6.3f}'})
      
      # Log the loss
      writer.add_scalar('loss', loss.item(), global_step)
      writer.flush()
      
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      
      # validation 
      run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, 
                     lambda msg: batch_iterator.write(msg), global_step, writer, num_examples=2)
      
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
  
  
