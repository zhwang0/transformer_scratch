import torch 
import torch.nn as nn
from torch.utils.data import Dataset

class BilinguaDataset(Dataset): 
  
  def __init__(self, ds, tokenizer_src, tokenizer_tgt, lang_src, lang_tgt, seq_len) -> None:
    super().__init__()
    
    self.ds = ds
    self.tokenizer_src = tokenizer_src
    self.tokenizer_tgt = tokenizer_tgt
    self.lang_src = lang_src
    self.lang_tgt = lang_tgt
    
    self.seq_len = seq_len
    
    self.sos_token = torch.tensor([self.tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
    self.eos_token = torch.tensor([self.tokenizer_src.token_to_id("[ESO]")], dtype=torch.int64)
    self.pad_token = torch.tensor([self.tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)
    
  def __len__(self):
    return len(self.ds)
  
  def __getitem__(self, idx):
    
    # extract the original pair from huggingface dataset
    src_target_pair = self.ds[idx]
    src_text = src_target_pair['translation'][self.lang_src]
    tgt_text = src_target_pair['translation'][self.lang_tgt]
    
    # split sentence to each word, then convert to token ids
    enc_input_tokens = self.tokenizer_src.encode(src_text).ids # array: input ids of each word in a sentence
    dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
    
    # pad the input to the same length
    enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # -2 for sos and eos tokens
    dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # for decoder, no need to pad eos/sos token
    
    if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
      raise ValueError('Input sequence is too long')
    
    # prepare input for encoder and decoder, and the target/label for decoder
    encoder_input = torch.cat(
      [
        # add SOS and EOS to the soruce text
        self.sos_token,
        torch.tensor(enc_input_tokens, dtype=torch.int64),
        self.eos_token,
        torch.tensor([self.pad_token]*enc_num_padding_tokens, dtype=torch.int64)
      ]
    )
    decoder_input = torch.cat(
      [
        # add SOS to the decoder input
        self.sos_token,
        torch.tensor(dec_input_tokens, dtype=torch.int64),
        torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
      ]
    )
    label = torch.cat(
      [
        # add EOS to the decoder label (what the model should predict)
        torch.tensor(dec_input_tokens, dtype=torch.int64),
        self.eos_token,
        torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
      ]
    )
    
    assert encoder_input.size(0) == self.seq_len
    assert decoder_input.size(0) == self.seq_len
    assert label.size(0) == self.seq_len

    return {
      'encoder_input': encoder_input, # (seq_len)
      'decoder_input': decoder_input, # (seq_len)
      'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
      'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, 1, seq_len) & (1, seq_len, seq_len)
      'label': label, # (seq_len)
      'src_text': src_text,
      'tgt_text': tgt_text
    }
    
def causal_mask(size):
  # return upper triangular part, others will be zeros. diagonal=1 means the diagonal and above it will be retained 
  mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int) # give anything above the diagonal a value of 1
  return mask == 0