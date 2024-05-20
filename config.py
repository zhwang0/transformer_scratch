from pathlib import Path

def get_config(): 
  return {
    "batch_size": 8,
    "num_epochs": 20,
    "lr": 1e-4,
    "seq_len": 350, 
    "d_model": 512,
    "lang_src": "en",
    "lang_tgt": "it",
    "model_folder": "weights",
    "model_basename": "transformer",
    "preload": None, # if you want to load a pretrained model
    "tokenzier_file": "tokenizer_{0}.json",
    "experiment_name": "runs/transformer" # experiment name for tensorboard
  }
  
def get_weights_file_path(config, epoch: str): 
  model_folder = config['model_folder']
  model_basename = config['model_basename']
  model_filename = f"{model_basename}_{epoch}.pt"
  return str(Path('.') / model_folder / model_filename)