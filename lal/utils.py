import os
import re
import torch
import logging
import argparse

logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s", 
                    datefmt="%Y-%m-%d %H:%M:%S")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def process_sample(sample):
    # Regular expression to match the identifier (alphanumeric) and the sentence
    match = re.match(r"^(\w+)\s+(.+)$", sample)
    if not match:
        raise ValueError("The sample format is incorrect.")
    
    identifier = match.group(1)
    sentence = match.group(2)
    
    # Regular expression to identify Chinese characters and English words
    pattern = re.compile(r'[\u4e00-\u9fff]|[a-zA-Z]+')
    
    # Find all matches in the sentence
    elements = pattern.findall(sentence)
    
    return elements


def preprocess_files(folder):
    wavscp = os.path.join(folder, "wav.scp")
    text = os.path.join(folder, "text")
    with open(wavscp, 'r') as f1:
        wave_data = f1.readlines()
    with open(text, 'r') as f2:
        trans_data = f2.readlines()
    audio_list = [audio_line.strip('\n').split(' ')[1] for audio_line in wave_data]
    text_list = [' '.join(text_line.split(' ')[1:]).lower().strip('\n') for text_line in trans_data]
    return audio_list, text_list

# def preprocess_files(folder):
#       wavscp = os.path.join(folder, "wav.scp")
#       text = os.path.join(folder, "text")
#       with open(wavscp, 'r') as f1:
#             wave_data = f1.readlines()
#       with open(text, 'r') as f2:
#             trans_data = f2.readlines()
#       audio_list = [audio_line.strip().split()[1] for audio_line in wave_data]
#       text_list = [' '.join(process_sample(text_line)).lower().strip() for text_line in trans_data]
#       return audio_list, text_list



def save_best_checkpoints(model, step, error_rate, loss, save_dir="checkpoints", ckpt_records=None, max_ckpts=3):
    """
    Saves the top 3 checkpoints with the lowest error rates by updating `ckpt_records`.
    
    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        step (int): Current epoch number.
        error_rate (float): Error rate of the model.
        save_dir (str): Directory to save checkpoints.
        ckpt_records (list, optional): List to track best checkpoints [(error_rate, filepath)].
        max_ckpts (int): Maximum number of best checkpoints to retain.
    
    Returns:
        list: Updated checkpoint records sorted by error rate.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Initialize checkpoint record list if not provided
    if ckpt_records is None:
        ckpt_records = []

    # Path for new checkpoint
    ckpt_path = os.path.join(save_dir, f"loss_{loss:.3f}_step_{step}_wer_{error_rate:.4f}.pt")

    torch.save(model, ckpt_path)
    
    # Add new checkpoint and sort by error rate
    ckpt_records.append((error_rate, ckpt_path))
    ckpt_records.sort(key=lambda x: x[0])  # Sort ascending by error rate

    # Keep only the top `max_ckpts`
    while len(ckpt_records) > max_ckpts:
        _, worst_ckpt = ckpt_records.pop()  # Remove worst (highest error rate)
        os.remove(worst_ckpt)  # Delete the removed checkpoint

    logging.info(f"Checkpoint saved: {ckpt_path}")
    logging.info(f"Best Checkpoints: {[ckpt[1] for ckpt in ckpt_records]}")
    
    return ckpt_records  # Return updated checkpoint list
