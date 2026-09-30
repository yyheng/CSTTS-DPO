import torch
import torchaudio
import whisper
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import WhisperProcessor
from langdetect import detect

def load_wave(wave_path) -> torch.Tensor:
      waveform, sr = torchaudio.load(wave_path, normalize=True)
      if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
      return waveform

class WhisperDatasetLAL(torch.utils.data.Dataset):
      def __init__(self, audio_list, text_list, featureprocessor, tokenizer, sample_rate=16000):
            super().__init__()
            self.audio_list = audio_list
            self.text_list = text_list
            self.sample_rate = sample_rate
            self.tokenizer = tokenizer
            self.feat_extractor = featureprocessor
            assert len(audio_list) == len(text_list), "audio and text not aligned"
      
      def __len__(self):
            return len(self.audio_list)

      def __getitem__(self, idx):
            audio_path = self.audio_list[idx]
            text = self.text_list[idx]
            item = {"audio": audio_path, "text": text}
            audio = load_wave(audio_path).squeeze(0).numpy()
            feat = self.feat_extractor(audio, sampling_rate=16_000, return_tensors="pt")['input_features']
            item["mel"] = feat
            tokenized_output = self.tokenizer(text, max_length=1024, truncation=True, return_offsets_mapping=True)
            token_ids = tokenized_output["input_ids"]
            token_offsets = tokenized_output["offset_mapping"]
            item["label"] = token_ids
            lang_id = []
            special_token_ids = [50258, 50260, 50359, 50363]
            
            for token_id, (start, end) in zip(token_ids[:-1], token_offsets[:-1]):
                  if token_id in special_token_ids:
                        lang_id.append(-100)   # skip special tokens entirely
                  elif token_id == 220:
                        lang_id.append(2)  # <blank> token
                  else:
                        token_text = text[start:end]
                        # print(token_text)
                        try:
                              lang = detect(token_text)
                              if lang in ['zh-cn', 'ko', 'zh-tw', 'ja']:
                                    lang_id.append(0) # lang = 'zh-cn'
                              else:
                                    lang_id.append(1) # lang = 'en'
                        except:
                              lang_id.append(3) # lang = "others"

            item["lang"] = torch.LongTensor(lang_id)
            # logging.info(f"single lang_id: {lang_id}")
            return item
      
      def collate_whisper(self, batch):

            feature_list = [{"input_features": item["mel"]} for item in batch]
            label_list = [{"input_ids": item["label"]} for item in batch]
            lang_id_list = [item["lang"] for item in batch]

            feats = self.feat_extractor.pad(feature_list, return_tensors="pt")
            labels_batch = self.tokenizer.pad(label_list, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            lang_labels = pad_sequence(lang_id_list, batch_first=True, padding_value=-100) # shape (batch, n_tokens_max)
            print(f"pad batch lang_labels: {lang_labels}")
            print(f"pad batch labels: {labels_batch['input_ids'][:,:-1]}")   
            return feats['input_features'].squeeze(1), labels_batch["input_ids"][:,:-1], labels, lang_labels

