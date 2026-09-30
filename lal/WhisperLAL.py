import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

import torch
torch.autograd.set_detect_anomaly(True)  # super helpful around backward()

# for reference
class WhisperWithLAL(nn.Module):
      """
      :param whisper_model: e.g., "openai/whisper-small"
      :param layer_index: index of the layer to use (optional, default -1)
      """
      def __init__(self, whisper_model, layer_index=-1):
            super(WhisperWithLAL, self).__init__()
            self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_model)
            hidden_size = self.whisper.config.d_model
            self.language_cls = nn.Linear(hidden_size, 4)
            self.layer_index = layer_index
      
      def forward(self, feat, decode_ids):
            if self.training:
                  output = self.whisper(input_features=feat, decoder_input_ids=decode_ids, output_attentions=True)
                  logits = output.logits
                  cross_atten = output.cross_attentions[-1]
                  encoder_outputs = self.whisper.model.encoder(feat).last_hidden_state
                  lang_logits = self.language_cls(encoder_outputs)
                  return logits, cross_atten, lang_logits
            else:
                  decoded_ids = self.whisper.generate(inputs=feat, forced_decoder_ids=decode_ids, max_new_tokens=150)
                  return decoded_ids



class LanguageAlignmentLoss(nn.Module):
    """
    :param lang_change: where the lang changes within token dict
    :param n_tokens: dict size
    :param n_lang: num. of language tokens (including others)
    """
    def __init__(self, weight=None, n_lang=4, threshold=None):
        super(LanguageAlignmentLoss, self).__init__()

        if weight is not None:
            self.weight = torch.FloatTensor(weight)
        else:
            self.weight = None
        self.criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=-100)
        self.n_langs = n_lang
        self.threshold = threshold

    def forward(self, frames, labels, cross_atten):
        """

        :param frames: shape (batch, n_frames, n_lang_class)
        :param labels: shape (batch, n_tokens)
        :param cross_atten: shape (batch, n_heads, n_token_class, n_frames)
        :return:
        """ 
        print(f"labels.size(-1): {labels.size(-1)}, cross_atten.size(-2): {cross_atten.size(-2)}")
        assert labels.size(-1) == cross_atten.size(-2)

        frames = frames.view(-1, self.n_langs)

        language_matrix = cross_atten.clone().detach() # (batch, n_heads, n_token_class, n_frames)
        lang_atten_mat = language_matrix.mean(dim=1) # (batch, n_token_class, n_frames)
        max_value_, index_ = torch.max(lang_atten_mat, dim=1)
        if self.threshold > 0:
            index_[max_value_<self.threshold] = 2

        lang_label_mat = labels.clone()

        lang_label_mat_ = lang_label_mat.gather(1, index_).view(-1)

        return self.criterion(frames, lang_label_mat_)
