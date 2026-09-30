import os
import torch
import torch.utils
import torch.utils.data
import argparse
from jiwer import wer as calculate_wer
from WhisperDataPreLAL import WhisperDatasetLAL
from WhisperLAL import LanguageAlignmentLoss, WhisperWithLAL
from whisper.normalizers import EnglishTextNormalizer
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast, WhisperProcessor, get_scheduler
import logging
from utils import preprocess_files, save_best_checkpoints, str2bool

logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s", 
                    datefmt="%Y-%m-%d %H:%M:%S")

normalizer = EnglishTextNormalizer()

# SEAME Kaldi dirs: $CMI_DPO_DATA_ROOT from config/paths.env (exported by slurm/sb / the sbatch bodies)
DATA_ROOT = os.environ.get('CMI_DPO_DATA_ROOT', 'data/SEAME_Segmented')

def main():
      parser = argparse.ArgumentParser(description='paras for making data')
      parser.add_argument('--train', type=str, default=os.path.join(DATA_ROOT, 'train'))
      parser.add_argument('--dev', type=str, default=os.path.join(DATA_ROOT, 'valid'))
      parser.add_argument('--devman', type=str, default=os.path.join(DATA_ROOT, 'devman'))
      parser.add_argument('--devsge', type=str, default=os.path.join(DATA_ROOT, 'devsge'))
      parser.add_argument('--model', type=str, default="openai/whisper-small")
      parser.add_argument('--epochs', type=int, default=8)
      parser.add_argument('--lr', type=float, default=1e-6)
      parser.add_argument('--save_every', type=int, default=10000)
      parser.add_argument('--zeroshot', type=str2bool, default=False)
      parser.add_argument('--batch', type=int, default=4)
      parser.add_argument('--lang', type=str, default="zh")
      parser.add_argument('--lal', type=float, default=0.1)
      parser.add_argument('--accumulation', type=int, default=2)
      parser.add_argument('--threshold', type=float, default=0)
      parser.add_argument('--warmup', type=int, default=10000)
      parser.add_argument('--module', type=str, default='all')
      parser.add_argument('--save_dir', type=str, default="exp2")
      parser.add_argument('--layer_index', type=int, default=-1, help='Index of the layer to use in WhisperWithLAL')
      args = parser.parse_args()

      normalizer = EnglishTextNormalizer()

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      # for single GPU, may need DDP for multiple GPUs

      train_path = args.train
      dev_path = args.dev

      devman_path = args.devman
      devsge_path = args.devsge
      train_audio_list, train_text_list = preprocess_files(train_path)
      dev_audio_list, dev_text_list = preprocess_files(dev_path)

      devman_audio_list, devman_text_list = preprocess_files(devman_path)
      devsge_audio_list, devsge_text_list = preprocess_files(devsge_path)
      

      tokenizer = WhisperTokenizerFast.from_pretrained(args.model, language=args.lang, task="transcribe")
      processor = WhisperProcessor.from_pretrained(args.model, language=args.lang, task="transcribe")
      feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model)

      model = WhisperWithLAL(args.model, layer_index=args.layer_index).to(device)
      
      # Module freezing logic
      if args.module == 'encoder':
            for param in model.whisper.model.decoder.parameters():
                  param.requires_grad = False
            train_module = 'encoder'
      elif args.module == 'decoder':
            for param in model.whisper.model.encoder.parameters():
                  param.requires_grad = False
            train_module = 'decoder'
      elif args.module == 'all':
            train_module = 'all'
      else:
            raise ValueError(f"Invalid module specified: {args.module}. Expected 'encoder', 'decoder', or 'all'.")
      logging.info(f"Trainable module: {train_module}")

      trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
      non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
      logging.info(f"Model params: {trainable_params+non_trainable_params}\n #Non-train: {non_trainable_params}\n #Trainable: {trainable_params}")  

      model.whisper.config.forced_decoder_ids = None
      model.whisper.config.suppress_tokens = []

      def evaluate(model, dev_data, text_list):
            forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.lang, task="transcribe")
            with torch.no_grad():
                  all_pred, all_gt = [], []
                  for idx_, (feats, label_batch, labels, lang_labels) in enumerate(dev_data):
                        generated_ids = model(feat=feats.to(device), decode_ids=forced_decoder_ids)
                        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        pred = normalizer(generated_text)
                        pred = pred if len(pred) > 0 else '<UNK>'
                        gt = normalizer(text_list[idx_])
                        gt = gt if len(gt) > 0 else '<UNK>'

                        all_pred.append(pred)
                        all_gt.append(gt)

                  return calculate_wer(all_gt, all_pred)

      logging.info("Training data preparing...")
      batch_size = args.batch
      train_dataset = WhisperDatasetLAL(train_audio_list, train_text_list, feature_extractor, tokenizer)
      train_data = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               collate_fn=train_dataset.collate_whisper)
      train_data_size = len(train_data)
      num_iterations = train_data_size//batch_size
      logging.info("Dev data preparing...")
      dev_dataset = WhisperDatasetLAL(dev_audio_list, dev_text_list, feature_extractor, tokenizer)
      dev_data = torch.utils.data.DataLoader(dataset=dev_dataset, 
                                             batch_size=1,
                                             collate_fn=dev_dataset.collate_whisper)

      devman_dataset = WhisperDatasetLAL(devman_audio_list, devman_text_list, feature_extractor, tokenizer)
      devman_data = torch.utils.data.DataLoader(dataset=devman_dataset, 
                                             batch_size=1,
                                             collate_fn=dev_dataset.collate_whisper)

      devsge_dataset = WhisperDatasetLAL(devsge_audio_list, devsge_text_list, feature_extractor, tokenizer)
      devsge_data = torch.utils.data.DataLoader(dataset=devsge_dataset, 
                                             batch_size=1,
                                             collate_fn=dev_dataset.collate_whisper)
      
      logging.info(f"Training started, total epochs {args.epochs}")
      logging.info(f"Iterations per eps {num_iterations}\n Total iterations: {num_iterations * args.epochs}")

      optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)
      loss_fn = torch.nn.CrossEntropyLoss(ignore_index = -100)
      loss_align = LanguageAlignmentLoss(n_lang=4, threshold=args.threshold)
      gradient_accumulation = args.accumulation
      save_every = args.save_every
      total_training_steps = args.epochs * len(train_data)

      scheduler = get_scheduler(name="linear", 
                                optimizer=optimizer, 
                                num_warmup_steps=args.warmup, 
                                num_training_steps=total_training_steps)

      save_dir = args.save_dir
      os.system(f"mkdir -p {save_dir}")

      ckpt_records = None

      step, pre_step, loss = 0, 0, 0
      best_loss, best_wer = 100000, 100000

      lal_weight = args.lal
      if args.zeroshot:
            model.eval()
            dev_wer = evaluate(model, dev_data, dev_text_list)
            logging.info(f"zero shot performance: WER {dev_wer}")

      model.train()
      for epoch in range(args.epochs):
            logging.info(f"Epoch: {epoch+1} starts")
            for _, (feats, label_batch, labels, lang_labels) in enumerate(train_data):
                  
                  y_out = labels[:, 1:].to(device)
                  logits, cross_atten, lang_logits = model(feat=feats.to(device), decode_ids=label_batch.to(device))
                  loss_asr = loss_fn(logits.transpose(1,2), y_out)
                  print(lang_logits.shape, lang_labels.shape, cross_atten.shape)
                  loss_lal = loss_align(lang_logits, lang_labels.to(device), cross_atten)
                  loss = loss_asr + lal_weight*loss_lal
                  
                  loss.backward()
                  step += 1

                  if (step+1) % gradient_accumulation == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                  if (step+1) % 500 == 0:
                        logging.info(f"Epoch: {epoch+1} ( step {step+1}): loss {loss:.3f} loss_asr: {loss_asr:.3f} loss align: {loss_lal:.3f}")

                  if (step+1) % save_every == 0:
                        model.eval()
                        dev_wer = evaluate(model, dev_data, dev_text_list)
                        logging.info(f"evaluation in epoch: {epoch+1} ( step {step+1}): WER {dev_wer}")
                        model.train()
                        ckpt_records = save_best_checkpoints(model=model, 
                                                             step=step+1, 
                                                             error_rate=dev_wer, 
                                                             loss=loss, 
                                                             save_dir=save_dir, 
                                                             ckpt_records=ckpt_records)
      
      model = torch.load(ckpt_records[0], map_location=device)
      model.to(device)
      model.eval()

      devman_wer = evaluate(model, devman_data, devman_text_list)
      devsge_wer = evaluate(model, devsge_data, devsge_text_list)
      logging.info(f"evaluation in for devman: WER {devman_wer}")
      logging.info(f"evaluation in for devsge: WER {devsge_wer}")


if __name__ == "__main__":
    main()