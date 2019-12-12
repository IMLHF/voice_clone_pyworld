import argparse
import os
import re
import numpy as np
import soundfile as sf
from synthesizer.textnorm import get_pinyin
from synthesizer import inference as syn_infer
from synthesizer import audio
from functools import partial
import pypinyin
from synthesizer.hparams import hparams
# import lpcnet
import pyworld as pw


def run_eval_part1(args):
  syn_ckpt = args.syn_checkpoint
  speaker_name = args.speaker_name
  eval_results_dir = os.path.join(args.eval_results_dir,
                                  speaker_name)
  if not os.path.exists(eval_results_dir):
    os.makedirs(eval_results_dir)

  speaker_id = int(speaker_name)

  sentences = [
    "Either measure the temperature with a bath thermometer or test the water with your wrist",
    "A test is a deliberate action or experiment to find out how well something works",
    "This was demonstrated in a laboratory experiment with rats",
    "This evidence supports the view that there is too much violence on television",
  ]

  # sentences = [sen.upper() for sen in sentences]

  print('eval part1> model: %s.' % syn_ckpt)
  syner = syn_infer.Synthesizer(syn_ckpt)

  ckpt_step = re.compile(r'.*?\.ckpt\-([0-9]+)').match(syn_ckpt)
  ckpt_step = "step-"+str(ckpt_step.group(1)) if ckpt_step else syn_ckpt

  speaker_embed = np.eye(251, dtype=np.float32)[speaker_id]
  for i, text in enumerate(sentences):
    path = os.path.join(eval_results_dir,
                        "%s-%s-eval-%03d_%s.wav" % (speaker_name, ckpt_step, i, "lpcnet"))
    print('[{:<10}]: {}'.format('processing', path))
    mel_spec = syner.synthesize_spectrograms([text], [speaker_embed])[
        0]  # batch synthesize
    print('[{:<10}]:'.format('text:'), text)
    print(np.shape(mel_spec))
    # mel_spec is world output feat
    f0, sp, ap = np.split(mel_spec, [1, 514])
    f0 *= 100.0
    sp /= 1000.0
    f0 = np.ascontiguousarray(f0.T, dtype=np.float64)
    sp = np.ascontiguousarray(sp.T, dtype=np.float64)
    ap = np.ascontiguousarray(ap.T, dtype=np.float64)
    f0 = np.squeeze(f0, -1)
    print(np.shape(f0), np.shape(sp), np.shape(ap))
    wav = pw.synthesize(f0, sp, ap, hparams.sample_rate)
    audio.save_wav(wav, path, hparams.sample_rate)


def main():
  # os.environ['CUDA_VISIBLE_DEVICES']= ''
  parser = argparse.ArgumentParser()
  parser.add_argument('syn_checkpoint',
                      # required=True,
                      help='Path to synthesizer model checkpoint.')
  parser.add_argument('speaker_name',
                      help='Path to target speaker audio.')
  parser.add_argument('--speaker_encoder_checkpoint', default='encoder/saved_models/pretrained.pt',
                      help='Path to speaker encoder nodel checkpoint.')
  parser.add_argument('--vocoder_checkpoint', default='vocoder/saved_models/vocoder/pretrained.pt',
                      help='Path to speaker encoder nodel checkpoint.')
  parser.add_argument('--eval_results_dir', default='overall_eval_results',
                      help='Overall evaluation results will be saved here.')
  args = parser.parse_args()
  hparams.set_hparam("tacotron_num_gpus", 1) # set tacotron_num_gpus=1 to synthesizer single wav.
  run_eval_part1(args)


if __name__ == '__main__':
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
  main()


# python3 overall_eval.py syn_ckpt_dir biaobei_speaker
