from multiprocessing.pool import Pool
from synthesizer import audio
from functools import partial
from itertools import chain
from pathlib import Path
from utils import logmmse
from tqdm import tqdm
import numpy as np
import librosa
from synthesizer.textnorm import get_pinyin
import os
import struct
import webrtcvad
from scipy.ndimage.morphology import binary_dilation
# import lpcnet
import pyworld as pw

# region aishell2
def preprocess_aishell2(datasets_root: Path, dataset: str, out_dir: Path, n_processes: int,
                        skip_existing: bool, hparams, detach_label_and_embed_utt):
    dataset_root = datasets_root.joinpath("aishell2")
    input_dir = dataset_root.joinpath("data/wav")
    print("\n    Using data from:" + str(input_dir))
    assert input_dir.exists(), str(input_dir)+" not exist."
    all_sub_dirs=list(input_dir.glob("*"))
    speaker_dirs=[]
    for _dir in all_sub_dirs:
        if _dir.is_file(): continue
        speaker_dirs.append(_dir)
    speaker_dirs.sort()
    trans_fpath = dataset_root.joinpath('data', 'trans.txt')
    assert trans_fpath.exists(), str(input_dir)+" not exist."
    with trans_fpath.open("r") as trans_f:
        lines = [_line.split() for _line in trans_f]
    trans_dict = {}
    for _line in lines:
        trans_dict[_line[0]] = _line[1]

    # process_speaker_fn_params: params for _preprocess_speaker_aishell2
    process_speaker_fn_params = {"trans_dict": trans_dict,
                                 "detach_label_and_embed_utt": detach_label_and_embed_utt}
    _preprocess_speakers(speaker_dirs, dataset, "wav", _preprocess_speaker_aishell2, process_speaker_fn_params,
                         out_dir, n_processes, skip_existing, hparams)


def _preprocess_speaker_aishell2(speaker_dir, suffix, out_dir: Path, skip_existing: bool, hparams, others_params):
    speaker_dir, spk_id = speaker_dir
    trans_dict = others_params["trans_dict"]
    detach_label_and_embed_utt = others_params["detach_label_and_embed_utt"]
    metadata = []
    wav_fpath_list = speaker_dir.glob("*."+suffix)

    utt_fpath_list = list(speaker_dir.glob("*."+suffix))
    utt_num = len(utt_fpath_list)
    # Iterate over each wav
    for wav_fpath in wav_fpath_list:
        assert wav_fpath.exists(), str(wav_fpath)+" not exist."

        # Process each utterance
        wav, _=librosa.load(str(wav_fpath), hparams.sample_rate)
        wav_abs_max = np.max(np.abs(wav))
        wav_abs_max = wav_abs_max if wav_abs_max > 0.0 else 1e-8
        wav=wav / wav_abs_max * hparams.rescaling_max  # norm
        # wav_bak = wav

        # denoise
        if len(wav) > hparams.sample_rate*(0.3+0.1):
          noise_wav = np.concatenate([wav[:int(hparams.sample_rate*0.15)],
                                      wav[-int(hparams.sample_rate*0.15):]])
          profile = logmmse.profile_noise(noise_wav, hparams.sample_rate)
          wav = logmmse.denoise(wav, profile, eta=0)

        # trim silence
        wav = audio.trim_silence(wav, 30)  # top_db: smaller for noisy
        # audio.save_wav(wav_bak, str(wav_fpath.name), hparams.sample_rate)
        # audio.save_wav(wav, str(wav_fpath.name).replace('.wav','_trimed.wav'),
        #                hparams.sample_rate)

        text = trans_dict[wav_fpath.stem]

        # Chinese to Pinyin
        pinyin = " ".join(get_pinyin(text, std=True, pb=True))

        # print(wav_fpath.name, wav_fpath.stem)
        random_uttBasename_forSpkEmbedding = None
        if detach_label_and_embed_utt:
            random_uttBasename_forSpkEmbedding = utt_fpath_list[np.random.randint(
                utt_num)].stem
        metadata.append(process_utterance(wav, pinyin, out_dir, wav_fpath.stem,
                                          skip_existing, hparams, random_uttBasename_forSpkEmbedding))
    return [m for m in metadata if m is not None]

# endregion aishell2

# region SLR38
def preprocess_SLR38(datasets_root: Path, dataset: str, out_dir: Path, n_processes: int,
                     skip_existing: bool, hparams, detach_label_and_embed_utt):
    dataset_root = datasets_root.joinpath("SLR38")
    input_dir = dataset_root.joinpath("ST-CMDS-speaker-separated")
    print("\n    Using data form:" + str(input_dir))
    assert input_dir.exists(), str(input_dir)+" not exist."
    all_sub_dirs = list(input_dir.glob("*"))
    speaker_dirs = []
    for _dir in all_sub_dirs:
        if _dir.is_file(): continue
        speaker_dirs.append(_dir)

    others_params = {"detach_label_and_embed_utt": detach_label_and_embed_utt}
    _preprocess_speakers(speaker_dirs, dataset, "wav", _preprocess_speaker_SLR38, others_params,
                         out_dir, n_processes, skip_existing, hparams)

def _preprocess_speaker_SLR38(speaker_dir, suffix, out_dir: Path, skip_existing: bool, hparams, others_params):
    speaker_dir, spk_id = speaker_dir
    detach_label_and_embed_utt = others_params["detach_label_and_embed_utt"]
    wav_fpath_list = speaker_dir.glob("*."+suffix)
    text_fpath_list = speaker_dir.glob("*.txt")
    metadata = []
    # Iterate over each wav
    utt_fpath_list = list(speaker_dir.glob("*."+suffix))
    utt_num = len(utt_fpath_list)
    for wav_fpath, txt_fpath in zip(wav_fpath_list, text_fpath_list):
        assert wav_fpath.exists(), str(wav_fpath)+" not exist."
        assert txt_fpath.exists(), str(wav_fpath)+" not exist."

        # Process each utt
        wav, _ = librosa.load(str(wav_fpath), hparams.sample_rate)
        wav = wav / np.max(np.abs(wav)) * hparams.rescaling_max
        # wav_bak = wav

        # denoise
        if len(wav) > hparams.sample_rate*(0.3+0.1):
            noise_wav = np.concatenate([wav[:int(hparams.sample_rate*0.15)],
                                        wav[-int(hparams.sample_rate*0.15):]])
            profile = logmmse.profile_noise(noise_wav, hparams.sample_rate)
            wav = logmmse.denoise(wav, profile, eta=0)

        # trim silence
        wav = audio.trim_silence(wav, 30)
        # audio.save_wav(wav_bak, str(wav_fpath.name), hparams.sample_rate)
        # audio.save_wav(wav, str(wav_fpath.name).replace('.wav','_trimed.wav'),
        #                hparams.sample_rate)

        # get text
        text = txt_fpath.read_text()

        # Chinese to Pinyin
        pinyin = " ".join(get_pinyin(text, std=True, pb=True))

        # print(wav_fpath.name, wav_fpath.stem)
        random_uttBasename_forSpkEmbedding=None
        if detach_label_and_embed_utt:
            random_uttBasename_forSpkEmbedding=utt_fpath_list[np.random.randint(utt_num)].stem
        metadata.append(process_utterance(wav, pinyin, out_dir, wav_fpath.stem,
                                          skip_existing, hparams, random_uttBasename_forSpkEmbedding))
    return [m for m in metadata if m is not None]

# endregion SLR38

# region XX SLR68
def preprocess_SLR68(datasets_root: Path, dataset: str, out_dir: Path, n_processes: int,
                     skip_existing: bool, hparams, detach_label_and_embed_utt):
    dataset_root = datasets_root.joinpath("SLR68")
    input_dir = dataset_root.joinpath("train")
    print("\n    Using data from:" + str(input_dir))
    assert input_dir.exists(), str(input_dir)+" not exist."
    all_sub_dirs = list(input_dir.glob("*"))
    speaker_dirs = []
    for _dir in all_sub_dirs:
        if _dir.is_file(): continue
        speaker_dirs.append(_dir)

    trans_fpath = input_dir.joinpath("TRANS.txt")
    assert trans_fpath.exists(), str(input_dir)+" not exist."
    with trans_fpath.open("r") as trans_f:
        lines = [_line.split() for _line in trans_f]
    lines = lines[1:]
    trans_dict = {}
    for _line in lines:
        trans_dict[_line[0]]={"speaker_id": _line[1], "text": _line[2]}

    # process_speaker_fn_params: params for _preprocess_speaker_SLR68
    process_speaker_fn_params = {"trans_dict": trans_dict,
                                 "detach_label_and_embed_utt": detach_label_and_embed_utt}
    _preprocess_speakers(speaker_dirs, dataset, "wav", _preprocess_speaker_SLR68, process_speaker_fn_params,
                         out_dir, n_processes, skip_existing, hparams)


def _preprocess_speaker_SLR68(speaker_dir, suffix, out_dir: Path, skip_existing: bool, hparams, others_params):
    speaker_dir, spk_id = speaker_dir
    trans_dict = others_params["trans_dict"]
    detach_label_and_embed_utt = others_params["detach_label_and_embed_utt"]
    metadata = []
    wav_fpath_list = speaker_dir.glob("*."+suffix)

    utt_fpath_list = list(speaker_dir.glob("*."+suffix))
    utt_num = len(utt_fpath_list)
    # Iterate over each wav
    for wav_fpath in wav_fpath_list:
        assert wav_fpath.exists(), str(wav_fpath)+" not exist."

        # Process each utterance
        wav, _ = librosa.load(str(wav_fpath), hparams.sample_rate)
        # wav_bak = wav
        wav = wav / np.max(np.abs(wav)) * hparams.rescaling_max # norm

        # denoise
        if len(wav) > hparams.sample_rate*(0.3+0.1):
          noise_wav = np.concatenate([wav[:int(hparams.sample_rate*0.15)],
                                      wav[-int(hparams.sample_rate*0.15):]])
          profile = logmmse.profile_noise(noise_wav, hparams.sample_rate)
          wav = logmmse.denoise(wav, profile, eta=0)

        # trim silence
        wav = audio.trim_silence(wav, 20) # top_db: smaller for noisy
        # audio.save_wav(wav_bak, str(wav_fpath.name), hparams.sample_rate)
        # audio.save_wav(wav, str(wav_fpath.name).replace('.wav','_trimed.wav'),
        #                hparams.sample_rate)

        text = trans_dict[wav_fpath.name]["text"]

        # Chinese to Pinyin
        pinyin = " ".join(get_pinyin(text, std=True, pb=True))

        # print(wav_fpath.name, wav_fpath.stem)
        random_uttBasename_forSpkEmbedding=None
        if detach_label_and_embed_utt:
            random_uttBasename_forSpkEmbedding=utt_fpath_list[np.random.randint(utt_num)].stem
        metadata.append(process_utterance(wav, pinyin, out_dir, wav_fpath.stem,
                                          skip_existing, hparams, random_uttBasename_forSpkEmbedding))
    return [m for m in metadata if m is not None]
# endregion SLR68

# region librispeech
def preprocess_librispeech(datasets_root: Path, dataset: str, out_dir: Path, n_processes: int,
                           skip_existing: bool, hparams, detach_label_and_embed_utt):
    del detach_label_and_embed_utt
    # Gather the input directories
    dataset_root = datasets_root.joinpath("LibriSpeech")
    input_dirs = [
        dataset_root.joinpath("train-clean-100"),
        # dataset_root.joinpath("train-clean-360"),
        # dataset_root.joinpath("train-other-500"),
    ]
    print("\n    ".join(map(str, ["Using data from:"] + input_dirs)))
    assert all(input_dir.exists() for input_dir in input_dirs)
    all_sub_dirs = list(chain.from_iterable(input_dir.glob("*") for input_dir in input_dirs))
    speaker_dirs = []
    for _dir in all_sub_dirs:
        if _dir.is_file(): continue
        speaker_dirs.append(_dir)
    speaker_dirs.sort()
    _preprocess_speakers(speaker_dirs, dataset, "flac", _preprocess_speaker_librispeech, None,
                         out_dir, n_processes, skip_existing, hparams)


def _preprocess_speaker_librispeech(speaker_dir, suffix, out_dir: Path, skip_existing: bool, hparams, others_params):
    speaker_dir, spk_id = speaker_dir
    speaker_dir = Path(speaker_dir)
    metadata = []
    for book_dir in speaker_dir.glob("*"):
        # Gather the utterance audios and texts
        try:
            alignments_fpath = next(book_dir.glob("*.alignment.txt"))
            with alignments_fpath.open("r") as alignments_file:
                alignments = [line.rstrip().split(" ") for line in alignments_file]
        except StopIteration:
            # A few alignment files will be missing
            continue

        # Iterate over each entry in the alignments file
        for wav_fname, words, end_times in alignments:
            wav_fpath = book_dir.joinpath(".".join([wav_fname, suffix]))
            assert wav_fpath.exists(), str(wav_fpath)
            words = words.replace("\"", "").split(",")
            end_times = list(map(float, end_times.replace("\"", "").split(",")))

            # Process each sub-utterance
            wavs, texts = split_on_silences(wav_fpath, words, end_times, hparams)
            for i, (wav, text) in enumerate(zip(wavs, texts)):
                sub_basename = "%s_%02d-id-%s" % (wav_fname, i, str(spk_id))
                metadata.append(process_utterance(wav, text, out_dir, sub_basename,
                                                  skip_existing, hparams))
    return [m for m in metadata if m is not None]
# endregion librispeech


def _preprocess_speakers(speaker_dirs: list, dataset: str, wav_suffix: str, preprocess_speaker_fn, process_speaker_fn_params,
                         out_dir: Path, n_processes: int, skip_existing: bool, hparams):
    # per-speaker
    # Create the output directories for each output file type
    out_dir.joinpath("mels").mkdir(exist_ok=True)
    out_dir.joinpath("audio").mkdir(exist_ok=True)

    # Create a metadata file
    metadata_fpath = out_dir.joinpath("train.txt")
    metadata_file = metadata_fpath.open("a" if skip_existing else "w", encoding="utf-8")

    # Preprocess the dataset
    func = partial(preprocess_speaker_fn, suffix=wav_suffix, out_dir=out_dir,
                   skip_existing=skip_existing, hparams=hparams, others_params=process_speaker_fn_params)
    speaker_dirs_with_id = []
    spl_list_f = open('speaker_id.txt', 'w')
    for i, speaker_dir in enumerate(speaker_dirs):
      speaker_dirs_with_id.append((speaker_dir, i))
      spl_list_f.write(str(i)+"  "+str(speaker_dir)+"\n")
    spl_list_f.close()

    # for speaker_dir_and_i in speaker_dirs_with_id: # DEBUG
    #     print(speaker_dir_and_i)
    #     speaker_metadata = func(speaker_dir_and_i)
    #     for metadatum in speaker_metadata:
    #         metadata_file.write("|".join(str(x) for x in metadatum) + "\n")
    #     break
    # exit(0)

    job = Pool(n_processes).imap(func, speaker_dirs_with_id)
    for speaker_metadata in tqdm(job, dataset, len(speaker_dirs), unit="speakers"):
        for metadatum in speaker_metadata:
            metadatum = list(metadatum)
            embed_dir = metadatum[2]
            audio_get_embed = str(out_dir.joinpath("audio", embed_dir.replace('embed-', 'audio-')))
            # audio may not exist (filted by "skip utterances that are too short" in process_utterance).
            if not (os.path.exists(audio_get_embed) and os.path.isfile(audio_get_embed)):
                metadatum[2] = metadatum[0].replace('audio-', 'embed-')
            # print(metadatum[:3], flush=True)
            metadata_file.write("|".join(str(x) for x in metadatum) + "\n")
    metadata_file.close()

    # Verify the contents of the metadata file
    with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
    mel_frames = sum([int(m[4]) for m in metadata])
    timesteps = sum([int(m[3]) for m in metadata])
    sample_rate = hparams.sample_rate
    hours = (timesteps / sample_rate) / 3600
    print("The dataset consists of %d utterances, %d mel frames, %d audio timesteps (%.2f hours)." %
          (len(metadata), mel_frames, timesteps, hours))
    print("Max input length (text chars): %d" % max(len(m[5]) for m in metadata))
    print("Max mel frames length: %d" % max(int(m[4]) for m in metadata))
    print("Max audio timesteps length: %d" % max(int(m[3]) for m in metadata))


def split_on_silences(wav_fpath, words, end_times, hparams):
    # Load the audio waveform
    wav, _ = librosa.load(str(wav_fpath), hparams.sample_rate)
    wav = wav / np.abs(wav).max() * hparams.rescaling_max

    # denoise
    if len(wav) > hparams.sample_rate*(0.3+0.1):
      noise_wav = np.concatenate([wav[:int(hparams.sample_rate*0.1)],
                                  wav[-int(hparams.sample_rate*0.1):]])
      profile = logmmse.profile_noise(noise_wav, hparams.sample_rate)
      wav = logmmse.denoise(wav, profile, eta=0)

    words = np.array(words)
    start_times = np.array([0.0] + end_times[:-1])
    end_times = np.array(end_times)
    assert len(words) == len(end_times) == len(start_times)
    assert words[0] == "" and words[-1] == ""

    # Find pauses that are too long
    mask = (words == "") & (end_times - start_times >= hparams.silence_min_duration_split)
    mask[0] = mask[-1] = True
    breaks = np.where(mask)[0] # first dim indexs

    # Profile the noise from the silences and perform noise reduction on the waveform
    silence_times = [[start_times[i], end_times[i]] for i in breaks]
    silence_times = (np.array(silence_times) * hparams.sample_rate).astype(np.int)
    noisy_wav = np.concatenate([wav[stime[0]:stime[1]] for stime in silence_times])
    if len(noisy_wav) > hparams.sample_rate * 0.02:
        profile = logmmse.profile_noise(noisy_wav, hparams.sample_rate)
        wav = logmmse.denoise(wav, profile, eta=0)

    # Re-attach(Re-join) segments that are too short
    segments = list(zip(breaks[:-1], breaks[1:]))
    segment_durations = [start_times[end] - end_times[start] for start, end in segments]
    i = 0
    while i < len(segments) and len(segments) > 1:
        if segment_durations[i] < hparams.utterance_min_duration:
            # See if the segment can be re-attached with the right or the left segment
            left_duration = float("inf") if i == 0 else segment_durations[i - 1]
            right_duration = float("inf") if i == len(segments) - 1 else segment_durations[i + 1]
            joined_duration = segment_durations[i] + min(left_duration, right_duration)

            # Do not re-attach if it causes the joined utterance to be too long
            if joined_duration > hparams.hop_size * hparams.max_mel_frames / hparams.sample_rate:
                i += 1
                continue

            # Re-attach the segment with the neighbour of shortest duration
            j = i - 1 if left_duration <= right_duration else i
            segments[j] = (segments[j][0], segments[j + 1][1])
            segment_durations[j] = joined_duration
            del segments[j + 1], segment_durations[j + 1]
        else:
            i += 1

    # Split the utterance
    segment_times = [[end_times[start], start_times[end]] for start, end in segments]
    segment_times = (np.array(segment_times) * hparams.sample_rate).astype(np.int)
    wavs = [wav[segment_time[0]:segment_time[1]] for segment_time in segment_times] # [N_seg, seg_time]
    texts = [" ".join(words[start + 1:end]).replace("  ", " ") for start, end in segments] # [N_seg]

    # # DEBUG: play the audio segments (run with -n=1)
    # import sounddevice as sd
    # if len(wavs) > 1:
    #     print("This sentence was split in %d segments:" % len(wavs))
    # else:
    #     print("There are no silences long enough for this sentence to be split:")
    # for wav, text in zip(wavs, texts):
    #     # Pad the waveform with 1 second of silence because sounddevice tends to cut them early
    #     # when playing them. You shouldn't need to do that in your parsers.
    #     wav = np.concatenate((wav, [0] * 16000))
    #     print("\t%s" % text)
    #     sd.play(wav, 16000, blocking=True)
    # print("")

    return wavs, texts


int16_max = (2 ** 15) - 1
vad_moving_average_width = 8
vad_max_silence_length = 6

def trim_long_silences(wav, vad_window_length, sampling_rate):
    """
    Ensures that segments without voice in the waveform remain no longer than a
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000

    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)

    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    return wav[audio_mask == 1]


def process_utterance(wav: np.ndarray, text: str, out_dir: Path, basename: str,
                      skip_existing: bool, hparams, random_uttBasename_forSpkEmbedding=None):
    '''
    random_uttBasename_forSpkEmbedding: if not None, use the utterance to generate speaker embedding in synthesizer training.
    '''
    ## FOR REFERENCE:
    # For you not to lose your head if you ever wish to change things here or implement your own
    # synthesizer.
    # - Both the audios and the mel spectrograms are saved as numpy arrays
    # - There is no processing done to the audios that will be saved to disk beyond volume
    #   normalization (in split_on_silences)
    # - However, pre-emphasis is applied to the audios before computing the mel spectrogram. This
    #   is why we re-apply it on the audio on the side of the vocoder.
    # - Librosa pads the waveform before computing the mel spectrogram. Here, the waveform is saved
    #   without extra padding. This means that you won't have an exact relation between the length
    #   of the wav and of the mel spectrogram. See the vocoder data loader.


    # Skip existing utterances if needed
    mel_fpath = out_dir.joinpath("mels", "mel-%s.npy" % basename)
    wav_fpath = out_dir.joinpath("audio", "audio-%s.npy" % basename)
    if skip_existing and mel_fpath.exists() and wav_fpath.exists():
        return None

    # Skip utterances that are too short
    if len(wav) < hparams.utterance_min_duration * hparams.sample_rate:
        return None

    wav = trim_long_silences(wav, hparams.vad_window_length, hparams.sample_rate)

    # Compute the mel spectrogram
    wav = wav.astype(np.float64)

    # feature extraction
    f0, sp, ap = pw.wav2world(wav, hparams.sample_rate)
    n_frames = len(f0)

    # reduce the dimension of ap from 513 to 2
    enc_ap = pw.code_aperiodicity(ap, hparams.sample_rate)

    # feature normalization
    lf0 = audio.f0_normalize(f0)
    mgc = audio.sp_normalize(sp, hparams)
    bap = audio.ap_normalize(enc_ap)
    # print(lf0.dtype, mgc.dtype, bap.dtype, flush=True)
    # print(np.shape(lf0), np.shape(mgc), np.shape(bap), flush=True)
    # wav233 = audio.synthesize(lf0,mgc,bap,hparams)
    # audio.save_wav(wav233, "/home/zhangwenbo5/worklhf/english_voice_clone/Voice_Cloning_byid_pyworld/test_wav.wav", hparams.sample_rate)
    ######################

    # f0, sp, ap = pw.wav2world(wav, hparams.sample_rate)
    # f0 = f0.astype(np.float32)
    # sp = sp.astype(np.float32)
    # ap = ap.astype(np.float32)
    # # print(f0.dtype, sp.dtype, ap.dtype, flush=True)
    # # f0 /= 100.0
    # # sp *= 1000.0
    # n_frames = np.shape(f0)[0]
    # f0 = np.reshape(f0, [n_frames, 1])
  ###########################################

    # mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32) # [80, frame]
    # mel_frames = mel_spectrogram.shape[1]
    lf0 = np.reshape(lf0, [n_frames, 1])
    mel_spectrogram = np.concatenate((lf0, mgc, bap), axis=-1) # [frame, 1+60+1], 1027

    # Skip utterances that are too long
    if n_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        return None

    # Write the spectrogram, embed and audio to disk
    np.save(mel_fpath, mel_spectrogram, allow_pickle=False)
    np.save(wav_fpath, wav, allow_pickle=False)

    # Return a tuple describing this training example
    embed_basename = basename
    if random_uttBasename_forSpkEmbedding is not None:
        embed_basename = random_uttBasename_forSpkEmbedding
    return wav_fpath.name, mel_fpath.name, "embed-%s.npy" % embed_basename, len(wav), n_frames, text


def embed_utterance(fpaths, spk_num):
    # Compute the speaker embedding of the utterance
    wav_fpath, embed_fpath = fpaths
    spk_id = str(wav_fpath).split('-')[-1].split('.')[0]
    embed = np.eye(spk_num, dtype=np.float32)[int(spk_id)]
    np.save(embed_fpath, embed, allow_pickle=False)


def create_embeddings(synthesizer_root: Path, speaker_num: int, encoder_model_fpath: Path, n_processes: int, datasets_root: Path):
    del datasets_root
    wav_dir = synthesizer_root.joinpath("audio")
    metadata_fpath = synthesizer_root.joinpath("train.txt")
    assert wav_dir.exists() and metadata_fpath.exists()
    embed_dir = synthesizer_root.joinpath("embeds")
    embed_dir.mkdir(exist_ok=True)

    # Gather the input wave filepath and the target output embed filepath
    with metadata_fpath.open("r") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
        fpaths = [(wav_dir.joinpath(m[2].replace('embed-', 'audio-')),
                   embed_dir.joinpath(m[2])) for m in metadata]

    # TODO: improve on the multiprocessing, it's terrible. Disk I/O is the bottleneck here.
    # Embed the utterances in separate threads
    func = partial(embed_utterance, spk_num=speaker_num)
    # for fpath_pair in fpaths:
    #     # print("fpath_pair", fpath_pair) # DEBUG
    #     func(fpath_pair)
    job = Pool(n_processes).imap(func, fpaths)
    list(tqdm(job, "Embedding", len(fpaths), unit="utterances"))
