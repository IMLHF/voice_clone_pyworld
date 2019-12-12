if [ -z "$1" ]; then
  echo "Need a step param:"
  echo "encoder_preprocess_train"
  echo "encoder_preprocess_test"
  echo "encoder_train"
  echo "synthesizer_preprocess_audio"
  echo "synthesizer_preprocess_embeds"
  echo "synthesizer_train"
  echo "vocoder_preprocess"
  echo "vocoder_train"
  exit -1
fi

GPU_DEVICES="4"
step=$1

if [ "$step" = "synthesizer_preprocess_audio" ]; then
    OMP_NUM_THREADS=1 python synthesizer_preprocess_audio.py /home/zhangwenbo5/worklhf/english_voice_clone/alldata LibriSpeech --n_processes=56 \
                                           2>&1 | tee -a log_lhf/synthesizer_preprocess_audio.log
elif [ "$step" = "synthesizer_preprocess_embeds" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES python synthesizer_preprocess_embeds.py /home/zhangwenbo5/worklhf/english_voice_clone/alldata \
                                                                              251 \
                                                                              2>&1 | tee -a log_lhf/synthesizer_preprocess_embeds.log
elif [ "$step" = "synthesizer_train" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES python synthesizer_train.py synthesizer /home/zhangwenbo5/worklhf/english_voice_clone/alldata \
                                                                  2>&1 | tee -a log_lhf/synthesizer_train.log
else
    echo "step param error." && exit -1
fi
