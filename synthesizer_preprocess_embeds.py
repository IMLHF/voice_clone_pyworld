from synthesizer.preprocess import create_embeddings
from utils.argutils import print_args
from pathlib import Path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates embeddings for the synthesizer from the LibriSpeech utterances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # parser.add_argument("synthesizer_root", type=Path,
    #                     help="Path to the synthesizer training data that contains the audios and the train.txt file. "
    #                     "If you let everything as default, it should be <datasets_root>/SV2TTS/synthesizer/.")
    parser.add_argument("datasets_root", type=Path,
                        help="Path to the directory containing your datasets 'SV2TTS-byid'.")
    parser.add_argument("speaker_num", type=int, help="train set speaker num")
    parser.add_argument("-e", "--encoder_model_fpath", type=Path,
                        default="encoder/saved_models/pretrained.pt",
                        help="Path your trained encoder model.")
    parser.add_argument("-n", "--n_processes", type=int, default=26,
                        help="Number of parallel processes. An encoder is created for each, so you may need to lower "
                        "this value on GPUs with low memory. Set it to 1 if CUDA is unhappy.")
    args = parser.parse_args()

    # Process the arguments
    args.synthesizer_root = args.datasets_root.joinpath("SV2TTS_pyworld_byid_small", "synthesizer")

    # Preprocess the dataset
    print_args(args, parser)
    create_embeddings(**vars(args))
