from typing import Sequence
from scipy.io import wavfile
import scipy.signal as sps
from datasets import Audio, Dataset
import soundfile
from pathlib import Path
from tqdm import tqdm
from glob import glob

AUDIO_PATH = r'C:\projects\hCPC\data\tira-asr\himidan'
RESAMPLED_PATH = r'C:\projects\hCPC\data\tira-asr-resampled'

def resample_scipy(audio_fps: Sequence[str], out_dir: str, new_sr: int = 16000):
    # taken from Jeremy Cochoy and Rachid Riad
    # https://stackoverflow.com/questions/30619740/downsampling-wav-audio-file
    for audio_fp in tqdm(audio_fps, desc='Resampling using scipy signal...'):
        old_sr, array = wavfile.read(audio_fp)
        num_samples = round(len(array) * float(new_sr) / old_sr)
        resampled = sps.resample(array, num=num_samples)
        audio_filename = Path(audio_fp).name
        out_fp = out_dir/audio_filename
        soundfile.write(str(out_fp), resampled, new_sr)

def resample_hf(audio_fps: Sequence[str], out_dir: str, new_sr: int = 16000):
    print('Resampling using HuggingFace datasets...')
    audio_ds = Dataset.from_dict({'audio': audio_fps}).cast_column('audio', Audio(sampling_rate=new_sr))
    print('Saving output...')
    def save_audio(row: dict) -> None:
        audio_name = Path(row['audio']['path']).name
        out_path = out_dir/audio_name
        audio = row['audio']
        audio_array = audio['array']
        sample_rate = audio['sampling_rate']
        soundfile.write(str(out_path), audio_array, samplerate=sample_rate)
    audio_ds.map(save_audio)

    return audio_ds

def main():
    audio_fps = glob(AUDIO_PATH+'\\**\\*.wav', recursive=True)
    scipy_path = Path(RESAMPLED_PATH)/'scipy\\'
    hf_path = Path(RESAMPLED_PATH)/'hf\\'
    scipy_path.mkdir(exist_ok=True)
    hf_path.mkdir(exist_ok=True)
    resample_scipy(audio_fps, scipy_path)
    resample_hf(audio_fps, hf_path)

if __name__ == '__main__':
    main()