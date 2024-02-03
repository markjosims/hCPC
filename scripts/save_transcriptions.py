from datasets import load_dataset
from typing import Mapping
from pathlib import Path

ACUTE = '\u0301'
GRAVE = '\u0300'
MACRN = '\u0304'
CARON = '\u0304'
CIRCM = '\u0302'
TONES = [ACUTE, GRAVE, MACRN, CARON, CIRCM]

DATA_DIR = '/mnt/cube/home/AD/mjsimmons/hCPC/data/tira-asr/himidan'

def strip_tone(text: str) -> str:
    for t in TONES:
        text = text.replace(t, '')
    return text

def save_transcription_file(row: Mapping, split_dir: str) -> None:
    text = row['sentence']
    processed_text = strip_tone(text) # may end up needing more processing
    
    filename = row['audio']['path']
    filestem = Path(filename).stem
    savepath = str(split_dir/filestem)+'.lab'
    with open(savepath, 'w') as f:
        f.write(processed_text)
    

if __name__ == '__main__':
    ds = load_dataset('markjosims/tira-asr')
    for split, split_ds in ds.items():
        split_dir = Path(DATA_DIR)/split
        split_ds.map(lambda row: save_transcription_file(row, split_dir))