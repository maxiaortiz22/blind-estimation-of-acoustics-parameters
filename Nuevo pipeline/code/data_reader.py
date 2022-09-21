import glob
from pathlib import Path

import pandas as pd

NUMBERS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def read_dataset(path, sample_frac=1.0, random_state=None):
    metadata = []
    for filename in glob.glob(path + '/*/*.wav'):
        filename = Path(filename)
        number = str(filename.absolute()).split('/')[-2]
        metadata.append({
            'audio_filename': filename.absolute(),
            'number': number,
            'label': NUMBERS.index(number),
            'speaker': filename.stem.split('_')[0],
            'utterance': filename.stem.split('_')[-1],
        })
    return pd.DataFrame(metadata).sample(frac=sample_frac, random_state=random_state)
