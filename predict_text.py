import os
import omegaconf
import time
from datetime import datetime

import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.models.encodec import CompressionModel
from audiocraft.models.builders import get_lm_model
from audiocraft.data.audio import audio_write

# load constituent models and states
model_dir = 'triniti_weights'
cfg_file = os.path.join(model_dir, 'triniti_cfg.yaml')
cfg = omegaconf.OmegaConf.load(cfg_file)
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

# load compression model
compression_model = CompressionModel.get_pretrained('facebook/encodec_32khz')
compression_model.to(device=cfg.device)
print('loaded compression model')

# load language model
lm = get_lm_model(cfg)
lm_dict = torch.load(
        os.path.join(model_dir, 'state_dict.bin')
    )
lm.load_state_dict(
    lm_dict['best_state']
)
lm = lm.to(device=cfg.device)
print('loaded lm model')

model = MusicGen(
    name='triniti',
    lm=lm,
    compression_model=compression_model,
    max_duration=30.0
)
model.set_generation_params(duration=30)

# load some audio
audio, sample_rate = torchaudio.load('short/dreams.mp3')
audio = audio[:1, :]

descriptions=["a test text prompt", "and another"]
# musical_symbols=[
#     {
#         'chords': "A:min E:min C:maj E:7",
#         'downbeats': [0.1, 0.9, 1.7, 2.5]
#     }
# ]
musical_symbols=[
    {
        'chords': "A:min E:min",
        'downbeats': [0.1, 5.0, 10.0]
    },
    {
        'chords': "A:min E:min",
        'downbeats': [0.1, 5.0, 10.0]
    }
]


# generate
print(f'starting generation: {datetime.now()}')
start = time.time()
wavs = model.generate_with_text_chroma(
    descriptions=descriptions,
    musical_symbols=musical_symbols,
    bpm=[120]*2, 
    meter=[4]*2
)

print(f'finished generation: {datetime.now()}')
print(f'time elapsed = {time.time() - start} seconds')

for n, wav in enumerate(wavs):
    audio_write(stem_name=f"{descriptions[n]}", 
                wav=wav.cpu(),
                sample_rate=32000,
                strategy='loudness',
                loudness_compressor=True)
    
