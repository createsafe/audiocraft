import os

import torch
import soundfile as sf

from audiocraft.models.builders import get_lm_model, get_compression_model
from audiocraft.models.loaders import load_compression_model, load_lm_model
from audiocraft.models.encodec import CompressionModel
from audiocraft.models.lm import LMModel
from audiocraft.models import MusicGen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_state_file = os.path.join('export', 'state_dict.bin')
compression_state_file = os.path.join('export', 'compression_state_dict.bin')

lm = load_lm_model(model_state_file, device=device)
comp = load_compression_model(compression_state_file, device=device)

model = MusicGen(name="uhhh", 
                 compression_model=comp,
                 lm=lm)


model.set_generation_params(
    use_sampling=True,
    top_k=250,
    duration=30
)

output = model.generate(
    descriptions=[
        #'80s pop track with bassy drums and synth',
        #'90s rock song with loud guitars and heavy drums',
        #'Progressive rock drum and bass solo',
        #'Punk Rock song with loud drum and power guitar',
        #'Bluesy guitar instrumental with soulful licks and a driving rhythm section',
        #'Jazz Funk song with slap bass and powerful saxophone',
        'drum and bass beat with intense percussions'
    ],
    progress=True, return_tokens=True
)

sf.write("output.wav", output[0].squeeze().cpu().numpy(), samplerate=model.sample_rate)
