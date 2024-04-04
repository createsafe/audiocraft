"""
Basic text-to-music example with command line args.

```console
python -m demos.inference --prompt "emocore drum and bass"
```

author: javanasse
"""

import os
import argparse

import torch
import soundfile as sf

from audiocraft.models.builders import get_lm_model, get_compression_model
from audiocraft.models.loaders import load_compression_model, load_lm_model
from audiocraft.models.encodec import CompressionModel
from audiocraft.models.lm import LMModel
from audiocraft.models import MusicGen

p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("-p", "--prompt",
               type=str,
               help="text prompt condition")
p.add_argument("-o", "--output",
               type=str,
               help="output file or directory name",
               default="output")
args = p.parse_args()

# setup model
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

# run inference
output = model.generate(
    descriptions=[args.prompt],
    progress=True, return_tokens=True
)

output_path = None
# if os.path.isdir(args.output):
#     if not os.path.exists(args.output):
#         os.mkdir(args.output)
#     output_path = os.path.join(args.output, f"{args.prompt}.wav")

if not output_path:
    output_path = f"{args.prompt}.wav"

sf.write(output_path, output[0].squeeze().cpu().numpy(), samplerate=model.sample_rate)
