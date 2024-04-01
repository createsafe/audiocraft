import os

from audiocraft.utils import export
from audiocraft import train

sig = '5eda0d60'

# output directory
cwd = os.getcwd()
export_dir = os.path.join(cwd, 'export')

# Export the MusicGen model
xp = train.main.get_xp_from_sig(sig)
export.export_lm(xp.folder / 'checkpoint.th', os.path.join(export_dir, 'state_dict.bin'))

# Export the pretrained EnCodec model
export.export_pretrained_compression_model('facebook/encodec_32khz', os.path.join(export_dir, 'compression_state_dict.bin'))
