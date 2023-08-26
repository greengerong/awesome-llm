import math
import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from audiocraft.utils.notebook import display_audio

# 选择模型，四选一 small/medium/melody/large
# 如果没有则会从 huggingface 自动下载
# 请先设置系统环境变量 MUSICGEN_ROOT
model = MusicGen.get_pretrained('melody')

# 产生一段间歇的bip bip旋律
def get_bip_bip(bip_duration=0.125, frequency=440,
                duration=0.5, sample_rate=32000, device="cuda"):
    """Generates a series of bip bip at the given frequency."""
    t = torch.arange(
        int(duration * sample_rate), device="cuda", dtype=torch.float) / sample_rate
    wav = torch.cos(2 * math.pi * 440 * t)[None]
    tp = (t % (2 * bip_duration)) / (2 * bip_duration)
    envelope = (tp >= 0.5).float()
    return wav * envelope

# 音乐时长设为12秒
model.set_generation_params(duration=12)

# 根据bipbip声的旋律，及两段prompt，生成新的音乐
wav = model.generate_continuation(
    get_bip_bip(0.125).expand(2, -1, -1), 
    32000, ['Jazz jazz and only jazz', 
            'Heartful EDM with beautiful synths and chords'], 
    progress=True)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'test3_{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

print("finsh all music")