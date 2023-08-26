import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# 选择模型，四选一 small/medium/melody/large
# 如果没有则会从 huggingface 自动下载
# 请先设置系统环境变量 MUSICGEN_ROOT
model = MusicGen.get_pretrained('melody')

# 音乐时长设为12秒
model.set_generation_params(duration=12)

wav = model.generate_unconditional(num_samples=3, progress=True)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'uncond_{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

print("finsh all music")