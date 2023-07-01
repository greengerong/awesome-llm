import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# 选择模型，四选一 small/medium/melody/large
# 如果没有则会从 huggingface 自动下载
# 请先设置系统环境变量 MUSICGEN_ROOT
model = MusicGen.get_pretrained('melody')

# 音乐时长设为12秒
model.set_generation_params(duration=2 * 60)

# 根据三段prompt生成音乐
descriptions = [
    # "a piano playing a sad chambers music, canon style",
    # "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions",
    # "A grand orchestral arrangement with thunderous percussion, epic brass fanfares, and soaring strings, creating a cinematic atmosphere fit for a heroic battle.",
    "《仓央嘉措情歌》",
   '那一日闭目在经殿香雾中,蓦然听见是你颂经中的真言,那一夜摇动啊所有的经筒,不为超度只为触摸你的指尖,那一年磕长头匍匐在山路,不为觐见只为贴着你的温暖,那一世转山啊转水转佛塔,不为来世只为途中与你相见,那一瞬我已飞,喔飞成仙,不为来世只为有你,喜乐平安,那一瞬我已飞,不为来世只为有你,那一日闭目在经殿香雾中,蓦然听见是你颂经中的真言,那一夜摇动啊所有的经筒,不为超度只为触摸你的指尖,那一年磕长头匍匐在山路,不为觐见只为贴着你的温暖,那一世转山啊转水转佛塔,不为来世只为途中与你相见,那一瞬我已飞,喔飞成仙,不为来世只为有你,喜乐平安,那一瞬我已飞,不为来世只为有你'
  ]
wav = model.generate(
    descriptions,
    progress=True
)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'text_{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

print("finsh all music")


