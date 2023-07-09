import whisper

## wget https://github.com/bjnortier/whisper-ios-demo/raw/main/resources/aragorn.wav

model = whisper.load_model("base")
text = model.transcribe("aragorn.wav")

print(text['text'])