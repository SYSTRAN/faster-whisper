from flask import Flask, request, render_template
import os
from faster_whisper import WhisperModel

app = Flask(__name__)

# Create the 'temp' directory if it doesn't exist
if not os.path.exists('temp'):
    os.makedirs('temp')

@app.route('/')
def index():
    return render_template('index.html')

# ...

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    audio_files = request.files.getlist('audio_files')

    if not audio_files:
        return "No files provided."

    try:
        # Initialize the WhisperModel outside the if block for better efficiency
        model_size = "large-v2"
        # Run on CPU with INT8
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

        for audio_file in audio_files:
            # Determine the file extension
            file_extension = os.path.splitext(audio_file.filename)[1].lower()

            if file_extension in ('.mp3', '.mp4'):
                # Save the uploaded file to a temporary location
                audio_file_path = os.path.join("temp", audio_file.filename)
                audio_file.save(audio_file_path)

                segments, info = model.transcribe(audio_file_path, beam_size=5)

                print(f"Detected language '{info.language}' with probability {info.language_probability}")

                for segment in segments:
                    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

                # Remove the temporary audio file
                os.remove(audio_file_path)
            else:
                return f"Invalid file type: {audio_file.filename}. Supported types: .mp3 and .mp4"

        return "Transcription complete"
    except Exception as e:
        # Handle exceptions here
        return "An error occurred during transcription: " + str(e)

# ...

if __name__ == '__main__':
    app.run(debug=True)
