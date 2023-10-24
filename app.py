from flask import Flask, request, render_template, send_file, jsonify
import os
import openai
import re
import shutil
import zipfile  # Import the zipfile module
from faster_whisper import WhisperModel

app = Flask(__name__)

# Define global variables to store transcription results and text file paths
transcription_results = []
txt_file_paths = []  # Initialize the list for text file paths

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
        model_size = "medium.en"
        # Run on CPU with INT8
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

        # Clear the 'temp' directory if it contains any files
        temp_directory = 'temp'
        for filename in os.listdir(temp_directory):
            file_path = os.path.join(temp_directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")

        # Remove previously generated .zip file
        for filename in os.listdir(os.getcwd()):
            if filename.endswith('.zip'):
                os.unlink(filename)

        transcription_results = []  # Store transcription results
        txt_file_paths = []

        for audio_file in audio_files:
            # Determine the file extension
            file_extension = os.path.splitext(audio_file.filename)[1].lower()

            if file_extension in ('.mp3', '.mp4'):
                # Save the uploaded file to a temporary location
                audio_file_path = os.path.join("temp", audio_file.filename)
                audio_file.save(audio_file_path)

                segments, info = model.transcribe(audio_file_path, beam_size=5)

                print(f"Detected language '{info.language}' with probability {info.language_probability}")

                transcription_text = ""

                for segment in segments:
                    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                    transcription_text += f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"

                transcription_results.append((audio_file.filename, transcription_text))

                # Remove the temporary audio file
                os.remove(audio_file_path)
            else:
                return f"Invalid file type: {audio_file.filename}. Supported types: .mp3 and .mp4"

        # Create and save .txt and .vtt files
        txt_file_paths = []
        vtt_file_paths = []

        for filename, transcription_text in transcription_results:
            txt_file_path = os.path.join('temp', f'{os.path.splitext(filename)[0]}.txt')
            with open(txt_file_path, 'w') as text_file:
                text_file.write(transcription_text)
            txt_file_paths.append(txt_file_path)

            vtt_file_path = os.path.join('temp', f'{os.path.splitext(filename)[0]}.vtt')
            with open(vtt_file_path, 'w') as vtt_file:
                vtt_file.write("WEBVTT\n\n")
                segments = transcription_text.strip().split('\n')
                for i, segment in enumerate(segments):
                    if segment.strip():
                        vtt_file.write(f"{i + 1}\n")
                        vtt_file.write(f"{segment}\n\n")
            vtt_file_paths.append(vtt_file_path)

        # Generate a single .zip file containing both .txt and .vtt files
        if len(txt_file_paths) > 0:
            zip_filename = 'transcriptions.zip'
            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                for txt_file_path in txt_file_paths:
                    zipf.write(txt_file_path, os.path.basename(txt_file_path))
                    vtt_file_path = txt_file_path.replace('.txt', '.vtt')
                    if os.path.exists(vtt_file_path):
                        zipf.write(vtt_file_path, os.path.basename(vtt_file_path))

        # Automatically download the .zip file
        if os.path.exists(zip_filename):
            response = send_file(zip_filename, as_attachment=True)
            return response

        return "No valid transcription files found."

    except Exception as e:
        # Handle exceptions here
        return "An error occurred during transcription: " + str(e)

# ...

@app.route('/generate_descriptions', methods=['POST'])
def generate_descriptions():
    try:
        # Initialize the OpenAI API key
        openai.api_key = "OPENAI_API_KEY"

        temp_directory = 'temp'
        response_descriptions = {}

        for filename in os.listdir(temp_directory):
            if filename.endswith(".txt"):
                txt_file_path = os.path.join(temp_directory, filename)
                with open(txt_file_path, 'r') as text_file:
                    transcription_text = text_file.read()

                # Use regular expressions to extract the text within square brackets
                timestamps_removed = re.sub(r'\[\d+\.\d+s -> \d+\.\d+s\] ', '', transcription_text)

                # Concatenate the text into a single block
                resulting_text = ' '.join(timestamps_removed.splitlines())

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "Provide a 2 or 3 sentence description of what happens in the lesson you are provided"
                        },
                        {
                            "role": "user",
                            "content": resulting_text
                        }
                    ],
                    temperature=0,
                    max_tokens=1024,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )

                # Get the generated description from the API response
                description = response.choices[0].message["content"]

                # Print the description and transcription text to the console
                print(f"Generated Description for {filename}:")
                print(description)

                response_descriptions[filename] = description

        # Return the descriptions as a JSON object
        return jsonify(response_descriptions)

    except Exception as e:
        # Handle exceptions here
        return jsonify({"error": "An error occurred during description generation: " + str(e)})

# ...

if __name__ == '__main__':
    app.run(debug=True)