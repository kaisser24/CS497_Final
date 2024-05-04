from pydub import AudioSegment
import os

#used for dataset creation
#Split all wav files in the input directory into chunks of 10 seconds (1000 ms)

def split_wav_files(input_dir, output_dir, chunk_length_ms=10000):
    print("Splitting files in directory:", input_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all folders in the input directory
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            #If theres a folder in the directory, parse through it
            output_subdir = os.path.join(output_dir, item)
            split_wav_files(item_path, output_subdir, chunk_length_ms)
        elif item.endswith(".wav"):
            # If there's a wav file, split it
            #Get artist name from the directory (labels)
            artist_name = os.path.basename(os.path.dirname(item_path))
            output_file_dir = os.path.join(output_dir, artist_name)
            os.makedirs(output_file_dir, exist_ok=True)
            # Split each wav file in the directory
            split_wav_file(item_path, output_file_dir, chunk_length_ms, artist_name)

def split_wav_file(input_file, output_dir, chunk_length_ms=10000, artist_name="unknown"):
    audio = AudioSegment.from_wav(input_file)
    total_duration_ms = len(audio)
    num_chunks = total_duration_ms // chunk_length_ms + 1

    for i in range(num_chunks):
        start_time = i * chunk_length_ms
        end_time = min((i + 1) * chunk_length_ms, total_duration_ms)
        chunk = audio[start_time:end_time]
        # Use "a_" prefix in file names
        # this was for 2 and 3 addition to dataset, to speed up moving files to train dataset (prevent duplicates)
        output_file_name = os.path.splitext(os.path.basename(input_file))[0]  # Get file name without extension
        chunk.export(os.path.join(output_dir, f"a_{output_file_name}_{i+1}.wav"), format="wav")

# Grab input directory and output directory, Combo_Splits is the full dataset
input_directory = "Download_Wavs"
output_directory = "Download_Splits"
split_wav_files(input_directory, output_directory)


