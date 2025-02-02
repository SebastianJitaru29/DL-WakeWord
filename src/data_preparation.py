import os
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm
import openwakeword
import openwakeword.data as owdata
import datasets
from openwakeword.utils import download_models

def initialize_audio_feature_extractor(melspec_path, embedding_path):
    """Initialize the OpenWakeWord audio feature extractor."""
    return openwakeword.utils.AudioFeatures(melspec_model_path=melspec_path,
                                            embedding_model_path=embedding_path)

def filter_audio_clips(paths, min_length, max_length):
    """Filter audio clips based on length constraints."""
    clips, durations = owdata.filter_audio_paths(paths, 
                                                 min_length_secs=min_length, 
                                                 max_length_secs=max_length, 
                                                 duration_method="header")
    print(f"{len(clips)} clips after filtering, representing ~{sum(durations) // 3600} hours")
    return clips, durations

def create_audio_dataset(audio_paths, sample_rate=16000):
    """Create an audio dataset from file paths."""
    dataset = datasets.Dataset.from_dict({"audio": audio_paths})
    return dataset.cast_column("audio", datasets.Audio(sampling_rate=sample_rate))

def process_audio_dataset(audio_dataset, feature_extractor, output_file, output_shape, batch_size=64, clip_size=3, sample_rate=16000):
    """Process an audio dataset to extract features and store them in a memory-mapped file."""
    clip_samples = sample_rate * clip_size
    
    if os.path.exists(output_file):
        fp = open_memmap(output_file, mode='r+', dtype=np.float32, shape=output_shape)
        print(f"Existing file loaded: {output_file}")
    else:
        fp = open_memmap(output_file, mode='w+', dtype=np.float32, shape=output_shape)
        print(f"New file created: {output_file}")
    
    row_counter = np.count_nonzero(fp[:, 0, 0])
    print(f"Resuming from row {row_counter}")
    
    for i in tqdm(range(0, len(audio_dataset), batch_size), desc=f"Processing {output_file}"):
        batch = [clip["array"] * 32767 for clip in audio_dataset[i:i + batch_size]["audio"]]
        batch = owdata.stack_clips(batch, clip_size=clip_samples).astype(np.int16)
        features = feature_extractor.embed_clips(x=batch, batch_size=1024, ncpu=8)
        
        if row_counter + features.shape[0] > output_shape[0]:
            fp[row_counter:output_shape[0], :, :] = features[:output_shape[0] - row_counter]
            fp.flush()
            break
        else:
            fp[row_counter:row_counter + features.shape[0], :, :] = features
            row_counter += features.shape[0]
            fp.flush()
    
    del fp  # Release lock
    try:
        owdata.trim_mmap(output_file)
        print(f"Trimmed {output_file} successfully.")
    except PermissionError:
        print(f"Warning: Could not delete {output_file}. Ensure it is not in use.")
    
    print(f"Processing completed for {output_file}")

