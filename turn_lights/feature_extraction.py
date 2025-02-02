from data_preparation import (
    initialize_audio_feature_extractor,
    filter_audio_clips,
    create_audio_dataset,
    process_audio_dataset
)

# Paths and model initialization
melspec_model = "models/melspectrogram.onnx"
embedding_model = "models/embedding_model.onnx"
feature_extractor = initialize_audio_feature_extractor(melspec_model, embedding_model)

# Filter audio clips
negative_clips, negative_durations = filter_audio_clips([
    "data/fma_sample", "data/fsd50k_sample", "data/cv11_test_clips/en_test_0"
], min_length=1.0, max_length=1800.0)

positive_clips, _ = filter_audio_clips(["data/turn_on_the_office_lights"], min_length=1.0, max_length=2.0)

# Create datasets
audio_dataset_neg = create_audio_dataset(negative_clips)
audio_dataset_pos = create_audio_dataset(positive_clips)

# Define output paths
negative_output_file = "data/negative_features.npy"
positive_output_file = "data/positive_features.npy"

# Compute feature shapes
n_feature_cols = feature_extractor.get_embedding_shape(3)
neg_output_shape = (int(sum(negative_durations) // 3), n_feature_cols[0], n_feature_cols[1])
pos_output_shape = (len(positive_clips), n_feature_cols[0], n_feature_cols[1])

# Process datasets
process_audio_dataset(audio_dataset_neg, feature_extractor, negative_output_file, neg_output_shape)
process_audio_dataset(audio_dataset_pos, feature_extractor, positive_output_file, pos_output_shape)