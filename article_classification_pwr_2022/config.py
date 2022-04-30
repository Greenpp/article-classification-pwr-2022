RANDOM_SEED = 42


class TrainingConfig:
    segment_size = 250
    segment_overlap = 50
    batch_size = 2
    classes = 11
    epochs = 10
    learning_rate = 1e-5
    encoding_dim = 128
    aggregation_dim = 64
