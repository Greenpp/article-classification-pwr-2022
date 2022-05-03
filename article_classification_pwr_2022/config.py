RANDOM_SEED = 42


class TrainingConfig:
    model = "distilbert-base-uncased"
    segment_size = 250
    segment_overlap = 50
    batch_size = 4
    classes = 11
    epochs = 10
    learning_rate = 1e-5
    encoding_dim = 128
    aggregation_dim = 64
    processing_batch_size = 256
    tokenization_batch_size = 1

    project_name = "article-classification-pwr-2022"
    run_name = "test-full"
    processed_train = "data/processed_train.pkl"
    processed_val = "data/processed_val.pkl"
    processed_test = "data/processed_test.pkl"
