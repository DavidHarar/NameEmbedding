from src.experiment import pipeline

general_config = {
    'model_type':'RoBERTa',
    'experiments_folder':'./experiments',
    'training_set_path':"./data/raw/text/training_names_processed.txt",
    'validation_set_path':"./data/raw/text/validation_names_processed.txt",
}

tokenizer_config = {
    'model_type':general_config['model_type'],
    'vocab_size' : 5000,
    'min_frequency' : 2,
    'max_len':30,
    'add_special_tokens' : True,
    'pad_to_max_length' : True,
    'return_attention_mask' : True,
    'return_tensors' : 'pt',
    'create_a_new_tokenizer': True,
    'tokenizer_location':'./experiments/experiment_1/tokenizer'
}

model_config = {
    'model_type':general_config['model_type'],
    "vocab_size": tokenizer_config['vocab_size'],
    "max_position_embeddings": 32,
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "type_vocab_size": 1,
}

train_config = {
    "train_epochs": 10,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "train_batch_size": 512,
    "valid_batch_size": 32,
    "max_len": 30,
    "mlm_probability": 0.15,
    'save_steps':50,
    'eval_steps':50,
}
# trainer, model, eval_dataset = experiment(general_config, tokenizer_config, model_config, train_config)
