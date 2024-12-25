import os
import logging
import torch
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments, TrainerCallback, DataCollatorForLanguageModeling
from src.utils import initiate_new_experiment, setup_logger
from src.model import model_factory
from src.tokenizer import BPE_Based_Tokenizer
from src.dataset import CustomDataset
from src.post_modeling_analysis import PostModelingReport

def setup_logger(log_folder):
    """
    Sets up the logger to save logs to the specified log folder.
    """
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, "experiment.log")
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)
    return logger


class LoggingCallback(TrainerCallback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            for key, value in logs.items():
                self.logger.info(f"{key}: {value}")

    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        # Log training and eval loss at the end of each epoch
        if logs:
            training_loss = logs.get("loss", "N/A")
            eval_loss = logs.get("eval_loss", "N/A")
            self.logger.info(f"Epoch {state.epoch} completed. "
                             f"Training loss: {training_loss} "
                             f"Eval loss: {eval_loss}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Log metrics after evaluation
        if metrics:
            for key, value in metrics.items():
                self.logger.info(f"Evaluation Metric - {key}: {value}")

    def on_train_end(self, args, state, control, **kwargs):
        # Log after training is complete
        self.logger.info(f"Training completed at epoch {state.epoch}")
        self.logger.info(f"Final training loss: {state.best_metric if hasattr(state, 'best_metric') else 'N/A'}")


def pipeline(general_config, tokenizer_config, model_config, train_config):
    
    # -------------------
    # Setup
    # -------------------
    # load data
    with open(general_config.get("training_set_path"), "r") as file:
        training_names_txt = [line.strip() for line in file]
    with open(general_config.get("validation_set_path"), "r") as file:
        validation_names_txt = [line.strip() for line in file]
    
    # create a new experiment folder
    experiment_location, experiment_index = initiate_new_experiment(general_config.get('experiments_folder'))
    log_folder = os.path.join("logs", "training_logs", f"experiment_{experiment_index}")
    print(f'Saving log to {log_folder}')
    logger = setup_logger(log_folder)
    logging_callback = LoggingCallback(logger)

    logger.info("Starting experiment %d", experiment_index)
    logger.info("Experiment location: %s", experiment_location)

    # initiate a tokenizer
    logger.info("Setting up the tokenizer")
    tokenizer_obj = BPE_Based_Tokenizer(tokenizer_config)

    if tokenizer_config['create_a_new_tokenizer']:
        tokenizer_config['tokenizer_location'] = experiment_location + '/tokenizer'
        tokenizer_obj.tokenizer_location = experiment_location + '/tokenizer'
        os.mkdir(experiment_location+'/tokenizer')
        # train the tokenizer, wrap it with HF tokenizer
        logger.info("Training new tokenizer")
        tokenizer_obj.train_and_save(training_names_txt)
    else:
        assert "tokenizer_location" in tokenizer_config.keys(), 'If a tokenizer is loaded, a location must be provided'
        logger.info("Loading tokenizer from %s", tokenizer_config['tokenizer_location'])

    # load BERT-type tokenizer
    tokenizer_obj.load_and_wrap_tokenizer()

    # initiate a model
    logger.info("Initializing the model")
    model = model_factory(model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # -------------------
    # data processing
    # -------------------
    logger.info("Preparing datasets")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer_obj.tokenizer, mlm=True, mlm_probability=0.15
    )
    
    train_dataset = CustomDataset(
        text_list=training_names_txt,
        tokenizer=tokenizer_obj.tokenizer,
        max_len=tokenizer_config['max_len'],
        include_attention_mask=True
    )
    eval_dataset = CustomDataset(
        text_list=validation_names_txt,
        tokenizer=tokenizer_obj.tokenizer,
        max_len=tokenizer_config['max_len'], 
        include_attention_mask=True
    )
    
    # -------------------
    # Training
    # -------------------
    logger.info("Starting training")
    training_args = TrainingArguments(
        output_dir="model_folder",
        overwrite_output_dir=True,
        eval_strategy='epoch',
        num_train_epochs=train_config["train_epochs"],
        learning_rate=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        per_device_train_batch_size=train_config["train_batch_size"],
        per_device_eval_batch_size=train_config["valid_batch_size"],
        save_steps=train_config['save_steps'],
        eval_steps=train_config['eval_steps'],
        save_total_limit=1,
        max_grad_norm=1.0,
        logging_steps = 20,
        logging_dir=log_folder
    )

    # Create the trainer for our model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[logging_callback]
    )

    # Train the model
    trainer.train()

    # Save model
    logger.info("Training complete. Saving model.")
    os.mkdir(experiment_location+'/model')
    model.save_pretrained(experiment_location+'/model')

    # TODO: Save config
    pass
    
    post_modeling = PostModelingReport(
        model=model,
        tokenizer=tokenizer_obj.tokenizer,
        tokenizer_config=tokenizer_config,
        eval_dataset=eval_dataset,
        experiment_location=experiment_location,
        device=device
    )

    post_modeling.generate_post_modeling_report(trainer)


    return trainer, model, eval_dataset
    
