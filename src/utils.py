import os
import logging

def initiate_new_experiment(model_folder):
    """
    This function checks which is the most up-to-date experiment and initiates a new folder.
    In the new folder, it initiates the relevant folders and saves the config in text.
    """
    num_current_experiments = len(os.listdir(model_folder))
    new_experiment_index = num_current_experiments + 1
    new_experiment_location = os.path.join(model_folder, f'experiment_{new_experiment_index}')
    os.mkdir(new_experiment_location)
    return new_experiment_location, new_experiment_index

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