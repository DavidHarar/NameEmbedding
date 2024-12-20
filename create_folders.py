# %%
# Create folders
import os

# Define the folder structure
folders = [
    "data/raw/text",
    "data/raw/audio",
    "data/processed/text",
    "data/processed/audio",
    "data/processed/embeddings/text",
    "data/processed/embeddings/audio",
    "data/examples",
    "models/roberta/tokenizer",
    "models/roberta/pre-trained",
    "models/roberta/fine-tuned",
    "models/audio/pre-trained",
    "models/audio/fine-tuned",
    "models/checkpoints",
    "src",
    "notebooks",
    "tests",
    "config",
    "logs/training_logs",
    "logs/inference_logs",
    "logs/evaluation_logs",
    "results/visualizations",
    "results/reports",
    "results/predictions"
]

# Create each folder if it doesn't already exist
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create additional files
files = [
    "data/raw/metadata.csv",
    "requirements.txt",
    "Dockerfile",
    ".gitignore",
    "README.md",
    "LICENSE"
]

for file in files:
    # Create the file and add a placeholder if it doesn't exist
    if not os.path.exists(file):
        with open(file, 'w') as f:
            f.write(f"# {file} placeholder\n")

print("Project folder structure created successfully!")
# %%
