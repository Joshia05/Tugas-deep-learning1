import os

# Function to create directory structure
def create_project_structure():
    # Define the folder structure
    dirs = [
        'data/train',
        'data/val',
        'data/test',
        'models',
        'models/vit_model.py',
        'models/deit_model.py',
        'utils',
        'utils/train.py',
        'utils/eval.py',
        'utils/metrics.py',
        'results/logs',
        'results/figures',
        'results/model_weights',
        'requirements.txt',
        'README.md',
        'main.py'
    ]

    # Create directories and files
    for dir in dirs:
        if dir.endswith('.py') or dir == 'requirements.txt' or dir == 'README.md' or dir == 'main.py':
            # Create empty Python files or text files
            with open(dir, 'w') as f:
                if dir == 'requirements.txt':
                    f.write("torch\ntransformers\ntimm\npytorch-lightning\nmatplotlib\n")
                elif dir == 'README.md':
                    f.write("# Vision Transformer Comparison\n\nThis project compares different Vision Transformer models.\n")
                elif dir == 'main.py':
                    f.write("# Main script to train and evaluate models\n")
                    f.write("import torch\nfrom models.vit_model import ViTModel\nfrom utils.train import train_model\n\n")
                    f.write("def main():\n")
                    f.write("    # Setup and load data\n")
                    f.write("    # Train and evaluate models\n")
                    f.write("    pass\n\n")
                else:
                    f.write("# Placeholder for model or utility code\n")
        else:
            # Create directories
            os.makedirs(dir, exist_ok=True)
    print("Project structure created successfully.")

# Run the function to create the structure
create_project_structure()
