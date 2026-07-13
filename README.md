# Cats vs Dogs Classification (PyTorch Tutorial)

This repository is a beginner-friendly computer vision project for classifying images of cats and dogs using a Convolutional Neural Network (CNN) in PyTorch.

It is designed as a tutorial for students: you can run a baseline model first, then move to TensorBoard tracking and transfer learning experiments.

## What You Will Learn

- How to structure an image dataset for `torchvision.datasets.ImageFolder`
- How to build and train a CNN in PyTorch
- How to evaluate model predictions on validation data
- How to log metrics and images with TensorBoard

## Project Structure

```text
Cats_dogs_code/
|-- dataset/
|   |-- training_set/
|   |   |-- cats/
|   |   `-- dogs/
|   `-- test_set/
|       |-- cats/
|       `-- dogs/
|-- artifacts/                    # Saved model and TensorBoard logs
|-- research/
|   `-- CNN_cats_vs_dogs.ipynb
`-- src/
	|-- main.py                   # Baseline training script
	|-- main_with_tensore_board.py# Training with TensorBoard logging
	|-- models/
	|   |-- cnn.py
	|   `-- transfer_learning.py
	|-- train/
	|   `-- train.py
	`-- utils.py
```

## Prerequisites

- Python 3.10+ (recommended)
- `pip`

Install dependencies from the project root:

```bash
pip install torch torchvision matplotlib numpy tqdm tensorboard jupyter
```

If you have a CUDA-capable GPU and the default `torch` install is CPU-only, install the correct PyTorch build from the official PyTorch website.

## Dataset Setup (Important)

The scripts expect the following exact folders:

- `dataset/training_set/cats`
- `dataset/training_set/dogs`
- `dataset/test_set/cats`
- `dataset/test_set/dogs`

Notes:

- Your current repository includes `dataset/training_set/` but training code also requires `dataset/test_set/`.
- Keep class folder names consistent (`cats`, `dogs`).
- Place images directly inside class folders.

## Quick Start

Run commands from the repository root.

### 1) Train Baseline CNN

```bash
python src/main.py
```

What this script does:

- Applies image transforms and normalization
- Loads data with `ImageFolder`
- Trains `ConvolutionalNetwork` for 20 epochs
- Saves best model weights to:

```text
artifacts/best_mode_weight_CNN_dogs_cats.pth
```

### 2) Train With TensorBoard

```bash
python src/main_with_tensore_board.py
```

Then open TensorBoard:

```bash
tensorboard --logdir artifacts/tensorboard
```

This version logs:

- Train/validation loss
- Train/validation accuracy
- Learning rate
- Sample predictions and dataset preview images

## Notebook Option

For guided exploration, open:

```text
research/CNN_cats_vs_dogs.ipynb
```

Use the notebook if you want to inspect steps interactively, visualize batches, and experiment with small changes.

## How the Model Works

The baseline CNN in `src/models/cnn.py` has:

- 2 convolution layers + max pooling
- Flattening
- 3 fully connected layers
- 2 output classes (cat, dog)

Training loop (`src/train/train.py`) computes:

- Cross entropy loss
- Per-epoch train and validation accuracy
- Best checkpoint based on validation accuracy

## Common Issues

1. `FileNotFoundError` for dataset paths

- Ensure `dataset/test_set/` exists and has `cats/` and `dogs/` subfolders.

2. Import errors when running scripts

- Run from project root: `python src/main.py` (not from inside `src/`).

3. TensorBoard command not found

- Install: `pip install tensorboard`
- Then run again: `tensorboard --logdir artifacts/tensorboard`

## Suggested Student Exercises

1. Change epochs, batch size, and learning rate. Compare results.
2. Add data augmentation (for example `ColorJitter`) and measure impact.
3. Try transfer learning using `src/models/transfer_learning.py`.
4. Add early stopping based on validation loss.
5. Create a confusion matrix after evaluation.

## Next Step

If you want, I can also generate a `requirements.txt` file and a small script to split one raw dataset into train/test folders automatically.
