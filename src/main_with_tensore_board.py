from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models.cnn import ConvolutionalNetwork
from train.train import Train
from utils import (
    plot_10,
    plot_train_test_loss,
    plot_val_images,
)


def denormalize_image(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
    return torch.clamp((img * std) + mean, 0, 1)


def make_labeled_batch_figure(
    images: torch.Tensor,
    labels: torch.Tensor,
    classes: list,
    limit: int = 12,
):
    count = min(limit, images.size(0))
    cols = 4
    rows = (count + cols - 1) // cols
    fig = plt.figure(figsize=(14, 3.5 * rows))

    for idx in range(count):
        ax = fig.add_subplot(rows, cols, idx + 1)
        img = (
            denormalize_image(images[idx])
            .detach()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )
        label_idx = int(labels[idx].item())
        class_name = (
            classes[label_idx] if label_idx < len(classes) else str(label_idx)
        )
        ax.imshow(img)
        ax.set_title(f"Label: {class_name}")
        ax.axis("off")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )

    # Path to train and test datasets stored outside src.
    path_to_train_data = "dataset/training_set"
    path_to_test_data = "dataset/test_set"
    batch_size = 32

    train_dataset = torchvision.datasets.ImageFolder(
        root=path_to_train_data,
        transform=train_transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_set = torchvision.datasets.ImageFolder(
        root=path_to_test_data,
        transform=test_transform,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
    )

    plot_10(train_loader, classes=train_dataset.classes)

    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"artifacts/tensorboard/{run_stamp}"
    writer = SummaryWriter(log_dir=log_dir)

    sample_images, sample_labels = next(iter(train_loader))
    preview_fig = make_labeled_batch_figure(
        sample_images,
        sample_labels,
        train_dataset.classes,
    )
    writer.add_figure(
        "Dataset/TrainSamplesWithLabels",
        preview_fig,
        global_step=0,
    )
    plt.close(preview_fig)

    classes_text = "\n".join(
        [f"{idx}: {name}" for idx, name in enumerate(train_dataset.classes)]
    )
    writer.add_text("Dataset/Classes", classes_text, 0)

    model = ConvolutionalNetwork()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    writer.add_graph(model, sample_images.to(device))

    trainer = Train(
        train_loader,
        test_loader,
        model,
        criterion,
        optimizer,
        device=device,
        batch_size=batch_size,
        save_path="artifacts",
        writer=writer,
        class_names=train_dataset.classes,
    )
    (
        train_losses,
        test_losses,
        train_correct,
        test_correct,
        CNNmodel,
    ) = trainer.run(epochs=10)

    plot_val_images(
        test_loader,
        CNNmodel,
        device=device,
        classes=test_set.classes,
    )
    plot_train_test_loss(train_losses, test_losses)

    writer.close()
