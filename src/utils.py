import torch
import matplotlib.pyplot as plt
import numpy as np

def image_convert(img: torch.Tensor):
    img = img.clone().cpu().numpy()
    img = img.transpose(1,2,0)
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img


def plot_10(train_loader: torch.utils.data.DataLoader, classes: list):
        iter_ = iter(train_loader)
        images, labels = next(iter_)
        an_ = {str(i): classes[i] for i in range(len(classes))}

        plt.figure(figsize=(20,10))
        for idx in range(10):
            plt.subplot(2,5,idx+1)
            img = image_convert(images[idx])
            label = labels[idx]
            plt.imshow(img)
            plt.title(an_[str(label.numpy())])
        plt.show()

def plot_val_images(test_loader: torch.utils.data.DataLoader, CNNmodel: torch.nn.Module, device: str, classes: list):

    iter_ = iter(test_loader)
    images, labels = next(iter_)
    images = images.to(device)
    labels = labels.to(device)


    img_out = CNNmodel(images)
    _, index_val = torch.max(img_out, 1)

    fig = plt.figure(figsize=(35,9))
    for idx in np.arange(10):
        ax = fig.add_subplot(2,5,idx+1)
        plt.imshow(image_convert(images[idx]))
        label = labels[idx]
        pred_label = index_val[idx]
        ax.set_title('Act {},pred {}'.format(classes[label.item()], classes[pred_label.item()]))


def plot_train_test_loss(train_losses: list, test_losses: list):
    plt.figure(figsize=(10,5))
    plt.title("Train and Test Loss")
    plt.plot(train_losses,label="train")
    plt.plot(test_losses,label="test")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
