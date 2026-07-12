import torch
import torchvision
from train.train import Train
from torchvision import transforms  
from models.cnn import ConvolutionalNetwork
from utils import plot_10, plot_val_images, plot_train_test_loss


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize((224, 224)),             # resize shortest side to 224 pixels
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    #path to the train and test data pathes in a folder outside the src:
    path_to_train_data = 'dataset/training_set'
    path_to_test_data = 'dataset/test_set'
    batch_size = 32
    train_dataset = torchvision.datasets.ImageFolder(root=path_to_train_data, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_set = torchvision.datasets.ImageFolder(root=path_to_test_data, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)    
    plot_10(train_loader, classes=train_dataset.classes)
    # Initialize model, criterion, and optimizer
    model = ConvolutionalNetwork()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initialize training class
    trainer = Train(train_loader, test_loader, model, criterion, optimizer, device=device, batch_size=batch_size, save_path='artifacts')
    train_losses, test_losses, _, _, CNNmodel  = trainer.run(epochs=20)
    plot_val_images(test_loader, model, device=device, classes=test_set.classes)

    plot_train_test_loss(train_losses, test_losses)
