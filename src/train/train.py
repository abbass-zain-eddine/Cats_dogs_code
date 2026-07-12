import os
import time

import matplotlib.pyplot as plt
import torch
import tqdm


def _denormalize_image(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
    return torch.clamp((img * std) + mean, 0, 1)


class Train:
    def __init__(
        self,
        train_loader,
        test_loader,
        CNNmodel,
        criterion,
        optimizer,
        device,
        batch_size,
        save_path,
        writer=None,
        class_names=None,
    ):
        self.save_path = save_path
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.CNNmodel = CNNmodel
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.writer = writer
        self.class_names = list(class_names) if class_names is not None else []
        os.makedirs(self.save_path, exist_ok=True)

    def _prediction_figure(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        preds: torch.Tensor,
        limit: int = 8,
    ):
        count = min(limit, images.size(0))
        cols = 4
        rows = (count + cols - 1) // cols
        fig = plt.figure(figsize=(16, 4 * rows))

        for idx in range(count):
            ax = fig.add_subplot(rows, cols, idx + 1)
            img = (
                _denormalize_image(images[idx])
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            true_idx = int(labels[idx].item())
            pred_idx = int(preds[idx].item())
            true_name = (
                self.class_names[true_idx]
                if true_idx < len(self.class_names)
                else str(true_idx)
            )
            pred_name = (
                self.class_names[pred_idx]
                if pred_idx < len(self.class_names)
                else str(pred_idx)
            )
            ax.imshow(img)
            ax.set_title(f"True: {true_name} | Pred: {pred_name}")
            ax.axis("off")

        fig.tight_layout()
        return fig

    def run(self, epochs=10):
        start_time = time.time()
        train_losses = []
        test_losses = []
        train_correct = []
        test_correct = []

        self.CNNmodel.to(self.device)
        best_val_acc = 0.0

        for epoch in tqdm.tqdm(range(epochs)):
            trn_corr = 0
            tst_corr = 0
            train_loss_sum = 0.0
            self.CNNmodel.train()

            for X_train, y_train in tqdm.tqdm(self.train_loader):
                X_train = X_train.to(self.device)
                y_train = y_train.to(self.device)
                y_pred = self.CNNmodel(X_train)
                loss = self.criterion(y_pred, y_train)
                train_loss_sum += loss.item() * X_train.size(0)

                # True predictions.
                predicted = torch.max(y_pred.data, 1)[1]
                batch_corr = (predicted == y_train).sum()
                trn_corr += batch_corr

                # Update parameters.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_epoch_loss = train_loss_sum / len(self.train_loader.dataset)
            train_epoch_acc = (
                trn_corr.item() * 100 / len(self.train_loader.dataset)
            )
            print(
                f"epoch: {epoch} loss: {train_epoch_loss:.5f} "
                f"accuracy: {train_epoch_acc:7.3f}%"
            )
            train_losses.append(train_epoch_loss)
            train_correct.append(trn_corr)
            with torch.no_grad():
                self.CNNmodel.eval()
                val_loss_sum = 0.0
                vis_batch = None

                for X_test, y_test in self.test_loader:
                    X_test = X_test.to(self.device)
                    y_test = y_test.to(self.device)
                    y_val = self.CNNmodel(X_test)
                    loss = self.criterion(y_val, y_test)
                    val_loss_sum += loss.item() * X_test.size(0)
                    predicted = torch.max(y_val.data, 1)[1]
                    batch_corr = (predicted == y_test).sum()
                    tst_corr += batch_corr

                    if vis_batch is None:
                        vis_batch = (
                            X_test.detach().clone(),
                            y_test.detach().clone(),
                            predicted.detach().clone(),
                        )

                val_acc = tst_corr.item() * 100 / len(self.test_loader.dataset)
                val_loss = val_loss_sum / len(self.test_loader.dataset)
                test_losses.append(val_loss)
                test_correct.append(tst_corr)

                if self.writer is not None:
                    global_step = epoch + 1
                    self.writer.add_scalar(
                        "Loss/train", train_epoch_loss, global_step
                    )
                    self.writer.add_scalar("Loss/val", val_loss, global_step)
                    self.writer.add_scalar(
                        "Accuracy/train", train_epoch_acc, global_step
                    )
                    self.writer.add_scalar(
                        "Accuracy/val", val_acc, global_step
                    )
                    self.writer.add_scalar(
                        "LearningRate",
                        self.optimizer.param_groups[0]["lr"],
                        global_step,
                    )

                    if vis_batch is not None:
                        fig = self._prediction_figure(*vis_batch)
                        self.writer.add_figure(
                            "Validation/Predictions", fig, global_step
                        )
                        plt.close(fig)

                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    torch.save(
                        self.CNNmodel.state_dict(),
                        f"{self.save_path}/best_mode_weight_CNN_dogs_cats.pth",
                    )

        print(f"\nDuration: {time.time() - start_time:.0f} seconds")

        if self.writer is not None:
            self.writer.flush()

        return (
            train_losses,
            test_losses,
            train_correct,
            test_correct,
            self.CNNmodel,
        )
