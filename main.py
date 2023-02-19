import tempfile
import time
from pathlib import Path

import joblib  # NOQA: F401 (imported but unused)
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torchvision.datasets import MNIST

# from safetensors import safe_open
# from safetensors.torch import save_file


class Net(nn.Module):
    """Baseline Network for MNIST.

    Args:
        nn (_type_): _description_
    """

    def __init__(self):
        """Initialize the model."""
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """Froward pass of the network.

        Args:
            x (torch.Tensor): processed Image

        Returns:
            torch.Tensor: Logits for each class
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def test(model, device, test_loader, epoch):
    """Test the model.

    Args:
        model (torch.nn): Model to test
        device (torch.device): Backend device
        test_loader (torch.utils.data.DataLoader): Test data loader
        epoch (int): Current epoch number
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

    mlflow.log_metrics({"test_loss": test_loss}, step=epoch)
    mlflow.log_metrics({"accuracy": 100.0 * correct / len(test_loader.dataset)}, step=epoch)


def train(model, device, train_loader, optimizer, epoch):
    """_summary_

    Args:
        model (_type_): _description_
        device (_type_): _description_
        train_loader (_type_): _description_
        optimizer (torch.optim): _description_
        epoch (_type_): _description_
    """
    model.train()
    global_step = (epoch - 1) * len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:

            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            mlflow.log_metrics({"loss": loss.item()}, step=global_step)


def main():
    """Main Logic for training the model."""
    # Training settings
    CFG = {
        "batch_size": 64,
        "num_workers": 1,
        "pin_memory": True,
        "test_batch_size": 1000,
        "epochs": 5,
        "lr": 1.0,
        "gamma": 0.7,
        "no_cuda": True,
        "seed": 1,
        "log_interval": 10,
        "save_model": False,
    }
    batch_size = CFG["batch_size"]
    num_workers = CFG["num_workers"]
    pin_memory = CFG["pin_memory"]
    test_batch_size = CFG["test_batch_size"]
    epochs = CFG["epochs"]
    lr = CFG["lr"]
    gamma = CFG["gamma"]
    no_cuda = CFG["no_cuda"]
    seed = CFG["seed"]
    log_interval = CFG["log_interval"]  # NOQA: F841
    save_model = CFG["save_model"]  # NOQA: F841

    use_cuda = not no_cuda and torch.cuda.is_available()

    # Check that MPS is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )

        # choose CPU or GPU based on availability
        device = torch.device("cuda" if use_cuda else "cpu")

    else:
        print("MPS is available! (but not necessarily enabled.")
        device = torch.device("mps")

    torch.manual_seed(seed)

    kwargs = {"num_workers": num_workers, "pin_memory": pin_memory} if use_cuda else {}

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = MNIST("../data", train=True, download=False, transform=transform)
    dataset2 = MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        dataset1, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset2, batch_size=test_batch_size, shuffle=True, **kwargs
    )

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch)
        scheduler.step()

    # if (save_model):
    #     torch.save(model.state_dict(),Path(tmpdir, "model.pkl"))
    with tempfile.TemporaryDirectory() as tmpdir:
        # move to safetensors
        torch.save(model.state_dict(), Path(tmpdir, "model.pkl"))
        # joblib.dump(model, Path(tmpdir, "model.pkl"))
        # Log model
        mlflow.log_artifacts(tmpdir)
    # Log parameters
    mlflow.log_params(CFG)


if __name__ == "__main__":

    import platform

    print("Python version: ", platform.platform())
    print("PyTorch version: ", torch.__version__)
    print("MLflow version: ", mlflow.__version__)

    # Set tracking URI
    MODEL_REGISTRY = Path("experiments")
    Path(MODEL_REGISTRY).mkdir(exist_ok=True)  # create experiments dir
    mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))

    # Set experiment name
    mlflow.set_experiment(experiment_name="baselines")

    tick = time.time()
    main()
    tock = time.time()
    print("Time taken: ", tock - tick)
