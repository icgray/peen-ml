"""
Module: Displacement Prediction using CNN with Attention Mechanisms

This module contains classes and functions for loading and processing simulation data,
defining neural network models with attention mechanisms, and training and evaluating the models
for displacement prediction.

Features:
1. Loading .npy files from simulation datasets.
2. Custom PyTorch Dataset classes for checkerboard and displacement data.
3. Channel and spatial attention modules for feature enhancement.
4. A CNN model for displacement prediction.
5. Data loader creation, training, and evaluation utilities.

Author:
    Jiachen Zhong
Date:
    Dec 10, 2024
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt


# 1. Load All Numpy Files Function
def load_all_npy_files(base_folder,
                        load_files=("checkerboard", "displacements"),
                          skip_missing=True):
    """
    Load specified .npy files from multiple simulation folders.

    Args:
        base_folder (str): The base folder containing simulation subfolders.
        load_files (tuple): Names of the files to load (default: ("checkerboard", "displacements")).
        skip_missing (bool): If True, skip missing files; otherwise, raise an error.

    Returns:
        dict: A dictionary containing loaded data arrays for the specified files.
              Keys are file names, and values are stacked arrays.
    """
    # Find all folders matching the pattern "Simulation_\d+"
    simulation_folders = [
        folder for folder in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, folder)) and folder.startswith("Simulation_")
    ]

    # Sort folders numerically by the index after "Simulation_"
    simulation_folders.sort(key=lambda x: int(x.split("_")[1]))

    # Initialize dictionaries to store data
    data_dict = {key: [] for key in load_files}

    for simulation_folder in simulation_folders:
        simulation_path = os.path.join(base_folder, simulation_folder)

        for file_name in load_files:
            data_file_path = os.path.join(simulation_path, f"{file_name}.npy")

            if os.path.exists(data_file_path):
                # Load the file and append to the respective list
                data_dict[file_name].append(np.load(data_file_path))
                print(f"{file_name.capitalize()} from {simulation_folder} loaded successfully!")
            else:
                # Handle missing files
                if skip_missing:
                    print(f"{file_name.capitalize()} File not found in {simulation_folder}! Skipping...")
                else:
                    raise FileNotFoundError(f"{file_name.capitalize()} File not found in {simulation_folder}!")

    # Stack data from all simulations along a new axis
    stacked_data = {}
    for key, data_list in data_dict.items():
        if data_list:
            stacked_data[key] = np.stack(data_list)  # Stack along a new axis
        else:
            stacked_data[key] = None  # No data loaded for this key

    print("All specified data loaded and stacked successfully!")
    return stacked_data

# 2. Dataset Classes
class CheckerboardDataset(Dataset):
    """
    A PyTorch Dataset class for checkerboard patterns and displacement data.

    Args:
        checkerboards (numpy array): Array of checkerboard patterns (batch_size, height, width).
        displacements (numpy array): Array of displacements (batch_size, num_nodes, 3).
    """
    def __init__(self, checkerboards, displacements):
        """
        Args:
            checkerboards (numpy array): Array of checkerboard patterns (batch_size, height, width).
            displacements (numpy array): Array of displacements (batch_size, num_nodes, 3).
        """
        self.checkerboards = checkerboards
        self.displacements = displacements

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.checkerboards)

    def __getitem__(self, idx):
        """
        Retrieves a sample by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the checkerboard tensor and the displacement tensor.
        """
        checkerboard = self.checkerboards[idx]
        displacement = self.displacements[idx]

        # Add a channel dimension to checkerboard (1 channel) to match with CNN expectations
        checkerboard = torch.tensor(checkerboard, dtype=torch.float32).unsqueeze(0)  # (1, height, width)
        displacement = torch.tensor(displacement, dtype=torch.float32)  # (num_nodes, 3)

        return checkerboard, displacement

class NormalizedDataset(Dataset):
    """
    A wrapper for normalizing datasets. Takes a base dataset and applies normalization to its features.

    Args:
        base_dataset (Dataset): The original dataset to normalize.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.checkerboards = torch.cat([data[0] for data in base_dataset], dim=0)  # Collect all checkerboards
        self.min_val = self.checkerboards.min()
        self.max_val = self.checkerboards.max()

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        Retrieves a sample by index and normalizes the checkerboard.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the normalized checkerboard tensor and the displacement tensor.
        """
        checkerboard, displacement = self.base_dataset[idx]
        normalized_checkerboard = (checkerboard - self.min_val) / (self.max_val - self.min_val)
        return normalized_checkerboard, displacement

# 3. Attention Modules
class ChannelAttention(nn.Module):
    """
    Channel Attention module for emphasizing relevant feature channels.

    Args:
        channels (int): Number of input channels.
        reduction (int): Reduction ratio for channel compression (default: 16).
    """
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the Channel Attention module.

        Args:
            x (Tensor): Input feature map.

        Returns:
            Tensor: Feature map after channel attention.
        """
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)  # Global average pooling
        max_pool = torch.max(torch.max(x, dim=2, keepdim=True).values, dim=3, keepdim=True).values  # Global max pooling
        scale = self.fc1(avg_pool) + self.fc1(max_pool)
        scale = self.fc2(torch.relu(scale))
        return self.sigmoid(scale) * x

class SpatialAttention(nn.Module):
    """
    Spatial Attention module for emphasizing relevant spatial regions.
    """
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the Spatial Attention module.

        Args:
            x (Tensor): Input feature map.

        Returns:
            Tensor: Feature map after spatial attention.
        """
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # Channel-wise average
        max_pool = torch.max(x, dim=1, keepdim=True).values  # Channel-wise max
        scale = torch.cat([avg_pool, max_pool], dim=1)
        return self.sigmoid(self.conv1(scale)) * x

# 4. CNN Model with Attention
class DisplacementPredictor(nn.Module):
    """
    A CNN model with channel and spatial attention for displacement prediction.

    Args:
        input_channels (int): Number of input channels.
        num_nodes (int): Number of nodes in the displacement data.
        checkerboard_size (int): Spatial size of the checkerboard input (G for a G×G grid).
            All three conv layers use same-padding, so the feature map stays G×G.
            The FC input size is computed as 128 * G * G automatically.
            Defaults to 5 (matches the original Abaqus dataset).
    """
    def __init__(self, input_channels, num_nodes, checkerboard_size=5):
        super(DisplacementPredictor, self).__init__()

        # Store so forward() can reshape correctly and for inspection
        self.num_nodes = num_nodes
        self.checkerboard_size = checkerboard_size

        # Convolutional layers for spatial feature extraction
        # All use padding=1 so spatial size is preserved: output remains G×G
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.ca1 = ChannelAttention(32)
        self.sa1 = SpatialAttention()

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.ca2 = ChannelAttention(64)
        self.sa2 = SpatialAttention()

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.ca3 = ChannelAttention(128)
        self.sa3 = SpatialAttention()

        # Fully connected layers for displacement prediction.
        # After 3 same-padding conv layers the feature map is (batch, 128, G, G),
        # so the flattened size is 128 * G * G — computed dynamically here.
        _fc_in = 128 * checkerboard_size * checkerboard_size
        self.fc = nn.Sequential(
            nn.Linear(_fc_in, 512),
            nn.ReLU(),
            nn.Linear(512, num_nodes * 3)  # Output size = num_nodes * 3 (displacement components)
        )

    def forward(self, x):
        """
        Forward pass of the displacement predictor model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            Tensor: Predicted displacement tensor of shape (batch_size, num_nodes, 3).
        """
        x = self.conv1(x)
        x = self.ca1(x)
        x = self.sa1(x)

        x = self.conv2(x)
        x = self.ca2(x)
        x = self.sa2(x)

        x = self.conv3(x)
        x = self.ca3(x)
        x = self.sa3(x)

        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # Reshape output to (batch_size, num_nodes, 3)
        return x.view(x.size(0), -1, 3)

# 5. Data Loader Creation Function
def create_data_loaders(base_folder, load_files=("checkerboard", "displacements"), skip_missing=True, batch_size=15):
    """
    Create PyTorch DataLoaders for training, validation, and testing.

    Args:
        base_folder (str): Path to the folder containing simulation data.
        num_simulations (int): Number of simulation subfolders to process.
        load_files (tuple): Names of the files to load (default: ("checkerboard", "displacements")).
        skip_missing (bool): Whether to skip missing files or raise an error.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        tuple: DataLoaders for training, validation, and testing, and the loaded data dictionary.
    """
    loaded_data = load_all_npy_files(base_folder, load_files, skip_missing)
    checkerboard = loaded_data["checkerboard"]
    displacements = loaded_data["displacements"]

    # Set Random State for Reproducibility
    torch.manual_seed(2024)
    np.random.seed(2024)

    # Create dataset
    full_dataset = CheckerboardDataset(checkerboard, displacements)

    # Split into train, validation, and test sets
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Wrap subsets with normalization
    train_dataset = NormalizedDataset(train_dataset)
    val_dataset = NormalizedDataset(val_dataset)
    test_dataset = NormalizedDataset(test_dataset)

    # pin_memory speeds up CPU->GPU transfers when a CUDA GPU is present.
    # num_workers=0 avoids Windows multiprocessing issues with CUDA.
    _pin = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=_pin)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=_pin)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=_pin)

    return train_loader, val_loader, test_loader, loaded_data

# 6. Model Creation Function
def create_model(input_channels, num_nodes, checkerboard_size=5):
    """
    Create a DisplacementPredictor model.

    Args:
        input_channels (int): Number of input channels.
        num_nodes (int): Number of nodes in the displacement data.
        checkerboard_size (int): Spatial size G of the G×G checkerboard input.
            The FC layer input is set to 128 * G * G automatically.
            Defaults to 5 (matches the original Abaqus dataset).

    Returns:
        DisplacementPredictor: The instantiated model.
    """
    model = DisplacementPredictor(input_channels, num_nodes, checkerboard_size)
    return model


def infer_dataset_shape(base_folder):
    """Scan the first Simulation_* subfolder to detect num_nodes and checkerboard_size.

    This makes model creation fully dynamic — no hardcoded shapes needed.
    The function sorts simulation folders numerically and reads the first one.

    Args:
        base_folder (str): The parent directory containing Simulation_0/, Simulation_1/, ...

    Returns:
        tuple: (num_nodes (int), checkerboard_size (int))

    Raises:
        FileNotFoundError: If no Simulation_* folders or required .npy files are found.
        ValueError: If the checkerboard is not square or displacements has unexpected shape.

    Example:
        num_nodes, cb_size = infer_dataset_shape("./Dataset_Python")
        model = create_model(input_channels=1,
                             num_nodes=num_nodes,
                             checkerboard_size=cb_size)
    """
    sim_folders = sorted(
        [d for d in os.listdir(base_folder)
         if os.path.isdir(os.path.join(base_folder, d))
         and d.startswith("Simulation_")
         and d[len("Simulation_"):].isdigit()],
        key=lambda x: int(x.split("_")[1])
    )

    if not sim_folders:
        raise FileNotFoundError(
            f"No 'Simulation_<N>' subfolders found in: {base_folder}\n"
            "Expected structure:\n"
            "  <base_folder>/\n"
            "      Simulation_0/\n"
            "          checkerboard.npy\n"
            "          displacements.npy\n"
            "      Simulation_1/\n"
            "          ..."
        )

    # Walk through folders until we find one with both required files
    for sim_name in sim_folders:
        sim_dir = os.path.join(base_folder, sim_name)
        disp_path = os.path.join(sim_dir, "displacements.npy")
        cb_path   = os.path.join(sim_dir, "checkerboard.npy")

        if not os.path.exists(disp_path) or not os.path.exists(cb_path):
            continue  # try next folder

        disp = np.load(disp_path)
        cb   = np.load(cb_path)

        if disp.ndim != 2 or disp.shape[1] != 3:
            raise ValueError(
                f"displacements.npy in {sim_name} has unexpected shape {disp.shape}. "
                "Expected (N_nodes, 3)."
            )
        if cb.ndim != 2:
            raise ValueError(
                f"checkerboard.npy in {sim_name} has unexpected shape {cb.shape}. "
                "Expected a 2-D array."
            )
        if cb.shape[0] != cb.shape[1]:
            raise ValueError(
                f"checkerboard.npy in {sim_name} is not square: {cb.shape}. "
                "Only square G×G checkerboards are supported."
            )

        num_nodes        = disp.shape[0]
        checkerboard_size = cb.shape[0]

        print(f"[infer_dataset_shape] Detected from {sim_name}: "
              f"num_nodes={num_nodes}, checkerboard_size={checkerboard_size}x{checkerboard_size}")
        return num_nodes, checkerboard_size

    raise FileNotFoundError(
        f"Found {len(sim_folders)} Simulation_* folder(s) in {base_folder} "
        "but none contained both 'checkerboard.npy' and 'displacements.npy'."
    )

# 7. Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10, patience=5, device=None, plot_save_path=None):
    """
    Train the model with early stopping.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs to wait for improvement before stopping early.
        device (torch.device | None): Device to run on. Auto-detected if None.

    Returns:
        tuple: Lists of training and validation losses per epoch.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    best_val_loss = float('inf')
    early_stop_counter = 0

    train_losses = []
    val_losses = []

    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    line1, = ax.plot([], [], label='Training Loss', color='blue')
    line2, = ax.plot([], [], label='Validation Loss', color='orange')
    ax.legend()

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0

        for checkerboard, displacement in train_loader:
            checkerboard = checkerboard.to(device)
            displacement = displacement.to(device)
            optimizer.zero_grad()
            predicted_displacements = model(checkerboard)
            loss = criterion(predicted_displacements, displacement)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for checkerboard, displacement in val_loader:
                checkerboard = checkerboard.to(device)
                displacement = displacement.to(device)
                predicted_displacements = model(checkerboard)
                loss = criterion(predicted_displacements, displacement)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        line1.set_xdata(range(1, len(train_losses) + 1))
        line1.set_ydata(train_losses)
        line2.set_xdata(range(1, len(val_losses) + 1))
        line2.set_ydata(val_losses)
        ax.relim()
        ax.autoscale_view()

        # Print Losses
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.10f}, Validation Loss: {val_loss:.10f}")

    if plot_save_path:
        plt.savefig(plot_save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return train_losses, val_losses

# 8. Evaluation Function
def smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE).

    Args:
        y_true (Tensor): Ground truth tensor.
        y_pred (Tensor): Predicted tensor.

    Returns:
        float: sMAPE value.
    """
    numerator = torch.abs(y_true - y_pred)
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2
    smape_value = torch.mean(numerator / denominator)
    return smape_value

def evaluate_model(model, test_loader, criterion, device=None):
    """
    Evaluate the model on the test set.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for test data.
        criterion (nn.Module): Loss function.
        device (torch.device | None): Device to run on. Auto-detected if None.

    Returns:
        float: Overall Mean Squared Error (MSE) on the test set.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    total_mse = 0.0  # Initialize total MSE for all batches
    total_smape = 0.0  # Initialize total sMAPE for all batches

    batch_count = 0


    with torch.no_grad():
        for checkerboard, displacement in test_loader:
            checkerboard = checkerboard.to(device)
            displacement = displacement.to(device)
            # Forward pass to get predictions
            predicted_displacements = model(checkerboard)

            # Calculate batch MSE
            batch_mse = criterion(predicted_displacements, displacement).item()  # Compute MSE loss for the batch
            total_mse += batch_mse

            # Calculate batch sMAPE
            batch_smape = smape(displacement, predicted_displacements).item() # sMAPE
            total_smape += batch_smape

            batch_count += 1

            # Display results for the first batch
            if batch_count == 1:
                print("\nCheckerboard Input:")
                print(checkerboard[0][0].cpu().numpy())  # Show first checkerboard in the batch
                print("\nPredicted Displacement (First 5 Nodes):")
                print(predicted_displacements[0][:5].cpu().numpy())  # Predicted displacement for first 5 nodes
                print("\nGround Truth Displacement (First 5 Nodes):")
                print(displacement[0][:5].cpu().numpy())  # Ground truth displacement for first 5 nodes

    # Calculate and print overall MSE
    overall_mse = total_mse / batch_count
    overall_smape = total_smape / batch_count

    print(f"Overall Mean Squared Error (MSE) on Test Set: {overall_mse:.10f}")
    print(f"Overall Symmetric Mean Absolute Percentage Error (sMAPE) on Test Set: {overall_smape * 100:.10f}%")
    return overall_mse

# 9. Main Function
def main():
    """
    Main function to load data, train the model, and evaluate it.

    Steps:
    1. Load data from the specified folder.
    2. Create the model and initialize training components.
    3. Train the model with early stopping.
    4. Evaluate the model on the test set.
    5. Save the trained model.
    """
    ### Change the path to your local data directory
    data_path1 = r"C:\Users\Lenovo\Desktop\CSE 583 Software Development for Data Scientists\Project\Dataset1_Random_Board\Dataset1_Random_Board"


    # Create DataLoaders
    print("Loading data...")
    train_loader, val_loader, test_loader, _ = create_data_loaders(
        base_folder=data_path1,
        load_files=("checkerboard", "displacements")
    )

    # Auto-detect GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Detect num_nodes and checkerboard_size automatically from the data
    num_nodes, checkerboard_size = infer_dataset_shape(data_path1)

    # Model, Loss, and Optimizer
    input_channels = 1  # Checkerboard has 1 channel
    model = create_model(input_channels, num_nodes, checkerboard_size)
    model = model.to(device)
    print("Model created.")

    criterion = nn.MSELoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)  # Reduce LR every 2 epochs

    # Training
    epochs = 10
    patience = 5  # Number of epochs to wait for improvement before stopping early
    print("Starting training...")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        patience=patience,
        device=device,
    )
    print(
        f"Training completed. The last training loss is: {train_losses[-1]:.10f}, "
        f"and the last validation loss is: {val_losses[-1]:.10f}."
    )
    # Testing and Evaluation
    print("Evaluating model on test set...")
    evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
    )
    print("Evaluation completed.")

if __name__ == "__main__":
    main()


def train_save_gui(data_path):
    """
    Train the displacement-prediction CNN on the dataset at *data_path* and
    save the trained model to disk.

    This is the entry point called by the GUI 'Train' button.  It is a
    streamlined version of ``main()`` that skips the test-set evaluation step
    and writes the final model to a ``saved_model/`` sub-directory inside the
    dataset folder so that the 'Load Model' screen can locate it without any
    extra configuration.

    Args:
        data_path (str): Path to the parent folder that contains the
            ``Simulation_<N>/`` sub-folders produced by
            ``native_dataset_gen.py`` (or Abaqus export scripts).
            The function calls ``infer_dataset_shape`` to detect
            ``num_nodes`` and ``checkerboard_size`` automatically — no
            hard-coded constants are needed.

    Side-effects:
        Saves the trained model to::

            <data_path>/saved_model/trained_displacement_predictor_full_model.pth

    Raises:
        FileNotFoundError: Propagated from ``infer_dataset_shape`` if no valid
            ``Simulation_<N>/`` sub-folders are found.
    """
    # Auto-detect GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)} — training on CUDA.")
    else:
        print("No GPU detected — training on CPU.")

    # Detect num_nodes and checkerboard_size automatically from the data
    print("Inspecting dataset...")
    num_nodes, checkerboard_size = infer_dataset_shape(data_path)
    print(f"  num_nodes={num_nodes}, checkerboard_size={checkerboard_size}x{checkerboard_size}")

    # Create DataLoaders
    print("Loading data...")
    train_loader, val_loader, _, _ = create_data_loaders(
        base_folder=data_path,
        load_files=("checkerboard", "displacements")
    )

    # Model, Loss, and Optimizer
    input_channels = 1  # Checkerboard has 1 channel
    model = create_model(input_channels, num_nodes, checkerboard_size)
    model = model.to(device)
    print("Model created.")

    criterion = nn.MSELoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)  # Reduce LR every 2 epochs

    # Training
    epochs = 10
    patience = 5  # Number of epochs to wait for improvement before stopping early
    # Create save directory before training so the loss curve can be written there
    save_dir = Path(data_path) / "saved_model"
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_save_path = str(save_dir / "training_loss_curve.png")

    print("Starting training...")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        patience=patience,
        device=device,
        plot_save_path=plot_save_path,
    )
    print(
        f"Training completed. The last training loss is: {train_losses[-1]:.10f}, "
        f"and the last validation loss is: {val_losses[-1]:.10f}."
    )
    save_path = save_dir / "trained_displacement_predictor_full_model.pth"

    torch.save(model, save_path)
    print(f"Trained model has been saved to {save_path}.")

    # Save the reference mesh node coordinates so load_and_evaluate_model_gui
    # can spatially interpolate predictions onto evaluation meshes of different size.
    _ref_src = next(
        (p / "node_coords.npy"
         for p in sorted(Path(data_path).glob("Simulation_*"))
         if (p / "node_coords.npy").exists()),
        None,
    )
    if _ref_src is not None:
        import shutil as _shutil
        _shutil.copy2(str(_ref_src), str(save_dir / "reference_node_coords.npy"))
        print("Reference node coordinates saved alongside model for mesh interpolation.")


# ============================================================
# Convolutional Decoder Architecture
# ============================================================

class FieldDataset(Dataset):
    """Dataset that serves (checkerboard, disp_field) pairs.

    Displacements are stored flat (N, 3) in .npy files but the nodes lie on a
    regular H×W grid (X outer-loop, Y inner-loop).  This class reshapes them
    to (3, H, W) so a convolutional decoder can predict the full spatial field.
    """
    def __init__(self, checkerboards, displacements, grid_H, grid_W):
        self.checkerboards = checkerboards
        self.displacements = displacements
        self.grid_H = grid_H
        self.grid_W = grid_W

    def __len__(self):
        return len(self.checkerboards)

    def __getitem__(self, idx):
        cb = torch.tensor(self.checkerboards[idx], dtype=torch.float32).unsqueeze(0)
        disp = self.displacements[idx]  # (N, 3), N = H*W
        # Reshape flat array to spatial field: (H, W, 3) → (3, H, W)
        field = torch.tensor(
            disp.reshape(self.grid_H, self.grid_W, 3).transpose(2, 0, 1),
            dtype=torch.float32,
        )
        return cb, field


def infer_grid_shape(data_path):
    """Return (grid_H, grid_W) by counting unique X and Y values in node_coords.npy."""
    for sim in sorted(os.listdir(data_path), key=lambda s: int(s.split('_')[1])
                      if s.startswith('Simulation_') and s[len('Simulation_'):].isdigit() else 9999):
        nc_path = os.path.join(data_path, sim, 'node_coords.npy')
        if os.path.exists(nc_path):
            nc = np.load(nc_path)
            H = len(np.unique(np.round(nc[:, 0], 8)))
            W = len(np.unique(np.round(nc[:, 1], 8)))
            if H * W == len(nc):
                print(f"[infer_grid_shape] Grid detected: {H}×{W} from {sim}")
                return H, W
    raise ValueError(f"Cannot determine grid shape from node_coords.npy in {data_path}")


def create_field_data_loaders(data_path, batch_size=15):
    """Load data and build DataLoaders serving (checkerboard, disp_field) pairs."""
    loaded = load_all_npy_files(data_path, ('checkerboard', 'displacements'))
    grid_H, grid_W = infer_grid_shape(data_path)

    torch.manual_seed(2024); np.random.seed(2024)
    full_ds = FieldDataset(loaded['checkerboard'], loaded['displacements'], grid_H, grid_W)
    n = len(full_ds)
    tr, va = int(0.7 * n), int(0.15 * n)
    te = n - tr - va
    train_ds, val_ds, test_ds = random_split(full_ds, [tr, va, te])

    _pin = torch.cuda.is_available()
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=_pin),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=_pin),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=_pin),
        grid_H, grid_W,
    )


class ConvDecoderPredictor(nn.Module):
    """CNN encoder + convolutional decoder for displacement field prediction.

    Unlike DisplacementPredictor (Linear(512, N*3) output), this model decodes
    to a fixed (3, out_H, out_W) spatial field.  Any mesh is evaluated by
    bilinear-sampling the field at its (x, y) node coordinates — so node count
    never appears in the model parameters.

    Parameters
    ----------
    input_channels : int
    out_H, out_W   : spatial size of the predicted displacement field
    """
    def __init__(self, input_channels=1, out_H=51, out_W=51):
        super().__init__()
        self.out_H = out_H
        self.out_W = out_W

        # Encoder — identical to DisplacementPredictor (3 conv+attention blocks)
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, 32, 3, padding=1),
                                   nn.BatchNorm2d(32), nn.ReLU())
        self.ca1 = ChannelAttention(32);  self.sa1 = SpatialAttention()

        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1),
                                   nn.BatchNorm2d(64), nn.ReLU())
        self.ca2 = ChannelAttention(64);  self.sa2 = SpatialAttention()

        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),
                                   nn.BatchNorm2d(128), nn.ReLU())
        self.ca3 = ChannelAttention(128); self.sa3 = SpatialAttention()

        # Decoder: upsample to training grid resolution, then refine with convolutions
        self.decoder = nn.Sequential(
            nn.Upsample(size=(out_H, out_W), mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1),  # 3 channels = ux, uy, uz; no activation (displacements can be negative)
        )

    def forward(self, x):
        """Return predicted displacement field (B, 3, out_H, out_W)."""
        x = self.sa1(self.ca1(self.conv1(x)))
        x = self.sa2(self.ca2(self.conv2(x)))
        x = self.sa3(self.ca3(self.conv3(x)))
        return self.decoder(x)


def sample_field_at_coords(field, node_xy):
    """Bilinearly sample a predicted (B, 3, H, W) field at arbitrary node coordinates.

    The field spatial convention: H-axis = X-coordinate, W-axis = Y-coordinate
    (matching the X-outer, Y-inner node ordering produced by the dataset generator).

    Parameters
    ----------
    field    : (B, 3, H, W) tensor on any device
    node_xy  : (N, 2) float32 tensor with columns [x, y] in the same coordinate
               range as the training node_coords (typically [0, 1]).

    Returns
    -------
    (B, N, 3) tensor of sampled displacements.
    """
    import torch.nn.functional as F
    B, _, H, W = field.shape
    N = node_xy.shape[0]

    x_norm = node_xy[:, 0]  # X maps to H dimension
    y_norm = node_xy[:, 1]  # Y maps to W dimension

    # Normalise coordinate range to [-1, 1] from whatever range the coords are in
    x_min, x_max = x_norm.min(), x_norm.max()
    y_min, y_max = y_norm.min(), y_norm.max()
    x_01 = (x_norm - x_min) / (x_max - x_min + 1e-12)
    y_01 = (y_norm - y_min) / (y_max - y_min + 1e-12)

    # F.grid_sample convention: grid[..., 0]=gx → W, grid[..., 1]=gy → H
    gx = (2.0 * y_01 - 1.0).view(1, N, 1, 1).expand(B, N, 1, 1)
    gy = (2.0 * x_01 - 1.0).view(1, N, 1, 1).expand(B, N, 1, 1)
    grid = torch.cat([gx, gy], dim=-1)  # (B, N, 1, 2)

    sampled = F.grid_sample(field, grid, mode='bilinear',
                            align_corners=False, padding_mode='border')
    # sampled: (B, 3, N, 1) → (B, N, 3)
    return sampled.squeeze(-1).permute(0, 2, 1)


def train_save_conv_gui(data_path, epochs=20):
    """Train ConvDecoderPredictor on *data_path* and save to saved_model_conv/.

    Args:
        data_path (str): Parent folder containing Simulation_N/ sub-folders.
        epochs    (int): Training epochs (default 20; conv decoder converges faster).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)} — training on CUDA.")
    else:
        print("No GPU — training on CPU.")

    print("Loading data...")
    train_loader, val_loader, _, grid_H, grid_W = create_field_data_loaders(data_path)
    _, G = infer_dataset_shape(data_path)  # checkerboard size

    model = ConvDecoderPredictor(input_channels=1, out_H=grid_H, out_W=grid_W).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ConvDecoder: grid={grid_H}×{grid_W}  params={n_params:,}  "
          f"({n_params*4/1e6:.2f} MB weights)")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    save_dir = Path(data_path) / "saved_model_conv"
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_path = str(save_dir / "training_loss_curve.png")

    print("Starting training...")
    train_losses, val_losses = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        epochs=epochs, patience=7, device=device, plot_save_path=plot_path,
    )
    print(f"Training done. train={train_losses[-1]:.4e}  val={val_losses[-1]:.4e}")

    save_path = save_dir / "trained_conv_decoder_full_model.pth"
    torch.save(model, save_path)
    print(f"ConvDecoder saved to {save_path}")

    # Save reference node coords (same as train_save_gui) for any post-processing that needs them
    _ref_src = next(
        (p / "node_coords.npy"
         for p in sorted(Path(data_path).glob("Simulation_*"))
         if (p / "node_coords.npy").exists()), None)
    if _ref_src:
        import shutil as _sh
        _sh.copy2(str(_ref_src), str(save_dir / "reference_node_coords.npy"))
        print("Reference node coords saved.")


def load_and_evaluate_conv_gui(model_path, test_data_path, pred_save_dir):
    """Load a ConvDecoderPredictor and run inference on *test_data_path*.

    The model predicts a (3, H, W) displacement field per sample, then bilinearly
    samples it at the node coordinates from test_data_path.  This makes it
    compatible with any mesh resolution — no re-training needed.

    Predictions are saved as:
        <pred_save_dir>/Simulation_<idx>/pred_displacements.npy  (N, 3)
        <pred_save_dir>/Simulation_<idx>/pred_displacements.csv
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    # Load checkerboard(s) from test folder
    if os.path.exists(os.path.join(test_data_path, 'checkerboard.npy')):
        cbs = np.stack([np.load(os.path.join(test_data_path, 'checkerboard.npy'))])
    else:
        loaded = load_all_npy_files(test_data_path, ('checkerboard',), skip_missing=True)
        cbs = loaded['checkerboard']

    # Load node coordinates for sampling
    nc_path = os.path.join(test_data_path, 'node_coords.npy')
    if not os.path.exists(nc_path) and not os.path.exists(os.path.join(test_data_path, 'checkerboard.npy')):
        # Try first simulation sub-folder
        for d in sorted(os.listdir(test_data_path)):
            candidate = os.path.join(test_data_path, d, 'node_coords.npy')
            if os.path.exists(candidate):
                nc_path = candidate
                break
    node_coords = np.load(nc_path).astype(np.float32) if os.path.exists(nc_path) else None

    os.makedirs(pred_save_dir, exist_ok=True)
    criterion = nn.MSELoss()

    for idx, cb in enumerate(cbs):
        cb_t = torch.tensor(cb, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            field = model(cb_t)  # (1, 3, H, W)

        if node_coords is not None:
            nc_t = torch.tensor(node_coords[:, :2], dtype=torch.float32).to(device)
            pred = sample_field_at_coords(field, nc_t)[0].cpu().numpy()  # (N, 3)
        else:
            # No node coords — flatten field to (H*W, 3)
            pred = field[0].permute(1, 2, 0).reshape(-1, 3).cpu().numpy()

        batch_dir = os.path.join(pred_save_dir, f"Simulation_{idx}")
        os.makedirs(batch_dir, exist_ok=True)
        np.save(os.path.join(batch_dir, "pred_displacements.npy"), pred)
        np.savetxt(os.path.join(batch_dir, "pred_displacements.csv"), pred, delimiter=",")

    print(f"ConvDecoder evaluation complete. Predictions saved to {pred_save_dir}")


### Evaluation_GUI part
def create_test_loader(test_data_path, load_files=("checkerboard", "displacements"), batch_size=1):
    """
    Create a DataLoader using the entire dataset from test_data_path.

    Args:
        test_data_path (str): Path to the folder containing the test data.
        load_files (tuple): Names of the files to load (default: ("checkerboard", "displacements")).
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: DataLoader for the entire dataset in test_data_path.
    """
    # Detect whether test_data_path is a single simulation folder (contains
    # checkerboard.npy directly) or a parent folder with Simulation_N/ sub-folders.
    # The GUI's "Step 2 — Select Peen Intensity" asks the user to pick a single
    # Simulation_N/ folder, but load_all_npy_files() expects a *parent* folder.
    # We handle both cases here so either selection works.
    if os.path.exists(os.path.join(test_data_path, "checkerboard.npy")):
        # ---- Single simulation folder ----------------------------------------
        # Load .npy files directly and wrap each in a list so np.stack produces
        # the (1, ...) batch shape that CheckerboardDataset expects.
        print(f"Single simulation folder detected: {os.path.basename(test_data_path)}")
        loaded_data = {}
        for file_name in load_files:
            file_path = os.path.join(test_data_path, f"{file_name}.npy")
            if os.path.exists(file_path):
                loaded_data[file_name] = np.stack([np.load(file_path)])
                print(f"{file_name.capitalize()} loaded successfully!")
            else:
                loaded_data[file_name] = None
                print(f"Warning: {file_name}.npy not found in folder.")
    else:
        # ---- Parent folder with Simulation_N/ sub-folders --------------------
        loaded_data = load_all_npy_files(test_data_path, load_files, skip_missing=True)

    checkerboards = loaded_data.get("checkerboard")
    displacements = loaded_data.get("displacements")

    # Validate that the checkerboard data was found before building the DataLoader.
    if checkerboards is None:
        raise FileNotFoundError(
            f"No checkerboard data found in: {test_data_path}\n"
            "Select either:\n"
            "  - A single Simulation_N/ folder (containing checkerboard.npy), or\n"
            "  - A parent folder that contains Simulation_N/ sub-folders."
        )

    # displacements.npy is required for ground-truth metrics.
    # Raise a clear error rather than silently producing wrong MSE/sMAPE values.
    if displacements is None:
        raise FileNotFoundError(
            f"No displacements.npy found in: {test_data_path}\n"
            "The file is needed for ground-truth comparison during evaluation.\n"
            "If you want pure inference (no metrics), place a zero-filled\n"
            "displacements.npy with shape (num_nodes, 3) in the folder."
        )

    # Create a dataset using the entire loaded data
    full_dataset = CheckerboardDataset(checkerboards, displacements)
    normalized_dataset = NormalizedDataset(full_dataset)

    # Create a DataLoader for the entire dataset.
    # pin_memory speeds up CPU->GPU transfers; num_workers=0 avoids Windows CUDA issues.
    _pin = torch.cuda.is_available()
    test_loader = DataLoader(normalized_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=_pin)

    return test_loader


def _infer_trained_grid_size(model):
    """Return G for which the model's FC layer was built: fc[0].in_features = 128*G*G."""
    try:
        return int(round((model.fc[0].in_features / 128) ** 0.5))
    except (AttributeError, IndexError):
        return None


def _interpolate_displacements(pred_flat, ref_coords, eval_coords):
    """
    Spatially map (N_train, 3) predicted displacements onto (N_eval, 3) target
    mesh nodes via thin-plate-spline RBF in the XY plane.  Appropriate for
    flat-plate meshes where Z is constant (or nearly so).

    Args:
        pred_flat  (np.ndarray): (N_train, 3) model output reshaped.
        ref_coords (np.ndarray): (N_train, 3) training-mesh node coordinates.
        eval_coords(np.ndarray): (N_eval,  3) evaluation-mesh node coordinates.

    Returns:
        np.ndarray: (N_eval, 3) interpolated displacements.
    """
    from scipy.interpolate import RBFInterpolator
    interp = RBFInterpolator(
        ref_coords[:, :2], pred_flat,
        kernel='thin_plate_spline', smoothing=1e-6,
    )
    return interp(eval_coords[:, :2])


def evaluate_model_gui(model, test_loader, criterion, pred_save_dir, device=None,
                       ref_node_coords=None, eval_node_coords=None):
    """
    Evaluate the model on the test set and save predictions.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for test data.
        criterion (nn.Module): Loss function.
        pred_save_dir (str): Directory to save the predicted displacements.
        device (torch.device | None): Device to run on. Auto-detected if None.

    Returns:
        float: Overall Mean Squared Error (MSE) on the test set.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    total_mse    = 0.0
    total_smape  = 0.0
    batch_count  = 0
    metric_count = 0  # batches where MSE was computable

    os.makedirs(pred_save_dir, exist_ok=True)

    trained_G = _infer_trained_grid_size(model)

    with torch.no_grad():
        for batch_idx, (checkerboard, displacement) in enumerate(test_loader):
            # checkerboard : (batch, 1,      G_eval, G_eval)
            # displacement : (batch, N_eval, 3)
            checkerboard = checkerboard.to(device)
            displacement = displacement.to(device)

            # ---- Layer 1: checkerboard resolution interpolation ----
            eval_G = checkerboard.shape[-1]
            if trained_G is not None and eval_G != trained_G:
                print(f"[Interp] Checkerboard {eval_G}x{eval_G} -> {trained_G}x{trained_G}")
                checkerboard = torch.nn.functional.interpolate(
                    checkerboard,
                    size=(trained_G, trained_G),
                    mode='bilinear',
                    align_corners=False,
                )

            # Forward pass — model returns (batch, N_train, 3)
            predicted_displacements = model(checkerboard)

            # ---- Layer 2: output mesh spatial interpolation ----
            N_train = predicted_displacements.shape[1]
            N_eval  = displacement.shape[1]

            if N_train != N_eval:
                if ref_node_coords is not None and eval_node_coords is not None:
                    print(f"[Interp] Displacement nodes {N_train} -> {N_eval} "
                          f"via thin-plate-spline RBF")
                    batch_sz = predicted_displacements.shape[0]
                    pred_np  = predicted_displacements.cpu().numpy()  # (batch, N_train, 3)
                    interped = np.zeros((batch_sz, N_eval, 3), dtype=np.float32)
                    for b in range(batch_sz):
                        interped[b] = _interpolate_displacements(
                            pred_np[b], ref_node_coords, eval_node_coords
                        )
                    predicted_displacements = torch.tensor(
                        interped, dtype=torch.float32, device=device
                    )
                else:
                    print(f"[Warning] Node count mismatch ({N_train} vs {N_eval}). "
                          f"reference_node_coords.npy not found next to model — "
                          f"saving raw model output; MSE skipped for this batch.")

            # Save predictions — always as (N, 3), never with a leading batch dim.
            # data_viz.compute_deformed_mesh loads displacements.npy and indexes it
            # as displacements[node_idx] expecting shape (3,); a (1, N, 3) array would
            # give (N, 3) instead and raise a broadcast error.
            batch_dir = os.path.join(pred_save_dir, f"Simulation_{batch_idx}")
            os.makedirs(batch_dir, exist_ok=True)
            pred_to_save = predicted_displacements.cpu().numpy()   # (B, N, 3)
            pred_2d = pred_to_save[0] if pred_to_save.ndim == 3 else pred_to_save
            np.save(os.path.join(batch_dir, "pred_displacements.npy"), pred_2d)
            np.savetxt(
                os.path.join(batch_dir, "pred_displacements.csv"),
                pred_2d,
                delimiter=",",
            )

            batch_count += 1
            if batch_count == 1:
                print("\nCheckerboard Input:")
                print(checkerboard[0][0].cpu().numpy())
                print("\nPredicted Displacement (First 5 Nodes):")
                print(pred_2d[:5])
                print("\nGround Truth Displacement (First 5 Nodes):")
                print(displacement[0, :5].cpu().numpy())

            # Compute loss only when output and ground-truth shapes match
            if predicted_displacements.shape == displacement.shape:
                total_mse   += criterion(predicted_displacements, displacement).item()
                total_smape += smape(displacement, predicted_displacements).item()
                metric_count += 1

    if metric_count == 0:
        print("Warning: MSE/sMAPE could not be computed (shape mismatch, no node coords).")
        return float('nan')

    overall_mse   = total_mse   / metric_count
    overall_smape = total_smape / metric_count
    print(f"Overall Mean Squared Error (MSE) on Test Set: {overall_mse:.10f}")
    print(f"Overall Symmetric Mean Absolute Percentage Error (sMAPE) on Test Set: {overall_smape * 100:.10f}%")
    return overall_mse


def load_and_evaluate_model_gui(model_path, test_data_path, pred_save_dir):
    """
    Load a previously saved model and run inference on a new peen-intensity
    folder, saving the predicted displacements for later visualisation.

    This is the entry point called by the GUI '1. Evaluate Model' button.
    It loads the entire ``Simulation_<N>/`` folder at *test_data_path* as a
    single-sample DataLoader (``batch_size=1``) and writes one
    ``Simulation_<idx>/pred_displacements.npy`` file per simulation into
    *pred_save_dir*.

    Args:
        model_path (str): Path to the ``.pth`` model file produced by
            ``train_save_gui`` (e.g. ``saved_model/
            trained_displacement_predictor_full_model.pth``).
        test_data_path (str): Path to a folder containing at least one
            ``Simulation_<N>/`` sub-folder with ``checkerboard.npy`` and
            ``displacements.npy``.  Typically this is a single simulation
            folder selected in the GUI.
        pred_save_dir (str): Output directory.  One ``Simulation_<idx>/``
            sub-folder is created per sample; each contains
            ``pred_displacements.npy`` and ``pred_displacements.csv``.

    Side-effects:
        Prints checkerboard input, first-5-node predictions and ground truth
        for the first batch, plus overall MSE and sMAPE to stdout.
    """
    # Auto-detect GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)} — inference on CUDA.")
    else:
        print("No GPU detected — inference on CPU.")

    # Load the model.
    # weights_only=False is required when the .pth was saved with torch.save(model, ...)
    # (i.e. the full model object, not just the state_dict).  PyTorch >= 2.6 changed
    # the default to weights_only=True which rejects pickled custom classes.
    # This file is written by train_save_gui() in this same codebase, so it is trusted.
    model = torch.load(model_path, weights_only=False, map_location=device)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # Use the entire test data as the DataLoader
    test_loader = create_test_loader(test_data_path, batch_size=1)
    print("Test data loaded successfully.")

    # ---- Load node-coordinate arrays for mesh interpolation ----
    # reference_node_coords.npy is saved alongside the model by train_save_gui.
    # node_coords.npy lives in the evaluation simulation folder.
    model_dir = os.path.dirname(os.path.abspath(model_path))
    ref_coords_path  = os.path.join(model_dir, "reference_node_coords.npy")
    eval_coords_path = os.path.join(test_data_path, "node_coords.npy")

    ref_node_coords  = np.load(ref_coords_path)  if os.path.exists(ref_coords_path)  else None
    eval_node_coords = np.load(eval_coords_path) if os.path.exists(eval_coords_path) else None

    if ref_node_coords is None:
        print("[Info] reference_node_coords.npy not found next to model — "
              "output mesh interpolation disabled.")
    if eval_node_coords is None:
        print("[Info] node_coords.npy not found in test folder — "
              "output mesh interpolation disabled.")

    # Define loss function
    criterion = nn.MSELoss()

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model_gui(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        pred_save_dir=pred_save_dir,
        device=device,
        ref_node_coords=ref_node_coords,
        eval_node_coords=eval_node_coords,
    )
    print("Evaluation completed, Predicted Displacements saved.")


# ---------------------------------------------------------------------------
# Curved-surface ML inference (Layer 3)
# ---------------------------------------------------------------------------

def curved_surface_checkerboard(
    stl_surface,
    trajectory,
    G: int,
    h_nozzle: float = 0.15,
    theta_div: float = 0.2618,
    V_mean: float = 50.0,
    sigma_V_frac: float = 0.10,
    n_shots_per_step: int = 10,
    V_exit_min: float = 5.0,
    seed: int = 42,
) -> np.ndarray:
    """Build a (G, G) checkerboard by integrating shot coverage over a nozzle trajectory on an STL surface.

    Projects the cumulative shot energy density (V_n²) over all trajectory
    steps onto a G×G orthographic grid aligned to the STL's XY bounding box.

    Args:
        stl_surface       : STLSurface instance (from stl_surface.py).
        trajectory        : NozzleTrajectory with (T, 3) positions.
        G                 : Checkerboard grid resolution.
        h_nozzle          : Standoff height for Gaussian spread (m).
        theta_div         : Jet cone half-angle (rad).
        V_mean, sigma_V_frac, n_shots_per_step, V_exit_min : shot physics params.
        seed              : RNG seed.

    Returns:
        np.ndarray: (G, G) float32 checkerboard in [0, 1].
    """
    from gaussian_nozzle_dataset_gen import sample_gaussian_nozzle_shots

    bounds   = stl_surface.bounds()
    x_min    = float(bounds[0, 0])
    y_min    = float(bounds[0, 1])
    Lx       = max(float(bounds[1, 0]) - x_min, 1e-9)
    Ly       = max(float(bounds[1, 1]) - y_min, 1e-9)
    sigma_V  = V_mean * sigma_V_frac
    rng      = np.random.default_rng(seed)

    all_xyz: list = []
    all_vn:  list = []

    for pos in trajectory.positions:
        nx, ny, nz = float(pos[0]), float(pos[1]), float(pos[2])
        h_eff      = max(abs(nz), 0.001)

        centres_2d, V_norm, _, _ = sample_gaussian_nozzle_shots(
            h_nozzle=h_eff, theta_div=theta_div,
            V_mean=V_mean, sigma_V=sigma_V,
            n_shots=n_shots_per_step,
            Lx=Lx, Ly=Ly,
            nozzle_x=nx - x_min,
            nozzle_y=ny - y_min,
            V_exit_min=V_exit_min, rng=rng,
        )
        shot_xy = centres_2d.copy()
        shot_xy[:, 0] += x_min
        shot_xy[:, 1] += y_min

        hit_xyz, _, _ = stl_surface.project_shots_onto_surface(shot_xy, z_nozzle=nz)
        all_xyz.append(hit_xyz)
        all_vn.append(V_norm)

    if all_xyz:
        xyz_np = np.concatenate(all_xyz, axis=0)
        vn_np  = np.concatenate(all_vn,  axis=0)
        return stl_surface.shots_to_checkerboard(xyz_np, vn_np ** 2, G)
    return np.zeros((G, G), dtype=np.float32)


def curved_surface_inference(
    model_path: str,
    stl_path: str,
    trajectory_or_checkerboard,
    G: Optional[int] = None,
    pred_save_dir: Optional[str] = None,
    **traj_kwargs,
) -> dict:
    """ML inference on an arbitrary curved 3D surface (Layers 1-3).

    Pipeline:
      1. Build a (G, G) checkerboard from STL + trajectory (or use a
         precomputed checkerboard array directly).
      2. Resize checkerboard to the model's trained grid size if needed (Layer 1).
      3. Run the flat-plate CNN forward pass.
      4. Spatially interpolate predicted displacements onto STL vertices via
         thin-plate-spline RBF (Layer 2, reuses _interpolate_displacements).
      5. Rotate displacements from flat-plate [0,0,1] frame into per-vertex
         surface normals using Rodrigues' formula (Layer 3).

    Args:
        model_path   : Path to the .pth model file from train_save_gui.
        stl_path     : Path to STL file.
        trajectory_or_checkerboard : Either a NozzleTrajectory or a (G, G)
            numpy array (precomputed checkerboard).
        G            : Checkerboard resolution. Required when
            trajectory_or_checkerboard is a NozzleTrajectory.
        pred_save_dir: Where to write prediction .npy files (optional).
        **traj_kwargs: Forwarded to curved_surface_checkerboard() when a
            NozzleTrajectory is supplied. Supported keys: h_nozzle, theta_div,
            V_mean, sigma_V_frac, n_shots_per_step, V_exit_min, seed.

    Returns:
        dict with keys:
            displacements_flat   : (N_train, 3) raw model output
            displacements_on_stl : (V, 3) interpolated + surface-normal-rotated
            vertex_normals       : (V, 3) STL vertex normals
            checkerboard         : (G, G) array used as model input
            stl_surface          : STLSurface instance
    """
    import torch
    from stl_surface import STLSurface

    surface = STLSurface(stl_path)

    # ---- Build or validate checkerboard ----
    if isinstance(trajectory_or_checkerboard, np.ndarray):
        cb = np.asarray(trajectory_or_checkerboard, dtype=np.float32)
        if cb.ndim != 2 or cb.shape[0] != cb.shape[1]:
            raise ValueError(
                f"Precomputed checkerboard must be a square 2D array, got {cb.shape}"
            )
    else:
        if G is None:
            raise ValueError("G must be specified when passing a NozzleTrajectory.")
        _defaults = dict(
            h_nozzle=0.15, theta_div=0.2618, V_mean=50.0,
            sigma_V_frac=0.10, n_shots_per_step=10, V_exit_min=5.0, seed=42,
        )
        _defaults.update(traj_kwargs)
        cb = curved_surface_checkerboard(
            stl_surface=surface,
            trajectory=trajectory_or_checkerboard,
            G=G,
            **_defaults,
        )

    # ---- Load model and run forward pass ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = torch.load(model_path, weights_only=False, map_location=device)
    model.eval()

    trained_G  = _infer_trained_grid_size(model)
    cb_tensor  = torch.tensor(cb[None, None, :, :], dtype=torch.float32, device=device)
    if trained_G is not None and cb_tensor.shape[-1] != trained_G:
        cb_tensor = torch.nn.functional.interpolate(
            cb_tensor, size=(trained_G, trained_G),
            mode="bilinear", align_corners=False,
        )

    with torch.no_grad():
        raw_out = model(cb_tensor)

    # ---- Layer 2: map predictions onto STL vertices ----
    # ConvDecoderPredictor outputs a (1, 3, H, W) field — bilinearly sample it at
    # each STL vertex (x, y) coordinate.  This is exact and requires no reference
    # node file.  DisplacementPredictor outputs (1, N_train, 3) — use the existing
    # thin-plate-spline RBF path with coordinate normalisation.
    if isinstance(model, ConvDecoderPredictor):
        stl_xy = surface.vertices[:, :2].astype(np.float32)
        nc_t   = torch.tensor(stl_xy, dtype=torch.float32, device=device)
        disp_on_stl = sample_field_at_coords(raw_out, nc_t)[0].cpu().numpy()  # (V, 3)
        pred_np     = raw_out[0].permute(1, 2, 0).reshape(-1, 3).cpu().numpy()  # (H*W, 3) for saving
        print(f"[curved_surface_inference] ConvDecoder: bilinear-sampled "
              f"{surface.n_vertices} STL vertices from ({model.out_H}×{model.out_W}) field.")
    else:
        pred_np = raw_out[0].cpu().numpy()        # (N_train, 3)

        model_dir       = os.path.dirname(os.path.abspath(model_path))
        ref_coords_path = os.path.join(model_dir, "reference_node_coords.npy")

        if os.path.exists(ref_coords_path):
            ref_coords = np.load(ref_coords_path)

            # Normalise both coordinate systems to [0,1] before RBF interpolation so
            # that different unit scales (e.g. training plate in metres, STL in mm)
            # don't cause wild extrapolation.
            ref_xy  = ref_coords[:, :2].astype(np.float64)
            stl_xy  = surface.vertices[:, :2].astype(np.float64)

            r_min, r_max = ref_xy.min(axis=0), ref_xy.max(axis=0)
            s_min, s_max = stl_xy.min(axis=0), stl_xy.max(axis=0)
            r_range = np.maximum(r_max - r_min, 1e-12)
            s_range = np.maximum(s_max - s_min, 1e-12)

            ref_norm = np.column_stack([(ref_xy - r_min) / r_range, np.zeros(len(ref_xy))])
            stl_norm = np.column_stack([(stl_xy - s_min) / s_range, np.zeros(len(stl_xy))])

            disp_on_stl = _interpolate_displacements(
                pred_np, ref_norm.astype(np.float32), stl_norm.astype(np.float32)
            )
        else:
            disp_on_stl = pred_np
            print(
                "[curved_surface_inference] reference_node_coords.npy not found "
                "next to model — skipping spatial interpolation onto STL vertices."
            )

    # ---- Layer 3: rotate displacements into local surface normals ----
    if len(disp_on_stl) == surface.n_vertices:
        R_matrices   = surface.vertex_normal_rotation_matrices()   # (V, 3, 3)
        disp_rotated = np.einsum("vij,vj->vi", R_matrices, disp_on_stl).astype(np.float32)
    else:
        disp_rotated = disp_on_stl.astype(np.float32)
        print(
            f"[curved_surface_inference] Vertex count mismatch "
            f"({len(disp_on_stl)} vs {surface.n_vertices}) — "
            "skipping normal-frame rotation."
        )

    # ---- Save ----
    if pred_save_dir is not None:
        os.makedirs(pred_save_dir, exist_ok=True)
        np.save(os.path.join(pred_save_dir, "pred_displacements_flat.npy"),   pred_np)
        np.save(os.path.join(pred_save_dir, "pred_displacements_on_stl.npy"), disp_rotated)
        # Save as both names: checkerboard_used.npy (descriptive) and checkerboard.npy
        # so that visualize_all() / visualize_checkerboard() can find it by the standard name.
        np.save(os.path.join(pred_save_dir, "checkerboard_used.npy"),         cb)
        np.save(os.path.join(pred_save_dir, "checkerboard.npy"),              cb)
        np.save(os.path.join(pred_save_dir, "stl_vertex_normals.npy"),        surface.vertex_normals)
        surface.save_arrays(pred_save_dir)
        print(f"[curved_surface_inference] Predictions saved to: {pred_save_dir}")

    return {
        "displacements_flat":   pred_np,
        "displacements_on_stl": disp_rotated,
        "vertex_normals":       surface.vertex_normals,
        "checkerboard":         cb,
        "stl_surface":          surface,
    }
