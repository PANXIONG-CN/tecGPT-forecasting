import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class TECDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed TEC data (.npz files).
    """
    def __init__(self, file_path: str, p_history: int, s_forecast: int, n_nodes: int, c_in: int):
        """
        Args:
            file_path (str): Path to the .npz file.
            p_history (int): Number of historical time steps.
            s_forecast (int): Number of future time steps to predict.
            n_nodes (int): Number of spatial nodes (N_lat * N_lon).
            c_in (int): Number of input channels/features per node per time step.
        """
        logger.info(f"Loading data from: {file_path}")
        try:
            data = np.load(file_path)
            self.X = data['X']  # Shape: [Num_Samples, P, N_nodes, C_in]
            self.Y = data['Y']  # Shape: [Num_Samples, S, N_nodes, 1]
            logger.info(f"Loaded X with shape: {self.X.shape}, Y with shape: {self.Y.shape}")
        except FileNotFoundError:
            logger.error(f"Error: Data file not found at {file_path}")
            # Handle this case: maybe raise error, or allow empty dataset
            self.X = np.empty((0, p_history, n_nodes, c_in), dtype=np.float32)
            self.Y = np.empty((0, s_forecast, n_nodes, 1), dtype=np.float32)
            # raise # Or return gracefully
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            self.X = np.empty((0, p_history, n_nodes, c_in), dtype=np.float32)
            self.Y = np.empty((0, s_forecast, n_nodes, 1), dtype=np.float32)
            # raise

        # Validate shapes based on config (P, S, N_nodes, C_in)
        if self.X.shape[0] > 0: # If data was loaded
            if self.X.shape[1] != p_history:
                logger.warning(f"X P_history mismatch: expected {p_history}, got {self.X.shape[1]} from {file_path}")
            if self.X.shape[2] != n_nodes:
                logger.warning(f"X N_nodes mismatch: expected {n_nodes}, got {self.X.shape[2]} from {file_path}")
            if self.X.shape[3] != c_in:
                logger.warning(f"X C_in mismatch: expected {c_in}, got {self.X.shape[3]} from {file_path}")
            
            if self.Y.shape[1] != s_forecast:
                logger.warning(f"Y S_forecast mismatch: expected {s_forecast}, got {self.Y.shape[1]} from {file_path}")
            if self.Y.shape[2] != n_nodes:
                logger.warning(f"Y N_nodes mismatch: expected {n_nodes}, got {self.Y.shape[2]} from {file_path}")
            if self.Y.shape[3] != 1:
                logger.warning(f"Y C_out (should be 1) mismatch: got {self.Y.shape[3]} from {file_path}")

            if self.X.shape[0] != self.Y.shape[0]:
                logger.error(f"Number of samples in X and Y do not match in {file_path}!")
                # This is a critical error, might need to raise it.

        self.num_samples = self.X.shape[0]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a single sample (X, Y).
        X: [P, N_nodes, C_in]
        Y: [S, N_nodes, 1] (target)
        """
        x_sample = torch.from_numpy(self.X[idx]).float()
        y_sample = torch.from_numpy(self.Y[idx]).float()
        return x_sample, y_sample

class TECDataModule:
    """
    DataModule for TEC prediction. Handles loading of train, val, test .npz files
    and provides DataLoaders.
    This class is designed to be instantiated by Hydra.
    """
    def __init__(self,
                 processed_data_dir: str,
                 train_file: str = "train.npz",
                 val_file: str = "val.npz",
                 test_file: str = "test.npz",
                 p_history: int = 12,
                 s_forecast: int = 6,
                 n_lat: int = 41,
                 n_lon: int = 71,
                 c_in: int = 12, # TEC_scaled (1) + SW_scaled (5) + TimeFeat (6)
                 batch_size: int = 32,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 shuffle_train: bool = True,
                 shuffle_val: bool = False,
                 shuffle_test: bool = False,
                 **kwargs: Any # To catch other parameters from Hydra config if any
                ):
        super().__init__()
        self.processed_data_dir = processed_data_dir
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        
        self.p_history = p_history
        self.s_forecast = s_forecast
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.n_nodes = n_lat * n_lon
        self.c_in = c_in
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.shuffle_test = shuffle_test

        # Placeholder for datasets
        self.train_dataset: Optional[TECDataset] = None
        self.val_dataset: Optional[TECDataset] = None
        self.test_dataset: Optional[TECDataset] = None
        
        logger.info(f"TECDataModule initialized with P={p_history}, S={s_forecast}, N_nodes={self.n_nodes}, C_in={c_in}")

    def prepare_data(self) -> None:
        """
        Downloads or verifies data. In this case, .npz files are expected to be pre-generated.
        This method is called once per node in distributed training.
        """
        # Check if preprocessed files exist
        for split_name, fname in [("train", self.train_file), ("val", self.val_file), ("test", self.test_file)]:
            fpath = os.path.join(self.processed_data_dir, fname)
            if not os.path.exists(fpath):
                logger.warning(f"Preprocessed data file for {split_name} not found: {fpath}. Ensure preprocess_hdf5.py has been run.")
            else:
                logger.info(f"Found preprocessed data for {split_name}: {fpath}")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Assigns train/val/test datasets. Called on every GPU in DDP.
        Args:
            stage: 'fit', 'validate', 'test', 'predict'. None for all.
        """
        common_args = {
            "p_history": self.p_history,
            "s_forecast": self.s_forecast,
            "n_nodes": self.n_nodes,
            "c_in": self.c_in
        }

        if stage == 'fit' or stage is None:
            train_path = os.path.join(self.processed_data_dir, self.train_file)
            self.train_dataset = TECDataset(file_path=train_path, **common_args)
            if len(self.train_dataset) == 0:
                logger.error(f"Training dataset is empty. Check {train_path}")
                # Decide on behavior: raise error or allow training to proceed (and likely fail)

            val_path = os.path.join(self.processed_data_dir, self.val_file)
            if os.path.exists(val_path):
                self.val_dataset = TECDataset(file_path=val_path, **common_args)
                if len(self.val_dataset) == 0:
                     logger.warning(f"Validation dataset is empty. Check {val_path}")
            else:
                logger.warning(f"Validation file {val_path} not found. Validation dataloader will be empty.")

        if stage == 'validate' and not self.val_dataset: # If called for validate stage specifically
             val_path = os.path.join(self.processed_data_dir, self.val_file)
             if os.path.exists(val_path):
                self.val_dataset = TECDataset(file_path=val_path, **common_args)
             else:
                logger.warning(f"Validation file {val_path} not found for validate stage.")

        if stage == 'test' or stage is None:
            test_path = os.path.join(self.processed_data_dir, self.test_file)
            if os.path.exists(test_path):
                self.test_dataset = TECDataset(file_path=test_path, **common_args)
                if len(self.test_dataset) == 0:
                     logger.warning(f"Test dataset is empty. Check {test_path}")
            else:
                logger.warning(f"Test file {test_path} not found. Test dataloader will be empty.")
        
        if stage == 'predict' and not self.test_dataset: # Predict often uses test data
            test_path = os.path.join(self.processed_data_dir, self.test_file)
            if os.path.exists(test_path):
                self.test_dataset = TECDataset(file_path=test_path, **common_args)
            else:
                logger.warning(f"Test file {test_path} not found for predict stage.")

    def train_dataloader(self) -> Optional[DataLoader]:
        if self.train_dataset and len(self.train_dataset) > 0:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle_train,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=True # Often useful for training
            )
        logger.warning("Train dataset not available or empty, returning None for train_dataloader.")
        return None

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset and len(self.val_dataset) > 0:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle_val,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False
            )
        logger.info("Validation dataset not available or empty, returning None for val_dataloader.")
        return None

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset and len(self.test_dataset) > 0:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle_test,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False
            )
        logger.warning("Test dataset not available or empty, returning None for test_dataloader.")
        return None

    def predict_dataloader(self) -> Optional[DataLoader]:
        # Typically uses the test_dataset or a specific prediction dataset
        if self.test_dataset and len(self.test_dataset) > 0: # Assuming predict uses test data by default
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False, # No shuffle for prediction typically
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False
            )
        logger.warning("Predict dataset (using test_dataset) not available or empty, returning None for predict_dataloader.")
        return None

# Example usage (for testing this script directly):
if __name__ == '__main__':
    # Create dummy .npz files for testing
    P_TEST, S_TEST, N_NODES_TEST, C_IN_TEST = 3, 3, 10, 12 # Example values
    BATCH_SIZE_TEST = 2
    
    processed_dir = "../../processed_data_dummy_dataset" # Adjust path as needed
    os.makedirs(processed_dir, exist_ok=True)

    # Create dummy train.npz
    num_train_samples = 20
    X_train = np.random.rand(num_train_samples, P_TEST, N_NODES_TEST, C_IN_TEST).astype(np.float32)
    Y_train = np.random.rand(num_train_samples, S_TEST, N_NODES_TEST, 1).astype(np.float32)
    np.savez_compressed(os.path.join(processed_dir, "train.npz"), X=X_train, Y=Y_train)
    logger.info(f"Created dummy train.npz in {processed_dir}")

    # Create dummy val.npz
    num_val_samples = 10
    X_val = np.random.rand(num_val_samples, P_TEST, N_NODES_TEST, C_IN_TEST).astype(np.float32)
    Y_val = np.random.rand(num_val_samples, S_TEST, N_NODES_TEST, 1).astype(np.float32)
    np.savez_compressed(os.path.join(processed_dir, "val.npz"), X=X_val, Y=Y_val)
    logger.info(f"Created dummy val.npz in {processed_dir}")

    # Instantiate DataModule
    # Note: n_lat, n_lon are not directly used by TECDataset if n_nodes is given,
    # but TECDataModule calculates n_nodes from them.
    dm = TECDataModule(
        processed_data_dir=processed_dir,
        train_file="train.npz",
        val_file="val.npz",
        test_file="test_nonexistent.npz", # To test missing file scenario
        p_history=P_TEST,
        s_forecast=S_TEST,
        n_lat=1, # Dummy, n_nodes will be N_NODES_TEST
        n_lon=N_NODES_TEST, # So n_nodes = 1 * N_NODES_TEST = N_NODES_TEST
        c_in=C_IN_TEST,
        batch_size=BATCH_SIZE_TEST,
        num_workers=0 # Simpler for direct testing
    )

    dm.prepare_data() # Check for files
    dm.setup(stage='fit')   # Load train and val datasets

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    if train_loader:
        logger.info(f"Train DataLoader created. Number of batches: {len(train_loader)}")
        for i, batch in enumerate(train_loader):
            x_batch, y_batch = batch
            logger.info(f"Train Batch {i}: X shape {x_batch.shape}, Y shape {y_batch.shape}")
            if i == 0: # Check one batch
                # Example: pass a batch to a dummy model input structure
                # This dict matches the expected input for ST_LLM.forward
                model_input_dict = {
                    'tec_history': x_batch[..., 0:1],          # Assuming TEC is the first channel
                    'sw_history': x_batch[..., 1:6],           # Assuming SW are next 5 channels
                    'time_features_history': x_batch[..., 6:12] # Assuming Time are last 6 channels
                }
                logger.info(f"  model_input_dict['tec_history'] shape: {model_input_dict['tec_history'].shape}")
                logger.info(f"  model_input_dict['sw_history'] shape: {model_input_dict['sw_history'].shape}")
                logger.info(f"  model_input_dict['time_features_history'] shape: {model_input_dict['time_features_history'].shape}")
            break # only check first batch
    else:
        logger.error("Train DataLoader is None.")

    if val_loader:
        logger.info(f"Validation DataLoader created. Number of batches: {len(val_loader)}")
        for i, batch in enumerate(val_loader):
            x_batch, y_batch = batch
            logger.info(f"Val Batch {i}: X shape {x_batch.shape}, Y shape {y_batch.shape}")
            break # only check first batch
    else:
        logger.info("Validation DataLoader is None (as expected if val file was empty or not found).")
    
    # Test test_dataloader (expected to be None or empty if test_nonexistent.npz doesn't exist)
    dm.setup(stage='test')
    test_loader = dm.test_dataloader()
    if test_loader:
        logger.info(f"Test DataLoader created. Number of batches: {len(test_loader)}")
    else:
        logger.info("Test DataLoader is None (as expected).")

    logger.info("TECDataModule direct test finished.")
    # Clean up dummy directory
    # import shutil
    # shutil.rmtree(processed_dir)
 