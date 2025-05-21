import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR # Add others as needed
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import wandb
from omegaconf import DictConfig, OmegaConf
import logging
from typing import Dict, Any, Optional, Tuple, List

from src.models.tec_gpt import ST_LLM # Main model
from src.utils.scaler import StandardScaler as TecStandardScaler # Renamed to avoid confusion with sklearn
from src.utils.metrics import calculate_mae, calculate_rmse, calculate_wmape, calculate_r_squared
from src.utils.helpers import get_device, seed_everything

logger = logging.getLogger(__name__)

class TecTrainer:
    """
    封装了 TecGPT 模型的训练和评估逻辑。
    设计为可由 Hydra 驱动。
    """
    def __init__(self, cfg: DictConfig, model: ST_LLM, 
                 train_loader: Optional[DataLoader] = None, 
                 val_loader: Optional[DataLoader] = None, 
                 test_loader: Optional[DataLoader] = None, 
                 scaler: Optional[TecStandardScaler] = None):
        """
        Args:
            cfg (DictConfig): Hydra 配置对象 (通常是全局配置).
            model (ST_LLM): 要训练的 ST_LLM (TecGPT) 模型实例.
            train_loader (Optional[DataLoader]): 训练数据加载器.
            val_loader (Optional[DataLoader]): 验证数据加载器.
            test_loader (Optional[DataLoader]): 测试数据加载器.
            scaler (Optional[TecStandardScaler]): 用于逆转换输出以计算原始尺度指标的缩放器.
        """
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler # Instance of TecStandardScaler from utils.scaler

        # Trainer specific config
        trainer_cfg = cfg.trainer
        self.epochs = trainer_cfg.epochs
        self.learning_rate = trainer_cfg.learning_rate
        self.weight_decay = trainer_cfg.weight_decay
        self.gradient_clip_val = trainer_cfg.get('gradient_clip_val', None)
        self.log_every_n_steps = trainer_cfg.get('log_every_n_steps', 50)
        
        # Device setup
        self.device_name = cfg.get('device', 'auto') # Global device config
        self.device = get_device(self.device_name)
        self.model.to(self.device)

        # Optimizer
        self.optimizer = self._configure_optimizer(trainer_cfg)

        # Scheduler (optional)
        self.scheduler = self._configure_scheduler(trainer_cfg, self.optimizer)

        # Criterion (Loss function)
        self.criterion = self._configure_criterion(trainer_cfg)
        
        # Early stopping (optional)
        self.early_stopping_cfg = trainer_cfg.get('early_stopping', None)
        if self.early_stopping_cfg and self.early_stopping_cfg.get('monitor', None):
            self.early_stopping_patience = self.early_stopping_cfg.patience
            self.early_stopping_min_delta = self.early_stopping_cfg.min_delta
            self.early_stopping_counter = 0
            self.best_val_metric = float('inf') if self.early_stopping_cfg.mode == "min" else float('-inf')
            logger.info(f"Early stopping enabled: monitor '{self.early_stopping_cfg.monitor}', mode '{self.early_stopping_cfg.mode}', patience {self.early_stopping_patience}")
        else:
            self.early_stopping_cfg = None # Disable if not properly configured
            
        # Checkpointing config (managed by this trainer)
        self.checkpoint_cfg = trainer_cfg.get('checkpoint', {})
        if not self.checkpoint_cfg: # Fallback if PL style checkpoint config is used in trainer.yaml
            self.checkpoint_cfg = {
                'dirpath': trainer_cfg.get('checkpoint_dir', 'checkpoints'),
                'monitor': trainer_cfg.get('checkpoint_monitor', 'val/loss'),
                'mode': trainer_cfg.get('checkpoint_mode', 'min'),
                'save_top_k': trainer_cfg.get('checkpoint_save_top_k', 1),
                'save_last': trainer_cfg.get('checkpoint_save_last', True),
                'filename_prefix': trainer_cfg.get('checkpoint_filename_prefix', 'tecllm')
            }
        self.best_checkpoint_metric = float('inf') if self.checkpoint_cfg.get('mode', 'min') == "min" else float('-inf')
        self.top_k_checkpoints = [] # Stores (metric_val, path)

        # Wandb setup (optional, from global config)
        self.wandb_cfg = cfg.get('wandb', None)
        if self.wandb_cfg and self.wandb_cfg.get('project', None):
            # wandb.init will be called in the main train.py script
            logger.info("Wandb logging is configured.")
        else:
            self.wandb_cfg = None
            logger.info("Wandb logging is not configured.")

    def _configure_optimizer(self, trainer_cfg: DictConfig) -> optim.Optimizer:
        opt_name = trainer_cfg.optimizer_name.lower()
        opt_params = trainer_cfg.get('optimizer_params', {})
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())

        if opt_name == "adamw":
            optimizer = optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay, **opt_params)
        elif opt_name == "adam":
            optimizer = optim.Adam(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay, **opt_params)
        # TODO: Add more optimizers like RMSprop, SGD etc.
        else:
            raise ValueError(f"Unsupported optimizer: {trainer_cfg.optimizer_name}")
        logger.info(f"Optimizer: {trainer_cfg.optimizer_name} configured.")
        return optimizer

    def _configure_scheduler(self, trainer_cfg: DictConfig, optimizer: optim.Optimizer) -> Optional[optim.lr_scheduler._LRScheduler]:
        sched_name = trainer_cfg.get('lr_scheduler_name', None)
        if not sched_name:
            return None
        
        sched_params = trainer_cfg.get('lr_scheduler_params', {})
        sched_name = sched_name.lower()

        if sched_name == "reducelronplateau":
            # Ensure monitor key is compatible, e.g. val_loss, not val/loss for PL
            # Our custom loop will use keys like 'val_loss' or 'val_mae'
            scheduler = ReduceLROnPlateau(optimizer, mode=sched_params.get('mode','min'), factor=sched_params.get('factor',0.1), patience=sched_params.get('patience',10))
        elif sched_name == "cosineannealinglr":
            scheduler = CosineAnnealingLR(optimizer, T_max=sched_params.get('T_max', self.epochs), eta_min=sched_params.get('eta_min', 0))
        elif sched_name == "steplr":
            scheduler = StepLR(optimizer, step_size=sched_params.get('step_size', 30), gamma=sched_params.get('gamma', 0.1))
        else:
            raise ValueError(f"Unsupported scheduler: {sched_name}")
        logger.info(f"Scheduler: {sched_name} configured.")
        return scheduler

    def _configure_criterion(self, trainer_cfg: DictConfig) -> nn.Module:
        crit_name = trainer_cfg.criterion_name.lower()
        # crit_params = trainer_cfg.get('criterion_params', {})
        if crit_name == "mseloss":
            criterion = nn.MSELoss() # reduction='mean' by default
        elif crit_name == "l1loss" or crit_name == "maeloss":
            criterion = nn.L1Loss()
        # TODO: Add more loss functions if needed
        else:
            raise ValueError(f"Unsupported criterion: {trainer_cfg.criterion_name}")
        logger.info(f"Criterion: {trainer_cfg.criterion_name} configured.")
        return criterion

    def _prepare_batch(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts batch from DataLoader (X, Y_raw_tec) to model input dictionary.
        Assumes X from DataLoader is [B, P, N_nodes, C_in=12]
        Y_raw_tec is [B, S, N_nodes, 1]
        """
        x_data, y_target_raw = batch
        # x_data: [B, P, N_nodes, C_in], where C_in: TEC(1), SW(5), Time(6)
        # y_target_raw: [B, S, N_nodes, 1] (raw TEC values for loss calculation if needed, or just for metrics)

        # The model.forward expects a dict with specific keys.
        # These keys match the ST_LLM.forward method's expectations.
        # We need to slice C_in from x_data.
        # Assuming order in C_in: tec_history (1), sw_history (5), time_features_history (6)
        model_input_dict = {
            'tec_history': x_data[..., 0:1].to(self.device),          
            'sw_history': x_data[..., 1:6].to(self.device),           
            'time_features_history': x_data[..., 6:12].to(self.device)
            # 'llm_attention_mask' can be added here if used
        }
        # Target for loss calculation (model output is scaled, so loss is on scaled values unless inverse transformed first)
        # Y from dataloader is RAW. Model outputs SCALED.
        # So, for loss, we need SCALED Y. 
        # This implies that Y in the .npz file should be the scaled version if loss is computed directly.
        # OR, the dataloader needs to provide scaled Y, OR the trainer transforms Y_raw to Y_scaled.
        # Let's assume for now the target for the loss is the SCALED version of Y.
        # The current preprocess_hdf5.py saves Y as RAW values.
        # This needs to be reconciled. For now, let's assume y_target_raw is what we need and scaler will be used later for metrics.
        # For loss calculation, if model predicts scaled values, target must also be scaled.
        # Let's make the trainer responsible for scaling the target if scaler is available.
        
        y_target_for_loss = y_target_raw.squeeze(-1) # [B, S, N_nodes]
        if self.scaler:
            # This scales the raw target Y to the scaled domain for loss calculation against model's scaled output
            # The scaler needs to handle [B, S, N_nodes] -> permute to [B, N_nodes, S] then scale then permute back
            # Or, if NodeScaler.transform can handle [B, S, N_nodes] if N_nodes is last dim.
            # Let's reshape to [B*S, N_nodes] for tec_scaler.transform, then reshape back.
            B, S, N = y_target_raw.shape[0], y_target_raw.shape[1], y_target_raw.shape[2]
            y_target_reshaped = y_target_raw.reshape(B * S, N)
            y_scaled_reshaped = self.scaler.transform_tec(y_target_reshaped) # Scaler.transform_tec expects [..., N_nodes]
            y_target_for_loss = y_scaled_reshaped.reshape(B, S, N) # [B, S, N_nodes]
        
        model_input_dict['target_scaled'] = y_target_for_loss.to(self.device)
        model_input_dict['target_raw'] = y_target_raw.squeeze(-1).to(self.device) # For metrics calculation after inverse transform
        return model_input_dict

    def train_epoch(self, epoch_num: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        # Store all predictions and targets for epoch-level metrics if needed
        # all_preds_scaled = [] 
        # all_targets_raw = []

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch_num+1}/{self.epochs} [Train]", leave=False)
        for batch_idx, batch_data in enumerate(progress_bar):
            # batch_data from dataloader is (X, Y_raw_tec)
            # X: [B,P,N,C_in], Y_raw_tec: [B,S,N,1]
            
            model_inputs = self._prepare_batch(batch_data)
            # model_inputs = {k: v.to(self.device) for k,v in model_inputs.items()}
            
            self.optimizer.zero_grad()
            
            # Model forward pass
            # predictions_scaled shape: [B, N_nodes, S_forecast]
            predictions_scaled = self.model(model_inputs) # ST_LLM.forward expects the dict
            
            # Target for loss is y_target_scaled from _prepare_batch, shape [B, S, N_nodes]
            # Predictions are [B, N_nodes, S]. Need to match shapes for loss.
            target_scaled_for_loss = model_inputs['target_scaled'].permute(0, 2, 1) # -> [B, N_nodes, S]
            
            loss = self.criterion(predictions_scaled, target_scaled_for_loss)
            
            loss.backward()
            if self.gradient_clip_val:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if self.wandb_cfg and (batch_idx % self.log_every_n_steps == 0):
                wandb.log({
                    f"train/step_loss": loss.item(), 
                    "train/lr": self.optimizer.param_groups[0]['lr'],
                    "epoch": epoch_num,
                    "batch": batch_idx
                })
            progress_bar.set_postfix(loss=loss.item(), lr=self.optimizer.param_groups[0]['lr'])

            # all_preds_scaled.append(predictions_scaled.detach().cpu())
            # all_targets_raw.append(model_inputs['target_raw'].detach().cpu())

        avg_loss = total_loss / len(self.train_loader)
        epoch_metrics = {"train/loss": avg_loss, "train/lr_final": self.optimizer.param_groups[0]['lr']}
        
        # TODO: Calculate epoch-level metrics (MAE, RMSE etc. on raw scale) if needed for training logs
        # This would involve inverse transforming all_preds_scaled and comparing with all_targets_raw

        logger.info(f"Epoch {epoch_num+1} Train: Avg Loss: {avg_loss:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        if self.wandb_cfg: wandb.log(epoch_metrics, step=epoch_num) # Log epoch metrics
        return {"loss": avg_loss}

    def eval_epoch(self, epoch_num: int, dataloader: DataLoader, split_name: str = "val") -> Dict[str, float]:
        self.model.eval()
        total_loss_scaled = 0 # Loss on scaled values
        all_preds_raw_list = []
        all_targets_raw_list = []

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num+1} [{split_name.capitalize()}]", leave=False)
        with torch.no_grad():
            for batch_data in progress_bar:
                model_inputs = self._prepare_batch(batch_data)
                # model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

                predictions_scaled = self.model(model_inputs) # [B, N_nodes, S]
                target_scaled_for_loss = model_inputs['target_scaled'].permute(0, 2, 1) # -> [B, N_nodes, S]
                
                loss_scaled = self.criterion(predictions_scaled, target_scaled_for_loss)
                total_loss_scaled += loss_scaled.item()
                
                # Inverse transform predictions to raw scale for metrics
                # predictions_scaled: [B, N_nodes, S]
                if self.scaler:
                    preds_raw = self.scaler.inverse_transform_tec(predictions_scaled.cpu().numpy())
                    preds_raw = torch.from_numpy(preds_raw).to(self.device)
                else:
                    preds_raw = predictions_scaled # Cannot convert to raw if no scaler
                
                all_preds_raw_list.append(preds_raw) # [B, N_nodes, S]
                all_targets_raw_list.append(model_inputs['target_raw']) # [B, S, N_nodes]

        avg_loss_scaled = total_loss_scaled / len(dataloader)
        metrics = {f"{split_name}/loss_scaled": avg_loss_scaled}

        # Concatenate all batch results for metric calculation
        # Ensure targets are permuted to match preds: [B, N_nodes, S]
        all_preds_raw_cat = torch.cat(all_preds_raw_list, dim=0) # [Total_Samples, N_nodes, S]
        all_targets_raw_cat = torch.cat(all_targets_raw_list, dim=0).permute(0, 2, 1) # -> [Total_Samples, N_nodes, S]
        
        # Calculate metrics on raw scale
        # TODO: Add mask handling if there are fill values in raw target/preds that should be ignored
        # For now, assuming all values are valid after inverse transform.
        # If scaler correctly handles fill values (e.g. -1 becomes -1), then mask based on that.
        valid_mask = None
        if self.scaler and self.scaler.fill_value_tec is not None:
            # Create mask based on original target values having the fill_value_tec
            # This assumes target_raw still has fill values if they existed.
            valid_mask = (all_targets_raw_cat != self.scaler.fill_value_tec)
        
        # Flatten N_nodes and S dimensions for metric functions if they expect 2D [Samples, Features]
        # Or calculate per step / per node if desired. For now, overall metrics.
        # y_true_flat = all_targets_raw_cat.reshape(-1)
        # y_pred_flat = all_preds_raw_cat.reshape(-1)
        # mask_flat = valid_mask.reshape(-1) if valid_mask is not None else None

        # Calculate metrics on the full concatenated tensors
        metrics[f"{split_name}/mae"] = calculate_mae(all_targets_raw_cat, all_preds_raw_cat, mask=valid_mask).item()
        metrics[f"{split_name}/rmse"] = calculate_rmse(all_targets_raw_cat, all_preds_raw_cat, mask=valid_mask).item()
        metrics[f"{split_name}/wmape"] = calculate_wmape(all_targets_raw_cat, all_preds_raw_cat, mask=valid_mask).item()
        metrics[f"{split_name}/r2"] = calculate_r_squared(all_targets_raw_cat, all_preds_raw_cat, mask=valid_mask).item()
        
        # Per-step metrics (optional)
        for step_idx in range(self.cfg.model.s_forecast):
            pred_step = all_preds_raw_cat[:, :, step_idx]
            true_step = all_targets_raw_cat[:, :, step_idx]
            mask_step = valid_mask[:, :, step_idx] if valid_mask is not None else None
            
            metrics[f"{split_name}/mae_step{step_idx+1}"] = calculate_mae(true_step, pred_step, mask=mask_step).item()
            metrics[f"{split_name}/rmse_step{step_idx+1}"] = calculate_rmse(true_step, pred_step, mask=mask_step).item()

        log_str = f"Epoch {epoch_num+1} {split_name.capitalize()}: Loss Scaled: {avg_loss_scaled:.4f}, MAE: {metrics[f'{split_name}/mae']:.4f}, RMSE: {metrics[f'{split_name}/rmse']:.4f}"
        logger.info(log_str)
        if self.wandb_cfg: wandb.log(metrics, step=epoch_num)
        return metrics

    def _check_early_stopping(self, current_metric_val: float, epoch_num: int) -> bool:
        """ Returns True if training should stop. """
        if not self.early_stopping_cfg: return False

        mode = self.early_stopping_cfg.mode
        improved = False
        if mode == "min":
            if current_metric_val < self.best_val_metric - self.early_stopping_min_delta:
                improved = True
        elif mode == "max":
            if current_metric_val > self.best_val_metric + self.early_stopping_min_delta:
                improved = True
        
        if improved:
            logger.info(f"EarlyStopping: {self.early_stopping_cfg.monitor} improved from {self.best_val_metric:.4f} to {current_metric_val:.4f}.")
            self.best_val_metric = current_metric_val
            self.early_stopping_counter = 0
            return False # Do not stop
        else:
            self.early_stopping_counter += 1
            logger.info(f"EarlyStopping: {self.early_stopping_cfg.monitor} did not improve from {self.best_val_metric:.4f}. Counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
            if self.early_stopping_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch_num + 1}.")
                return True # Stop training
            return False

    def _save_checkpoint(self, epoch_num: int, current_metric_val: Optional[float] = None):
        if not self.checkpoint_cfg or not self.checkpoint_cfg.get('dirpath'):
            return

        save_dir = self.checkpoint_cfg['dirpath']
        os.makedirs(save_dir, exist_ok=True)
        
        state = {
            'epoch': epoch_num,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_metric_early_stop': self.best_val_metric if self.early_stopping_cfg else None,
            'best_checkpoint_metric': self.best_checkpoint_metric,
            'config': OmegaConf.to_container(self.cfg, resolve=True) # Save config for reproducibility
        }

        # Save last checkpoint
        if self.checkpoint_cfg.get('save_last', True):
            last_ckpt_path = os.path.join(save_dir, f"{self.checkpoint_cfg.get('filename_prefix', 'model')}-last.ckpt")
            torch.save(state, last_ckpt_path)
            logger.info(f"Saved last checkpoint to {last_ckpt_path} at epoch {epoch_num + 1}")
            if self.wandb_cfg and self.wandb_cfg.get('log_model') == 'all':
                # Log as artifact
                artifact = wandb.Artifact(f"{self.cfg.project_name}-last_checkpoint", type="model")
                artifact.add_file(last_ckpt_path)
                wandb.log_artifact(artifact)

        # Save top-k checkpoints
        save_top_k = self.checkpoint_cfg.get('save_top_k', 0)
        if save_top_k > 0 and current_metric_val is not None:
            monitor_mode = self.checkpoint_cfg.get('mode', 'min')
            improved = False
            if monitor_mode == 'min':
                if current_metric_val < self.best_checkpoint_metric:
                    improved = True
            else: # mode == 'max'
                if current_metric_val > self.best_checkpoint_metric:
                    improved = True
            
            if improved or len(self.top_k_checkpoints) < save_top_k:
                if improved:
                    logger.info(f"Checkpoint metric {self.checkpoint_cfg.get('monitor')} improved from {self.best_checkpoint_metric:.4f} to {current_metric_val:.4f}.")
                    self.best_checkpoint_metric = current_metric_val
                
                ckpt_filename = f"{self.checkpoint_cfg.get('filename_prefix', 'model')}-epoch={epoch_num+1:03d}-{self.checkpoint_cfg.get('monitor').replace('/','_')}={current_metric_val:.3f}.ckpt"
                ckpt_path = os.path.join(save_dir, ckpt_filename)
                torch.save(state, ckpt_path)
                logger.info(f"Saved top-k checkpoint to {ckpt_path}")
                self.top_k_checkpoints.append((current_metric_val, ckpt_path))
                # Sort and prune
                self.top_k_checkpoints.sort(key=lambda x: x[0], reverse=(monitor_mode == 'max'))
                if len(self.top_k_checkpoints) > save_top_k:
                    worst_of_top_k = self.top_k_checkpoints.pop()
                    if os.path.exists(worst_of_top_k[1]):
                        os.remove(worst_of_top_k[1])
                        logger.info(f"Removed old top-k checkpoint: {worst_of_top_k[1]}")
                
                # Wandb logging for best model
                if self.wandb_cfg and self.wandb_cfg.get('log_model', False) and improved:
                    # Log only if 'log_model' is True (or 'all') and it's the best so far
                    best_ckpt_path = self.top_k_checkpoints[0][1] # Current best
                    artifact_name = f"{self.cfg.project_name}-best_model"
                    # Remove previous "best_model" artifact if using a simple name, or version it
                    # wandb.run.summary["best_model_path"] = best_ckpt_path
                    # wandb.save(best_ckpt_path, base_path=save_dir) # Saves to wandb files dir
                    artifact = wandb.Artifact(artifact_name, type="model", metadata=state)
                    artifact.add_file(best_ckpt_path)
                    wandb.log_artifact(artifact, aliases=["best", f"epoch_{epoch_num+1}"])
                    logger.info(f"Logged best model artifact to WandB: {artifact_name}")

    def load_checkpoint(self, checkpoint_path: str, load_optimizer_scheduler: bool = True, load_config: bool = False) -> int:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return 0
        
        try:
            state = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e:
            logger.error(f"Error loading checkpoint file: {e}")
            return 0

        self.model.load_state_dict(state['model_state_dict'])
        logger.info("Model state loaded.")

        if load_optimizer_scheduler:
            if 'optimizer_state_dict' in state and self.optimizer:
                self.optimizer.load_state_dict(state['optimizer_state_dict'])
                logger.info("Optimizer state loaded.")
            if 'scheduler_state_dict' in state and self.scheduler and state['scheduler_state_dict']:
                self.scheduler.load_state_dict(state['scheduler_state_dict'])
                logger.info("Scheduler state loaded.")
        
        start_epoch = state.get('epoch', -1) + 1
        self.best_val_metric = state.get('best_val_metric_early_stop', self.best_val_metric)
        self.best_checkpoint_metric = state.get('best_checkpoint_metric', self.best_checkpoint_metric)
        
        if load_config and 'config' in state:
            loaded_cfg_dict = state['config']
            # Be careful when overwriting self.cfg. For now, just log or compare.
            logger.info(f"Loaded config from checkpoint (first 5 keys): {list(loaded_cfg_dict.keys())[:5]}")
            # self.cfg = OmegaConf.create(loaded_cfg_dict) # Potentially dangerous

        logger.info(f"Resuming training from epoch {start_epoch}")
        return start_epoch

    def run_training(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        start_epoch = 0
        if resume_from_checkpoint:
            start_epoch = self.load_checkpoint(resume_from_checkpoint, load_optimizer_scheduler=True)

        for epoch in range(start_epoch, self.epochs):
            train_metrics = self.train_epoch(epoch)
            
            val_metrics = {}
            if self.val_loader:
                val_metrics = self.eval_epoch(epoch, self.val_loader, split_name="val")
            else:
                logger.warning(f"Epoch {epoch+1}: No validation loader provided. Skipping validation.")
                # If no val_loader, use train_loss for checkpointing/early_stopping if configured
                val_metrics["val/loss"] = train_metrics.get("loss", float('inf')) 
                val_metrics["val/mae"] = float('inf') # Dummy for checkpointing if MAE is monitored

            # Scheduler step (if any, and if it needs validation metric)
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    # Monitor key for scheduler (e.g., val_loss, val/loss)
                    sched_monitor_key = self.scheduler.optimizer.param_groups[0].get('monitor', 'val/loss') #Hacky way to get monitor if stored by user
                    if 'monitor' in self.cfg.trainer.get('lr_scheduler_params',{}):
                         sched_monitor_key = self.cfg.trainer.lr_scheduler_params.monitor
                    else: # default
                         sched_monitor_key = "val/loss_scaled" if "val/loss_scaled" in val_metrics else "val/loss"
                    
                    metric_for_scheduler = val_metrics.get(sched_monitor_key, None)
                    if metric_for_scheduler is not None:
                        self.scheduler.step(metric_for_scheduler)
                        logger.info(f"Scheduler ReduceLROnPlateau step with {sched_monitor_key}: {metric_for_scheduler:.4f}")
                    else:
                        logger.warning(f"Metric '{sched_monitor_key}' not found for LR scheduler. Scheduler will not step.")
                else:
                    self.scheduler.step() # For schedulers like CosineAnnealingLR, StepLR
            
            # Checkpointing
            metric_for_checkpointing = val_metrics.get(self.checkpoint_cfg.get('monitor', 'val/loss'), None)
            if metric_for_checkpointing is None:
                logger.warning(f"Metric '{self.checkpoint_cfg.get('monitor')}' not found for checkpointing. Using val/loss_scaled if available.")
                metric_for_checkpointing = val_metrics.get('val/loss_scaled', float('inf'))
            self._save_checkpoint(epoch, metric_for_checkpointing)

            # Early stopping
            if self.early_stopping_cfg:
                metric_for_early_stop = val_metrics.get(self.early_stopping_cfg.monitor, None)
                if metric_for_early_stop is None:
                    logger.warning(f"Metric '{self.early_stopping_cfg.monitor}' not found for early stopping. Early stopping may not work as expected.")
                    metric_for_early_stop = float('inf') if self.early_stopping_cfg.mode == 'min' else float('-inf')
                
                if self._check_early_stopping(metric_for_early_stop, epoch):
                    break
        
        logger.info("Training finished.")
        if self.wandb_cfg and wandb.run:
            # Log best metrics if available
            # wandb.summary["best_val_loss"] = self.best_val_metric if self.early_stopping_cfg else val_metrics.get("val/loss", None)
            # wandb.summary["best_model_checkpoint_metric_val"] = self.best_checkpoint_metric
            if self.top_k_checkpoints:
                 wandb.summary["best_model_path_on_disk"] = self.top_k_checkpoints[0][1]
            wandb.finish()
        
        return {"best_val_metric": self.best_val_metric, "final_epoch": epoch}

    def evaluate_on_test(self, checkpoint_path: Optional[str] = None) -> Dict[str, float]:
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path, load_optimizer_scheduler=False)
        elif self.top_k_checkpoints: # Use best saved checkpoint if no path given
            best_ckpt_path = self.top_k_checkpoints[0][1]
            logger.info(f"No checkpoint path provided for testing. Using best saved checkpoint: {best_ckpt_path}")
            self.load_checkpoint(best_ckpt_path, load_optimizer_scheduler=False)
        else:
            logger.warning("No checkpoint specified and no best checkpoint found. Evaluating with current model state.")

        if not self.test_loader:
            logger.error("Test loader not available. Cannot evaluate on test set.")
            return {}
        
        logger.info("Starting evaluation on the test set...")
        # Pass a dummy epoch number like -1 for logging consistency if eval_epoch expects it
        test_metrics = self.eval_epoch(epoch_num=-1, dataloader=self.test_loader, split_name="test") 
        
        logger.info("Test set evaluation finished.")
        logger.info(f"Test Metrics: {test_metrics}")
        if self.wandb_cfg and wandb.run: # If run is still active or re-init for test
            wandb.log(test_metrics) # Log as final test metrics
            # wandb.finish() # Ensure it finishes if not part of run_training call
        return test_metrics


# For direct testing of the trainer (requires a full setup)
if __name__ == '__main__':
    # This requires a lot of setup: dummy config, dummy model, dummy dataloaders, dummy scaler.
    # It's more practical to test this via src/train.py with a minimal config.
    logger.info("TecTrainer direct testing requires full environment setup (config, model, data, scaler).")
    logger.info("Please test via `src/train.py` with a suitable configuration.")

    # Example of what would be needed:
    # 1. Create dummy OmegaConf config (similar to ST_LLM test but more extensive)
    #    cfg = OmegaConf.create({...}) 
    #    cfg.trainer.epochs = 2 # For quick test
    #    cfg.wandb = None # Disable wandb for this test
    
    # 2. Initialize model
    #    seed_everything(cfg.seed)
    #    model = ST_LLM(cfg)
    
    # 3. Create dummy DataLoaders (as in tec_dataset.py test)
    #    # dummpy_X = torch.randn(10, cfg.model.p_history, cfg.data.n_lat * cfg.data.n_lon, 12)
    #    # dummpy_Y = torch.randn(10, cfg.model.s_forecast, cfg.data.n_lat * cfg.data.n_lon, 1)
    #    # dummy_dataset = torch.utils.data.TensorDataset(dummpy_X, dummpy_Y)
    #    # train_loader = DataLoader(dummy_dataset, batch_size=2)
    #    # val_loader = DataLoader(dummy_dataset, batch_size=2)
    
    # 4. Create dummy Scaler (as in scaler.py test)
    #    # class DummyScaler: ...
    #    # scaler = DummyScaler(...) 
    
    # 5. Instantiate Trainer
    #    # trainer = TecTrainer(cfg, model, train_loader, val_loader, scaler=scaler)
    
    # 6. Run training (short)
    #    # trainer.run_training()
    
    # 7. Evaluate (optional)
    #    # trainer.evaluate_on_test(test_loader=val_loader) # Using val_loader as dummy test
    
    print("End of TecTrainer placeholder for direct testing.") 