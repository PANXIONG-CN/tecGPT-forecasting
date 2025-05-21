import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import torch
import wandb
import optuna # For Optuna specific exceptions or types if needed

# Import your project components
from src.datasets.tec_dataset import TECDataModule
from src.models.tec_gpt import ST_LLM
from src.trainers.tec_trainer import TecTrainer
from src.utils.scaler import StandardScaler as TecStandardScaler
from src.utils.helpers import seed_everything

logger = logging.getLogger(__name__) # Module-level logger

# Global variable to store the scaler, as it's loaded once per study if data doesn't change.
# This is a simplification; for more complex scenarios, consider passing it or managing state differently.
# If data_preprocessing or scaler generation depends on tunable params, this needs rethinking.
_SCALER_CACHE = None
_DATAMODULE_CACHE = None # For train/val loaders if data config is fixed across trials

def objective(cfg: DictConfig) -> float:
    """
    Optuna 目标函数，由 Hydra Optuna Sweeper 调用。
    它接收一个包含当前试验超参数的配置对象 `cfg`。
    """
    global _SCALER_CACHE, _DATAMODULE_CACHE
    try:
        # --- 初始设置 (部分可能已由 Hydra Sweeper 处理) ---
        # Seed for this trial (Optuna might handle this, or we can ensure it)
        # If Hydra's Optuna sweeper sets a seed per trial, it might be in cfg.seed
        current_trial_seed = cfg.get("seed", None)
        if current_trial_seed is None:
            current_trial_seed = optuna.trial.TrialState.RUNNING # Get Optuna trial if available
            if hasattr(optuna.trial, 'number'): # Check if we are in an Optuna trial context
                 current_trial_seed = cfg.get("seed_base", 42) + optuna.trial.number # Make seed vary per trial
            else:
                 current_trial_seed = cfg.get("seed_base", 42) # Fallback if not in Optuna trial or no number
        seed_everything(current_trial_seed)
        logger.info(f"Optuna Trial Seed: {current_trial_seed}")

        # Potentially re-use scaler if data config is fixed for the sweep
        # This assumes cfg.data and paths for scaler don't change with hyperparams
        if _SCALER_CACHE is None or cfg.get("force_reload_scaler_per_trial", False):
            # --- Scaler 实例化 (copied from train.py, ensure paths are correct) ---
            processed_data_dir_for_scaler = cfg.data.processed_data_dir
            if not os.path.isabs(processed_data_dir_for_scaler) and hydra.utils.get_original_cwd() != os.getcwd():
                 processed_data_dir_for_scaler = os.path.join(hydra.utils.get_original_cwd(), processed_data_dir_for_scaler)
            
            scaler_path = os.path.join(processed_data_dir_for_scaler, cfg.data.scaler_filename)
            if not os.path.exists(scaler_path):
                logger.error(f"Scaler file not found at {scaler_path} for Optuna trial. Preprocessing needed.")
                raise FileNotFoundError(f"Scaler file for Optuna: {scaler_path}")
            _SCALER_CACHE = TecStandardScaler(scaler_path=scaler_path)
            logger.info(f"Scaler (re)loaded for Optuna trial from {scaler_path}")
        scaler = _SCALER_CACHE

        # --- 数据模块实例化 (copied from train.py) ---
        # Potentially cache datamodule if data params are fixed across trials
        if _DATAMODULE_CACHE is None or cfg.get("force_reload_data_per_trial", False):
            logger.info(f"Instantiating DataModule <{cfg.data._target_}> for Optuna trial.")
            # Ensure processed_data_dir in cfg.data is absolute or correctly relative for hydra
            # If cfg.data.processed_data_dir is relative to original CWD:
            if not os.path.isabs(cfg.data.processed_data_dir) and hydra.utils.get_original_cwd() != os.getcwd():
                # Create a mutable copy of cfg.data to modify path temporarily
                temp_data_cfg = cfg.data.copy() # Create a mutable copy
                temp_data_cfg.processed_data_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.data.processed_data_dir)
                datamodule: TECDataModule = hydra.utils.instantiate(temp_data_cfg)
            else:
                datamodule: TECDataModule = hydra.utils.instantiate(cfg.data)
            
            datamodule.prepare_data()
            datamodule.setup(stage='fit')
            _DATAMODULE_CACHE = datamodule # Cache it
            logger.info("DataModule (re)loaded for Optuna trial.")
        else:
            datamodule = _DATAMODULE_CACHE
            logger.info("Using cached DataModule for Optuna trial.")
            
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        if train_loader is None:
            logger.error("Training dataloader is None for Optuna trial. Aborting trial.")
            # Optuna needs a float. Indicate failure with a bad value.
            return float('inf') if cfg.hydra.sweeper.direction == "minimize" else float('-inf')

        # --- 模型实例化 (using current trial's hyperparams from cfg.model) ---
        logger.info(f"Instantiating model <{cfg.model._target_}> for Optuna trial.")
        # ST_LLM needs the full cfg for n_lat, n_lon from cfg.data potentially
        model: ST_LLM = hydra.utils.instantiate(cfg.model, cfg=cfg)

        # --- Wandb 初始化 (for this specific trial) ---
        # The main tune.py script will handle the overall Wandb run for the sweep.
        # Here, we might want to log trial-specific details if Wandb is used *per trial*.
        # However, Hydra's Optuna Sweeper typically logs to a parent run.
        # For simplicity, let's assume wandb.init is called by the launcher script (tune.py main).
        # If we want individual wandb runs per trial, we'd init here:
        trial_wandb_run = None
        if cfg.get("wandb_per_trial", False) and cfg.wandb and cfg.wandb.project:
            trial_name = f"trial_{optuna.trial.number}_{os.path.basename(os.getcwd())}" if hasattr(optuna.trial, 'number') else f"trial_unknown_{os.path.basename(os.getcwd())}"
            trial_wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.get("entity"),
                name=trial_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                group=cfg.wandb.get("group_name", "optuna_sweep"), # Group trials by sweep
                reinit=True, # Allow re-initialization for each trial
                dir=os.getcwd(),
                job_type="hpo_trial",
                mode=cfg.wandb.get("mode", "online")
            )
            logger.info(f"Wandb initialized for Optuna trial '{trial_name}'.")

        # --- 训练器实例化和运行 ---
        logger.info(f"Instantiating trainer <{cfg.trainer._target_}> for Optuna trial.")
        trainer: TecTrainer = hydra.utils.instantiate(
            cfg.trainer, 
            cfg=cfg, 
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            scaler=scaler
        )
        
        # For HPO, we might want to run fewer epochs or use specific callbacks for pruning.
        # This can be configured in tune.yaml by overriding trainer.epochs for the sweep.
        logger.info(f"Starting training for Optuna trial (Epochs: {cfg.trainer.epochs}).")
        training_results = trainer.run_training() # No checkpoint resume during HPO usually

        # --- 获取并返回目标指标 ---
        # The metric to optimize is typically the one monitored by early stopping/checkpointing.
        # Ensure this key exists in `training_results` or `val_metrics` from trainer.
        # `TecTrainer.run_training` returns `{"best_val_metric": self.best_val_metric, ...}`
        # where `best_val_metric` is the value of the metric monitored by early stopping.
        
        optimized_metric = training_results.get("best_val_metric")
        
        if optimized_metric is None:
            logger.warning("Objective metric ('best_val_metric') not found in training results. Optuna might fail or use a default.")
            # Return a very bad value based on optimization direction
            optimized_metric = float('inf') if cfg.hydra.sweeper.direction == "minimize" else float('-inf')
        else:
            logger.info(f"Optuna trial finished. Monitored metric value: {optimized_metric:.4f}")

        # Optuna Pruning: If integrated, check if trial should be pruned.
        # This is typically handled by callbacks passed to trainer or checked within the loop.
        # For Hydra + Optuna, pruner is configured in tune.yaml and Hydra handles it.
        # If a pruner reports Pruned, Optuna raises optuna.TrialPruned.
        # We just need to return the metric value. Hydra sweeper handles pruning exceptions.

        if trial_wandb_run: # If we had a wandb run per trial
            wandb.log({"final_objective_metric": optimized_metric}) # Log the final metric for this trial
            trial_wandb_run.finish()

        return optimized_metric

    except optuna.exceptions.TrialPruned as e:
        logger.info(f"Optuna Trial Pruned: {e}")
        raise # Re-raise for Hydra to handle
    except FileNotFoundError as e:
        logger.error(f"File not found during Optuna trial: {e}. Reporting failure to Optuna.")
        # Report a bad value to Optuna to mark this trial as failed due to setup issues.
        # Should not prune, but mark as failed.
        # For Optuna, returning a high/low value can be one way, or raise specific error.
        # However, Hydra Optuna sweeper might just mark it as failed if an exception occurs.
        # Re-raising might be the cleanest.
        raise
    except Exception as e:
        logger.error(f"Exception in Optuna objective function: {e}", exc_info=True)
        # Report a bad value to Optuna to mark this trial as failed.
        # Based on Optuna direction, return a very bad value
        # Or re-raise to let Hydra handle it (might mark as FAIL).
        raise # Re-raise for Hydra to see the failure

@hydra.main(config_path="../conf", config_name="tune", version_base=None)
def run_optuna_sweep(cfg: DictConfig) -> None:
    """
    主入口点，用于启动由 Hydra Optuna Sweeper 管理的超参数优化过程。
    Hydra 会多次调用 `objective` 函数（上面定义的）。

    `tune.yaml` 中应包含 Optuna Sweeper 的配置：
    ```yaml
    defaults:
      - override hydra/sweeper: optuna
    
    hydra:
      sweeper:
        _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper
        direction: minimize  # or "maximize"
        study_name: "tecgpt_tuning"
        storage: null # e.g., "sqlite:///optuna_study.db"
        n_trials: 50
        n_jobs: 1 # Parallel trials
        sampler:
          _target_: optuna.samplers.TPESampler
          # ... sampler args
        pruner:
          _target_: optuna.pruners.MedianPruner
          # ... pruner args
    
    # Parameters to tune are also defined in tune.yaml by overriding main config values.
    # e.g.:
    # trainer:
    #   learning_rate:
    #     type: float
    #     low: 1e-5
    #     high: 1e-3
    #     log: True
    ```
    这个 `run_optuna_sweep` 函数本身在被 Hydra Optuna Sweeper 调用时，
    其主要职责是确保 `objective` 函数能够正确运行。
    Hydra 的多重运行 (multirun) 机制会自动处理对 `objective` 的调用。
    我们在这里主要是为了提供一个清晰的入口点，并可以添加整体 sweep 的设置 (如 Wandb run for the entire sweep).
    """
    logger.info("Starting Optuna hyperparameter sweep...")
    logger.info("Configuration for sweep (from tune.yaml and merged):")
    logger.info(OmegaConf.to_yaml(cfg))

    # The Hydra Optuna Sweeper plugin will execute the `objective` function based on `tune.yaml`.
    # This `run_optuna_sweep` function, when `tune.py` is the main script for `hydra.main` with Optuna sweeper,
    # essentially just sets up the context. The actual sweeping logic (calling `objective` multiple times)
    # is handled by Hydra's multirun mechanism when the Optuna sweeper is active.

    # Optional: Initialize a global Wandb run for the entire sweep if not done per trial.
    # This is if you want one wandb run to track all optuna trials.
    # Individual trial logging might be better handled by Hydra's logging or if objective itself logs to wandb.
    if cfg.wandb and cfg.wandb.project and not cfg.get("wandb_per_trial", False):
        sweep_run_name = cfg.wandb.get("name", f"optuna_sweep_{cfg.hydra.sweeper.study_name if cfg.hydra.sweeper.study_name else 'default_study'}")
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity"),
            name=sweep_run_name,
            config=OmegaConf.to_container(cfg, resolve=True), # Log the base sweep config
            dir=os.getcwd(), # Save in hydra's multirun directory
            job_type="hpo_sweep",
            tags=cfg.wandb.get("tags", ["optuna", "sweep"]),
            mode=cfg.wandb.get("mode", "online")
        )
        logger.info(f"Overall Wandb run for Optuna sweep initialized: '{sweep_run_name}'.")
        # Note: Hydra's Optuna Sweeper plugin might have its own Wandb integration capabilities or callbacks.
        # Refer to hydra-plugins documentation for best practices on combining.

    # At this point, Hydra will take over and run the trials using the `objective` function.
    # The `objective` function will be called by Hydra for each trial. No explicit loop here.
    # If this script is run as `python src/tune.py -m ...`, Hydra executes the sweep.
    # The return value of `objective` is used by Optuna.
    # This function itself doesn't directly return the best trial's value; Optuna/Hydra handles that.
    logger.info("Hydra Optuna Sweeper will now manage the execution of trials.")

    # If there was an overall wandb run, finish it after the sweep (Hydra should manage this)
    # However, Hydra runs the objective in separate processes for multirun, so a single wandb.finish() here
    # might not be correct if objective itself does not init/finish wandb runs when `wandb_per_trial` is false.
    # It's often better if the objective function itself handles its logging if wandb_per_trial is false, 
    # or if hydra plugins for wandb handle this.
    # For now, if we started a sweep-level wandb run, we might need a way to close it. 
    # But typically, the hydra process running this main `run_optuna_sweep` finishes, and that should end the wandb run.

if __name__ == "__main__":
    # To run Optuna sweep:
    # python src/tune.py -m 
    # (This enables multirun, which the Optuna sweeper uses)
    # You can add overrides, e.g.:
    # python src/tune.py -m hydra.sweeper.n_trials=10 trainer.epochs=5
    run_optuna_sweep() 