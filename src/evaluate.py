import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import torch
import wandb # Optional, for logging evaluation results

from src.datasets.tec_dataset import TECDataModule
from src.models.tec_gpt import ST_LLM
from src.trainers.tec_trainer import TecTrainer # Re-use eval_epoch and checkpoint loading
from src.utils.scaler import StandardScaler as TecStandardScaler
from src.utils.helpers import seed_everything

logger = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Hydra驱动的主评估函数。
    加载指定的模型检查点并在测试集上进行评估。

    Args:
        cfg (DictConfig): Hydra加载的配置对象。
                         Key evaluation configs:
                         - cfg.checkpoint_path: Path to the model checkpoint to evaluate.
                         - cfg.data: Data configuration.
                         - cfg.model: Model configuration (matching the checkpoint).
                         - cfg.trainer: Trainer configuration (subset needed for eval).
                         - cfg.wandb: Optional Wandb config for logging results.
    """
    # --- 初始设置 ---
    seed_everything(cfg.seed)
    logger.info(f"Global seed set to: {cfg.seed}")
    logger.info(f"Hydra current working directory: {os.getcwd()}")
    logger.info(f"Original working directory: {hydra.utils.get_original_cwd()}")
    
    if cfg.verbose:
        logger.info("Full configuration for evaluation:\n" + OmegaConf.to_yaml(cfg))

    # --- Wandb 初始化 (可选, 用于记录评估结果) ---
    # It might be part of a larger experiment, or a standalone evaluation run.
    if cfg.wandb and cfg.wandb.project and cfg.get("log_evaluation_to_wandb", True):
        run_name = cfg.wandb.get("name", None)
        if run_name is None:
            run_name = f"eval_{os.path.basename(os.getcwd())}"
            if cfg.get("checkpoint_path"): 
                 run_name += f"_{os.path.splitext(os.path.basename(cfg.checkpoint_path))[0]}"

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity"),
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            dir=os.getcwd(),
            job_type="evaluation", # Mark this run as an evaluation job
            tags=cfg.wandb.get("tags", ["evaluation"]), 
            mode=cfg.wandb.get("mode", "online")
        )
        logger.info(f"Wandb initialized for evaluation: project='{cfg.wandb.project}', run '{wandb.run.name}'")
    else:
        logger.info("Wandb not configured for evaluation or logging disabled. Skipping Wandb initialization.")

    # --- 数据模块实例化 ---
    logger.info(f"Instantiating DataModule <{cfg.data._target_}>")
    datamodule: TECDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data() # Checks for files
    datamodule.setup(stage='test')  # Creates test dataset
    
    test_loader = datamodule.test_dataloader()
    if test_loader is None:
        logger.error("Test dataloader is None. Cannot proceed with evaluation.")
        if wandb.run: wandb.finish(exit_code=1)
        return

    # --- Scaler 实例化 ---
    scaler_path = os.path.join(datamodule.processed_data_dir, cfg.data.scaler_filename)
    if not os.path.isabs(scaler_path) and hydra.utils.get_original_cwd() != os.getcwd():
        scaler_path_abs = os.path.join(hydra.utils.get_original_cwd(), scaler_path)
        if os.path.exists(scaler_path_abs):
            scaler_path = scaler_path_abs
        else:
            logger.warning(f"Scaler file not found at {scaler_path_abs}, trying {scaler_path}")
            if not os.path.exists(scaler_path):
                 logger.error(f"Scaler file not found at {scaler_path} or {scaler_path_abs}. Cannot proceed.")
                 if wandb.run: wandb.finish(exit_code=1)
                 return
    try:
        scaler = TecStandardScaler(scaler_path=scaler_path)
        logger.info(f"Scaler loaded from {scaler_path}")
    except Exception as e:
        logger.error(f"Error loading scaler: {e}. Evaluation cannot proceed without scaler for metrics.")
        if wandb.run: wandb.finish(exit_code=1)
        return

    # --- 模型实例化 ---
    # Model config should match the one used for training the checkpoint.
    # It's good practice to save the config with the checkpoint, but here we rely on current cfg.model.
    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: ST_LLM = hydra.utils.instantiate(cfg.model, cfg=cfg) # Pass full cfg

    # --- Trainer 实例化 (主要用于加载 checkpoint 和使用 eval_epoch) ---
    # We don't need train_loader or full optimizer/scheduler setup for evaluation.
    # A subset of trainer config might be relevant (device, criterion for loss_scaled).
    logger.info(f"Instantiating a lightweight trainer for evaluation purposes.")
    # Create a minimal trainer config for evaluation context if needed
    eval_trainer_cfg = cfg.trainer.copy() # Start with trainer config
    # Override parts not needed for eval to avoid issues, e.g., optimizer params if model is frozen
    # eval_trainer_cfg.optimizer_name = 'adamw' # Dummy, won't be used for steps
    # eval_trainer_cfg.lr_scheduler_name = None

    trainer = TecTrainer(
        cfg=cfg, # Pass full cfg, trainer will pick what it needs
        model=model,
        test_loader=test_loader, # Pass test_loader here
        scaler=scaler
        # train_loader and val_loader can be None for evaluation only
    )
    
    # --- 加载模型检查点 ---
    checkpoint_path_to_load = cfg.get("checkpoint_path", None)
    if not checkpoint_path_to_load:
        logger.error("No checkpoint_path specified in config for evaluation. Cannot proceed.")
        if wandb.run: wandb.finish(exit_code=1)
        return

    if not os.path.isabs(checkpoint_path_to_load):
        original_cwd_ckpt_path = os.path.join(hydra.utils.get_original_cwd(), checkpoint_path_to_load)
        if os.path.exists(original_cwd_ckpt_path):
            checkpoint_path_to_load = original_cwd_ckpt_path
        else:
            logger.warning(f"Checkpoint {checkpoint_path_to_load} (rel to orig_cwd as {original_cwd_ckpt_path}) not found. Trying relative to hydra output dir.")
            # If not found, it might be relative to hydra output dir. Check if exists.
            if not os.path.exists(checkpoint_path_to_load):
                logger.error(f"Checkpoint file not found at either {original_cwd_ckpt_path} or {checkpoint_path_to_load} (rel to hydra output dir). Cannot proceed.")
                if wandb.run: wandb.finish(exit_code=1)
                return

    logger.info(f"Attempting to load checkpoint: {checkpoint_path_to_load}")
    # Load checkpoint using trainer's method. Don't need optimizer/scheduler states for eval.
    # The `load_checkpoint` method in TecTrainer should put model on the correct device.
    trainer.load_checkpoint(checkpoint_path_to_load, load_optimizer_scheduler=False, load_config=cfg.get("load_config_from_checkpoint", False))
    logger.info(f"Model loaded from checkpoint: {checkpoint_path_to_load}")

    # --- 执行评估 ---
    logger.info("Starting evaluation on the test set...")
    # The evaluate_on_test method in TecTrainer uses its self.test_loader and self.scaler
    # It calls eval_epoch internally.
    test_metrics = trainer.evaluate_on_test(checkpoint_path=None) # Checkpoint already loaded
    
    logger.info("--- Test Set Evaluation Metrics ---")
    for metric_name, metric_value in test_metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    logger.info("-----------------------------------")

    # --- (可选) 保存评估结果 ---
    # Results are already logged to console and Wandb (if enabled).
    # Could also save to a JSON/YAML file in the output directory.
    output_metrics_file = os.path.join(os.getcwd(), "test_metrics.yaml")
    try:
        with open(output_metrics_file, 'w') as f:
            OmegaConf.save(config=OmegaConf.create(test_metrics), f=f)
        logger.info(f"Test metrics saved to: {output_metrics_file}")
    except Exception as e:
        logger.error(f"Failed to save test metrics to file: {e}")

    if wandb.run:
        wandb.finish()

    logger.info("Evaluation script finished.")

if __name__ == "__main__":
    # 运行: python src/evaluate.py checkpoint_path=/path/to/your/model.ckpt [other overrides]
    main() 