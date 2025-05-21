import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import torch
import wandb

from src.datasets.tec_dataset import TECDataModule
from src.models.tec_gpt import ST_LLM
from src.trainers.tec_trainer import TecTrainer
from src.utils.scaler import StandardScaler as TecStandardScaler # Renamed
from src.utils.helpers import seed_everything

# 获取一个logger实例
# Hydra会自动配置日志，所以我们通常只需要获取logger
logger = logging.getLogger(__name__) # Use __name__ for module-level logger

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> Optional[float]:
    """
    Hydra驱动的主训练函数。

    Args:
        cfg (DictConfig): Hydra加载的配置对象。

    Returns:
        Optional[float]: Optuna优化时需要返回的目标指标 (例如, 最小验证损失)。
    """
    # --- 初始设置 ---
    if cfg.get("deterministic"):
        # 设置torch.backends.cudnn.deterministic = True等
        # 这通常在seed_everything中处理，但可以根据需要在此处显式设置更严格的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Deterministic mode enabled in PyTorch backend.")
        
    seed_everything(cfg.seed)
    logger.info(f"Global seed set to: {cfg.seed}")
    logger.info(f"Hydra current working directory: {os.getcwd()}")
    logger.info(f"Original working directory: {hydra.utils.get_original_cwd()}")
    
    # 打印完整配置 (有助于调试)
    if cfg.verbose:
        logger.info("Full configuration:\n" + OmegaConf.to_yaml(cfg))

    # --- Wandb 初始化 (如果配置了) ---
    if cfg.wandb and cfg.wandb.project:
        # 使用Hydra的输出子目录作为wandb的run name，如果cfg.wandb.name未指定
        # run_name = os.path.basename(os.getcwd()) if cfg.wandb.get("name") is None else cfg.wandb.name
        # wandb.init(
        #     project=cfg.wandb.project,
        #     entity=cfg.wandb.entity, # Can be None
        #     name=run_name,
        #     config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        #     dir=os.getcwd(), # Save wandb files in hydra's output dir
        #     mode=cfg.wandb.get("mode", "online")
        # )
        # logger.info(f"Wandb initialized: project='{cfg.wandb.project}', run_name='{run_name}'")
        # Better: let hydra handle run name and dir for wandb if possible or define explicitly.
        # The following uses Hydra's output directory for wandb files and a generated name.
        
        # Construct a run name (can be customized further)
        # Example: experiment_name-timestamp or based on overrides
        run_name = cfg.wandb.get("name", None)
        if run_name is None:
            # Create a more descriptive name if not provided
            # For example, using some key parameters from the config
            # This is just an example, can be made more sophisticated
            run_name = f"train_{os.path.basename(os.getcwd())}" 

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.get("entity"), # Optional
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            dir=os.getcwd(), # Saves wandb metadata in hydra's output dir.
            # id= # Optional: for resuming runs
            # resume= # Optional: "allow", "must", "never", "auto"
            tags=cfg.wandb.get("tags", None), # Optional tags
            mode=cfg.wandb.get("mode", "online") # "online", "offline", "disabled"
        )
        logger.info(f"Wandb initialized for project '{cfg.wandb.project}', run '{wandb.run.name}'. Files in {os.getcwd()}/wandb")
        # Log model checkpoints to wandb if log_model is True
        if cfg.wandb.get("log_model", False):
            logger.info("Wandb model checkpoint logging enabled.")
            # This is usually handled by the trainer's checkpoint callback or explicitly
            # wandb.watch(model, log="all") # Can be too verbose for large models
    else:
        logger.info("Wandb not configured or project name missing. Skipping Wandb initialization.")

    # --- 数据模块实例化 ---
    logger.info(f"Instantiating DataModule <{cfg.data._target_}>")
    # data_cfg already contains p_history, s_forecast etc. needed by datamodule
    # Also n_lat, n_lon for n_nodes calculation within DataModule
    datamodule: TECDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data() # Checks for files
    datamodule.setup(stage='fit')   # Creates train/val datasets
    
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    if train_loader is None:
        logger.error("Training dataloader is None. Cannot proceed with training.")
        if wandb.run: wandb.finish(exit_code=1)
        return None # Or raise error

    # --- Scaler 实例化 ---
    # Scaler path is relative to processed_data_dir defined in cfg.data
    scaler_path = os.path.join(datamodule.processed_data_dir, cfg.data.scaler_filename)
    # Check if the path needs to be made absolute if processed_data_dir is relative to original CWD
    if not os.path.isabs(scaler_path) and hydra.utils.get_original_cwd() != os.getcwd():
        scaler_path_abs = os.path.join(hydra.utils.get_original_cwd(), scaler_path)
        if os.path.exists(scaler_path_abs):
            scaler_path = scaler_path_abs
        else: # Fallback to path relative to current hydra output dir if original CWD one not found
            logger.warning(f"Scaler file not found at {scaler_path_abs} (rel to orig_cwd), trying {scaler_path} (rel to hydra out dir)")
            if not os.path.exists(scaler_path):
                 logger.error(f"Scaler file not found at {scaler_path} or {scaler_path_abs}. Cannot proceed.")
                 if wandb.run: wandb.finish(exit_code=1)
                 return None
    
    try:
        scaler = TecStandardScaler(scaler_path=scaler_path)
        logger.info(f"Scaler loaded from {scaler_path}")
    except FileNotFoundError:
        logger.error(f"Scaler file not found at {scaler_path}. Preprocessing might be needed.")
        if wandb.run: wandb.finish(exit_code=1)
        return None # Or raise error
    except Exception as e:
        logger.error(f"Error loading scaler: {e}")
        if wandb.run: wandb.finish(exit_code=1)
        return None

    # --- 模型实例化 ---
    # Model config (cfg.model) should have all necessary params for ST_LLM constructor
    # including sub-configs for embeddings, PFA_LLM, etc.
    # ST_LLM constructor uses cfg.model and cfg.data (for n_lat, n_lon if n_nodes not given)
    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: ST_LLM = hydra.utils.instantiate(cfg.model, cfg=cfg) # Pass the full config to ST_LLM
    
    # Log model architecture to wandb if enabled
    # if wandb.run: wandb.watch(model, log='all', log_freq=100) # log_freq can be adjusted

    # --- 训练器实例化和运行 ---
    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: TecTrainer = hydra.utils.instantiate(
        cfg.trainer, 
        cfg=cfg, # Pass full config to trainer for access to wandb, device, etc.
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        scaler=scaler
    )
    
    resume_ckpt_path = cfg.get("resume_from_checkpoint", None)
    if resume_ckpt_path and not os.path.isabs(resume_ckpt_path):
        # Assume relative to original working directory if not absolute
        original_cwd_resume_path = os.path.join(hydra.utils.get_original_cwd(), resume_ckpt_path)
        if os.path.exists(original_cwd_resume_path):
            resume_ckpt_path = original_cwd_resume_path
        else:
            logger.warning(f"Resume checkpoint {resume_ckpt_path} (rel to orig_cwd as {original_cwd_resume_path}) not found. Trying relative to hydra output dir.")
            # If not found, it might be relative to hydra output dir (less common for resume)
            # Or it might be an error. The trainer.load_checkpoint will handle non-existent file.

    logger.info("Starting training...")
    training_results = trainer.run_training(resume_from_checkpoint=resume_ckpt_path)

    # --- Optuna: 返回被优化的指标 ---
    # 如果在进行超参数优化 (e.g., using hydra-optuna-sweeper),
    # 这个函数需要返回 Optuna 尝试最小化/最大化的值。
    # 这个值通常是最佳验证损失或某个关键验证指标。
    # 在 `tune.yaml` 中配置 `hydra.sweeper.direction` (minimize/maximize)
    # 和在 `train.py` (或 `tune.py` 的目标函数) 中返回的指标。
    # Example: return training_results.get("best_val_metric", float('inf'))
    # The key for the metric should match what's used in checkpointing/early_stopping monitor if possible.
    # Let's assume trainer.run_training returns a dict with 'best_val_metric' which holds the value of early_stopping_cfg.monitor
    
    objective_metric_key = cfg.trainer.get('early_stopping',{}).get('monitor', 'val/loss_scaled')
    if 'best_val_metric' in training_results and training_results['best_val_metric'] is not None:
        optimized_metric = training_results['best_val_metric']
        logger.info(f"Training finished. Optuna objective metric ({objective_metric_key} via best_val_metric): {optimized_metric}")
        return optimized_metric # For Optuna
    else:
        # Fallback if early stopping wasn't used or metric not found
        # Could be last validation loss, or some other default
        # For now, let's assume if we reach here, it might be an issue or a short run.
        logger.warning(f"'best_val_metric' not found in training_results or is None. Returning default for Optuna (inf/0).")
        # Based on Optuna direction, return a very bad value
        if cfg.get('hydra',{}).get('sweeper',{}).get('direction','minimize') == 'minimize':
            return float('inf') 
        else:
            return 0.0

if __name__ == "__main__":
    # Hydra的 @hydra.main 装饰器会处理命令行参数解析和配置加载。
    # 直接运行 `python src/train.py` (如果 PYTHONPATH 设置正确) 或
    # `python -m src.train` (从项目根目录) 即可启动训练。
    # 你可以通过命令行覆盖配置参数，例如：
    # python src/train.py trainer.epochs=10 data.batch_size=16 model.d_embed=64
    main() 