import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from pathlib import Path

import torch
from torch.backends import cuda, cudnn
import torch.onnx 

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy("file_system")

import hydra
import hdf5plugin
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelSummary

from config.modifier import dynamically_modify_train_config
from modules.utils.fetch import fetch_data_module, fetch_model_module
from modules.detection import Module


@hydra.main(config_path="config", config_name="val", version_base="1.2")
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    print("------ Configuration ------")
    print(OmegaConf.to_yaml(config))
    print("---------------------------")

    # ---------------------
    # GPU options
    # ---------------------
    gpus = config.hardware.gpus
    assert isinstance(gpus, int), "no more than 1 GPU supported"
    gpus = [gpus]

    # ---------------------
    # Data
    # ---------------------
    data_module = fetch_data_module(config=config)

    # ---------------------
    # Logging and Checkpoints
    # ---------------------
    logger = CSVLogger(save_dir="./validation_logs")
    ckpt_path = Path(config.checkpoint)

    # ---------------------
    # Model
    # ---------------------

    module = fetch_model_module(config=config)
    # from lightning.pytorch.utilities.cloud_io import load as pl_load
    # ckpt = pl_load(str(ckpt_path))
    # module = Module(**{"full_config": config})
    # module.load_state_dict(ckpt["state_dict"], strict=False)
    # checkpoint = torch.load(str(ckpt_path), map_location="cpu")

    # # 加载参数（strict=False 忽略不匹配的 key）
    # model = Module(full_config=config)
    # model.load_state_dict(checkpoint["state_dict"], strict=False)
    module = Module.load_from_checkpoint(str(ckpt_path), **{"full_config": config}, strict=False)
    # import pdb; pdb.set_trace()
    # ---------------------
    # Callbacks and Misc
    # ---------------------
    callbacks = [ModelSummary(max_depth=2)]

    # ---------------------
    # Validation
    # ---------------------

    trainer = pl.Trainer(
        accelerator="gpu",
        callbacks=callbacks,
        default_root_dir=None,
        devices=gpus,
        logger=logger,
        log_every_n_steps=100,
        precision=config.training.precision,
        # move_metrics_to_cpu=False,
    )
    # import pdb; pdb.set_trace()
    with torch.inference_mode():
        if config.use_test_set:
            trainer.test(model=module, datamodule=data_module, ckpt_path=str(ckpt_path))
        else:
            trainer.validate(
                model=module, datamodule=data_module, ckpt_path=str(ckpt_path)
            )


if __name__ == "__main__":
    main()
