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


# python /capstor/scratch/cscs/cyujie/ssms_event_cameras/RVT/onnx.py dataset=gen4 dataset.path=/capstor/scratch/cscs/cyujie/dataset/gen4 \
# checkpoint=/capstor/scratch/cscs/cyujie/ssms_event_cameras/onnx/onnx_test.ckpt use_test_set=1 hardware.gpus=0 +experiment/gen4="small.yaml" \
# batch_size.eval=12 model.postprocess.confidence_threshold=0.001

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
    module = Module.load_from_checkpoint(str(ckpt_path), **{"full_config": config})

    # ---------------------
    # Callbacks and Misc
    # ---------------------
    callbacks = [ModelSummary(max_depth=2)]
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
    # ---------------------
    # Validation
    # ---------------------

     # 转换的onnx格式的名称，文件后缀需为.onnx
    onnx_file_name = "dvs_ssm.onnx"
        # 我们需要转换的模型，将torch_model设置为自己的模型
        # model = torch_model
        # 加载权重，将model.pth转换为自己的模型权重
        # 如果模型的权重是使用多卡训练出来，我们需要去除权重中多的module. 具体操作可以见5.4节
        # model = model.load_state_dict(torch.load("model.pth"))
    model = module
    # 导出模型前，必须调用model.eval()或者model.train(False)
    model.eval()
    # dummy_input就是一个输入的实例，仅提供输入shape、type等信息 
    # import pdb; pdb.set_trace()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 1 # 随机的取值，当设置dynamic_axes后影响不大
    # dummy_input = get_dummy_input_from_dataloader(data_module)
    dummy_input = torch.randn(1, 12, 20, 384, 640, requires_grad=True).to(device) 
    # dummy_input = torch.randn(1, 12, 20, 360, 640, requires_grad=True).to(device) 
    
    # dummy_input = torch.randn(batch_size , 48, 20, 7, 7, requires_grad=True).to(device) 

    # 这组输入对应的模型输出
    # output = model(dummy_input)
    # out, _ = module.mdl.forward_backbone(x=dummy_input, previous_states=None, train_step=False)
    # 导出模型
    torch.onnx.export(model,        # 模型的名称
                    dummy_input,   # 一组实例化输入
                    onnx_file_name,   # 文件保存路径/名称
                    export_params=True,        #  如果指定为True或默认, 参数也会被导出. 如果你要导出一个没训练过的就设为 False.
                    opset_version=16,          # ONNX 算子集的版本，当前已更新到15
                    do_constant_folding=True,  # 是否执行常量折叠优化
                    input_names = ['input'],   # 输入模型的张量的名称
                    output_names = ['output'], # 输出模型的张量的名称
                    # dynamic_axes将batch_size的维度指定为动态，
                    # 后续进行推理的数据可以与导出的dummy_input的batch_size不同
                    dynamic_axes={'input' : {0 : 'batch_size'},    
                                    'output' : {0 : 'batch_size'}})
   
def get_dummy_input_from_dataloader(data_module):
    # 获取测试数据加载器
    test_dataloader = data_module.test_dataloader()
    # print(test_dataloader.dtype)
    # 获取第一个批次的数据
    for batch in test_dataloader:
        # 假设batch是一个元组，包含输入数据和标签
        inputs, _ = batch
        dummy_input = inputs
        break  # 只需要获取一个批次
    return dummy_input

if __name__ == "__main__":
    main()



