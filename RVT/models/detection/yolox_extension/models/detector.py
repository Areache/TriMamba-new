from typing import Any, Optional, Tuple, Union, Dict, List

import torch as th
from omegaconf import DictConfig

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from ...recurrent_backbone import build_recurrent_backbone
from .build import build_yolox_fpn, build_yolox_head
from utils.timers import TimerDummy as CudaTimer

from data.utils.types import BackboneFeatures, LstmStates
from enum import Enum, auto

from modules.utils.detection import (
    BackboneFeatureSelector,
    EventReprSelector,
    RNNStates,
    Mode,
    mode_2_string,
    merge_mixed_batches,
)
from data.utils.types import DataType, LstmStates, ObjDetOutput, DatasetSamplingMode
class YoloXDetector(th.nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        backbone_cfg = model_cfg.backbone
        fpn_cfg = model_cfg.fpn
        head_cfg = model_cfg.head

        self.backbone = build_recurrent_backbone(backbone_cfg)

        in_channels = self.backbone.get_stage_dims(fpn_cfg.in_stages)
        self.fpn = build_yolox_fpn(fpn_cfg, in_channels=in_channels)

        strides = self.backbone.get_strides(fpn_cfg.in_stages)
        self.yolox_head = build_yolox_head(
            head_cfg, in_channels=in_channels, strides=strides
        )

    def forward_backbone(
        self,
        x: th.Tensor,
        # previous_states: Optional[LstmStates] = None,
        token_mask: Optional[th.Tensor] = None,
        train_step: bool = True,
    ) -> Tuple[BackboneFeatures, LstmStates]:
        with CudaTimer(device=x.device, timer_name="Backbone"):
            # backbone_features, states = self.backbone(
            backbone_features = self.backbone(
                # x, previous_states, token_mask, train_step
                x, token_mask, train_step
            )
        return backbone_features
        # return backbone_features, states

    def forward_detect(
        self, backbone_features: BackboneFeatures, targets: Optional[th.Tensor] = None
    ) -> Tuple[th.Tensor, Union[Dict[str, th.Tensor], None]]:
        device = next(iter(backbone_features.values())).device
        with CudaTimer(device=device, timer_name="FPN"):
            fpn_features = self.fpn(backbone_features)
        # if self.training:
        #     assert targets is not None
        #     with CudaTimer(device=device, timer_name="HEAD + Loss"):
        #         outputs, losses = self.yolox_head(fpn_features, targets)
        #     return outputs, losses
        with CudaTimer(device=device, timer_name="HEAD"):
            outputs, losses = self.yolox_head(fpn_features)
        assert losses is None
        return outputs, losses

    def get_worker_id_from_batch(self, batch: Any) -> int:
        return batch["worker_id"]

    def get_data_from_batch(self, batch: Any):
        return batch["data"]

    # def forward(
    #     self,
    #     # batch: Any,
    #     x: th.Tensor,
    #     # mode: Mode,
    #     # previous_states: Optional[LstmStates] = None,
    #     retrieve_detections: bool = True,
    #     targets: Optional[th.Tensor] = None,
    # ) -> Tuple[Union[th.Tensor, None], Union[Dict[str, th.Tensor], None], LstmStates]:
    #     # # print(x.shape) #[1, 12, 20, 360, 640]
    #     # # backbone_features, states = self.forward_backbone(x, previous_states,train_step=False)
    #     # data = self.get_data_from_batch(batch)
    #     # worker_id = self.get_worker_id_from_batch(batch)

    #     # assert mode in (Mode.VAL, Mode.TEST)
    #     # ev_tensor_sequence = data[DataType.EV_REPR]
    #     # sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
    #     # is_first_sample = data[DataType.IS_FIRST_SAMPLE]

    #     # self.mode_2_rnn_states[mode].reset(
    #     #     worker_id=worker_id, indices_or_bool_tensor=is_first_sample
    #     # )

    #     # sequence_len = len(ev_tensor_sequence)
    #     # assert sequence_len > 0
    #     # batch_size = len(sparse_obj_labels[0])
    #     # if self.mode_2_batch_size[mode] is None:
    #     #     self.mode_2_batch_size[mode] = batch_size
    #     # else:
    #     #     assert self.mode_2_batch_size[mode] == batch_size

    #     # prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
    #     # backbone_feature_selector = BackboneFeatureSelector()
    #     # ev_repr_selector = EventReprSelector()
    #     # obj_labels = list()

    #     # ev_tensor_sequence = th.stack(
    #     #     ev_tensor_sequence
    #     # )  # shape: (sequence_len, batch_size, channels, height, width) = (L, B, C, H, W)
    #     # ev_tensor_sequence = ev_tensor_sequence.to(dtype=self.dtype)
    #     # ev_tensor_sequence = self.input_padder.pad_tensor_ev_repr(ev_tensor_sequence)

    #     # if self.mode_2_hw[mode] is None:
    #     #     self.mode_2_hw[mode] = tuple(ev_tensor_sequence.shape[-2:])
    #     # else:
    #     #     assert self.mode_2_hw[mode] == ev_tensor_sequence.shape[-2:]
    #     import pdb; pdb.set_trace()
    #     backbone_features = self.forward_backbone(x, train_step=False)
    #     outputs, losses = None, None
    #     # backbone_features = {
    #     #     k: v.squeeze(0) for k, v in backbone_features.items()
    #     # }
    #     # backbone_features = {
    #     #     k: v.squeeze(0) if v.dim() == 5 else v  # 仅当维度为5时压缩（避免意外操作）
    #     #     for k, v in backbone_features.items()
    #     # }
    #     if not retrieve_detections:
    #         assert targets is None
    #         return outputs, losses
    #     # if not retrieve_detections:
    #     #     assert targets is None
    #     #     # 返回相同维度的空结果（避免 None 导致 ONNX 问题）
    #     #     empty_output = {k: th.empty(0) for k in backbone_features.keys()}
    #     #     return None, empty_output
    #     # for tidx in range(sequence_len):
    #     #     collect_predictions = (tidx == sequence_len - 1) or (
    #     #         self.mode_2_sampling_mode[mode] == DatasetSamplingMode.STREAM
    #     #     )

    #     #     if collect_predictions:
    #     #         current_labels, valid_batch_indices = sparse_obj_labels[
    #     #             tidx
    #     #         ].get_valid_labels_and_batch_indices()
    #     #         # Store backbone features that correspond to the available labels.
    #     #         if len(current_labels) > 0:
    #     #             backbone_feature_selector.add_backbone_features(
    #     #                 backbone_features={
    #     #                     k: v[tidx] for k, v in backbone_features.items()
    #     #                 },
    #     #                 selected_indices=valid_batch_indices,
    #     #             )

    #     #             obj_labels.extend(current_labels)
    #     #             ev_repr_selector.add_event_representations(
    #     #                 event_representations=ev_tensor_sequence[tidx],
    #     #                 selected_indices=valid_batch_indices,
    #     #             )
    #     # # self.mode_2_rnn_states[mode].save_states_and_detach(
    #     # #     worker_id=worker_id, states=prev_states
    #     # # )
    #     # if len(obj_labels) == 0:
    #     #     return {ObjDetOutput.SKIP_VIZ: True}

    #     # selected_backbone_features = (
    #     #     backbone_feature_selector.get_batched_backbone_features()
    #     # )
    #     outputs, losses = self.forward_detect(
    #         backbone_features=backbone_features, targets=targets
    #     )
    #     # torch.Size([60, 5040, 8])
    #     return outputs, losses

    def forward(
        self,
        x: th.Tensor,
        # previous_states: Optional[LstmStates] = None,
        retrieve_detections: bool = True,
        targets: Optional[th.Tensor] = None,
    ) -> Tuple[Union[th.Tensor, None], Union[Dict[str, th.Tensor], None], LstmStates]:
        # import pdb; pdb.set_trace()
        # backbone_features, states = self.forward_backbone(x, previous_states)
        backbone_features = self.forward_backbone(x, train_step=False)
        outputs, losses = None, None
        if not retrieve_detections:
            assert targets is None
            return outputs, losses
        backbone_features = {
             k: v.flatten(0, 1) for k, v in backbone_features.items()}
        outputs, losses = self.forward_detect(
            backbone_features=backbone_features, targets=targets
        )
        return outputs, losses