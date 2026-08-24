import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import (TransformerLayerSequence,
                                         build_positional_encoding)
from torch import nn
from torch.nn.init import normal_

from ..builder import HEADS


class FutureStateSynthesizer(TransformerLayerSequence):
    """Transformer decoder used to synthesize a future image state."""

    def __init__(self,
                 *args,
                 post_norm_cfg=dict(type='LN'),
                 return_intermediate=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.post_norm = None
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]

    def forward(self, query, *args, **kwargs):
        if not self.return_intermediate:
            x = super().forward(query, *args, **kwargs)
            if self.post_norm:
                x = self.post_norm(x)[None]
            return x

        intermediate = []
        for layer in self.layers:
            query = layer(query, *args, **kwargs)
            if self.post_norm is not None:
                intermediate.append(self.post_norm(query))
            else:
                intermediate.append(query)
        return torch.stack(intermediate)


@HEADS.register_module()
class ForecastingModule(nn.Module):
    """Forecast future multi-camera image features from temporal context."""

    def __init__(self,
                 num_layers=4,
                 scale_dims=(64, 64, 128),
                 num_cams=6,
                 embed_dims=256,
                 num_heads=32,
                 mlp_ratio=4,
                 num_context=4,
                 num_future_frames=1):
        super().__init__()
        self._validate_configuration(
            num_layers=num_layers,
            scale_dims=scale_dims,
            num_cams=num_cams,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_context=num_context,
            num_future_frames=num_future_frames)

        self.num_layers = num_layers
        self.scale_channels = sum(scale_dims)
        self.embed_dims = embed_dims
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.scale_dims = tuple(scale_dims)
        self.num_context = num_context
        self.num_cams = num_cams
        self.num_future_frames = num_future_frames

        self.positional_encoding = build_positional_encoding(
            dict(type='SinePositionalEncoding',
                 num_feats=self.scale_channels // 2,
                 normalize=True))

        self.module_a = nn.ModuleList()
        self.future_embed = nn.ModuleList()
        self.temporal_embed = nn.ParameterList()
        self.cam_embed = nn.ParameterList()
        self.scale_factor = nn.ParameterList()

        for _ in range(self.num_future_frames):
            self.module_a.append(
                self._build_future_state_synthesizer(
                    self.num_layers, self.embed_dims, self.num_heads,
                    self.mlp_ratio))

            self.temporal_embed.append(
                nn.Parameter(torch.zeros(self.num_context, self.embed_dims, 1,
                                         1)))
            self.cam_embed.append(
                nn.Parameter(torch.zeros(self.num_cams, self.embed_dims, 1,
                                         1)))
            self.scale_factor.append(nn.Parameter(torch.tensor(1.0)))

            self.future_embed.append(self._build_future_projection())

        self.scale_embed = nn.ParameterList([
            nn.Parameter(torch.zeros(self.num_future_frames, dim))
            for dim in self.scale_dims
        ])
        for scale_embed in self.scale_embed:
            for horizon_idx in range(len(scale_embed)):
                normal_(scale_embed[horizon_idx])

        for temporal_embed in self.temporal_embed:
            normal_(temporal_embed)

        for camera_embed in self.cam_embed:
            normal_(camera_embed)

        self.init_weights()

    @staticmethod
    def _validate_configuration(num_layers, scale_dims, num_cams, embed_dims,
                                num_heads, mlp_ratio, num_context,
                                num_future_frames):
        integer_fields = {
            'num_layers': num_layers,
            'num_cams': num_cams,
            'embed_dims': embed_dims,
            'num_heads': num_heads,
            'mlp_ratio': mlp_ratio,
            'num_context': num_context,
            'num_future_frames': num_future_frames,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f'{name} must be an integer.')
            if value <= 0:
                raise ValueError(f'{name} must be positive.')

        if not isinstance(scale_dims, (list, tuple)) or not scale_dims:
            raise TypeError('scale_dims must be a non-empty sequence.')
        if any(isinstance(dim, bool) or not isinstance(dim, int)
               for dim in scale_dims):
            raise TypeError('scale_dims must contain integers.')
        if any(dim <= 0 for dim in scale_dims):
            raise ValueError('scale_dims must contain positive values.')
        if sum(scale_dims) != embed_dims:
            raise ValueError(
                'The sum of scale_dims must equal embed_dims; got '
                f'{sum(scale_dims)} and {embed_dims}.')
        if embed_dims % 2:
            raise ValueError('embed_dims must be even for positional encoding.')
        if embed_dims % num_heads:
            raise ValueError('embed_dims must be divisible by num_heads.')

    @staticmethod
    def _build_future_state_synthesizer(num_layers, embed_dims, num_heads,
                                        mlp_ratio):
        return FutureStateSynthesizer(
            return_intermediate=True,
            num_layers=num_layers,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=embed_dims,
                    feedforward_channels=embed_dims * mlp_ratio,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    with_cp=True,
                    add_identity=True),
                feedforward_channels=embed_dims * mlp_ratio,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None)

    def _build_future_projection(self):
        return nn.Sequential(
            nn.Linear(self.embed_dims, self.scale_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.scale_channels, self.scale_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.scale_channels, self.scale_channels))

    def init_weights(self):
        for parameter in self.module_a.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_normal_(parameter)

    def _add_scale_embedding(self, features, horizon_idx):
        batch_size, num_frames, _, height, width = features.shape
        scale_features = torch.split(features, self.scale_dims, dim=2)
        embedded_features = []
        for scale_idx, scale_feature in enumerate(scale_features):
            scale_embed = self.scale_embed[scale_idx][horizon_idx].view(
                1, 1, self.scale_dims[scale_idx], 1, 1)
            scale_embed = scale_embed.expand(batch_size, num_frames, -1,
                                             height, width)
            embedded_features.append(scale_feature + scale_embed)
        return torch.cat(embedded_features, dim=2)

    def _add_contextual_embedding(self, past_features, current_features,
                                  horizon_idx):
        context_features = torch.cat([past_features, current_features], dim=1)
        batch_size = context_features.shape[0]
        temporal_ids = torch.arange(self.num_context).repeat_interleave(
            self.num_cams).to(context_features.device)
        camera_ids = torch.arange(self.num_cams).repeat(
            self.num_context).to(context_features.device)
        temporal_embed = self.temporal_embed[horizon_idx][temporal_ids]
        camera_embed = self.cam_embed[horizon_idx][camera_ids]
        contextual_embed = temporal_embed + camera_embed
        contextual_embed = contextual_embed.unsqueeze(0).expand(
            batch_size, -1, -1, -1, -1)
        return context_features + contextual_embed

    def _init_future_state_queries(self, current_features):
        batch_size, _, channels, height, width = current_features.shape
        return current_features.permute(3, 4, 0, 1, 2).reshape(
            height * width, batch_size * self.num_cams, channels)

    def _validate_inputs(self, past_features, current_features):
        for name, features in (('past_feats', past_features),
                               ('curr_feats', current_features)):
            if not isinstance(features, torch.Tensor):
                raise TypeError(f'{name} must be a torch.Tensor.')
            if features.ndim != 5:
                raise ValueError(
                    f'{name} must have shape (B, T, C, H, W); got '
                    f'{tuple(features.shape)}.')

        if past_features.shape[0] != current_features.shape[0]:
            raise ValueError('past_feats and curr_feats batch sizes must match.')
        if past_features.shape[2:] != current_features.shape[2:]:
            raise ValueError(
                'past_feats and curr_feats channel and spatial dimensions '
                'must match.')
        if past_features.device != current_features.device:
            raise ValueError('past_feats and curr_feats must share a device.')
        if past_features.dtype != current_features.dtype:
            raise ValueError('past_feats and curr_feats must share a dtype.')

        expected_past_frames = (self.num_context - 1) * self.num_cams
        if past_features.shape[1] != expected_past_frames:
            raise ValueError(
                f'past_feats must contain {expected_past_frames} camera '
                f'features; got {past_features.shape[1]}.')
        if current_features.shape[1] != self.num_cams:
            raise ValueError(
                f'curr_feats must contain {self.num_cams} camera features; '
                f'got {current_features.shape[1]}.')
        if current_features.shape[2] != self.embed_dims:
            raise ValueError(
                f'Feature channels must equal embed_dims={self.embed_dims}; '
                f'got {current_features.shape[2]}.')

    def _build_positional_encoding(self, context_features):
        batch_size = context_features.shape[0]
        height = context_features.shape[-2]
        width = context_features.shape[-1]
        positional_mask = context_features.new_zeros(
            (batch_size * self.num_cams, height, width), dtype=torch.bool)
        return self.positional_encoding(positional_mask).flatten(2).permute(
            2, 0, 1)

    def _synthesize_future_state(self, query_features, context_features,
                                 positional_encoding, horizon_idx,
                                 output_channels):
        batch_size, _, channels, height, width = context_features.shape
        intermediate_features = []

        for layer_idx in range(self.num_layers):
            context_idx = layer_idx % self.num_context
            context_start = context_idx * self.num_cams
            context_end = context_start + self.num_cams
            key_value_features = context_features[:, context_start:
                                                  context_end]
            key_value_features = key_value_features.permute(3, 4, 0, 1, 2)
            key_value_features = key_value_features.reshape(
                -1, batch_size * self.num_cams, channels)

            layer = self.module_a[horizon_idx].layers[layer_idx]
            query_features = layer(
                query=query_features,
                key=key_value_features,
                value=key_value_features,
                query_pos=positional_encoding,
                key_pos=positional_encoding,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None)

            future_features = self.module_a[horizon_idx].post_norm(
                query_features)
            future_features = self.future_embed[horizon_idx](future_features)
            future_features = (
                future_features * self.scale_factor[horizon_idx])
            intermediate_features.append(
                future_features.permute(1, 2, 0).contiguous().reshape(
                    batch_size, self.num_cams, output_channels, height,
                    width))

        return query_features, intermediate_features

    @staticmethod
    def _merge_horizon_features(all_horizon_features, horizon_features):
        if not all_horizon_features:
            return horizon_features
        return [
            torch.cat([previous, current], dim=0)
            for previous, current in zip(all_horizon_features,
                                         horizon_features)
        ]

    def forward(self, past_feats, curr_feats):
        self._validate_inputs(past_feats, curr_feats)
        _, _, output_channels, _, _ = past_feats.shape
        current_features_original = curr_feats

        all_horizon_features = []
        for horizon_idx in range(self.num_future_frames):
            past_features = self._add_scale_embedding(past_feats, horizon_idx)
            current_features = self._add_scale_embedding(
                curr_feats, horizon_idx)
            context_features = self._add_contextual_embedding(
                past_features, current_features, horizon_idx)

            query_features = self._init_future_state_queries(curr_feats)

            positional_encoding = self._build_positional_encoding(
                context_features)

            query_features, horizon_features = self._synthesize_future_state(
                query_features, context_features, positional_encoding,
                horizon_idx, output_channels)
            all_horizon_features = self._merge_horizon_features(
                all_horizon_features, horizon_features)

        output_features = torch.cat(
            [current_features_original, all_horizon_features[-1]], dim=0)
        return output_features, all_horizon_features
