# Copyright (c) Phigent Robotics. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.models.backbones import VoVNet
from mmdet.models import DETECTORS
from mmdet.models.builder import build_loss

from .. import builder
from .bevdet import BEVStereo4D


@DETECTORS.register_module()
class ForecastOcc(BEVStereo4D):
    """Forecast semantic occupancy from current and historical camera views."""

    def __init__(self,
                 loss_occ=None,
                 out_dim=32,
                 num_classes=18,
                 use_predicter=True,
                 forecast_head=None,
                 future_state_alignment_loss=None,
                 num_future_frames=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim
        out_channels = out_dim if use_predicter else num_classes
        self.final_conv = ConvModule(
            self.img_view_transformer.out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv3d'))
        self.use_predicter = use_predicter
        if use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, num_classes),
            )
        self.pts_bbox_head = None
        self.num_classes = num_classes
        self.loss_occ = build_loss(loss_occ)
        self.align_after_view_transfromation = False

        if forecast_head is None:
            raise ValueError('forecast_head must be provided for ForecastOcc.')
        if (isinstance(num_future_frames, bool)
                or not isinstance(num_future_frames, int)
                or num_future_frames < 1):
            raise ValueError('num_future_frames must be a positive integer.')
        self.forecast_head = builder.build_head(forecast_head)
        if self.forecast_head.num_future_frames != num_future_frames:
            raise ValueError(
                'forecast_head.num_future_frames must match '
                'model.num_future_frames.')

        self.future_state_alignment_loss = future_state_alignment_loss
        if future_state_alignment_loss is not None:
            self.future_state_alignment_weight = \
                future_state_alignment_loss['weight']
            self._freeze_parameters(self.img_backbone)
            self._freeze_parameters(self.img_neck)

        self.extra_ref_frames_ = 2
        self.num_future_frames = num_future_frames

    @staticmethod
    def _freeze_parameters(module):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    @force_fp32()
    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x, feats = self.img_bev_encoder_neck(x)
        return x, feats

    def image_encoder(self, img, stereo=False):
        batch_size, num_cams, channels, image_height, image_width = img.shape
        img = img.view(batch_size * num_cams, channels, image_height,
                       image_width)
        x = self.img_backbone(img)
        if isinstance(self.img_backbone, VoVNet):
            backbone_features = []
            for stage in self.img_backbone._out_features:
                backbone_features.append(x[stage])
            x = tuple(backbone_features)
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]

        assert self.with_img_neck, \
            'ForecastOcc requires an image neck.'
        if self.with_img_neck:
            if len(self.img_neck.in_channels) < 4:
                x = x[1:]
            x = self.img_neck(x)
            if isinstance(x, (list, tuple)):
                x = x[0]
        _, output_dim, output_height, output_width = x.shape
        x = x.view(batch_size, num_cams, output_dim, output_height,
                   output_width)
        return x, stereo_feat

    def prepare_bev_feat(self, img, sensor2keyego, ego2global, intrin,
                         post_rot, post_tran, bda, mlp_input, feat_prev_iv,
                         k2s_sensor, extra_ref_frame):
        if extra_ref_frame:
            x, stereo_feat = self.image_encoder(img, stereo=True)
            return None, None, stereo_feat, x
        x, stereo_feat = self.image_encoder(img, stereo=True)
        metas = dict(k2s_sensor=k2s_sensor,
                     intrins=intrin,
                     post_rots=post_rot,
                     post_trans=post_tran,
                     frustum=self.img_view_transformer.cv_frustum.to(x),
                     cv_downsample=4,
                     downsample=self.img_view_transformer.downsample,
                     grid_config=self.img_view_transformer.grid_config,
                     cv_feat_list=[feat_prev_iv, stereo_feat])
        bev_feat, depth = self.img_view_transformer(
            [x, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda,
             mlp_input], metas)
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth, stereo_feat, x

    def _prepare_current_and_future_inputs(self, img_inputs, test_mode):
        prepared_inputs = []
        batch_size, num_images, channels, image_height, image_width = \
            img_inputs[0].shape
        num_cams = self.forecast_head.num_cams
        if num_images % num_cams:
            raise ValueError(
                f'img_inputs contains {num_images} images, which is not '
                f'divisible by the configured {num_cams} cameras.')
        source_frame_count = num_images // num_cams
        training_frame_count = self.num_frame + self.num_future_frames
        if test_mode:
            if source_frame_count != self.num_frame:
                raise ValueError(
                    f'ForecastOcc inference requires {self.num_frame} current '
                    f'and historical frames per camera, but received '
                    f'{source_frame_count}.')
            return img_inputs
        if source_frame_count != training_frame_count:
            raise ValueError(
                f'ForecastOcc training requires {training_frame_count} '
                f'current, historical, and future frames per camera, but '
                f'received {source_frame_count}.')

        for input_idx, input_tensor in enumerate(img_inputs):
            if input_idx == 0:
                image_sequence = input_tensor.view(
                    batch_size, num_cams, source_frame_count, channels,
                    image_height, image_width)
                future_sequences = []
                for horizon_idx in range(self.num_future_frames):
                    future_image = image_sequence[
                        :, :, self.num_frame + horizon_idx:
                        self.num_frame + horizon_idx + 1]
                    future_sequences.append(
                        torch.cat([
                            future_image,
                            image_sequence[:, :, :self.num_frame - 1]
                        ], dim=2))

                prepared_images = torch.cat([
                    image_sequence[:, :, :self.num_frame], *future_sequences
                ], dim=0)
                prepared_images = prepared_images.view(
                    prepared_images.shape[0], -1, channels, image_height,
                    image_width)
                prepared_inputs.append(prepared_images)
            elif input_tensor.shape[1] == num_images:
                future_sequences = []
                for horizon_idx in range(self.num_future_frames):
                    future_start = (self.num_frame + horizon_idx) * num_cams
                    future_end = future_start + num_cams
                    future_frame = input_tensor[:, future_start:future_end]
                    future_sequences.append(
                        torch.cat([
                            future_frame,
                            input_tensor[:, :(self.num_frame - 1) * num_cams]
                        ], dim=1))
                prepared_inputs.append(
                    torch.cat([
                        input_tensor[:, :self.num_frame * num_cams],
                        *future_sequences
                    ], dim=0))
            else:
                prepared_inputs.append(
                    torch.cat([input_tensor] *
                              (1 + self.num_future_frames), dim=0))

        return prepared_inputs

    @staticmethod
    def _camera_parameters(sensor2keyegos, ego2globals, intrins, post_rots,
                           post_trans, bda):
        return [
            sensor2keyegos[0], ego2globals[0], intrins[0], post_rots[0],
            post_trans[0], bda
        ]

    def _extract_image_context(self, images, sensor2keyegos, ego2globals,
                               intrins, post_rots, post_trans, bda,
                               curr2adjsensor, batch_size):
        bev_features = []
        past_image_features = []
        previous_stereo_features = None
        current_image_features = None
        current_depth = None

        for frame_idx in range(self.num_frame - 2, -1, -1):
            frame_images = images[frame_idx][:batch_size]
            sensor2keyego = sensor2keyegos[frame_idx][:batch_size]
            ego2global = ego2globals[frame_idx][:batch_size]
            intrin = intrins[frame_idx][:batch_size]
            post_rot = post_rots[frame_idx][:batch_size]
            post_tran = post_trans[frame_idx][:batch_size]
            is_current_frame = frame_idx == 0
            is_extra_reference = (
                frame_idx == self.num_frame - self.extra_ref_frames_)

            if not is_current_frame and not self.with_prev:
                continue

            mlp_input = self.img_view_transformer.get_mlp_input(
                sensor2keyegos[0][:batch_size],
                ego2globals[0][:batch_size], intrin, post_rot, post_tran,
                bda[:batch_size])
            frame_inputs = (
                frame_images, sensor2keyego, ego2global, intrin, post_rot,
                post_tran, bda[:batch_size], mlp_input,
                previous_stereo_features,
                curr2adjsensor[frame_idx][:batch_size], is_extra_reference)

            if is_current_frame:
                bev_feature, depth, stereo_features, image_features = \
                    self.prepare_bev_feat(*frame_inputs)
            else:
                with torch.no_grad():
                    bev_feature, depth, stereo_features, image_features = \
                        self.prepare_bev_feat(*frame_inputs)

            if is_current_frame:
                current_depth = depth
                current_image_features = image_features
            else:
                past_image_features.append(image_features)

            if not is_extra_reference:
                bev_features.append(bev_feature)
            previous_stereo_features = stereo_features

        with torch.no_grad():
            oldest_image_features, _ = self.image_encoder(
                images[-1][:batch_size], stereo=True)
        past_image_features.insert(0, oldest_image_features)

        return (bev_features, past_image_features, current_image_features,
                current_depth)

    def _encode_future_image_targets(self, images, batch_size, test_mode):
        if test_mode:
            return None
        with torch.no_grad():
            future_image_features, _ = self.image_encoder(
                images[0][batch_size:], stereo=True)
        return future_image_features

    def _forecast_image_features(self, past_image_features,
                                 current_image_features):
        image_context = torch.cat(past_image_features, dim=1)
        return self.forecast_head(image_context, current_image_features)

    def _lift_future_image_features(self, forecast_features, sensor2keyegos,
                                    intrins, post_rots, post_trans, bda,
                                    batch_size):
        view_transformer_meta = dict(
            cv_downsample=4,
            downsample=self.img_view_transformer.downsample,
            cv_feat_list=[None])
        mlp_input = self.img_view_transformer.get_mlp_input(
            sensor2keyegos[0][:batch_size], None,
            intrins[0][:batch_size], post_rots[0][:batch_size],
            post_trans[0][:batch_size], bda[:batch_size])
        mlp_input = torch.cat(
            [mlp_input] * self.num_future_frames, dim=0)

        future_bev_features, future_depth = self.img_view_transformer(
            [
                forecast_features[batch_size:],
                sensor2keyegos[0][:batch_size], None,
                intrins[0][:batch_size], post_rots[0][:batch_size],
                post_trans[0][:batch_size], bda[:batch_size], mlp_input
            ], view_transformer_meta)
        if self.pre_process:
            future_bev_features = self.pre_process_net(
                future_bev_features)[0]
        return future_bev_features, future_depth

    def _pad_missing_bev_history(self, bev_features):
        if self.with_prev:
            return bev_features

        current_bev_features = bev_features[0]
        missing_frames = self.num_frame - self.extra_ref_frames - 1
        if current_bev_features.ndim == 4:
            batch_size, channels, height, width = current_bev_features.shape
            empty_history = current_bev_features.new_zeros(
                (batch_size, channels * missing_frames, height, width))
        else:
            batch_size, channels, depth, height, width = \
                current_bev_features.shape
            empty_history = current_bev_features.new_zeros(
                (batch_size, channels * missing_frames, depth, height, width))
        return [empty_history, current_bev_features]

    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         test_mode=False,
                         **kwargs):
        assert not sequential, (
            'Sequential image feature extraction is not supported.')
        assert not pred_prev, (
            'Previous-frame prediction is not supported.')
        assert not self.align_after_view_transfromation, (
            'Post-view-transformation alignment is not supported by '
            'ForecastOcc.')

        self.img_backbone.eval()
        self.img_neck.eval()
        batch_size = img[0].shape[0]
        if batch_size != 1:
            raise ValueError(
                'ForecastOcc currently requires samples_per_gpu=1.')
        img = self._prepare_current_and_future_inputs(img, test_mode)
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
            bda, curr2adjsensor = self.prepare_inputs(img, stereo=True)
        camera_parameters = self._camera_parameters(
            sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda)

        bev_features, past_image_features, current_image_features, \
            current_depth = self._extract_image_context(
                imgs, sensor2keyegos, ego2globals, intrins, post_rots,
                post_trans, bda, curr2adjsensor, batch_size)
        future_image_features_gt = self._encode_future_image_targets(
            imgs, batch_size, test_mode)
        forecast_features, predicted_future_features = \
            self._forecast_image_features(past_image_features,
                                          current_image_features)
        output_image_features = forecast_features.clone()
        future_bev_features, future_depth = \
            self._lift_future_image_features(
                forecast_features, sensor2keyegos, intrins, post_rots,
                post_trans, bda, batch_size)
        bev_features = self._pad_missing_bev_history(bev_features)

        for frame_idx in range(len(bev_features) - 1):
            bev_features[frame_idx] = torch.cat(
                [bev_features[frame_idx], bev_features[frame_idx + 1]],
                dim=1)
        current_bev_features = bev_features.pop()
        for horizon_idx in range(self.num_future_frames):
            horizon_start = horizon_idx * batch_size
            horizon_end = horizon_start + batch_size
            bev_features.append(
                torch.cat([
                    current_bev_features,
                    future_bev_features[horizon_start:horizon_end]
                ], dim=1))

        encoded_bev, multi_scale_bev_features = self.bev_encoder(
            torch.cat(bev_features, dim=0))
        depth = torch.cat([current_depth, future_depth], dim=0)

        image_features = [
            encoded_bev, multi_scale_bev_features, output_image_features
        ]
        if self.future_state_alignment_loss is not None and not test_mode:
            alignment_losses = self.future_state_alignment_loss_fn(
                predicted_future_features, future_image_features_gt)
            image_features.append(alignment_losses)

        return image_features, depth, img_metas, camera_parameters

    def extract_feat(self, points, img, img_metas, test_mode=False, **kwargs):
        """Extract features from images and points."""
        img_feats, depth, img_metas, cam_params = self.extract_img_feat(
            img, img_metas, test_mode=test_mode, **kwargs)
        pts_feats = None
        return img_feats, pts_feats, depth, img_metas, cam_params

    def combined_loss(self,
                      pred,
                      target,
                      beta=2.0,
                      weight_huber=1.0,
                      weight_cos=1.0):
        huber = F.smooth_l1_loss(pred, target, beta=beta)

        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        cosine = 1 - F.cosine_similarity(pred_flat, target_flat, dim=1).mean()

        return weight_huber * huber + weight_cos * cosine

    def loss_single(self, voxel_semantics, preds):
        voxel_semantics = voxel_semantics.long().reshape(-1)
        preds = preds.reshape(-1, self.num_classes)
        return {'loss_occ': self.loss_occ(preds, voxel_semantics)}

    def _future_state_alignment_loss_single(self, predicted_future_features,
                                            future_features):
        loss_type = self.future_state_alignment_loss['type']
        if loss_type == 'mse':
            return F.mse_loss(predicted_future_features, future_features)
        if loss_type == 'huber':
            return F.smooth_l1_loss(
                predicted_future_features,
                future_features,
                beta=self.future_state_alignment_loss.get('beta', 1.0))
        if loss_type == 'huber_cosine':
            return self.combined_loss(
                predicted_future_features,
                future_features,
                beta=self.future_state_alignment_loss.get('beta', 1.0),
                weight_huber=self.future_state_alignment_loss.get(
                    'weight_huber', 1.0),
                weight_cos=self.future_state_alignment_loss.get(
                    'weight_cos', 1.0))
        return F.mse_loss(predicted_future_features, future_features)

    def future_state_alignment_loss_fn(self, predicted_future_features,
                                       future_features):
        batch_size = predicted_future_features[0].shape[0]
        layer_losses = []
        for predicted_features in predicted_future_features:
            sample_losses = [
                self._future_state_alignment_loss_single(
                    predicted_features[batch_idx],
                    future_features[batch_idx])
                for batch_idx in range(batch_size)
            ]
            layer_losses.append(torch.stack(sample_losses).mean())

        losses = {
            'loss_consistency':
            layer_losses[-1] * self.future_state_alignment_weight
        }
        for layer_idx, layer_loss in enumerate(layer_losses[:-1]):
            losses[f'd{layer_idx}.loss_consistency'] = (
                layer_loss * self.future_state_alignment_weight)
        return losses

    def _semantic_occupancy_logits(self, bev_features):
        occupancy_logits = self.final_conv(bev_features)
        occupancy_logits = occupancy_logits.permute(0, 4, 3, 2, 1)
        if self.use_predicter:
            occupancy_logits = self.predicter(occupancy_logits)
        return occupancy_logits

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Run occupancy forecasting without test-time augmentation."""
        kwargs.pop('test_mode', None)
        image_features, _, _, img_metas, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, test_mode=True, **kwargs)
        occupancy_logits = self._semantic_occupancy_logits(image_features[0])
        occupancy = occupancy_logits.softmax(-1).argmax(-1)
        occupancy = occupancy.cpu().numpy().astype(np.uint8)

        batch_size = len(img_metas)
        expected_outputs = batch_size * (1 + self.num_future_frames)
        if len(occupancy) != expected_outputs:
            raise RuntimeError(
                f'ForecastOcc produced {len(occupancy)} occupancy grids; '
                f'expected {expected_outputs}.')
        future_occupancy = occupancy[batch_size:].reshape(
            self.num_future_frames, batch_size, *occupancy.shape[1:])

        return [
            {'future_occ': future_occupancy[:, sample_idx]}
            for sample_idx in range(len(img_metas))
        ]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        image_features, _, depth, _, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        losses = {}

        if self.future_state_alignment_loss:
            alignment_losses = image_features[-1]
            image_features = image_features[:-1]
            losses.update(alignment_losses)

        future_fields = {
            name: kwargs[name]
            for name in (
                'future_voxel_semantics', 'future_gt_depth')
        }
        for name, values in future_fields.items():
            if not isinstance(values, list):
                raise TypeError(f'{name} must be a list of future horizons.')
            if len(values) != self.num_future_frames:
                raise ValueError(
                    f'{name} has {len(values)} horizons; expected '
                    f'{self.num_future_frames}.')

        voxel_semantics = torch.cat([
            kwargs['voxel_semantics'],
            *future_fields['future_voxel_semantics']
        ], dim=0)
        gt_depth = torch.cat(
            [kwargs['gt_depth'], *future_fields['future_gt_depth']], dim=0)

        losses['loss_depth'] = self.img_view_transformer.get_depth_loss(
            gt_depth, depth)

        occupancy_logits = self._semantic_occupancy_logits(image_features[0])
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        losses.update(self.loss_single(voxel_semantics, occupancy_logits))
        return losses

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        image_features, _, _, _, _ = self.extract_feat(
            points,
            img=img_inputs,
            img_metas=img_metas,
            test_mode=True,
            **kwargs)
        return self._semantic_occupancy_logits(image_features[0])
