import torch
from torch import nn
from transformers import DistilBertModel, DistilBertConfig, VideoMAEModel, VideoMAEImageProcessor, ViTImageProcessor, ViTMAEModel, Wav2Vec2Processor, Wav2Vec2Model
import config as CFG
from decord import VideoReader, cpu
import numpy as np
import time


class AudioEncoder(nn.Module):
    """
    Encode audio to a fixed size vector
    """

    def __init__(self, model_name=CFG.audio_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)

    def forward(self, batch):
        input_values = self.processor(batch, return_tensors="pt", sampling_rate=16000).input_values

        outputs = self.model(input_values.to(CFG.device))
        return torch.mean(outputs.last_hidden_state, dim=1)


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.processor = ViTImageProcessor.from_pretrained('facebook/vit-mae-base')
        self.model = ViTMAEModel.from_pretrained('facebook/vit-mae-base')
        # self.model = timm.create_model(
        #     model_name, pretrained, num_classes=0, global_pool="avg"
        # )
        # for p in self.model.parameters():
        #     p.requires_grad = trainable

    def forward(self, x):
        # print(x.shape)
        p_v = self.processor(images=x, return_tensors="pt").pixel_values
        outputs = self.model(p_v.to(CFG.device))
        return torch.mean(outputs.last_hidden_state, dim=1)
        # return self.model(x)


class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        # if pretrained:
        self.model = DistilBertModel.from_pretrained(model_name)
        # else:
        #     self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

class VideoEncoder(nn.Module):
    def __init__(self, model_name=CFG.video_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()

        # if pretrained:
        self.model = VideoMAEModel.from_pretrained(model_name)
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name, do_rescale=False)
        
    # def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    #     converted_len = int(clip_len * frame_sample_rate)
    #     end_idx = np.random.randint(converted_len, seg_len)
    #     start_idx = end_idx - converted_len
    #     indices = np.linspace(start_idx, end_idx, num=clip_len)
    #     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    #     return indices

    # def forward(self, x):
    #     print(x.shape)
    #     pixel_values = self.processor(x, return_tensors="pt").pixel_values

    #     num_frames = 16

    #     num_patches_per_frame = (self.model.config.image_size // self.model.config.patch_size) ** 2
    #     seq_length = (num_frames // self.model.config.tubelet_size) * num_patches_per_frame
    #     bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()

    #     outputs = self.model(pixel_values, bool_masked_pos=bool_masked_pos)

    #     return torch.mean(outputs.last_hidden_state, dim=1)

    def forward(self, batch_videos):
        batch_outputs = []

        # for video in batch_videos:
        #     # Process each frame in the video
        #     processed_frames = []
        #     for frame in video:
        #         # frame is now a tensor of shape [3, 244, 244]
        #         processed_frame = self.processor(frame, return_tensors="pt").pixel_values
        #         processed_frames.append(processed_frame)

        #     # Stack the processed frames back into a single tensor
        #     pixel_values = torch.stack(processed_frames)



        for x in batch_videos:
            # Process each video in the batch
            pixel_values = self.processor(list(x), return_tensors="pt").pixel_values

            num_frames = 16
            num_patches_per_frame = (self.model.config.image_size // self.model.config.patch_size) ** 2
            seq_length = (num_frames // self.model.config.tubelet_size) * num_patches_per_frame
            bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()

            outputs = self.model(pixel_values.to(CFG.device), bool_masked_pos=bool_masked_pos.to(CFG.device))

            # Aggregate the outputs for each video
            aggregated_output = torch.mean(outputs.last_hidden_state, dim=1)
            batch_outputs.append(aggregated_output)

        # Stack the aggregated outputs for the entire batch
        return torch.stack(batch_outputs)


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x