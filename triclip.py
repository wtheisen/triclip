import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

import config as CFG
from modules import ImageEncoder, TextEncoder, VideoEncoder, ProjectionHead

from decord import VideoReader, cpu

class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
        video_embedding=CFG.video_embedding,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()

        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.video_projection = ProjectionHead(embedding_dim=video_embedding)

        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        video_features = self.video_encoder(batch["video"])

        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        video_embeddings = self.video_projection(video_features)

        print(image_embeddings.shape)
        print(text_embeddings.shape)
        video_embeddings = video_embeddings.squeeze(1)
        print(video_embeddings.shape)

        # Calculating the Loss
        text_image_logits = (text_embeddings @ image_embeddings.T) / self.temperature
        text_video_logits = (text_embeddings @ video_embeddings.T) / self.temperature
        image_video_logits = (image_embeddings @ video_embeddings.T) / self.temperature

        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        videos_similarity = video_embeddings @ video_embeddings.T

        text_image_targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        image_video_targets = F.softmax(
            (images_similarity + videos_similarity) / 2 * self.temperature, dim=-1
        )
        text_video_targets = F.softmax(
            (texts_similarity + videos_similarity) / 2 * self.temperature, dim=-1
        )

        # Calculate loss for text-image pairs
        texts_images_loss = cross_entropy(text_image_logits, text_image_targets, reduction='none')
        images_texts_loss = cross_entropy(text_image_logits.T, text_image_targets.T, reduction='none')

        # Calculate loss for text-video pairs
        texts_videos_loss = cross_entropy(text_video_logits, text_video_targets, reduction='none')
        videos_texts_loss = cross_entropy(text_video_logits.T, text_video_targets.T, reduction='none')

        # Calculate loss for image-video pairs
        images_videos_loss = cross_entropy(image_video_logits, image_video_targets, reduction='none')
        videos_images_loss = cross_entropy(image_video_logits.T, image_video_targets.T, reduction='none')

        # Combine the losses
        loss = (texts_images_loss + images_texts_loss + texts_videos_loss + videos_texts_loss + images_videos_loss + videos_images_loss) / 6

        print(loss.mean())
        return loss.mean() 


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

if __name__ == '__main__':
    num_samples = 64

    images = torch.randn(num_samples, 3, 224, 224)
    input_ids = torch.randint(5, 300, size=(num_samples, 25))
    attention_mask = torch.ones(num_samples, 25)
    videos = list(torch.rand(num_samples, 16, 3, 224, 224))

    batch = {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'video': videos
    }

    CLIP = CLIPModel()
    loss = CLIP(batch)
    print("SNEED")
