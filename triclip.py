import torch
from torch import nn
import torch.nn.functional as F

import triplet_losses as TL
import contrastive_losses as CL

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

    def embed(self, batch):
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        video_features = self.video_encoder(batch["video"])

        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        video_embeddings = self.video_projection(video_features)
        video_embeddings = video_embeddings.squeeze(1)

        return video_embeddings, image_embeddings, text_embeddings

    def forward(self, batch):
        # Existing code to get embeddings
        with torch.no_grad():
            image_encodings = self.image_encoder(batch["image"])
            text_encodings = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            video_encodings = self.video_encoder(batch["video"])

        image_embeddings = self.image_projection(image_encodings)
        text_embeddings = self.text_projection(text_encodings)
        video_embeddings = self.video_projection(video_encodings).squeeze(1)

        # Assuming batch_size is the same for all modalities
        batch_size = image_embeddings.size(0)

        # total_loss = CL.contrastive_alpha(image_embeddings, text_embeddings, video_embeddings)

        # total_loss = TL.triplet_alfa(image_embeddings, text_embeddings, video_embeddings) 
        # total_loss = TL.triplet_bravo(image_embeddings, text_embeddings, video_embeddings) 
        # total_loss = TL.triplet_charlie(image_embeddings, text_embeddings, video_embeddings) 
        total_loss = TL.triplet_delta(image_embeddings, text_embeddings, video_embeddings) 

        # Average the loss over the batch
        total_loss /= batch_size

        return total_loss


if __name__ == '__main__':
    num_samples = 8

    images = torch.randn(num_samples, 3, 224, 224)
    input_ids = torch.randint(5, 300, size=(num_samples, 25))
    attention_mask = torch.ones(num_samples, 25)
    videos = torch.rand(num_samples, 16, 3, 224, 224)

    batch = {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'video': videos
    }

    CLIP = CLIPModel().to(CFG.device)
    loss = CLIP({k: v.to(CFG.device) if k != "video" else v for k, v in batch.items()})
    print("SNEED")
