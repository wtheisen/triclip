import torch
from torch import nn
import torch.nn.functional as F

import triplet_losses as TL
import contrastive_losses as CL

import numpy as np

import config as CFG
from modules import ImageEncoder, TextEncoder, VideoEncoder, ProjectionHead

import time

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
        # self.video_encoder = VideoEncoder()
        self.video_encoder = None

        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.video_projection = ProjectionHead(embedding_dim=video_embedding)

        self.temperature = temperature

    def embed(self, batch):
        with torch.no_grad():
            # image_features = self.image_encoder(batch["image"])
            # text_features = self.text_encoder(
            #     input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            # )
            # video_features = self.video_encoder(batch["video"])

            # Getting Image and Text Embeddings (with same dimension)
            image_embeddings = self.image_projection(batch["image"])
            text_embeddings = self.text_projection(batch["text"])
            video_embeddings = self.video_projection(batch["video"])
            # video_embeddings = video_embeddings.squeeze(1)

        return video_embeddings, image_embeddings, text_embeddings

    def forward(self, batch):
        # st = time.time()
        # Existing code to get embeddings
        # with torch.no_grad():
            # qt = time.time()
            # image_encodings = self.image_encoder(batch["image"])
            # xt = time.time()
            # print('image embedding time:', xt - qt)
            # qt = time.time()
            # text_encodings = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            # xt = time.time()
            # print('text embedding time:', xt - qt)
            # qt = time.time()
            # video_encodings = batch["video"]
            # xt = time.time()
            # print('video embedding time:', xt - qt)

        # et = time.time()
        # print('embedding time:', et - st)

        # st = time.time()
        image_embeddings = self.image_projection(batch["image"])
        # image_embeddings = self.image_projection(image_encodings)
        text_embeddings = self.text_projection(batch["text"])
        # text_embeddings = self.image_projection(text_encodings)
        video_embeddings = self.video_projection(batch["video"])
        # video_embeddings = self.video_projection(batch["video"]).squeeze(1)
        # et = time.time()
        # print('projecting time:', et - st)

        # Assuming batch_size is the same for all modalities

        # st = time.time()
        # total_loss = CL.contrastive_alpha(image_embeddings, text_embeddings, video_embeddings, CFG.temperature)
        # et = time.time()
        # print('loss time:', et - st)

        total_loss = TL.triplet_alfa(image_embeddings, text_embeddings, video_embeddings) 
        # total_loss = TL.triplet_bravo(image_embeddings, text_embeddings, video_embeddings) 
        # total_loss = TL.triplet_charlie(image_embeddings, text_embeddings, video_embeddings) 
        # total_loss = TL.triplet_delta(image_embeddings, text_embeddings, video_embeddings) 

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
