import torch
import torch.nn.functional as F


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

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
        video_embeddings = self.video_projection(video_features).squeeze(1)

        print(image_embeddings.shape)
        print(text_embeddings.shape)
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