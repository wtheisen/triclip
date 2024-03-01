import torch
import torch.nn.functional as F

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def contrastive_alpha(image_embeddings, text_embeddings, video_embeddings, temperature):

        # Calculating the Loss
        text_image_logits = (text_embeddings @ image_embeddings.T) / temperature
        text_video_logits = (text_embeddings @ video_embeddings.T) / temperature
        image_video_logits = (image_embeddings @ video_embeddings.T) / temperature

        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        videos_similarity = video_embeddings @ video_embeddings.T

        text_image_targets = F.softmax(
            (texts_similarity + images_similarity) / 2 * temperature, dim=-1
        )
        image_video_targets = F.softmax(
            (images_similarity + videos_similarity) / 2 * temperature, dim=-1
        )
        text_video_targets = F.softmax(
            (texts_similarity + videos_similarity) / 2 * temperature, dim=-1
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

        return loss.mean() 

def contrastive_alpha_quad(image_embeddings, text_embeddings, video_embeddings, audio_embeddings, temperature):
    # Calculating the Logits
    text_image_logits = (text_embeddings @ image_embeddings.T) / temperature
    text_video_logits = (text_embeddings @ video_embeddings.T) / temperature
    image_video_logits = (image_embeddings @ video_embeddings.T) / temperature
    text_audio_logits = (text_embeddings @ audio_embeddings.T) / temperature
    image_audio_logits = (image_embeddings @ audio_embeddings.T) / temperature
    video_audio_logits = (video_embeddings @ audio_embeddings.T) / temperature

    # Calculating the Similarities
    images_similarity = image_embeddings @ image_embeddings.T
    texts_similarity = text_embeddings @ text_embeddings.T
    videos_similarity = video_embeddings @ video_embeddings.T
    audios_similarity = audio_embeddings @ audio_embeddings.T

    # Calculating the Targets
    text_image_targets = F.softmax(
        (texts_similarity + images_similarity) / 2 * temperature, dim=-1
    )
    image_video_targets = F.softmax(
        (images_similarity + videos_similarity) / 2 * temperature, dim=-1
    )
    text_video_targets = F.softmax(
        (texts_similarity + videos_similarity) / 2 * temperature, dim=-1
    )
    text_audio_targets = F.softmax(
        (texts_similarity + audios_similarity) / 2 * temperature, dim=-1
    )
    image_audio_targets = F.softmax(
        (images_similarity + audios_similarity) / 2 * temperature, dim=-1
    )
    video_audio_targets = F.softmax(
        (videos_similarity + audios_similarity) / 2 * temperature, dim=-1
    )

    # Calculate loss for all pairs
    texts_images_loss = F.cross_entropy(text_image_logits, text_image_targets, reduction='none')
    images_texts_loss = F.cross_entropy(text_image_logits.T, text_image_targets.T, reduction='none')
    texts_videos_loss = F.cross_entropy(text_video_logits, text_video_targets, reduction='none')
    videos_texts_loss = F.cross_entropy(text_video_logits.T, text_video_targets.T, reduction='none')
    images_videos_loss = F.cross_entropy(image_video_logits, image_video_targets, reduction='none')
    videos_images_loss = F.cross_entropy(image_video_logits.T, image_video_targets.T, reduction='none')
    texts_audios_loss = F.cross_entropy(text_audio_logits, text_audio_targets, reduction='none')
    audios_texts_loss = F.cross_entropy(text_audio_logits.T, text_audio_targets.T, reduction='none')
    images_audios_loss = F.cross_entropy(image_audio_logits, image_audio_targets, reduction='none')
    audios_images_loss = F.cross_entropy(image_audio_logits.T, image_audio_targets.T, reduction='none')
    videos_audios_loss = F.cross_entropy(video_audio_logits, video_audio_targets, reduction='none')
    audios_videos_loss = F.cross_entropy(video_audio_logits.T, video_audio_targets.T, reduction='none')

    # Combine the losses
    total_loss = (texts_images_loss + images_texts_loss + texts_videos_loss + videos_texts_loss +
                  images_videos_loss + videos_images_loss + texts_audios_loss + audios_texts_loss +
                  images_audios_loss + audios_images_loss + videos_audios_loss + audios_videos_loss) / 12

    return total_loss.mean()