import torch

def triplet_loss(anchor, positive, negative):
        distance_positive = torch.norm(anchor - positive, dim=0)
        distance_negative = torch.norm(anchor - negative, dim=0)

        losses = torch.log(1 + torch.exp(distance_positive - distance_negative))

        # margin = 100
        # losses = torch.relu(distance_positive - distance_negative + margin)

        return losses.mean()

def triplet_alfa(image_embeddings, text_embeddings, video_embeddings):
    total_loss = 0.0

    # Assuming batch_size is the same for all modalities
    batch_size = image_embeddings.size(0)

    for i in range(batch_size):
        anchor_triple = (image_embeddings[i], text_embeddings[i], video_embeddings[i])
        positive_triple = anchor_triple
        negative_triple = (image_embeddings[(i + 1) % batch_size], 
                            text_embeddings[(i + 1) % batch_size],
                            video_embeddings[(i + 1) % batch_size])

        for i in range(3):
            for j in range(3):
                for q in range(3):
                    # if i != j:
                    total_loss += triplet_loss(anchor_triple[i], positive_triple[j], negative_triple[q])

    return total_loss * 0.001

def triplet_bravo(image_embeddings, text_embeddings, video_embeddings):
    total_loss = 0.0

    # Assuming batch_size is the same for all modalities
    batch_size = image_embeddings.size(0)

    for i in range(batch_size):
        # Choose anchors from each modality
        anchor_image = image_embeddings[i]
        anchor_text = text_embeddings[i]
        anchor_video = video_embeddings[i]

        # Positive examples are the corresponding items from different modalities
        positive_for_image = text_embeddings[i]  # Assuming text is the positive for image
        positive_for_text = video_embeddings[i]  # Assuming video is the positive for text
        positive_for_video = image_embeddings[i]  # Assuming image is the positive for video

        # Negative examples - randomly chosen from within the same modality but different from the anchor
        # Note: In a real scenario, ensure these are not the same as the anchor
        negative_for_image = image_embeddings[(i + 1) % batch_size]
        negative_for_text = text_embeddings[(i + 1) % batch_size]
        negative_for_video = video_embeddings[(i + 1) % batch_size]

        # Calculate triplet loss for each modality
        loss_image = triplet_loss(anchor_image, positive_for_image, negative_for_image)
        loss_text = triplet_loss(anchor_text, positive_for_text, negative_for_text)
        loss_video = triplet_loss(anchor_video, positive_for_video, negative_for_video)

        total_loss += loss_image + loss_text + loss_video

    return total_loss

def triplet_charlie(image_embeddings, text_embeddings, video_embeddings):

    # Initialize total loss
    total_loss = 0.0

    # Assuming batch_size is the same for all modalities
    batch_size = image_embeddings.size(0)

    # Loop over the batch to compute triplet loss for each example
    for i in range(batch_size):
        # Choose anchors from each modality
        anchor_image = image_embeddings[i]
        anchor_text = text_embeddings[i]
        anchor_video = video_embeddings[i]

        # Positive examples are the corresponding items from different modalities
        # positive_for_image = text_embeddings[i]  # Assuming text is the positive for image
        # positive_for_text = video_embeddings[i]  # Assuming video is the positive for text
        # positive_for_video = image_embeddings[i]  # Assuming image is the positive for video

        # Negative examples - randomly chosen from within the same modality but different from the anchor
        # Note: In a real scenario, ensure these are not the same as the anchor
        negative_for_image = image_embeddings[(i + 1) % batch_size]
        negative_for_text = text_embeddings[(i + 1) % batch_size]
        negative_for_video = video_embeddings[(i + 1) % batch_size]
        loss1 = triplet_loss(anchor_text, anchor_image, negative_for_image)
        loss2 = triplet_loss(anchor_text, anchor_video, negative_for_video)
        loss3 = triplet_loss(anchor_image, anchor_video, negative_for_text)

        # Aggregate the losses
        total_loss += loss1 + loss2 + loss3

    return total_loss

def triplet_delta(image_embeddings, text_embeddings, video_embeddings):
    total_loss = 0.0

    # Assuming batch_size is the same for all modalities
    batch_size = image_embeddings.size(0)

    # Concatenate embeddings from all modalities
    all_embeddings = torch.cat((image_embeddings, text_embeddings, video_embeddings), dim=0)

    for anchor_index in range(batch_size):
        for anchor_modality, anchor_embeddings in enumerate([image_embeddings, text_embeddings, video_embeddings]):
            anchor = anchor_embeddings[anchor_index]

            # Calculate distances from anchor to all other embeddings
            distances = torch.cdist(anchor.unsqueeze(0), all_embeddings)[0]

            # Mask to exclude embeddings from the same tuple
            exclude_same_tuple_mask = torch.arange(all_embeddings.size(0)) // batch_size != anchor_index
            valid_distances = distances[exclude_same_tuple_mask]

            if len(valid_distances) > 0:
                # Select the hardest negative from a different tuple
                hard_negative_index = torch.argmin(valid_distances)
                hard_negative = all_embeddings[exclude_same_tuple_mask][hard_negative_index]

                losses = []
                for i in range(3):
                    for j in range(3):
                        for q in range(3):
                            # if i != j:
                            losses.append(triplet_loss(anchor[i], anchor[j], hard_negative[q]))
                                # total_loss += triplet_loss(anchor[i], anchor[j], hard_negative[q])
                total_loss += sum(losses) / len(losses)

    return total_loss / batch_size
