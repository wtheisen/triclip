import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter

import torch
from torch import nn
from transformers import DistilBertTokenizer

import config as CFG
from dataset import CLIPTriplets, CLIPDataset, get_transforms
from triclip import CLIPModel
from utils import AvgMeter, get_lr, make_train_valid_dfs, build_loaders

from decord import VideoReader, cpu

import time

def test_retrieval(model, test_loader, num_trials=300):
    tqdm_object = tqdm(test_loader, total=len(test_loader))

    embed_dict = {}
    vid_embed_list = []
    img_embed_list = []
    txt_embed_list = []
    combined_embed_list = []

    modalities = ['VID', 'IMG', 'TXT']

    with torch.no_grad():
        for batch in tqdm_object:
            # gpu_batch = {k: v.to(CFG.device) if k != 'video_path' else v for k, v in batch.items()}
            # vid_embed, img_embed, txt_embed = model.embed(gpu_batch)
            vid_embed = batch["video"]
            img_embed = batch["image"]
            txt_embed = batch["text"]

            # video_embeddings = self.video_projection(video_features)
            # vid_embed = batch['video']
            for num in range(0, len(vid_embed)):
                embed_dict[batch["id"][num]] = (
                    vid_embed[num],
                    img_embed[num],
                    txt_embed[num]
                )

                vid_embed_list.append(vid_embed[num])
                img_embed_list.append(img_embed[num])
                txt_embed_list.append(txt_embed[num])

                combined_embed_list.append(vid_embed[num])
                combined_embed_list.append(img_embed[num])
                combined_embed_list.append(txt_embed[num])

    # for recall in [5]:

    recall_dict = {
        1: 0,
        5: 0,
        10: 0,
        25: 0
    }

    modality_correct = defaultdict(Counter)
    modality_counts = Counter()

    for _ in tqdm(range(0, num_trials), desc='Test trials'):
        # pick random embedding from all of them
        test_id = np.random.randint(0, len(combined_embed_list))
        test_embedding = combined_embed_list[test_id]

        # find the id of the triplet the randomly selected embedding was from
        true_id = None
        for id, triple in embed_dict.items():
            for i in range(3):
                if torch.equal(test_embedding, triple[i]):
                    true_id = (id, modalities[i])

        modality_counts[true_id[1]] += 1

        if true_id[1] == 'TXT':
            search_space = [vid_embed_list, img_embed_list]
        elif true_id[1] == 'IMG':
            search_space = [vid_embed_list, txt_embed_list]
        elif true_id[1] == 'VID':
            search_space = [txt_embed_list, img_embed_list]

        # search_space = [txt_embed_list, img_embed_list, vid_embed_list]
        # compute the cosine similarity of that random embedding to all others
        scores = defaultdict(list)
        for i, embed_list in enumerate(search_space):
            for embedding in embed_list:
                sim = nn.functional.cosine_similarity(test_embedding, embedding, dim=0)
                scores[modalities[i]].append((sim, embedding))

        # find the triplet ids of the N closest embeddings
        # for each of the three modalities find the N closest items
        # if the IDs are shared amongst the lists combine them into a single item
        closest_ids = defaultdict(list)
        for modality, score_list in scores.items():
            for score, embedding in sorted(score_list, key=lambda x: x[0], reverse=True):
                for id, triple in embed_dict.items():
                    for i in range(3):
                        if torch.equal(embedding, triple[i]):
                            closest_ids[modalities[i]].append((id, score))

        combined_id_scores = Counter()
        for modality, scores in closest_ids.items():
            # if modality == true_id[1]:
            #     continue

            for score in scores:
                combined_id_scores[score[0]] += score[1]

        for recall in [1, 5, 10, 25]:
            for rank, id_score in enumerate(combined_id_scores.most_common(recall)):
                if id_score[0] == true_id[0]:
                    recall_dict[recall] += 1
                    modality_correct[true_id[1]][recall] += 1

    print(f'Contrastive Alfa@ {num_trials} trials:')
    print(f'\tTrain/Val/Test: {CFG.num_train}/{CFG.num_val}/{CFG.num_test} - {CFG.epochs} Epochs')
    for recall, hits in recall_dict.items():
        print(f'\t\tRecall @ {recall}',  hits / num_trials)
    
    for recall, hits in recall_dict.items():
        recall_dict[recall] = hits / num_trials

    for modality, total_num in modality_counts.items():
        print(f'% correct for {modality} out of {total_num}:')

        for k, recall in modality_correct[modality].items():
            print(f'@{k}: {recall / total_num}')

    # print(modality_counts)
    # print(modality_correct)
    return recall_dict

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))


    for batch in tqdm_object:
        # batch = {k: v.to(CFG.device) for k, v in batch.items()}
        batch = {k: v.to(CFG.device) if k != 'video_path' else v for k, v in batch.items()}

        loss = model(batch)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) if k != 'video_path' else v for k, v in batch.items()}

        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():
    st = time.time()
    train_df, valid_df, test_df = make_train_valid_dfs()

    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")


    model = CLIPModel().to(CFG.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float('inf')
    best_model = None

    for epoch in tqdm(range(CFG.epochs), desc='Epochs'):
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)

        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            best_model = model

    torch.save(model.state_dict(), f"{CFG.num_train}t_{CFG.epochs}e_best.pt")
    print("Wrote Best Model...")

    et = time.time()
    test_loader = build_loaders(test_df, tokenizer, mode="valid")

    model.eval()
    with torch.no_grad():
        return test_retrieval(best_model, test_loader), et - st

if __name__ == "__main__":
    main()