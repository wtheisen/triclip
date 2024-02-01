import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from transformers import DistilBertTokenizer

import config as CFG
from dataset import CLIPTriplets, CLIPDataset, get_transforms
from triclip import CLIPModel
from utils import AvgMeter, get_lr
from sklearn.metrics import average_precision_score, recall_score

def test_retrieval(model, test_loader, num_trials=10):
    tqdm_object = tqdm(test_loader, total=len(test_loader))

    embed_dict = {}
    embed_list = []

    with torch.no_grad():
        for batch in tqdm_object:
            gpu_batch = {k: v.to(CFG.device) for k, v in batch.items()}
            vid_embed, img_embed, txt_embed = model.embed(gpu_batch)

            for num in range(0, len(vid_embed)):
                embed_dict[batch["id"][num]] = (
                    vid_embed[num],
                    img_embed[num],
                    txt_embed[num]
                )

                embed_list.append(vid_embed[num])
                embed_list.append(img_embed[num])
                embed_list.append(txt_embed[num])

    for recall in [1, 5, 10, 25]:
        trial_accuracy = 0

        for _ in range(0, num_trials):
            # pick random embedding from all of them
            test_id = np.random.randint(0, len(embed_list))
            test_embedding = embed_list[test_id]

            # compute the cosine similarity of that random embedding to all others
            scores = []
            for embedding in embed_list:
                if embedding is test_embedding:
                    continue

                sim = nn.functional.cosine_similarity(test_embedding, embedding, dim=0)
                scores.append((sim, embedding))

            # find the id of the triplet the randomly selected embedding was from
            true_id = None
            for id, triple in embed_dict.items():
                for t in triple:
                    if torch.equal(test_embedding.to(CFG.device), t):
                        true_id = id

            # find the triplet ids of the N closest embeddings
            closest_ids = []
            for score, embedding in sorted(scores, key=lambda x: x[0], reverse=True)[:recall]:
                for id, triple in embed_dict.items():
                    for t in triple:
                        if torch.equal(embedding.to(CFG.device), t):
                            closest_ids.append(id)

            # print(true_id, closest_ids)
            for id in closest_ids:
                if id == true_id:
                    trial_accuracy += 0.5

        print(f'Recall @ {recall}', trial_accuracy / num_trials)


def make_train_valid_dfs():
    dataframe = pd.read_csv(f"{CFG.captions_path}/video_captions.csv", delimiter='|')
    max_id = dataframe["id"].max() + 1 if not CFG.debug else 1000
    image_ids = np.arange(0, max_id)

    np.random.seed(42)
    val_test_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )

    train_ids = [id_ for id_ in image_ids if id_ not in val_test_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)

    val_ids = val_test_ids[:len(val_test_ids)//2]
    test_ids = val_test_ids[len(val_test_ids)//2:]
    test_ids = test_ids[:20]
    val_dataframe = dataframe[dataframe["id"].isin(val_ids)].reset_index(drop=True)
    test_dataframe = dataframe[dataframe["id"].isin(test_ids)].reset_index(drop=True)

    return train_dataframe, val_dataframe, test_dataframe


def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)

    # dataset = CLIPTriplets(
    #     dataframe["image"].values,
    #     dataframe["input_ids"].values,
    #     dataframe["attention_masks"].values,
    #     dataframe["video"].values
    # )

    dataset = CLIPTriplets(
        dataframe["id"].values,
        dataframe["caption"].values,
        dataframe["video_path"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def generate_dummy_data(num_samples):
    dataframe = pd.DataFrame(columns=["id", "image", "input_ids", "attention_masks", "video"])

    sample_size = 1
    for samp_num in range(0, num_samples):
        dataframe.loc[samp_num] = [
            samp_num,
            torch.randn(3, 224, 224),
            torch.randint(5, 300, size=(25,)),
            torch.ones(25),
            torch.rand(16, 3, 224, 224)
        ]

    max_id = dataframe["id"].max() + 1 if not CFG.debug else 1000
    image_ids = np.arange(0, max_id)

    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )

    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]

    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)

    return train_dataframe, valid_dataframe, dataframe

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))


    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items()}
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
        batch = {k: v.to(CFG.device) for k, v in batch.items()}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():
    train_df, valid_df, test_df = make_train_valid_dfs()
    # train_df, valid_df, _ = generate_dummy_data(1000)

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
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")

    test_loader = build_loaders(test_df, tokenizer, mode="valid")

    model.eval()
    with torch.no_grad():
        test_retrieval(best_model, test_loader)

if __name__ == "__main__":
    main()