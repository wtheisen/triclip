import json
import torch
import random

import pandas as pd
import numpy as np
import config as CFG

from dataset import CLIPTriplets, get_transforms

from collections import Counter
from decord import VideoReader, bridge, gpu, cpu
from tqdm import tqdm
from modules import VideoEncoder

bridge.set_bridge('torch')
video_encoder = VideoEncoder().to(CFG.device)

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def make_train_valid_dfs(size):
    print('Sneed')
    #dataframe = pd.read_csv(f"{CFG.captions_path}/fixed_video_captions.csv", delimiter='|')
    dataframe = pd.read_csv(f"{CFG.captions_path}/fixed_video_captions.csv")

    text_embeddings = np.load('text_embeddings.npy')
    # print(text_embeddings.shape)
    # print(text_embeddings)
    # exit()
    image_embeddings = np.load('image_embeddings.npy')
    video_embeddings = np.load('video_embeddings.npy')
 

    # users = [path.split('/')[5] for path in dataframe["video_path"]]
    # c = Counter(users)
    # print(c)
    # print(len(c.keys()))
    # exit()

    #max_id = len(image_embeddings)
    image_ids = list(dataframe["id"].values)[:51000]
    random.shuffle(image_ids)

    total_holdout = CFG.num_val + CFG.num_test

    np.random.seed(42)
    train_ids = np.random.choice(
        image_ids[:-total_holdout], size=size, replace=False
    )

    train_ids.sort()
    # test_ids = image_ids[-CFG.num_test:]

    #train_ids = [id_ for id_ in image_ids if id_ not in val_test_ids]
    # random.shuffle(train_ids)
    #train_ids = train_ids[:CFG.num_train]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    train_dataframe = train_dataframe.assign(text_embedding=[text_embeddings[x] for x in train_ids])
    train_dataframe = train_dataframe.assign(image_embedding=[image_embeddings[x] for x in train_ids])
    train_dataframe = train_dataframe.assign(video_embedding=[video_embeddings[x] for x in train_ids])


    # print(train_dataframe.columns.values)
    # for i in train_dataframe.iterrows():
    #     print(i[1]['text_embedding'].shape, i[1]['image_embedding'].shape, i[1]['video_embedding'].shape)
    # exit()

    val_test_ids = image_ids[-total_holdout:]
    random.shuffle(val_test_ids)

    val_ids = val_test_ids[:CFG.num_val]
    val_ids.sort()

    test_ids = val_test_ids[CFG.num_test:]
    test_ids.sort()
    #val_ids = val_test_ids[:len(val_test_ids)//2]
    # val_ids = val_ids[:CFG.num_val]

    #test_ids = val_test_ids[len(val_test_ids)//2:]
    # test_ids = test_ids[:CFG.num_test]

    # max_id = dataframe["id"].max() + 1
    # image_ids = np.arange(0, max_id)

    # np.random.seed(42)
    # val_test_ids = np.random.choice(
    #     image_ids, size=int(0.2 * len(image_ids)), replace=False
    # )

    # train_ids = [id_ for id_ in image_ids if id_ not in val_test_ids]
    # random.shuffle(train_ids)
    # train_ids = train_ids[:CFG.num_train]

    # train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)

    # val_ids = val_test_ids[:len(val_test_ids)//2]
    # val_ids = val_ids[:CFG.num_val]

    # test_ids = val_test_ids[len(val_test_ids)//2:]
    # test_ids = test_ids[:CFG.num_test]

    # videos = dataframe['video_path'].values[:10],
    # videos = videos[0].tolist()
    # print(videos)
    # videos = VideoLoader(videos,
    #                             ctx=[cpu(0)], 
    #                             shape=(10, 320, 240, 3),
    #                             interval=1,
    #                             skip=4,
    #                             shuffle=0)

    # print('seethe buddy')

    val_dataframe = dataframe[dataframe["id"].isin(val_ids)].reset_index(drop=True)
    val_dataframe = val_dataframe.assign(text_embedding=[text_embeddings[x] for x in val_ids])
    val_dataframe = val_dataframe.assign(image_embedding=[image_embeddings[x] for x in val_ids])
    val_dataframe = val_dataframe.assign(video_embedding=[video_embeddings[x] for x in val_ids])

    test_dataframe = dataframe[dataframe["id"].isin(test_ids)].reset_index(drop=True)
    test_dataframe = test_dataframe.assign(text_embedding=[text_embeddings[x] for x in test_ids])
    test_dataframe = test_dataframe.assign(image_embedding=[image_embeddings[x] for x in test_ids])
    test_dataframe = test_dataframe.assign(video_embedding=[video_embeddings[x] for x in test_ids])

    return train_dataframe, val_dataframe, test_dataframe

def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)

    # dataset = CLIPTriplets(
    #     dataframe["image"].values,
    #     dataframe["input_ids"].values,
    #     dataframe["attention_masks"].values,
    #     dataframe["video"].values
    # )


    # video_features = self.video_encoder(batch["video"])
    # videos = []
    # images = []
    # for video_path in tqdm(dataframe["video_path"].values, desc='Loading videos'):
    #     vr = VideoReader(video_path, num_threads=8, ctx=cpu(0), width=244, height=244)
    #     indices = np.random.randint(0, len(vr) - 1, size=17)
    #     videos.append(vr.get_batch(indices[:-1]))
    #     images.append(vr[indices[-1]])

    with torch.no_grad():
        dataset = CLIPTriplets(
            dataframe["id"].values,
            dataframe["caption"].values,
            dataframe["video_path"].values,
            # videoreader = VideoReader(self.videos[idx], num_threads=8, ctx=cpu(0), width=244, height=244)
            # [VideoReader(x, num_threads=8, ctx=gpu(0), width=244, height=244) for x in dataframe["video_path"]],
            dataframe['video_embedding'].values,
            dataframe['image_embedding'].values,
            dataframe['text_embedding'].values,
            # images,
            # dataframe["video_path"],
            # tokenizer=tokenizer,
            # transforms=transforms,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def account_stance_mapper():
    account_cluster_df = pd.read_csv('./account_cluster_map.csv')
    cluster_stance_df = pd.read_csv('./cluster_stance_map.csv')
    account_stance_dict = {}

    for _, row in account_cluster_df.iterrows():
        account = row[0]
        cluster = row[1]

        if cluster != 0 and cluster != 31:
            account_stance_dict[account] = cluster_stance_df.iloc[cluster]['stance']

    print(account_stance_dict)

    with open('stances.py', 'w') as f:
        json.dump(account_stance_dict, f)


