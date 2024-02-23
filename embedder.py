import torch

import pandas as pd
import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import DistilBertTokenizer
from decord import VideoReader, VideoLoader, bridge, cpu
from tqdm import tqdm

import config as CFG
from modules import VideoEncoder, ImageEncoder, TextEncoder

bridge.set_bridge('torch')

# def load_video(video_path):
#     """
#     Function to load a single video and its frames based on the given path.
#     Returns a tuple of the video frames and a single image frame.
#     """
#     vr = VideoReader(video_path, num_threads=32, ctx=cpu(), width=244, height=244)
#     indices = np.random.randint(0, len(vr) - 1, size=17)
#     video_frames = vr.get_batch(indices[:-1])
#     image_frame = vr[indices[-1]]
#     return video_frames, image_frame

with torch.no_grad():
    video_encoder = VideoEncoder().to(CFG.device)
    image_encoder = ImageEncoder().to(CFG.device)
    text_encoder = TextEncoder().to(CFG.device)

    dataframe = pd.read_csv(f"{CFG.captions_path}/video_captions.csv", delimiter='|')


    # for video_path in tqdm(dataframe["video_path"]):
    #     video_frames, image_frame = load_video(video_path)

    # exit()


    # with ProcessPoolExecutor(max_workers=1) as executor:
    #     # Submit all video loading jobs
    #     futures = [executor.submit(load_video, video_path) for video_path in dataframe["video_path"]]

    #     # future_to_video = {executor.submit(load_video, video_path): video_path for video_path in dataframe["video_path"]}
    #     for future in tqdm(as_completed(futures), total=len(dataframe["video_path"].values), desc='Loading videos'):
    #         video_frames, image_frame = future.result()
    #         videos.append(video_frames)
    #         images.append(image_frame)

    video_embeddings = []
    image_embeddings = []

    video_batch_size = 128

    video_path_list = list(dataframe["video_path"].values)
    for i in tqdm(range(0, len(video_path_list), video_batch_size), desc='Loading Videos'):
        video_batch = video_path_list[i:i+video_batch_size]

        vl = VideoLoader(video_batch, ctx=cpu(0), shape=(17, 244, 244, 3), interval=1, skip=1000000, shuffle=0)
        
        video_batch = []
        image_batch = []
        for batch in vl:
            video_batch.append(batch[0][:-1])
            image_batch.append(batch[0][-1])

        video_embeddings.extend(np.array(video_encoder(video_batch).squeeze().cpu()))
        image_embeddings.extend(np.array(image_encoder(image_batch).cpu()))

    # print(len(video_embeddings))
    # temp = np.asarray(video_embeddings)
    # temp_img = np.array(image_embeddings)

    np.save('video_embeddings.npy', np.asarray(video_embeddings))
    np.save('image_embeddings.npy', np.asarray(image_embeddings))
    # for video_path in tqdm(dataframe["video_path"].values, desc='Loading videos'):
    #     indices = np.random.randint(0, len(vr) - 1, size=17)
    #     videos.append(vr.get_batch(indices[:-1]))
    #     images.append(vr[indices[-1]])

    # video_embeddings = video_encoder(videos).squeeze()
    # image_embeddings = image_encoder(images)

    texts = dataframe['caption'].values

    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    text_tokens = tokenizer(
        list(texts), padding=True, truncation=True, max_length=CFG.max_length
    )
    input_ids = torch.tensor(text_tokens['input_ids'], dtype=torch.int).to(CFG.device)
    attention_masks = torch.Tensor(text_tokens['attention_mask']).to(CFG.device)
    text_embeddings = text_encoder(input_ids, attention_masks)

    np.save('text_embeddings.npy', np.array(text_embeddings.cpu()))
