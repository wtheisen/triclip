import csv
import torch
import sys

import pandas as pd
import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import DistilBertTokenizer
from decord import AVReader, bridge, cpu
from tqdm import tqdm

import config as CFG
from modules import VideoEncoder, ImageEncoder, TextEncoder, AudioEncoder

bridge.set_bridge('torch')

# def load_video(video_path):
#     """
#     Function to load a single video and its frames based on the given path.
#     Returns a tuple of the video frames and a single image frame.
#     """
#     return video_frames, image_frame

with torch.no_grad():
    video_encoder = VideoEncoder().to(CFG.device)
    audio_encoder = AudioEncoder().to(CFG.device)
    image_encoder = ImageEncoder().to(CFG.device)
    text_encoder = TextEncoder().to(CFG.device)
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)

    dataframe = pd.read_csv(sys.argv[1], delimiter='|')


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
    audio_embeddings = []
    image_embeddings = []
    text_embeddings = []

    valid_rows = [('id', 'video_path', 'caption')]
    video_batch_size = 128

    dataframe_tuple_list = [row for row in dataframe.itertuples()][:10]

    valid_ids = []
    for i in tqdm(dataframe_tuple_list, desc='Loading Videos'):
    #for i in tqdm(range(0, len(dataframe_tuple_list), video_batch_size), desc='Loading Videos'):

        # video_batch = [x[2] for x in dataframe_tuple_list[i:i+video_batch_size]]

        # vl = VideoLoader(video_batch, ctx=[cpu(0)], shape=(17, 244, 244, 3), interval=1, skip=1000000, shuffle=0)

        # vr = VideoReader(i[2], num_threads=8, ctx=cpu(), width=244, height=244)
        avr = AVReader(i[2], num_threads=8, ctx=cpu(), width=244, height=244, sample_rate=16000)
        indices = np.random.randint(0, len(avr) - 1, size=17)
        audio_frames, video_frames = avr.get_batch(indices[:-1])
        _, image_frame = avr[indices[-1]]


        # for batch in vl:
        valid_ids.append(i[0])
            # print('file:', batch[1].tolist()[0][0], batch[0].shape)
            # video_batch.append(batch[0][:-1])
            # image_batch.append(batch[0][-1])

        v_e = np.array(video_encoder([video_frames]).squeeze().cpu())
        video_embeddings.extend([v_e])

        a_s = torch.cat(audio_frames, dim=1).squeeze()
        a_e = np.array(audio_encoder(a_s).squeeze().cpu())
        print(a_e.shape)
        audio_embeddings.extend([a_e])

        i_e = np.array(image_encoder(image_frame).cpu())
        image_embeddings.extend(i_e)

        # caption_batch = [x[3] for x in dataframe_tuple_list[i:i+video_batch_size]]

        text_tokens = tokenizer(
            [i[2]], padding=True, truncation=True, max_length=CFG.max_length
        )

        input_ids = torch.tensor(text_tokens['input_ids'], dtype=torch.int).to(CFG.device)
        attention_masks = torch.Tensor(text_tokens['attention_mask']).to(CFG.device)
        t_e = np.array(text_encoder(input_ids, attention_masks).cpu())
        text_embeddings.extend(t_e)

        valid_rows.extend(i)

    # save_tag = sys.argv[1].split('/')[1].split('_')[3].split('.')[0]
    # np.save(f'video_embeddings_{save_tag}.npy', np.asarray(video_embeddings))
    # np.save(f'audio_embeddings_{save_tag}.npy', np.asarray(audio_embeddings))
    # np.save(f'image_embeddings_{save_tag}.npy', np.asarray(image_embeddings))
    print(np.asarray(image_embeddings).shape)
    print(np.asarray(audio_embeddings).shape)
    print(np.asarray(video_embeddings).shape)
    print(np.asarray(text_embeddings).shape)
    #print(len(valid_rows))
    # np.save(f'text_embeddings_{save_tag}.npy', np.array(text_embeddings))

    # Writing to the CSV file
    # with open( f'valid_rows_{save_tag}.csv', 'w+', newline='') as csvfile:
    # with open( f'valid_rows.csv', 'w+', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows(valid_rows)
        
