import pandas as pd
from decord import VideoReader, cpu
from tqdm import tqdm

dataframe = pd.read_csv(f"./video_captions.csv", delimiter='|')

invalid = []
for _, row in tqdm(dataframe.iterrows()):
    vr = VideoReader(row['video_path'], num_threads=8, ctx=cpu(0), width=244, height=244)