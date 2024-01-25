import os
import cv2
import torch
import albumentations as A

import config as CFG
import numpy as np

from decord import VideoReader, cpu

class CLIPTriplets(torch.utils.data.Dataset):
    def __init__(self, caption, video_path, tokenizer, transforms):
        self.images = ''
        self.captions = caption
        self.encoded_captions = tokenizer(
            list(caption), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.videos = video_path
        self.transforms = transforms

    # def __getitem__(self, idx):
    #     return {
    #         'image': self.images[idx],
    #         'input_ids': self.input_ids[idx],
    #         'attention_mask': self.attention_masks[idx],
    #         'video': self.videos[idx]
    #     }

    def preprocess_video(self, video_frames):
        # Reorder dimensions to bring the color channel to the second position
        video_frames = video_frames.transpose(0, 3, 1, 2)

        # Initialize an empty array for the resized frames
        resized_frames = np.empty((16, 3, 244, 244), dtype=np.uint8)

        # Resize each frame
        for i in range(16):
            for c in range(3):  # Iterate over each color channel
                resized_frames[i, c] = cv2.resize(video_frames[i, c], (244, 244), interpolation=cv2.INTER_AREA)

        return resized_frames

    def sample_frame_indices(self, clip_len, frame_sample_rate, seg_len):
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
        selected_image_frame = start_idx
        while selected_image_frame >= start_idx and selected_image_frame <= end_idx:
            selected_image_frame = np.random.randint(0, seg_len)

        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices, selected_image_frame

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        #get video frames and take an image out of the unchosen one
        videoreader = VideoReader(self.videos[idx], num_threads=1, ctx=cpu(0))
        videoreader.seek(0)
        indices, image_frame = self.sample_frame_indices(clip_len=16, frame_sample_rate=4, seg_len=len(videoreader))
        video = videoreader.get_batch(indices).asnumpy()
        item['video'] = self.preprocess_video(video)

        image = videoreader[image_frame].asnumpy()
        #preprocess video

        # image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image.astype(np.uint8)).permute(2, 0, 1).float()
        # item['caption'] = self.captions[idx]
        item['input_ids'] = np.asarray(self.encoded_captions['input_ids'][idx])
        item['attention_mask'] = np.asarray(self.encoded_captions['attention_mask'][idx])

        return item

    def __len__(self):
        return len(self.captions)

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)



def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )