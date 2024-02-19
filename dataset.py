import os
import cv2
import torch
import albumentations as A

import config as CFG
import numpy as np

from decord import VideoReader, VideoLoader, cpu, gpu, bridge

bridge.set_bridge('torch')

class CLIPTriplets(torch.utils.data.Dataset):
    def __init__(self, ids, caption, video_path, tokenizer, transforms):
        self.ids = ids
        self.images = ''
        self.captions = caption
        self.encoded_captions = tokenizer(
            list(caption), padding=True, truncation=True, max_length=CFG.max_length
        )
        # self.video_paths = video_path
        self.videos = video_path
        # print(self.video_paths)
            #     self.videos = VideoLoader(self.video_paths.tolist(), 
            #                             ctx=[cpu(0)], 
            #                             shape=(len(self.video_paths), 320, 240, 3),
            #                             interval=1,
            #                             skip=4,
            #                             shuffle=0)
            # except:
            #     print(v_p)
            #     exit(69)

        # print('dilate')
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
        item['video_path'] = self.videos[idx]

        #get video frames and take an image out of the unchosen one
        videoreader = VideoReader(self.videos[idx], num_threads=4, ctx=cpu(0), width=244, height=244)
        videoreader.seek(0)

        # try:
        #     indices, image_frame = self.sample_frame_indices(clip_len=16, frame_sample_rate=4, seg_len=len(videoreader))
        # except:
        #     print(self.videos[idx])
        #     exit(69)

        indices = np.random.randint(0, len(videoreader) - 1, size=17)

        # video = videoreader.get_batch(indices[:-1]).asnumpy()
        try:
            item['video'] = videoreader.get_batch(indices[:-1])
        except:
            print(self.videos[idx])
            print('Error Reading Batch')
            item['video'] = None

        # item['video'] = self.preprocess_video(video)
        # item['video'] = video.transpose(0, 3, 1, 2)

        # item['image'] = videoreader[indices[-1]].asnumpy()
        item['image'] = videoreader[indices[-1]]
        #preprocess video

        # image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = self.transforms(image=image)['image']
        # item['image'] = torch.tensor(image.astype(np.uint8)).permute(2, 0, 1).float()
        # item['caption'] = self.captions[idx]
        item['input_ids'] = np.asarray(self.encoded_captions['input_ids'][idx])
        item['attention_mask'] = np.asarray(self.encoded_captions['attention_mask'][idx])

        item["id"] = self.ids[idx]
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