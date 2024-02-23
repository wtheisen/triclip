import os
import cv2
import torch
import albumentations as A

import config as CFG
import numpy as np

from decord import VideoReader, cpu, gpu, bridge

bridge.set_bridge('torch')

class CLIPTriplets(torch.utils.data.Dataset):
    def __init__(self, ids, caption, video_embeddings, image_embeddings, text_embeddings):
        self.ids = ids
        self.images = ''
        self.captions = caption
        # self.video_paths = video_path
        self.videos = video_embeddings
        self.images = image_embeddings
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
        self.texts = text_embeddings

    # def __getitem__(self, idx):
    #     return {
    #         'image': self.images[idx],
    #         'input_ids': self.input_ids[idx],
    #         'attention_mask': self.attention_masks[idx],
    #         'video': self.videos[idx]
    #     }

    def __getitem__(self, idx):
        # item = {
        #     key: torch.tensor(values[idx])
        #     for key, values in self.encoded_captions.items()
        # }
        # # item['video_path'] = self.videos[idx]

        item = {}
        #get video frames and take an image out of the unchosen one
        # videoreader = VideoReader(self.videos[idx], ctx=gpu(0), width=244, height=244)
        item['video'] = self.videos[idx]
        # videoreader.seek(0)

        # try:
        #     indices, image_frame = self.sample_frame_indices(clip_len=16, frame_sample_rate=4, seg_len=len(videoreader))
        # except:
        #     print(self.videos[idx])
        #     exit(69)


        # video = videoreader.get_batch(indices[:-1]).asnumpy()
        # try:
        #     item['video'] = videoreader.get_batch(indices[:-1])
        # except:
        #     print(self.videos[idx])
        #     print('Error Reading Batch')
        #     item['video'] = None

        # item['video'] = self.preprocess_video(video)
        # item['video'] = video.transpose(0, 3, 1, 2)

        # item['image'] = videoreader[indices[-1]].asnumpy()
        item['image'] = self.images[idx]
        #preprocess video

        # image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = self.transforms(image=image)['image']
        # item['image'] = torch.tensor(image.astype(np.uint8)).permute(2, 0, 1).float()
        # item['caption'] = self.captions[idx]
        # item['input_ids'] = np.asarray(self.encoded_captions['input_ids'][idx])
        # item['attention_mask'] = np.asarray(self.encoded_captions['attention_mask'][idx])

        item["text"] = self.texts[idx]
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