import torch

debug = True
# image_path = "C:/Moein/AI/Datasets/Flicker-8k/Images"
# image_path = "/media/wtheisen/scratch3/Flickr8k/Flicker8k_Dataset"
# captions_path = "C:/Moein/AI/Datasets/Flicker-8k"
captions_path = "/media/wtheisen/scratch3/triclip/"
batch_size = 8
num_workers = 0
lr = 1e-3
weight_decay = 1e-3
patience = 2
factor = 0.5
epochs = 5
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# model_name = 'resnet50'
model_name = 'facebook/vit-mae-base'
image_embedding = 768
# text_encoder_model = "distilbert-base-uncased"
text_encoder_model = "distilbert-base-multilingual-cased"
text_embedding = 768
# text_tokenizer = "distilbert-base-uncased"
text_tokenizer = "distilbert-base-multilingual-cased"
max_length = 200

video_encoder_model = "MCG-NJU/videomae-base"
video_embedding = 768

pretrained = False # for both image encoder and text encoder
trainable = False # for both image encoder and text encoder
temperature = 1.0

# image size
size = 224

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256 
dropout = 0.1