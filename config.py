import torch

debug = True
captions_path = "/media/wtheisen/scratch3/triclip/"
batch_size = 128
num_workers = 0
lr = 5e-5
weight_decay = 1e-3
patience = 2
factor = 0.5
epochs = 50

num_train = 100
num_val = 100
num_test = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

model_name = 'facebook/vit-mae-base'
image_embedding = 768

text_encoder_model = "distilbert-base-multilingual-cased"
text_embedding = 768
text_tokenizer = "distilbert-base-multilingual-cased"
max_length = 128

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