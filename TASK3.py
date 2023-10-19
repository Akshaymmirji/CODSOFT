import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np

# Load a pre-trained image recognition model like VGG16
encoder = models.vgg16(pretrained=True)
modules = list(encoder.children())[:-1]
encoder = nn.Sequential(*modules)

# Define a simple LSTM-based captioning model
class DecoderRNN(nn.Module):
    def _init_(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self)._init_()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

# Load a pre-trained language model for captions
decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=vocab_size, num_layers=1)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Image captioning function
def caption_image(image_path, max_length=50):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    # Generate image features
    features = encoder(image).view(1, -1)

    # Generate captions
    sampled_ids = []
    inputs = features
    for i in range(max_length):
        hiddens, states = decoder.lstm(inputs)
        outputs = decoder.linear(hiddens.squeeze(1))
        _, predicted = outputs.max(1)
        sampled_ids.append(predicted.item())
        inputs = decoder.embed(predicted)
        inputs = inputs.unsqueeze(1)
    return sampled_ids

# You would need to have a vocabulary and set up data loading for training.

# Finally, you can call the `caption_image` function to generate captions for a given image.
