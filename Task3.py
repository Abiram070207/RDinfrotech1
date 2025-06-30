import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import json
import pickle
from collections import Counter
import matplotlib.pyplot as plt

# Vocabulary class for handling text processing
class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.word2idx)

# Build vocabulary from captions
def build_vocab(captions, threshold=4):
    """Build vocabulary from caption annotations"""
    counter = Counter()
    for caption in captions:
        tokens = caption.lower().split()
        counter.update(tokens)
    
    # Create vocabulary object
    vocab = Vocabulary()
    
    # Add special tokens
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    
    # Add words that occur more than threshold times
    for word, count in counter.items():
        if count >= threshold:
            vocab.add_word(word)
    
    return vocab

# CNN Encoder using pre-trained ResNet
class CNNEncoder(nn.Module):
    def __init__(self, embed_size):
        super(CNNEncoder, self).__init__()
        # Load pre-trained ResNet and remove final classification layer
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # Remove last fc layer
        self.resnet = nn.Sequential(*modules)
        
        # Linear layer to transform ResNet output to desired embedding size
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

# RNN Decoder with LSTM
class RNNDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        super(RNNDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search"""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids

# Complete Image Captioning Model
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = CNNEncoder(embed_size)
        self.decoder = RNNDecoder(embed_size, hidden_size, vocab_size, num_layers)
        
    def forward(self, images, captions, lengths):
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        return outputs
    
    def generate_caption(self, image, vocab):
        """Generate caption for a single image"""
        with torch.no_grad():
            features = self.encoder(image.unsqueeze(0))
            sampled_ids = self.decoder.sample(features)
            sampled_ids = sampled_ids[0].cpu().numpy()
            
            # Convert word IDs to words
            sampled_caption = []
            for word_id in sampled_ids:
                word = vocab.idx2word[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            
            # Join words and remove special tokens
            caption = ' '.join(sampled_caption[:-1])  # Remove <end> token
            return caption

# Dataset class for loading images and captions
class ImageCaptionDataset(Dataset):
    def __init__(self, image_paths, captions, vocab, transform=None):
        self.image_paths = image_paths
        self.captions = captions
        self.vocab = vocab
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption = self.captions[idx]
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        # Convert caption to tensor
        tokens = caption.lower().split()
        caption_tokens = [self.vocab('<start>')]
        caption_tokens.extend([self.vocab(token) for token in tokens])
        caption_tokens.append(self.vocab('<end>'))
        target = torch.Tensor(caption_tokens)
        
        return image, target

# Collate function for DataLoader
def collate_fn(data):
    # Sort data by caption length (descending)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Merge captions and create lengths tensor
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    
    return images, targets, lengths

# Training function
def train_model(model, data_loader, criterion, optimizer, num_epochs, device):
    model.train()
    total_step = len(data_loader)
    
    for epoch in range(num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            images = images.to(device)
            captions = captions.to(device)
            targets = nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward pass
            outputs = model(images, captions, lengths)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print log
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

# Image preprocessing
def get_transform(train=True):
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    return transform

# Example usage and training setup
def main():
    # Hyperparameters
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    num_epochs = 5
    batch_size = 32
    learning_rate = 0.001
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Example data (replace with your actual data)
    # For demonstration, using dummy data
    sample_captions = [
        "a cat sitting on a chair",
        "a dog running in the park",
        "a bird flying in the sky",
        "a car driving on the road",
        "a person walking on the street"
    ]
    
    # Build vocabulary
    vocab = build_vocab(sample_captions, threshold=1)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Initialize model
    model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, num_layers)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Model initialized successfully!")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Example of how to use the model for inference
    def demo_inference():
        model.eval()
        # Create a dummy image tensor
        dummy_image = torch.randn(3, 224, 224).to(device)
        
        with torch.no_grad():
            caption = model.generate_caption(dummy_image, vocab)
            print(f"Generated caption: {caption}")
    
    # Run demo
    demo_inference()
    
    return model, vocab

# Additional utility functions
def save_model(model, vocab, filepath):
    """Save model and vocabulary"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab': vocab
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath, embed_size, hidden_size, num_layers):
    """Load model and vocabulary"""
    checkpoint = torch.load(filepath, map_location='cpu')
    vocab = checkpoint['vocab']
    vocab_size = len(vocab)
    
    model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, num_layers)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, vocab

def generate_caption_for_image(image_path, model, vocab, device, transform):
    """Generate caption for a single image file"""
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Generate caption
    with torch.no_grad():
        caption = model.generate_caption(image.squeeze(0), vocab)
    
    return caption

if __name__ == "__main__":
    model, vocab = main()