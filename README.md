# Neural Image Captioning

A deep learning model that generates natural language descriptions for images using a combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN).

## 🌟 Features

- Pre-trained ResNet50 for image feature extraction
- LSTM-based decoder for caption generation
- Vocabulary management with special tokens
- Greedy search caption generation
- Support for both training and inference modes
- Batch processing with dynamic padding
- Custom dataset handling for images and captions

## 🛠️ Architecture

### CNN Encoder
- Uses pre-trained ResNet50 model
- Removes final classification layer
- Transforms features to desired embedding size
- Includes batch normalization

### RNN Decoder
- LSTM-based sequence generator
- Embedding layer for word representations
- Linear layer for vocabulary projection
- Support for teacher forcing during training

## 📋 Requirements

```bash
torch
torchvision
numpy
pillow
matplotlib
```

## 🚀 Usage

1. Install dependencies:
```bash
pip install torch torchvision numpy pillow matplotlib
```

2. Run the demo:
```bash
python Task3.py
```

3. For custom usage:
```python
# Load and initialize the model
model = ImageCaptioningModel(embed_size=256, hidden_size=512, vocab_size=len(vocab), num_layers=1)

# Generate caption for an image
caption = generate_caption_for_image(image_path, model, vocab, device, transform)
```

## 🔧 Model Parameters

- Embedding size: 256
- Hidden size: 512
- Number of layers: 1
- Maximum sequence length: 20
- Learning rate: 0.001
- Batch size: 32

## 💾 Save and Load Models

```python
# Save model
save_model(model, vocab, 'model_checkpoint.pth')

# Load model
model, vocab = load_model('model_checkpoint.pth', embed_size=256, hidden_size=512, num_layers=1)
```

## 📝 License

This project is open source and available under the MIT License.
