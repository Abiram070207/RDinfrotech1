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

# Rule-Based Chatbot

A simple implementation of a rule-based chatbot in Python that demonstrates basic natural language processing and conversation flow.

## Features

- Pattern-based response system
- Multiple response variations for each pattern
- Input preprocessing
- Rotating response selection
- Default responses for unknown queries
- Interactive command-line interface

## Response Categories

The chatbot can respond to:
- Greetings (hello)
- Well-being questions (how are you)
- Farewells (bye)
- Identity questions (name)
- Help requests (help)
- Weather queries
- Time queries
- Expressions of gratitude (thanks)

## Usage

1. Run the script:
```bash
python chatbot.py
```

2. Start chatting with the bot. Type messages and press Enter.

3. Type 'bye' to end the conversation.

## Example Conversation

```
ChatBot: Hi! I'm a simple rule-based chatbot. Type 'bye' to exit.
You: hello
ChatBot: Hi there!
You: what's your name
ChatBot: I'm ChatBot, nice to meet you!
You: how are you
ChatBot: I'm doing well, thank you!
You: bye
ChatBot: Goodbye!
```

## Extending the Chatbot

To add new response patterns:
1. Add new key-value pairs to the `response_rules` dictionary in the `__init__` method
2. The key should be the pattern to match
3. The value should be a list of possible responses

Example:
```python
'new_pattern': ['Response 1', 'Response 2', 'Response 3']
```

## Limitations

- Uses simple pattern matching
- No context awareness
- Cannot handle complex queries
- No memory of conversation history
- Limited to predefined responses

## Future Improvements

- Add regular expression pattern matching
- Implement context awareness
- Add conversation memory
- Integrate with external APIs
- Add natural language processing capabilities

I've created a comprehensive AI Tic-Tac-Toe game with multiple difficulty levels and a beautiful, interactive interface. Here are the key features:
## 🧠 AI Algorithms Implemented:
## 1. Unbeatable Mode (Minimax + Alpha-Beta Pruning):

Uses game theory to evaluate all possible moves
Alpha-Beta pruning optimizes performance by eliminating unnecessary branches
Guarantees optimal play - you can only tie or lose against this mode

## 2. Hard Mode (Standard Minimax):

Evaluates all possible game states without pruning
Still very strong but slightly slower than Alpha-Beta version

## 3. Medium Mode (Strategic AI):

Uses rule-based strategy: win if possible, block opponent, take center/corners
Challenging but beatable with good strategy

## 4. Easy Mode (Random):

Makes random valid moves
Good for beginners or casual play

## 🎮 Game Features:

Interactive UI: Modern glassmorphism design with smooth animations
Difficulty Selection: Choose from 4 AI difficulty levels
Score Tracking: Keeps track of wins, losses, and ties
First Player Toggle: Switch between going first or second
Real-time Status: Shows whose turn it is and game results
Responsive Design: Works on all screen sizes

## 🔧 Technical Implementation:

Game Tree Search: The AI explores all possible future game states
Evaluation Function: Assigns scores based on win/loss/tie outcomes
Depth Penalty: Prefers faster wins and slower losses
Optimized Performance: Alpha-Beta pruning reduces search time significantly

## 🎯 Game Theory Concepts Demonstrated:

Zero-sum Game: One player's gain equals the other's loss
Perfect Information: Both players can see the entire game state
Backward Induction: Working backwards from end states to choose optimal moves
Nash Equilibrium: Optimal strategy where neither player can improve by changing strategy

The unbeatable mode truly lives up to its name - it's mathematically impossible to win against it when it plays optimally. The best you can achieve is a tie! Try different difficulty levels to experience various AI behaviors and see how game theory algorithms work in practice!
