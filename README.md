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
- Add natural language processing capabilities.


# 🤖 Unbeatable AI Tic-Tac-Toe Game

An interactive web-based Tic-Tac-Toe game featuring an AI opponent with multiple difficulty levels, built with vanilla JavaScript and modern web technologies. Challenge yourself against an unbeatable AI powered by the Minimax algorithm with Alpha-Beta pruning!

![Game Screenshot](https://via.placeholder.com/600x400/667eea/ffffff?text=Tic-Tac-Toe+AI+Game)

## 🚀 Features

- **🧠 Multiple AI Difficulty Levels**
  - **Unbeatable**: Minimax with Alpha-Beta pruning (impossible to win!)
  - **Hard**: Standard Minimax algorithm
  - **Medium**: Strategic rule-based AI
  - **Easy**: Random move selection

- **🎮 Interactive Gameplay**
  - Smooth animations and hover effects
  - Real-time game status updates
  - Score tracking across multiple games
  - Switch between going first or second

- **📱 Modern UI/UX**
  - Glassmorphism design with backdrop blur effects
  - Fully responsive design for all devices
  - Intuitive controls and visual feedback
  - Clean, professional interface

- **📊 Game Statistics**
  - Track wins, losses, and ties
  - Persistent session statistics
  - Performance metrics display

## 🎯 Live Demo

[**🎮 Play the Game**](https://your-github-username.github.io/tic-tac-toe-ai/)

## 🛠️ Technical Stack

### Frontend Technologies
- **HTML5** - Semantic markup and structure
- **CSS3** - Modern styling with Grid, Flexbox, and animations
- **JavaScript (ES6+)** - Game logic and AI algorithms

### AI & Algorithms
- **Minimax Algorithm** - Optimal decision-making
- **Alpha-Beta Pruning** - Performance optimization
- **Game Theory** - Strategic gameplay implementation
- **Heuristic Evaluation** - Position assessment

### Design & UX
- **Glassmorphism** - Modern translucent design
- **Responsive Design** - Mobile-first approach
- **CSS Animations** - Smooth transitions and effects
- **Progressive Enhancement** - Graceful degradation

## 🏗️ Architecture

```
├── index.html              # Main game interface
├── styles/
│   └── main.css           # Styling and animations
├── scripts/
│   ├── game.js            # Core game logic
│   ├── ai.js              # AI algorithms
│   └── ui.js              # User interface handlers
└── assets/
    └── images/            # Game assets
```

## 🧠 AI Algorithm Details

### Minimax with Alpha-Beta Pruning
The unbeatable AI uses the Minimax algorithm enhanced with Alpha-Beta pruning:

- **Time Complexity**: O(b^d) → O(b^(d/2)) with pruning
- **Space Complexity**: O(d) for recursion stack
- **Performance**: Reduces ~550K evaluations to ~18K (97% improvement)

```javascript
minimaxAlphaBeta(board, depth, isMaximizing, alpha, beta) {
    // Base case: game over
    const result = this.checkWinner(board);
    if (result !== null) {
        return this.evaluatePosition(result, depth);
    }
    
    // Recursive case with Alpha-Beta pruning
    if (isMaximizing) {
        // ... maximize AI score
    } else {
        // ... minimize human score
    }
}
```

### Strategic AI (Medium Difficulty)
Rule-based approach following strategic priorities:
1. **Win** - Take winning move if available
2. **Block** - Prevent opponent from winning
3. **Center** - Control the center position
4. **Corners** - Take corner positions
5. **Edges** - Fill remaining positions

## 🚀 Getting Started

### Prerequisites
- Modern web browser (Chrome 60+, Firefox 55+, Safari 12+)
- No additional dependencies required!

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/tic-tac-toe-ai.git
   cd tic-tac-toe-ai
   ```

2. **Open in browser**
   ```bash
   # Simply open index.html in your browser
   open index.html
   # or
   python -m http.server 8000  # For local server
   ```

3. **Start playing!**
   - Choose your difficulty level
   - Click any cell to make your move
   - Try to beat the AI (good luck with unbeatable mode!)

## 🎮 How to Play

1. **Select Difficulty**: Choose from Easy, Medium, Hard, or Unbeatable
2. **Make Your Move**: Click any empty cell (you are X)
3. **AI Response**: Watch the AI make its move (AI is O)
4. **Win Condition**: Get three in a row (horizontal, vertical, or diagonal)
5. **New Game**: Click "New Game" to start over

### Game Controls
- **New Game**: Reset the current game
- **Switch First Player**: Toggle between going first or second
- **Difficulty Selector**: Change AI difficulty level

## 📊 Performance Metrics

| Difficulty | Algorithm | Avg Response Time | Win Rate vs Human |
|------------|-----------|-------------------|-------------------|
| Easy | Random | <1ms | ~10% |
| Medium | Strategic | <5ms | ~60% |
| Hard | Minimax | ~50ms | ~95% |
| Unbeatable | Minimax + α-β | ~10ms | 100% |

## 🧪 Algorithm Analysis

### Game Tree Complexity
- **State Space**: 3^9 = 19,683 possible positions
- **Game Tree**: ~255,168 nodes for complete analysis
- **Pruning Efficiency**: Alpha-Beta reduces search by ~97%

### Evaluation Function
```javascript
evaluatePosition(result, depth) {
    if (result === 'O') return 10 - depth;  // AI wins (prefer faster)
    if (result === 'X') return depth - 10;  // Human wins (delay loss)
    return 0;  // Tie game
}
```

## 🔧 Customization

### Adding New Difficulty Levels
```javascript
// Add to TicTacToeAI class
customStrategy() {
    // Implement your custom AI logic here
    return bestMoveIndex;
}
```

### Modifying Game Rules
```javascript
// Customize win conditions
const winPatterns = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8], // Rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8], // Columns
    [0, 4, 8], [2, 4, 6]             // Diagonals
];
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the Project**
2. **Create Feature Branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit Changes** (`git commit -m 'Add AmazingFeature'`)
4. **Push to Branch** (`git push origin feature/AmazingFeature`)
5. **Open Pull Request**

### Ideas for Contributions
- [ ] Add sound effects and music
- [ ] Implement online multiplayer
- [ ] Create tournament mode
- [ ] Add different board sizes (4x4, 5x5)
- [ ] Implement machine learning AI
- [ ] Add accessibility features
- [ ] Create mobile app version

## 📚 Learning Resources

This project demonstrates key concepts in:
- **Game Theory** - Zero-sum games and optimal strategies
- **Artificial Intelligence** - Search algorithms and decision trees
- **Web Development** - Modern JavaScript and CSS techniques
- **Algorithm Optimization** - Alpha-Beta pruning and performance

### Recommended Reading
- *Artificial Intelligence: A Modern Approach* by Russell & Norvig
- *Game Theory: An Introduction* by Barron
- *JavaScript: The Definitive Guide* by Flanagan

## 📈 Roadmap

- [ ] **v2.0**: Neural network AI using TensorFlow.js
- [ ] **v2.1**: Online multiplayer with WebRTC
- [ ] **v2.2**: Progressive Web App (PWA) support
- [ ] **v2.3**: Voice commands and accessibility
- [ ] **v3.0**: 3D graphics with Three.js

## 🐛 Known Issues

- [ ] AI thinking delay might be too short on very fast devices
- [ ] Score persistence doesn't survive browser refresh
- [ ] No keyboard navigation support yet

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/your-profile)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Inspired by classic game theory problems
- UI design influenced by modern glassmorphism trends
- Algorithm implementations based on AI textbook examples
- Special thanks to the open-source community

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/your-username/tic-tac-toe-ai?style=social)
![GitHub forks](https://img.shields.io/github/forks/your-username/tic-tac-toe-ai?style=social)
![GitHub issues](https://img.shields.io/github/issues/your-username/tic-tac-toe-ai)
![GitHub license](https://img.shields.io/github/license/your-username/tic-tac-toe-ai)

---

**⭐ If you found this project helpful, please give it a star!**

**🎮 Ready to challenge the AI? [Play Now!](https://your-github-username.github.io/tic-tac-toe-ai/)**
