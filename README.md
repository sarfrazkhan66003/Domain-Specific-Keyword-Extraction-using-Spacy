# Custom Named Entity Recognition (NER) Training System

## 📋 Overview
- This project implements a custom Named Entity Recognition (NER) training system using SpaCy, a powerful NLP library. The system allows you to train custom NER models to identify and classify specific entities in text data based on your labeled training examples.

- Key Capabilities:
  - Train custom NER models for domain-specific entity recognition
  - Support for multiple entity types (Food, Clothing, Technology, etc.)
  - Entity visualization using SpaCy's built-in displaCy
  - Model persistence for future use
  - Interactive testing interface

- 🎯 Use Cases
  - E-commerce: Extract product names, brands, and categories from product descriptions
  - Healthcare: Identify medical terms, diseases, and medications
  - Finance: Detect financial entities like company names, stock symbols, and monetary values
  - Customer Service: Extract key information from customer queries
  - Content Analysis: Automatically tag and categorize text content

## 🏗️ System Architecture
    ┌─────────────────────────────────────────────────────────────┐
    │                     INPUT DATA LAYER                        │
    ├─────────────────────────────────────────────────────────────┤
    │  • data_enhanced.txt  (Raw text data)                       │
    │  • labels_enhanced.csv (Entity labels: word → category)     │
    └─────────────────────────────────────────────────────────────┘
                                ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                  DATA PREPROCESSING                         │
    ├─────────────────────────────────────────────────────────────┤
    │  1. Text tokenization into sentences                        │
    │  2. Word extraction and cleaning                            │
    │  3. Entity position calculation (start, end)                │
    │  4. Format conversion to SpaCy training format              │
    └─────────────────────────────────────────────────────────────┘
                                ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                   TRAINING PIPELINE                         │
    ├─────────────────────────────────────────────────────────────┤
    │  • Create blank SpaCy model                                 │
    │  • Add NER pipeline component                               │
    │  • Register entity labels                                   │
    │  • Iterative training with dropout (0.2)                    │
    │  • Loss optimization using SGD                              │
    └─────────────────────────────────────────────────────────────┘
                                ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                   TRAINED MODEL                             │
    ├─────────────────────────────────────────────────────────────┤
    │  • Saved to disk for reuse                                  │
    │  • Can process new text inputs                              │
    │  • Identifies and classifies entities                       │
    └─────────────────────────────────────────────────────────────┘
                                ↓
    ┌─────────────────────────────────────────────────────────────┐
    │                   TESTING & OUTPUT                          │
    ├─────────────────────────────────────────────────────────────┤
    │  • Entity detection in test text                            │
    │  • Console output with labels and positions                 │
    │  • Visual representation using displaCy                     │
    └─────────────────────────────────────────────────────────────┘

## 🔬 Algorithm Workflow

- 1. Data Loading Phase
     Input Files:
      ├── data_enhanced.txt      # Contains sentences with entities
      └── labels_enhanced.csv    # Maps entities to their labels
          ├── Column 1: entities (e.g., "tomato", "shirt", "Python")
          └── Column 2: labels (e.g., "FOOD", "CLOTH", "TECH")
     
- 2. Preprocessing Phase
  - Sentence Segmentation: Text is split into individual sentences using SpaCy's sentencizer
  - Entity Matching: Each word is checked against the label dictionary
  - Position Calculation: Character-level start and end positions are calculated for each entity
  - Training Format Conversion:
        TRAIN_DATA = [
            ("I bought tomatoes", {"entities": [(10, 18, "FOOD")]}),
            ("Python is great", {"entities": [(0, 6, "TECH")]})
        ]
  
  - 3. Training Algorithm
    - The system uses Stochastic Gradient Descent (SGD) with the following parameters:
      - Iterations: 20 epochs (configurable)
      - Dropout Rate: 0.2 (prevents overfitting)
      - Batch Processing: One example at a time
      - Data Shuffling: Random shuffle each iteration for better generalization
  
  - Training Loop:
      - FOR each iteration (1 to 20):
        - 1. Shuffle training data
        - 2. FOR each (text, annotations) pair:
            - a. Feed text to model
            - b. Compare predictions with annotations
            - c. Calculate loss
            - d. Update model weights using optimizer
        - 3. Print loss values
        - 4. Continue to next iteration
  
  - 4. Model Persistence
    - Trained model is saved to disk with a custom name
    - Can be loaded later for inference without retraining
  
  - 5. Testing & Inference
    - User inputs test text
    - Model processes text and identifies entities
    - Results displayed with entity text, label, and position

## 📁 Project Structure
    ner-training-system/
    │
    ├── data_enhanced.txt          # Training text data
    ├── labels_enhanced.csv        # Entity labels mapping
    ├── train_ner.py              # Main training script
    ├── requirements.txt          # Python dependencies
    ├── README.md                 # This file
    │
    ├── screenshots/              # Documentation images
    │   ├── training_process.png
    │   ├── entity_detection.png
    │   └── visualization.png
    │
    └── models/                   # Saved trained models
        └── custom_ner_model/

## 🚀 Features

- ✅ Core Features
  - Custom Entity Training: Train models on your own labeled data
  - Multi-Label Support: Handle multiple entity types simultaneously
  - Position Tracking: Accurate character-level entity positions
  - Iterative Training: Configurable training iterations for optimal results
  - Model Saving: Persist trained models for future use

- 🎨 Advanced Features
  - Entity Visualization: Interactive HTML visualization using displaCy
  - Dropout Regularization: Prevents overfitting during training
  - Data Shuffling: Improves model generalization
  - Loss Tracking: Monitor training progress through loss values
  - Interactive Testing: Test model with custom inputs immediately
 
- 🛡️ Error Handling
  - Encoding support (CP1252) for special characters
  - Graceful handling of missing entities
  - Helpful suggestions for test inputs
  - Fallback options for visualization

## 🎓Training Process Explained

- Phase 1: Data Preparation (Lines 8-35)
  - Read raw text file
  - Load entity labels from CSV
  - Create dictionary mapping entities to labels
  - Tokenize text into sentences
  - Find entity positions in each sentence
  - Format data for SpaCy training

- Phase 2: Model Initialization (Lines 38-50)
  - Create blank English model
  - Add NER pipeline component
  - Register all entity labels from training data

- Phase 3: Training Loop (Lines 52-68)
  - Disable non-NER pipelines for efficiency
  - Initialize optimizer
  - For each iteration:
    - Shuffle training data (improve generalization)
    - Process each training example
    - Calculate prediction errors (losses)
    - Update model weights
    - Display loss values

- Phase 4: Model Persistence (Lines 73-74)
  - Save trained model to disk
  - Model can be reloaded without retraining

- Phase 5: Testing (Lines 77-96)
  - Accept user input
  - Process text through trained model
  - Display detected entities with details
  - Optional visualization
