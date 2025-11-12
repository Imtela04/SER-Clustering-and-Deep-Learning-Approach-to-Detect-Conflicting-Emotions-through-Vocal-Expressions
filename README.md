# Fuzzy-Based Emotional Conflict Detection in Speech

A novel hybrid deep ensemble framework for detecting overlapping and conflicting emotions in speech, going beyond traditional Speech Emotion Recognition (SER) systems.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

## 🎯 Overview

While conventional SER systems classify emotions as single, discrete categories, real human emotions are often mixed or conflicting—anger combined with disgust, or sadness with happiness. This research addresses that gap by introducing a **fuzzy-based emotional conflict detection framework** that:

- Detects both primary and secondary emotions
- Quantifies emotional ambiguity and overlap
- Provides interpretable conflict scores for uncertain predictions
- Bridges categorical and continuous emotional models

## 👥 Research Team

- **Imtela Islam** - BRAC University
- **Nuhash Kabir Neeha** - BRAC University
- **Nuzhat Rahman** - BRAC University

**Supervisor:** Mr. Dibyo Fabian Dofadar  
**Thesis Coordinator:** Dr. Md. Golam Rabiul Alam  
**Department Chair:** Dr. Sadia Hamid Kazi

## 🔬 Key Contributions

1. **Hybrid Deep Ensemble Architecture**
   - Combines CNN, Transformer (Wav2Vec2), BiLSTM, and GRU with attention mechanisms
   - Multi-branch architecture for robust emotion representation

2. **Fuzzy Clustering Integration**
   - Uses Fuzzy C-Means (FCM) to measure emotional overlap
   - Generates conflict scores for ambiguous predictions

3. **Emotional Conflict Detection**
   - First framework to systematically detect and quantify emotional conflicts in speech
   - Provides interpretable secondary emotion predictions

4. **Comprehensive Validation**
   - Empirical validation across three major datasets
   - Establishes new metrics for conflict rate and fuzzy partition coefficients

## 🏗️ Architecture

### Framework Components

```
┌─────────────────────────────────────────────────────────┐
│                  Data Augmentation                       │
│  (Noise, Pitch Shift, Time Stretch, SpecAugment)       │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                  Feature Extraction                      │
│  • MFCCs                    • Wav2Vec2 Embeddings       │
│  • Mel-spectrograms         • ZCR                       │
│  • Chroma Features          • Spectral Features         │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│              Hybrid Ensemble Model                       │
│  ┌─────────┐  ┌──────────┐  ┌────────────┐            │
│  │   CNN   │  │ RNN-Attn │  │Transformer │            │
│  │ Branch  │  │  Branch  │  │  (Wav2Vec2)│            │
│  └─────────┘  └──────────┘  └────────────┘            │
│       │            │              │                     │
│       └────────────┴──────────────┘                     │
│                    ↓                                     │
│            Fusion Layer (512-dim)                       │
│                    ↓                                     │
│       Adversarial Speaker Layer (GRL)                   │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│          Ensemble Averaging & Majority Voting           │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│           Fuzzy C-Means Conflict Analysis                │
│  • Conflict Score = 1 - Confidence(Primary)             │
│  • Secondary Emotion Detection                          │
│  • Membership Grade Calculation                         │
└─────────────────────────────────────────────────────────┘
```

### Model Branches

- **CNN Branch**: Captures spectral-spatial patterns from Mel-spectrograms
- **RNN Branch**: BiLSTM-GRU with attention for temporal emotion dynamics
- **Transformer Branch**: Wav2Vec2 for contextual dependencies
- **Fusion Layer**: Merges embeddings with residual connections and dropout
- **Adversarial Speaker Layer**: Minimizes speaker bias using Gradient Reversal Layer (GRL)

## 📊 Datasets

The framework was evaluated on three augmented benchmark datasets:

| Dataset | Utterances | Emotions | Notes |
|---------|-----------|----------|-------|
| **RAVDESS** | 2,401 | 8 emotions | Speech and song recordings |
| **SAVEE** | 1,596 | 7 emotions | British English speakers |
| **CREMA-D** | 16,801 | 6 emotions | Most diverse, multiple ethnicities |

### Data Augmentation
- Noise injection
- Pitch shifting
- Time stretching
- SpecAugment
- Volume scaling

## 📈 Results

### Primary Emotion Classification

| Dataset | Ensemble Accuracy | Mean Individual Accuracy | F1-Score | Cohen's κ |
|---------|------------------|-------------------------|----------|-----------|
| **RAVDESS** | 85.56% | 81.69% | 0.85 | 0.78 |
| **SAVEE** | **88.75%** | 80.69% | 0.86 | 0.79 |
| **CREMA-D** | 85.71% | 83.63% | 0.84 | 0.77 |

### Emotional Conflict Detection

| Dataset | Fuzzy Partition Coeff. | Conflict Rate | Avg. Conflict Score | Most Conflicted Emotions |
|---------|----------------------|---------------|--------------------|-----------------------|
| **RAVDESS** | 0.84 | 15.28% | 0.46 | Disgust, Neutral, Happiness |
| **SAVEE** | 0.82 | 21.25% | 0.48 | Anger, Fear, Neutral |
| **CREMA-D** | 0.81 | 22.50% | 0.44 | Sadness, Disgust, Neutral |

### Key Findings

- **High-confidence samples (>0.8)**: 96% accuracy
- **Low-confidence samples (<0.5)**: 41% accuracy (matches fuzzy conflict regions)
- **Ensemble improvement**: 4-6% over best single model
- **Conflict detection rate**: 15-21% across datasets
- **Average FPC**: 0.84 (indicates clear emotion-conflict boundaries)

## 🛠️ Technical Specifications

### Requirements
```
Python >= 3.8
TensorFlow >= 2.x
Librosa
Scikit-Fuzzy
NumPy
Pandas
Scikit-learn
```

### Training Parameters

- **Optimizer**: Adam (lr = 0.001)
- **Epochs**: 20-30
- **Batch size**: 16-64
- **Speaker loss weight (λ)**: 0.1
- **Fuzzy C-Means**: m=2, n=3
- **Dropout**: 0.2-0.3

### Conflict Detection Criteria

- Primary emotion: Highest probability class
- Conflict detected when:
  - Confidence < 0.5 AND
  - Secondary probability > 0.3
- Conflict Score = 1 - Confidence(Primary Emotion)

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/yourusername/fuzzy-emotion-conflict-detection.git
cd fuzzy-emotion-conflict-detection
pip install -r requirements.txt
```

### Quick Start

```python
# Load the trained ensemble model
from src.model import EmotionalConflictDetector

# Initialize detector
detector = EmotionalConflictDetector()
detector.load_weights('models/best_ensemble.h5')

# Predict with conflict detection
audio_path = 'path/to/audio.wav'
results = detector.predict_with_conflict(audio_path)

print(f"Primary Emotion: {results['primary']}")
print(f"Confidence: {results['confidence']:.2f}")
print(f"Conflict Score: {results['conflict_score']:.2f}")
if results['has_conflict']:
    print(f"Secondary Emotion: {results['secondary']}")
```

## 📊 Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **Cohen's Kappa (κ)**: Inter-rater agreement measure
- **Fuzzy Partition Coefficient (FPC)**: Cluster quality measure
- **Conflict Rate (%)**: Percentage of ambiguous predictions
- **Conflict Score**: Continuous measure of emotional overlap

## 🔍 Applications

This framework enables more nuanced emotion AI systems for:

- **Mental Health**: Therapy and counseling applications
- **Human-Robot Interaction**: Sentiment-aware robotics
- **Customer Service**: Empathetic chatbots and virtual assistants
- **Content Analysis**: Media and social media sentiment tracking
- **Voice User Interfaces**: Adaptive emotional response systems

## 📝 Key Innovations

1. **First fuzzy-based conflict detection** framework specifically for speech emotions
2. **Interpretable uncertainty quantification** through fuzzy membership grades
3. **Multi-view ensemble** combining spectral, temporal, and contextual features
4. **Speaker-adversarial training** for improved generalization
5. **Novel metrics** for measuring emotional conflict and ambiguity

## 🔬 Research Gap Addressed

Traditional SER systems assume:
- ❌ Single, static emotion labels
- ❌ Clear boundaries between emotion classes
- ❌ No overlapping or mixed emotions

Our framework provides:
- ✅ Detection of primary AND secondary emotions
- ✅ Quantification of emotional ambiguity
- ✅ Interpretable conflict scores
- ✅ Bridge between categorical and dimensional emotion theories

## ⚠️ Limitations

- Limited speaker diversity in training datasets
- Synthetic augmentation may not fully capture real-world variation
- Increased computational cost due to ensemble architecture
- Requires further validation on cross-cultural datasets

## 🔮 Future Work

- [ ] Real-time conflict-aware emotion detection
- [ ] Cross-cultural and multilingual dataset expansion
- [ ] Multimodal integration (speech + facial cues)
- [ ] Human perception studies on fuzzy conflict validity
- [ ] Lightweight model variants for edge deployment
- [ ] Integration with dimensional emotion models (valence-arousal)

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{neeha2023fuzzy,
  title={Fuzzy-Based Emotional Conflict Detection in Speech Using a Hybrid Deep Ensemble Framework},
  author={Neeha, Nuhash Kabir and Rahman, Nuzhat and Islam, Imtela},
  booktitle={IEEE Conference Proceedings},
  year={2023},
  organization={BRAC University}
}
```

## 🙏 Acknowledgments

We express our sincere gratitude to:
- **Mr. Dibyo Fabian Dofadar** - Research Supervisor
- **Dr. Md. Golam Rabiul Alam** - Thesis Coordinator
- **Dr. Sadia Hamid Kazi** - Department Chairperson
- Our peers, families, and friends for their unwavering support

## 📄 License

This project is part of academic research at BRAC University and is available for educational and research purposes.

## 📧 Contact

For questions, collaborations, or feedback:

- Imtela Islam: imtela.islam@g.bracu.ac.bd
- Nuhash Kabir Neeha: nuhash.kabir.neeha@g.bracu.ac.bd
- Nuzhat Rahman: nuzhat.rahman@g.bracu.ac.bd


## 🔗 Related Publications

1. V. Rajan et al., "ConflictNet: End-to-end learning for speech-based conflict intensity estimation," IEEE SPL, 2019
2. K. Zhou et al., "Mixed-EVC: Mixed emotion synthesis and control in voice conversion," 2023
3. A. Slimi et al., "Multiple models fusion for multi-label classification in SER systems," Procedia CS, 2022

---

**Department of Computer Science and Engineering**  
**BRAC University, Dhaka, Bangladesh**

*Advancing interpretable and reliable emotion AI through fuzzy conflict modeling*
