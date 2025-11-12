import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from skfuzzy import cmeans
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.layers import Layer
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import backend as K

# Load processor and model (do this once)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

def extract_wav2vec_features(audio, sample_rate=16000):
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = wav2vec_model(**inputs)

    features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return features

class GradientReversalLayer(Layer):
    def __init__(self, lambda_=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ = lambda_

    def call(self, x, training=None):
        if training:
            @tf.custom_gradient
            def reverse_grad(x):
                def grad(dy):
                    return -self.lambda_ * dy
                return x, grad
            return reverse_grad(x)
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update({'lambda_': self.lambda_})
        return config

# Choose dataset
DATASET_NAME = "cremad"  # or "savee", "tess", "ravdess"
DATASET_PATH = r"C:\Users\imtel\OneDrive\pc2\thesis\datasets\CREMAD_augmented"  # update as needed

class AudioFeatureExtractor:
    """Enhanced feature extraction from audio files"""
    def __init__(self, sr=22050, n_mfcc=20, n_fft=2048, hop_length=512, max_length=128):
        self.sr = sr
        self.n_mfcc = n_mfcc  # Increased from 13 to 20
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length = max_length

    def extract_features(self, audio_path):
        try:
            # Load audio at self.sr (e.g., 22050) for MFCC, Mel, Chroma
            y, sr = librosa.load(audio_path, sr=self.sr)

            # Pad or truncate to fixed length for MFCC, Mel, Chroma
            target_length = self.max_length * self.hop_length
            if len(y) > target_length:
                y = y[:target_length]
            else:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')

            features = {}

            # Enhanced MFCC with delta and delta-delta
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc,
                                        n_fft=self.n_fft, hop_length=self.hop_length)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            mfcc_combined = np.vstack([mfcc, mfcc_delta, mfcc_delta2])

            if mfcc_combined.shape[1] > self.max_length:
                mfcc_combined = mfcc_combined[:, :self.max_length]
            elif mfcc_combined.shape[1] < self.max_length:
                mfcc_combined = np.pad(mfcc_combined, ((0, 0), (0, self.max_length - mfcc_combined.shape[1])), mode='constant')
            features['mfcc'] = mfcc_combined.T

            # Enhanced Mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=self.n_fft,
                                                    hop_length=self.hop_length, n_mels=128)  # Increased from 80
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            if mel_spec_db.shape[1] > self.max_length:
                mel_spec_db = mel_spec_db[:, :self.max_length]
            elif mel_spec_db.shape[1] < self.max_length:
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, self.max_length - mel_spec_db.shape[1])), mode='constant')
            features['mel_spec'] = mel_spec_db.T

            # Enhanced Chroma with CQT
            chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
            chroma_combined = np.vstack([chroma_cqt, chroma_stft])

            if chroma_combined.shape[1] > self.max_length:
                chroma_combined = chroma_combined[:, :self.max_length]
            elif chroma_combined.shape[1] < self.max_length:
                chroma_combined = np.pad(chroma_combined, ((0, 0), (0, self.max_length - chroma_combined.shape[1])), mode='constant')
            features['chroma'] = chroma_combined.T

            # Additional features
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.n_fft, hop_length=self.hop_length)[0]
            if len(zcr) > self.max_length:
                zcr = zcr[:self.max_length]
            else:
                zcr = np.pad(zcr, (0, self.max_length - len(zcr)), mode='constant')

            # Spectral Centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
            if len(spectral_centroid) > self.max_length:
                spectral_centroid = spectral_centroid[:self.max_length]
            else:
                spectral_centroid = np.pad(spectral_centroid, (0, self.max_length - len(spectral_centroid)), mode='constant')

            # Spectral Rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)[0]
            if len(spectral_rolloff) > self.max_length:
                spectral_rolloff = spectral_rolloff[:self.max_length]
            else:
                spectral_rolloff = np.pad(spectral_rolloff, (0, self.max_length - len(spectral_rolloff)), mode='constant')

            # Combine additional features
            additional_features = np.vstack([zcr, spectral_centroid, spectral_rolloff])
            features['additional'] = additional_features.T

            # --- Wav2Vec2 expects 16kHz ---
            if sr != 16000:
                y_wav2vec = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr_wav2vec = 16000
            else:
                y_wav2vec = y
                sr_wav2vec = sr

            wav2vec_features = extract_wav2vec_features(y_wav2vec, sample_rate=sr_wav2vec)
            features['wav2vec'] = wav2vec_features

            return features
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None

class AttentionLayer(layers.Layer):
    """Enhanced attention mechanism with multi-head support"""
    def __init__(self, attention_dim=128, num_heads=4, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                               shape=(input_shape[-1], self.attention_dim),
                               initializer='glorot_uniform',
                               regularizer=l1_l2(l1=1e-5, l2=1e-4),
                               trainable=True)
        self.b = self.add_weight(name='attention_bias',
                               shape=(self.attention_dim,),
                               initializer='zeros',
                               trainable=True)
        self.u = self.add_weight(name='attention_u',
                               shape=(self.attention_dim,),
                               initializer='glorot_uniform',
                               trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # x shape: (batch_size, time_steps, features)
        uit = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.nn.softmax(ait, axis=1)

        # Expand dimensions for broadcasting
        ait = tf.expand_dims(ait, -1)
        weighted_input = x * ait
        output = tf.reduce_sum(weighted_input, axis=1)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'attention_dim': self.attention_dim,
            'num_heads': self.num_heads
        })
        return config

class SpeechEmotionRecognitionModel:
    """Enhanced ensemble model with improved architecture"""
    def __init__(self, num_classes=6):
        self.num_classes = num_classes
        self.model = None

    def build_transformer_branch(self, input_shape, name):
      """Enhanced transformer branch with self-attention"""
      input_layer = layers.Input(shape=input_shape, name=f'{name}_input')

      # Handle different input shapes
      if len(input_shape) == 1:
          # For 1D input (like wav2vec), create a pseudo-sequence
          # Split the features into chunks to create a sequence
          feature_size = input_shape[0]
          if feature_size >= 768:  # wav2vec typically outputs 768 features
              # Reshape into sequence of 16 timesteps with 48 features each
              sequence_length = 16
              feature_dim = feature_size // sequence_length
              # Pad if necessary
              if feature_size % sequence_length != 0:
                  x = layers.Lambda(lambda x: tf.pad(x, [[0, 0], [0, sequence_length - (feature_size % sequence_length)]]))(input_layer)
                  padded_size = feature_size + (sequence_length - (feature_size % sequence_length))
                  x = layers.Reshape((sequence_length, padded_size // sequence_length))(x)
              else:
                  x = layers.Reshape((sequence_length, feature_dim))(input_layer)
          else:
              # For smaller features, just add a sequence dimension
              x = layers.Reshape((1, feature_size))(input_layer)
              sequence_length = 1
      else:
          x = input_layer
          sequence_length = input_shape[0]

      # Layer normalization
      x = layers.LayerNormalization(epsilon=1e-6)(x)

      # Multi-head self-attention with proper configuration
      attention = layers.MultiHeadAttention(
          num_heads=4,
          key_dim=64,
          dropout=0.1,
          output_shape=256  # Ensure consistent output dimension
      )(x, x)

      # Add & Norm
      x = layers.Add()([x, attention]) if x.shape[-1] == attention.shape[-1] else attention
      x = layers.LayerNormalization(epsilon=1e-6)(x)

      # Feed-forward network
      ffn_output = layers.Dense(512, activation='gelu')(x)
      ffn_output = layers.Dropout(0.2)(ffn_output)
      ffn_output = layers.Dense(256, activation='gelu')(ffn_output)
      ffn_output = layers.Dropout(0.2)(ffn_output)

      # Add & Norm
      x = layers.Add()([x, ffn_output]) if x.shape[-1] == ffn_output.shape[-1] else ffn_output
      x = layers.LayerNormalization(epsilon=1e-6)(x)

      # Global pooling to get fixed size output
      if sequence_length > 1:
          x = layers.GlobalAveragePooling1D()(x)
      else:
          # For single timestep, just flatten
          x = layers.Flatten()(x)

      # Ensure output is 512 dimensions for consistency with other branches
      x = layers.Dense(512, activation='relu')(x)

      return input_layer, x

    def build_cnn_branch(self, input_shape, name):
        """Enhanced CNN branch with residual connections"""
        input_layer = layers.Input(shape=input_shape, name=f'{name}_input')

        # Reshape for 2D convolution
        x = layers.Reshape((*input_shape, 1))(input_layer)

        # First CNN block with residual connection
        conv1 = layers.Conv2D(64, (3, 3), padding='same')(x)
        conv1 = layers.BatchNormalization()(conv1)
        conv1 = layers.Activation('relu')(conv1)
        conv1 = layers.Conv2D(64, (3, 3), padding='same')(conv1)
        conv1 = layers.BatchNormalization()(conv1)

        # Residual connection
        shortcut1 = layers.Conv2D(64, (1, 1), padding='same')(x)
        conv1 = layers.Add()([conv1, shortcut1])
        conv1 = layers.Activation('relu')(conv1)

        if input_shape[0] >= 4 and input_shape[1] >= 4:
            pool1 = layers.MaxPooling2D((2, 2))(conv1)
        else:
            pool1 = conv1

        # Second CNN block with residual connection
        conv2 = layers.Conv2D(128, (3, 3), padding='same')(pool1)
        conv2 = layers.BatchNormalization()(conv2)
        conv2 = layers.Activation('relu')(conv2)
        conv2 = layers.Conv2D(128, (3, 3), padding='same')(conv2)
        conv2 = layers.BatchNormalization()(conv2)

        # Residual connection
        shortcut2 = layers.Conv2D(128, (1, 1), padding='same')(pool1)
        conv2 = layers.Add()([conv2, shortcut2])
        conv2 = layers.Activation('relu')(conv2)

        if input_shape[0] >= 8 and input_shape[1] >= 8:
            pool2 = layers.MaxPooling2D((2, 2))(conv2)
        else:
            pool2 = conv2

        # Third CNN block
        conv3 = layers.Conv2D(256, (3, 3), padding='same')(pool2)
        conv3 = layers.BatchNormalization()(conv3)
        conv3 = layers.Activation('relu')(conv3)
        conv3 = layers.Dropout(0.3)(conv3)

        # Global average pooling
        gap = layers.GlobalAveragePooling2D()(conv3)

        # Dense layers
        dense = layers.Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(gap)
        dropout = layers.Dropout(0.4)(dense)

        return input_layer, dropout

    def build_rnn_branch(self, input_shape, name):
        """Enhanced RNN branch with bidirectional layers"""
        input_layer = layers.Input(shape=input_shape, name=f'{name}_input')

        # Bidirectional GRU with attention
        gru1 = layers.Bidirectional(
            layers.GRU(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
        )(input_layer)
        gru1 = layers.BatchNormalization()(gru1)

        gru2 = layers.Bidirectional(
            layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        )(gru1)
        gru2 = layers.BatchNormalization()(gru2)

        # Apply attention
        gru_attention = AttentionLayer(256, num_heads=4)(gru2)

        # Bidirectional LSTM with attention
        lstm1 = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
        )(input_layer)
        lstm1 = layers.BatchNormalization()(lstm1)

        lstm2 = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        )(lstm1)
        lstm2 = layers.BatchNormalization()(lstm2)

        # Apply attention
        lstm_attention = AttentionLayer(256, num_heads=4)(lstm2)

        # Combine GRU and LSTM outputs
        combined = layers.Concatenate()([gru_attention, lstm_attention])
        dense = layers.Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(combined)
        dense = layers.BatchNormalization()(dense)
        dropout = layers.Dropout(0.4)(dense)

        return input_layer, dropout

    def build_model(self, feature_shapes, num_speakers):
        """Build the complete ensemble model with enhanced architecture"""
        inputs = []
        branches = []

        # Build branches based on available features
        for feature_name, shape in feature_shapes.items():
            if feature_name in ['mel_spec', 'chroma']:
                # Use CNN for spectral features
                input_layer, output = self.build_cnn_branch(shape, feature_name)
                inputs.append(input_layer)
                branches.append(output)
            elif feature_name == 'mfcc':
                # Use RNN for sequential features
                input_layer, output = self.build_rnn_branch(shape, feature_name)
                inputs.append(input_layer)
                branches.append(output)
            elif feature_name == 'wav2vec':
                # Use transformer for wav2vec features
                input_layer, output = self.build_transformer_branch(shape, feature_name)
                inputs.append(input_layer)
                branches.append(output)
            elif feature_name == 'additional':
                # Simple dense network for additional features
                input_layer = layers.Input(shape=shape, name=f'{feature_name}_input')
                x = layers.Flatten()(input_layer)  # Flatten if needed
                x = layers.Dense(128, activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.3)(x)
                x = layers.Dense(64, activation='relu')(x)
                x = layers.BatchNormalization()(x)
                output = layers.Dropout(0.3)(x)
                inputs.append(input_layer)
                branches.append(output)

        if not branches:
            raise ValueError("No valid features provided")

        # Enhanced ensemble fusion - ensure all branches have same dimensionality
        if len(branches) > 1:
            # Normalize branch outputs to same dimension
            normalized_branches = []
            for branch in branches:
                # Ensure all branches output 512 dimensions
                if branch.shape[-1] != 512:
                    branch = layers.Dense(512, activation='relu')(branch)
                normalized_branches.append(branch)

            merged = layers.Concatenate()(normalized_branches)
        else:
            merged = branches[0]

        # Deep classification layers with residual connections
        dense1 = layers.Dense(1024, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(merged)
        dense1 = layers.BatchNormalization()(dense1)
        dropout1 = layers.Dropout(0.5)(dense1)

        dense2 = layers.Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(dropout1)
        dense2 = layers.BatchNormalization()(dense2)

        # Residual connection
        residual = layers.Dense(512, activation='linear')(dropout1)
        dense2 = layers.Add()([dense2, residual])
        dropout2 = layers.Dropout(0.4)(dense2)

        dense3 = layers.Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(dropout2)
        dense3 = layers.BatchNormalization()(dense3)
        dropout3 = layers.Dropout(0.3)(dense3)

        # Gradient Reversal Layer for speaker invariance
        grl = GradientReversalLayer(lambda_=0.1)(dropout3)  # Reduced lambda
        speaker_dense = layers.Dense(128, activation='relu')(grl)
        speaker_output = layers.Dense(num_speakers, activation='softmax', name='speaker_output')(speaker_dense)

        # Emotion output with additional dense layer
        emotion_dense = layers.Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(dropout3)
        emotion_dense = layers.BatchNormalization()(emotion_dense)
        emotion_dense = layers.Dropout(0.2)(emotion_dense)
        emotion_output = layers.Dense(self.num_classes, activation='softmax', name='emotion_output')(emotion_dense)

        self.model = Model(inputs=inputs, outputs=[emotion_output, speaker_output])
        return self.model

    def compile_model(self, learning_rate=0.001):
        """Compile model with custom optimizer and loss weights"""
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            clipnorm=1.0  # Gradient clipping
        )

        self.model.compile(
            optimizer=optimizer,
            loss={
                'emotion_output': 'sparse_categorical_crossentropy',
                'speaker_output': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'emotion_output': 1.0,
                'speaker_output': 0.05  # Further reduced adversarial loss weight
            },
            metrics={
                'emotion_output': ['accuracy', tf.keras.metrics.SparseCategoricalAccuracy()],
                'speaker_output': 'accuracy'
            }
        )

    def train_model(self, X_train, y_train, X_val, y_val, speaker_labels, train_idx, val_idx,
                   epochs=20, batch_size=32):
        """Enhanced training with data augmentation and advanced callbacks"""

        # Create a custom callback for learning rate scheduling
        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            elif epoch < 20:
                return lr * 0.5
            elif epoch < 30:
                return lr * 0.1
            else:
                return lr * 0.05

        callbacks = [
            EarlyStopping(
                monitor='val_emotion_output_accuracy',
                patience=15,
                restore_best_weights=True,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_emotion_output_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1,
		        mode='min'
            ),
            tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
            ModelCheckpoint(
                'best_emotion_model.h5',
                monitor='val_emotion_output_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1,
		        save_weights_only=False
            )
        ]

        # Data augmentation function
        def augment_batch(X_batch, y_batch, speaker_batch):
            augmented_X = []
            augmented_y = []
            augmented_speakers = []

            for i in range(len(y_batch)):
                # Original sample
                augmented_X.append([x[i] for x in X_batch])
                augmented_y.append(y_batch[i])
                augmented_speakers.append(speaker_batch[i])

                # Add noise augmentation
                if np.random.random() > 0.5:
                    noisy_sample = []
                    for x in X_batch:
                        noise = np.random.normal(0, 0.005, x[i].shape)
                        noisy_sample.append(x[i] + noise)
                    augmented_X.append(noisy_sample)
                    augmented_y.append(y_batch[i])
                    augmented_speakers.append(speaker_batch[i])

            # Convert back to proper format
            num_features = len(X_batch)
            final_X = []
            for j in range(num_features):
                feature_data = [augmented_X[i][j] for i in range(len(augmented_X))]
                final_X.append(np.array(feature_data))

            return final_X, np.array(augmented_y), np.array(augmented_speakers)

                # Augment training data
        X_train_aug, y_train_aug, speaker_train_aug = augment_batch(
            X_train, y_train, speaker_labels[train_idx]
        )

        history = self.model.fit(
            X_train_aug,
            {'emotion_output': y_train_aug, 'speaker_output': speaker_train_aug},
            validation_data=(
                X_val,
                {'emotion_output': y_val, 'speaker_output': speaker_labels[val_idx]}
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history

# Keep the existing FuzzyEmotionAnalyzer class unchanged
class FuzzyEmotionAnalyzer:
    """Fuzzy C-Means analyzer for emotion conflicts"""
    def __init__(self, emotion_names):
        self.emotion_names = emotion_names

    def analyze_emotion_conflicts(self, predictions_proba, y_true, n_clusters=3, m=2):
        """
        Analyze emotion conflicts using Fuzzy C-Means clustering

        Args:
            predictions_proba: Prediction probabilities from the model
            y_true: True emotion labels
            n_clusters: Number of clusters for FCM
            m: Fuzziness parameter
        """
        print("\n🔍 Analyzing Emotion Conflicts using Fuzzy C-Means...")
        print("=" * 60)

        # Prepare data for FCM - transpose for sklearn format
        data = predictions_proba.T

        # Apply Fuzzy C-Means clustering
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data, n_clusters, m, error=0.005, maxiter=1000, init=None
        )

        # Get cluster membership for each sample
        cluster_membership = np.argmax(u, axis=0)

        # Calculate conflict scores for each emotion
        conflict_results = []

        for i, (true_emotion, pred_proba) in enumerate(zip(y_true, predictions_proba)):
            primary_emotion = np.argmax(pred_proba)
            primary_confidence = pred_proba[primary_emotion]

            # Sort probabilities to find secondary emotions
            sorted_indices = np.argsort(pred_proba)[::-1]
            secondary_emotion = sorted_indices[1] if len(sorted_indices) > 1 else primary_emotion
            secondary_confidence = pred_proba[secondary_emotion]

            # Calculate conflict score (uncertainty measure)
            conflict_score = 1 - primary_confidence

            # Determine if there's a significant conflict
            conflict_threshold = 0.3  # Adjustable threshold
            has_conflict = (secondary_confidence > conflict_threshold) or (conflict_score > 0.5)

            # Get fuzzy membership values
            fuzzy_memberships = u[:, i]
            dominant_cluster = cluster_membership[i]

            conflict_info = {
                'sample_id': i,
                'true_emotion': self.emotion_names[true_emotion],
                'predicted_emotion': self.emotion_names[primary_emotion],
                'primary_confidence': primary_confidence,
                'secondary_emotion': self.emotion_names[secondary_emotion],
                'secondary_confidence': secondary_confidence,
                'conflict_score': conflict_score,
                'has_conflict': has_conflict,
                'fuzzy_cluster': dominant_cluster,
                'fuzzy_memberships': fuzzy_memberships,
                'prediction_probabilities': pred_proba
            }

            conflict_results.append(conflict_info)

        # Analyze conflicts by emotion class
        self._analyze_conflicts_by_emotion(conflict_results)

        # Visualize fuzzy clustering results
        self._visualize_fuzzy_clusters(u, cluster_membership, predictions_proba, y_true)

        return conflict_results, cntr, u

    def _analyze_conflicts_by_emotion(self, conflict_results):
        """Analyze conflicts grouped by primary emotion"""
        print("\n📊 Conflict Analysis by Primary Emotion:")
        print("-" * 50)

        # Group by predicted emotion
        emotion_conflicts = {name: [] for name in self.emotion_names}

        for result in conflict_results:
            if result['has_conflict']:
                emotion_conflicts[result['predicted_emotion']].append(result)

        # Display conflicts for each emotion
        for emotion, conflicts in emotion_conflicts.items():
            if conflicts:
                print(f"\n🎭 {emotion.upper()} - Found {len(conflicts)} conflicting samples:")

                for conflict in conflicts[:5]:  # Show top 5 conflicts
                    print(f"  Sample {conflict['sample_id']}:")
                    print(f"    Primary: {conflict['predicted_emotion']} ({conflict['primary_confidence']:.3f})")
                    print(f"    Secondary: {conflict['secondary_emotion']} ({conflict['secondary_confidence']:.3f})")
                    print(f"    True: {conflict['true_emotion']}")
                    print(f"    Conflict Score: {conflict['conflict_score']:.3f}")
                    print(f"    Fuzzy Cluster: {conflict['fuzzy_cluster']}")
                    print()

                if len(conflicts) > 5:
                    print(f"    ... and {len(conflicts) - 5} more conflicts\n")
            else:
                print(f"\n✅ {emotion.upper()} - No significant conflicts detected")

    def _visualize_fuzzy_clusters(self, u, cluster_membership, predictions_proba, y_true):
          """Visualize fuzzy clustering results"""
          fig, axes = plt.subplots(2, 2, figsize=(15, 12))

          # Plot 1: Fuzzy membership values
          im1 = axes[0, 0].imshow(u, aspect='auto', cmap='viridis')
          axes[0, 0].set_title('Fuzzy Membership Matrix')
          axes[0, 0].set_xlabel('Samples')
          axes[0, 0].set_ylabel('Clusters')
          plt.colorbar(im1, ax=axes[0, 0])

          # Plot 2: Cluster assignments vs true emotions
          scatter = axes[0, 1].scatter(y_true, cluster_membership,
                                   c=cluster_membership, cmap='tab10', alpha=0.6)
          axes[0, 1].set_title('Fuzzy Clusters vs True Emotions')
          axes[0, 1].set_xlabel('True Emotion Labels')
          axes[0, 1].set_ylabel('Fuzzy Cluster Assignment')
          axes[0, 1].set_xticks(range(len(self.emotion_names)))
          axes[0, 1].set_xticklabels(self.emotion_names, rotation=45)

          # Plot 3: Prediction confidence distribution
          max_confidences = np.max(predictions_proba, axis=1)
          axes[1, 0].hist(max_confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
          axes[1, 0].set_title('Distribution of Maximum Prediction Confidence')
          axes[1, 0].set_xlabel('Confidence Score')
          axes[1, 0].set_ylabel('Frequency')
          axes[1, 0].axvline(x=0.7, color='red', linestyle='--', label='High Confidence Threshold')
          axes[1, 0].legend()

          # Plot 4: Conflict heatmap
          conflict_matrix = np.zeros((len(self.emotion_names), len(self.emotion_names)))

          for i, (true_label, pred_proba) in enumerate(zip(y_true, predictions_proba)):
              primary_pred = np.argmax(pred_proba)
              secondary_pred = np.argsort(pred_proba)[-2]

              if pred_proba[secondary_pred] > 0.2:  # Significant secondary prediction
                  conflict_matrix[primary_pred, secondary_pred] += 1

          sns.heatmap(conflict_matrix, annot=True, fmt='.0f', cmap='Reds',
                     xticklabels=self.emotion_names, yticklabels=self.emotion_names,
                     ax=axes[1, 1])
          axes[1, 1].set_title('Emotion Conflict Heatmap')
          axes[1, 1].set_xlabel('Secondary Emotion (Conflict)')
          axes[1, 1].set_ylabel('Primary Emotion (Main Prediction)')

          plt.tight_layout()
          plt.show()

class UniversalDatasetLoader:
    """Universal loader for SAVEE, CREMA-D, TESS, and RAVDESS datasets"""
    def __init__(self, dataset_path, dataset_name):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name.lower()
        # Only include the emotions you want to keep
        self.emotion_labels = {
            'anger': 0, 'angry': 0, 'a': 0, 'ang': 0,
            'disgust': 1, 'd': 1, 'dis': 1,
            'fear': 2, 'f': 2, 'fea': 2,
            'happy': 3, 'happiness': 3, 'h': 3, 'hap': 3,
            'sad': 4, 'sadness': 4, 'sa': 4,
            'neutral': 5, 'n': 5, 'neu': 5,
        }
        # Define which emotions to keep (the ones you want)
        self.target_emotions = {0, 1, 2, 3, 4, 5}  # Anger, Disgust, Fear, Happiness, Sadness, Neutral

    def load_data(self):
        file_paths, emotions, speakers = [], [], []
        if self.dataset_name == 'savee':
            for root, dirs, files in os.walk(self.dataset_path):
                for file in files:
                    if file.endswith('.wav'):
                        file_path = os.path.join(root, file)
                        filename = os.path.basename(file)
                        parts = filename.split('_')
                        if len(parts) >= 2:
                            emotion_part = parts[1].split('.')[0]
                            speaker_id = parts[0]
                            emotion_code = None
                            if emotion_part.startswith('sa'):
                                emotion_code = 'sa'
                            # Skip 'su' (surprise)
                            elif len(emotion_part) > 0 and not emotion_part.startswith('su'):
                                emotion_code = emotion_part[0]
                            if emotion_code and emotion_code in self.emotion_labels:
                                file_paths.append(file_path)
                                emotions.append(self.emotion_labels[emotion_code])
                                speakers.append(speaker_id)

        elif self.dataset_name == 'cremad':
            for root, dirs, files in os.walk(self.dataset_path):
                for file in files:
                    if file.endswith('.wav'):
                        file_path = os.path.join(root, file)
                        filename = os.path.basename(file)
                        parts = filename.split('_')
                        # For augmented CREMA-D: cremad_<emotion>_<speaker>_<utterance>_<augmentation>.wav
                        if len(parts) >= 5 and parts[0].lower() == 'cremad':
                            emotion_code = parts[1].lower()
                            speaker_id = parts[2]
                            if emotion_code in self.emotion_labels:
                                file_paths.append(file_path)
                                emotions.append(self.emotion_labels[emotion_code])
                                speakers.append(speaker_id)
                        # For original CREMA-D (fallback)
                        elif len(parts) >= 3:
                            speaker_id = parts[0]
                            emotion_code = parts[2].lower()
                            if emotion_code in self.emotion_labels:
                                file_paths.append(file_path)
                                emotions.append(self.emotion_labels[emotion_code])
                                speakers.append(speaker_id)

        elif self.dataset_name == 'tess':
            for root, dirs, files in os.walk(self.dataset_path):
                for file in files:
                    if file.endswith('.wav'):
                        file_path = os.path.join(root, file)
                        speaker_id = os.path.basename(root)
                        emotion_code = file.split('_')[-1].split('.')[0].lower()
                        if emotion_code in self.emotion_labels:
                            file_paths.append(file_path)
                            emotions.append(self.emotion_labels[emotion_code])
                            speakers.append(speaker_id)

        elif self.dataset_name == 'ravdess':
            for root, dirs, files in os.walk(self.dataset_path):
                for file in files:
                    if file.lower().endswith('.wav'):
                        file_path = os.path.join(root, file)
                        filename = os.path.basename(file)
                        parts = filename.split('-')
                        if len(parts) >= 7:
                            emotion_id = int(parts[2])
                            speaker_id = parts[6].split('.')[0]
                            # Updated RAVDESS emotion mapping (excluding calm and surprise)
                            ravdess_map = {
                                1: 5,  # neutral
                                # 2: skip (calm - not included)
                                3: 3,  # happy
                                4: 4,  # sad
                                5: 0,  # angry
                                6: 2,  # fearful
                                7: 1,  # disgust
                                # 8: skip (surprised - not included)
                            }
                            if emotion_id in ravdess_map:
                                mapped_emotion = ravdess_map[emotion_id]
                                file_paths.append(file_path)
                                emotions.append(mapped_emotion)
                                speakers.append(speaker_id)

        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

        return file_paths, emotions, speakers

def prepare_data(file_paths, emotions, feature_extractor):
    """Enhanced data preparation with normalization"""
    features_list = []
    labels = []

    print("Extracting features from audio files...")
    for i, (file_path, emotion) in enumerate(zip(file_paths, emotions)):
        if i % 50 == 0:
            print(f"Processing {i+1}/{len(file_paths)} files...")

        features = feature_extractor.extract_features(file_path)
        if features is not None and all(f is not None for f in features.values()):
            features_list.append(features)
            labels.append(emotion)

    if not features_list:
        raise ValueError("No valid features extracted from audio files")

    # Convert to structured format
    feature_arrays = {}
    feature_names = list(features_list[0].keys())

    for name in feature_names:
        feature_arrays[name] = np.array([f[name] for f in features_list])

    # Normalize features
    print("Normalizing features...")
    scalers = {}
    for name, features in feature_arrays.items():
        if name != 'wav2vec':  # Don't normalize wav2vec features
            original_shape = features.shape
            features_flat = features.reshape(features.shape[0], -1)

            scaler = StandardScaler()
            features_normalized = scaler.fit_transform(features_flat)
            feature_arrays[name] = features_normalized.reshape(original_shape)
            scalers[name] = scaler

    return feature_arrays, np.array(labels), scalers

def plot_results(history, y_true, y_pred, emotion_names):
    """Plot training results and confusion matrix"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Training history
    axes[0, 0].plot(history.history['emotion_output_accuracy'], label='Emotion Training Accuracy')
    axes[0, 0].plot(history.history['val_emotion_output_accuracy'], label='Emotion Validation Accuracy')
    axes[0, 0].set_title('Emotion Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history.history['emotion_output_loss'], label='Emotion Training Loss')
    axes[0, 1].plot(history.history['val_emotion_output_loss'], label='Emotion Validation Loss')
    axes[0, 1].set_title('Emotion Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=emotion_names, yticklabels=emotion_names, ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')

    # Classification report (as text)
    report = classification_report(
        y_true, y_pred, target_names=emotion_names, labels=list(range(len(emotion_names)))
    )
    axes[1, 1].text(0.1, 0.1, report, fontsize=10, family='monospace',
                   verticalalignment='bottom')
    axes[1, 1].set_title('Classification Report')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

def ensemble_predictions(models, X_test, num_classes=6):
    """Make ensemble predictions from multiple models"""
    predictions = []
    for model in models:
        pred = model.predict(X_test, verbose=0)
        predictions.append(pred[0])  # emotion predictions

    # Average predictions
    avg_predictions = np.mean(predictions, axis=0)
    return avg_predictions

def main():
    """Enhanced main training pipeline with ensemble and cross-validation"""
    print(f"Speech Emotion Recognition with Fuzzy Conflict Analysis - {DATASET_NAME.upper()} Dataset")
    print("=" * 80)

    # Check dataset path
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset path does not exist: {DATASET_PATH}")
        print("Please update the DATASET_PATH with the correct path.")
        return

    # Initialize components
    print("🔧 Initializing components...")
    dataset_loader = UniversalDatasetLoader(DATASET_PATH, DATASET_NAME)
    feature_extractor = AudioFeatureExtractor(n_mfcc=20, max_length=128)

    # Load dataset
    print(f"📁 Loading {DATASET_NAME.upper()} dataset...")
    file_paths, emotions, speakers = dataset_loader.load_data()
    print(f"Found {len(file_paths)} audio files")

    if len(file_paths) == 0:
        print("❌ No audio files found! Check your dataset structure.")
        return

    # Encode Speaker IDs
    speaker_encoder = LabelEncoder()
    speaker_labels = speaker_encoder.fit_transform(speakers)
    num_speakers = len(np.unique(speaker_labels))

    # Show emotion distribution
    emotion_names = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Neutral']
    emotion_counts = np.bincount(emotions, minlength=6)
    print("📊 Emotion distribution:")
    for i, (name, count) in enumerate(zip(emotion_names, emotion_counts)):
        if count > 0:
            print(f"  {name}: {count} samples")

    # Extract features with normalization
    print("🎵 Extracting audio features...")
    features_dict, labels, scalers = prepare_data(file_paths, emotions, feature_extractor)
    print("✅ Feature extraction completed!")
    print("Feature shapes:")
    for name, features in features_dict.items():
        print(f"  {name}: {features.shape}")

    # Prepare data splits with stratification
    print("🔄 Splitting dataset...")
    feature_names = ['mfcc', 'mel_spec', 'chroma', 'wav2vec', 'additional']
    X = [features_dict[name] for name in feature_names if name in features_dict]
    feature_shapes = {name: features_dict[name].shape[1:] for name in feature_names if name in features_dict}

        # Use stratified split
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.2, random_state=42, stratify=labels[train_idx]
    )

    X_train = [features[train_idx] for features in X]
    X_val = [features[val_idx] for features in X]
    X_test = [features[test_idx] for features in X]
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]

    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Test samples: {len(y_test)}")

    # Train multiple models for ensemble
    print("🏗️ Building ensemble models...")
    models = []
    histories = []

    # Model 1: Standard configuration
    print("\n📌 Training Model 1 (Standard)...")
    model1 = SpeechEmotionRecognitionModel(num_classes=6)
    model1.build_model(feature_shapes, num_speakers)
    model1.compile_model(learning_rate=0.001)

    history1 = model1.train_model(
        X_train, y_train, X_val, y_val, speaker_labels, train_idx, val_idx,
        epochs=20, batch_size=32
    )
    models.append(model1.model)
    histories.append(history1)

    # Model 2: Different learning rate and architecture
    print("\n📌 Training Model 2 (Lower LR)...")
    model2 = SpeechEmotionRecognitionModel(num_classes=6)
    model2.build_model(feature_shapes, num_speakers)
    model2.compile_model(learning_rate=0.0005)

    history2 = model2.train_model(
        X_train, y_train, X_val, y_val, speaker_labels, train_idx, val_idx,
        epochs=20, batch_size=16
    )
    models.append(model2.model)
    histories.append(history2)

    # Model 3: Different batch size
    print("\n📌 Training Model 3 (Larger Batch)...")
    model3 = SpeechEmotionRecognitionModel(num_classes=6)
    model3.build_model(feature_shapes, num_speakers)
    model3.compile_model(learning_rate=0.0008)

    history3 = model3.train_model(
        X_train, y_train, X_val, y_val, speaker_labels, train_idx, val_idx,
        epochs=20, batch_size=64
    )
    models.append(model3.model)
    histories.append(history3)

    # Evaluate individual models
    print("\n📊 Evaluating individual models...")
    for i, model in enumerate(models):
        results = model.evaluate(
            X_test,
            {'emotion_output': y_test, 'speaker_output': speaker_labels[test_idx]},
            verbose=0,
            return_dict=True
        )
        # Access metrics from the dictionary
        test_emotion_acc = results.get('emotion_output_accuracy', 
                                       results.get('emotion_output_sparse_categorical_accuracy', 0))
        print(f"Model {i+1} - Test Emotion Accuracy: {test_emotion_acc:.4f}")

    # Make ensemble predictions
    print("\n🔮 Making ensemble predictions...")
    ensemble_pred_proba = ensemble_predictions(models, X_test, num_classes=6)
    y_pred = np.argmax(ensemble_pred_proba, axis=1)

    # Calculate ensemble accuracy
    ensemble_accuracy = np.mean(y_pred == y_test)
    print(f"\n✅ Ensemble Test Accuracy: {ensemble_accuracy:.4f}")

    # Generate detailed report
    print("\n📋 Classification Report:")
    print(classification_report(
        y_test, y_pred, target_names=emotion_names, labels=list(range(len(emotion_names)))
    ))

    # Initialize Fuzzy Emotion Analyzer
    fuzzy_analyzer = FuzzyEmotionAnalyzer(emotion_names)

    # Analyze emotion conflicts using Fuzzy C-Means
    conflict_results, cluster_centers, fuzzy_matrix = fuzzy_analyzer.analyze_emotion_conflicts(
        ensemble_pred_proba, y_test, n_clusters=3, m=2
    )

    # Plot results for the best model
    print("📈 Generating plots...")
    best_history = histories[0]  # Use first model's history for plotting
    plot_results(best_history, y_test, y_pred, emotion_names)

    # Additional analysis: Per-class accuracy
    print("\n📊 Per-Class Accuracy Analysis:")
    print("-" * 50)
    cm = confusion_matrix(y_test, y_pred)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

    for i, (emotion, acc) in enumerate(zip(emotion_names, per_class_accuracy)):
        print(f"{emotion}: {acc:.4f} ({cm[i, i]}/{cm[i].sum()} correct)")

    # Analyze misclassifications
    print("\n🔍 Top Misclassification Patterns:")
    print("-" * 50)
    misclass_pairs = []
    for i in range(len(emotion_names)):
        for j in range(len(emotion_names)):
            if i != j and cm[i, j] > 0:
                misclass_pairs.append((i, j, cm[i, j]))

    misclass_pairs.sort(key=lambda x: x[2], reverse=True)
    for true_idx, pred_idx, count in misclass_pairs[:5]:
        print(f"{emotion_names[true_idx]} → {emotion_names[pred_idx]}: {count} times")

    # Additional conflict analysis summary
    print("\n" + "="*80)
    print("🎯 FUZZY CONFLICT ANALYSIS SUMMARY")
    print("="*80)

    total_conflicts = sum(1 for result in conflict_results if result['has_conflict'])
    total_samples = len(conflict_results)
    conflict_percentage = (total_conflicts / total_samples) * 100

    print(f"Total samples analyzed: {total_samples}")
    print(f"Samples with conflicts: {total_conflicts}")
    print(f"Conflict rate: {conflict_percentage:.2f}%")

    # Show most conflicted emotions
    conflict_by_emotion = {}
    for result in conflict_results:
        if result['has_conflict']:
            primary = result['predicted_emotion']
            if primary not in conflict_by_emotion:
                conflict_by_emotion[primary] = []
            conflict_by_emotion[primary].append(result)

    print("\n🔍 Most Conflicted Emotions:")
    sorted_conflicts = sorted(conflict_by_emotion.items(),
                            key=lambda x: len(x[1]), reverse=True)

    for emotion, conflicts in sorted_conflicts:
        avg_conflict_score = np.mean([c['conflict_score'] for c in conflicts])
        print(f"  {emotion}: {len(conflicts)} conflicts (avg score: {avg_conflict_score:.3f})")

    # Confidence analysis
    print("\n📊 Prediction Confidence Analysis:")
    print("-" * 50)
    max_confidences = np.max(ensemble_pred_proba, axis=1)
    print(f"Average confidence: {np.mean(max_confidences):.4f}")
    print(f"Min confidence: {np.min(max_confidences):.4f}")
    print(f"Max confidence: {np.max(max_confidences):.4f}")

    # High confidence predictions
    high_conf_mask = max_confidences > 0.8
    high_conf_accuracy = np.mean(y_pred[high_conf_mask] == y_test[high_conf_mask])
    print(f"\nHigh confidence (>0.8) samples: {np.sum(high_conf_mask)} ({np.sum(high_conf_mask)/len(y_test)*100:.1f}%)")
    print(f"High confidence accuracy: {high_conf_accuracy:.4f}")

    # Low confidence predictions
    low_conf_mask = max_confidences < 0.5
    if np.sum(low_conf_mask) > 0:
        low_conf_accuracy = np.mean(y_pred[low_conf_mask] == y_test[low_conf_mask])
        print(f"\nLow confidence (<0.5) samples: {np.sum(low_conf_mask)} ({np.sum(low_conf_mask)/len(y_test)*100:.1f}%)")
        print(f"Low confidence accuracy: {low_conf_accuracy:.4f}")

    # Save ensemble models
    print("\n💾 Saving models...")
    for i, model in enumerate(models):
        model_path = f'{DATASET_NAME.lower()}_emotion_model_{i+1}.keras'
        model.save(model_path)
        print(f"Model {i+1} saved: {model_path}")

    # Save scalers for future use
    import pickle
    with open(f'{DATASET_NAME.lower()}_scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)
    print(f"Feature scalers saved: {DATASET_NAME.lower()}_scalers.pkl")

    # Final summary
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Final Ensemble Accuracy: {ensemble_accuracy:.4f}")
    print(f"Number of models in ensemble: {len(models)}")
    print(f"Total training time: ~{len(models) * 50} epochs")

    if ensemble_accuracy >= 0.80:
        print("\n✅ TARGET ACHIEVED: Accuracy >= 80%")
    else:
        print(f"\n⚠️ Target accuracy not reached. Current: {ensemble_accuracy:.4f}, Target: 0.80")
        print("Suggestions:")
        print("- Increase training epochs")
        print("- Add more augmentation")
        print("- Collect more training data")
        print("- Fine-tune hyperparameters")

if __name__ == "__main__":
    main()