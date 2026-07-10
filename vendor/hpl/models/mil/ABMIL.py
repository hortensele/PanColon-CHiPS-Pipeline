import tensorflow as tf
import numpy as np

class ABMIL(tf.keras.Model):
    def __init__(self, head_dim=256, n_heads=8, dropout=0.0, 
                 n_branches=1, gated=False, num_classes=128, embedding_dim=256):
        super(ABMIL, self).__init__()
        self.gated = gated
        self.n_heads = n_heads
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.attention_heads = [
            tf.keras.Sequential([
                tf.keras.layers.Dense(head_dim, activation="tanh"),
                tf.keras.layers.Dropout(dropout)
            ]) for _ in range(n_heads)
        ]
        
        if self.gated:
            self.gating_layers = [
                tf.keras.Sequential([
                    tf.keras.layers.Dense(head_dim, activation="sigmoid"),
                    tf.keras.layers.Dropout(dropout)
                ]) for _ in range(n_heads)
            ]
        
        self.branching_layers = [tf.keras.layers.Dense(n_branches) for _ in range(n_heads)]
        
        if n_heads > 1:
            self.condensing_layer = tf.keras.layers.Dense(embedding_dim)
        
        # **Embedding Layer (for inference)**
        self.embedding_layer = tf.keras.layers.Dense(embedding_dim)

        # **Classification Layer (for training)**
        self.classification_layer = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, features, training=True, attn_mask=None):
        head_features = []
        head_attentions = []
        
        for i in range(self.n_heads):
            attention_vectors = self.attention_heads[i](features)
            
            if self.gated:
                gating_vectors = self.gating_layers[i](features)
                attention_vectors *= gating_vectors
            
            attention_scores = self.branching_layers[i](attention_vectors)
            
            if attn_mask is not None:
                attention_scores = tf.where(attn_mask[:, :, None], attention_scores, -1e9)
            
            attention_scores_softmax = tf.nn.softmax(attention_scores, axis=1)
            
            weighted_features = tf.einsum('bnr,bnf->brf', attention_scores_softmax, features)
            head_features.append(weighted_features)
            head_attentions.append(attention_scores)
        
        aggregated_features = tf.concat(head_features, axis=-1)
        if self.n_heads > 1:
            aggregated_features = self.condensing_layer(aggregated_features)
        
        # Compute Embeddings
        slide_embedding = self.embedding_layer(aggregated_features)

        # Choose Output Based on Training Mode
        if training:
            class_probs = self.classification_layer(aggregated_features)  # Classification for training
            return class_probs, tf.stack(head_attentions, axis=-1)
        else:
            return slide_embedding, tf.stack(head_attentions, axis=-1)  # Embeddings for inference




	