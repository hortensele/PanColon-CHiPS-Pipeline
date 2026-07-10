import tensorflow as tf
import h5py
import numpy as np
import argparse
import pandas as pd
import os
from models.mil.ABMIL import ABMIL
from sklearn.preprocessing import LabelEncoder

# Enable eager execution for TensorFlow 1.x (remove if using TensorFlow 2.x natively)
tf.compat.v1.enable_eager_execution()

def load_patch_embeddings(h5_path, embedding_key, slide_key):
    with h5py.File(h5_path, 'r') as f:
        patch_embeddings = np.array(f[embedding_key])
        slide_names = np.array(f[slide_key], dtype=str)
    return patch_embeddings, slide_names

def train_abmil(h5_path, embedding_key, slide_key, model_path, num_epochs=10, learning_rate=0.001):
    # Load patch embeddings and slide names
    patch_embeddings, slide_names = load_patch_embeddings(h5_path, embedding_key, slide_key)
    embedding_dim = patch_embeddings.shape[1]

    unique_slides = np.unique(slide_names)
    # Convert slide names (strings) into numerical labels
    label_encoder = LabelEncoder()
    slide_labels = label_encoder.fit_transform(unique_slides)  # Convert slide names to integers
    
    # Get unique slides and their encoded labels
    slide_label_dict = {slide: label for slide, label in zip(unique_slides, slide_labels)}

    # Define model, optimizer, and loss function
    model = ABMIL(embedding_dim=embedding_dim,num_classes=len(unique_slides))
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for slide in unique_slides:
            slide_patches = patch_embeddings[slide_names == slide]
            slide_patches_tensor = tf.convert_to_tensor(slide_patches, dtype=tf.float32)[None, :, :]
    
            label_tensor = tf.convert_to_tensor([slide_label_dict[slide]], dtype=tf.int32)  # Integer label
    
            with tf.GradientTape() as tape:
                class_probs, _ = model(slide_patches_tensor, training=True)
                loss = loss_fn(label_tensor, class_probs)
    
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
            total_loss += loss.numpy()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(unique_slides)}")
    
    model.save_weights(model_path)
    print(f"Model saved to {model_path}")

    return model

def load_trained_model(model_path, embedding_dim,num_classes):
    model = ABMIL(embedding_dim=embedding_dim,num_classes=num_classes)
    dummy_input = tf.random.normal([1, 100, embedding_dim])  # Example input tensor
    model(dummy_input)
    model.load_weights(model_path)
    print(f"Model loaded from {model_path}")
    return model

def compute_slide_embeddings(h5_path, embedding_key, slide_key, model, ssl_model, out_path):
    patch_embeddings, slide_names = load_patch_embeddings(h5_path, embedding_key, slide_key)
    unique_slides = np.unique(slide_names)
    
    for slide in unique_slides:
        slide_patches = patch_embeddings[slide_names == slide]
        slide_patches_tensor = tf.convert_to_tensor(slide_patches, dtype=tf.float32)[None, :, :]
        slide_embedding, _ = model(slide_patches_tensor, training=False)
        slide_embedding_np = slide_embedding.numpy().reshape(1, -1)
        
        csv_filename = os.path.join(out_path, f"{slide}_{ssl_model}_slide_features.csv")
        pd.DataFrame(slide_embedding_np).to_csv(csv_filename, index=False)
        print(f"Saved slide embedding to {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and infer slide embeddings using ABMIL.")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to the H5 file containing patch embeddings.")
    parser.add_argument("--embedding_key", type=str, required=True, help="Key for patch embeddings in the H5 file.")
    parser.add_argument("--slide_key", type=str, required=True, help="Key for slide names in the H5 file.")
    parser.add_argument("--ssl_model", type=str, required=True, help="SSL model name.")
    parser.add_argument("--out_path", type=str, required=True, help="Path to save the slide embeddings.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save/load the trained model.")
    
    args = parser.parse_args()
    os.makedirs(args.out_path, exist_ok=True)
    
    if os.path.exists(args.model_path):
        patch_embeddings, slide_names = load_patch_embeddings(args.h5_path, args.embedding_key, args.slide_key)
        embedding_dim = patch_embeddings.shape[1]
        num_classes = len(np.unique(slide_names))
        trained_model = load_trained_model(args.model_path, embedding_dim, num_classes)
    else:
        trained_model = train_abmil(args.h5_path, args.embedding_key, args.slide_key, args.model_path, args.num_epochs, args.learning_rate)
    
    compute_slide_embeddings(args.h5_path, args.embedding_key, args.slide_key, trained_model, args.ssl_model, args.out_path)
