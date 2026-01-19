import numpy as np
import tensorflow as tf
import mne
from keras.models import Model
from keras.layers import (Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
                          AveragePooling2D, Dropout, Dense, Flatten, 
                          BatchNormalization, Activation, GlobalAveragePooling2D)
from keras.constraints import max_norm
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import os

# ----------------------------
# PARAMETERS - OPTIMIZED
# ----------------------------
sfreq = 256
tmin, tmax = -0.25,1.00  # YOUR ACTUAL WORKING VERSION!
baseline = (0, 0)  
reject_criteria = dict(eeg=200e-6)  
event_id = {'T1': 1, 'T2': 0}  # SWAPPED - this was the bug!

base_path = "/Users/carterlawrence/Downloads/preprocessed_eeg_V2"
wanted_runs = ["R04", "R08", "R12"]
mne.set_log_level('ERROR')

# ----------------------------
# DATA AUGMENTATION FUNCTIONS
# ----------------------------
def temporal_shift(X, max_shift=20):
    """Randomly shift signals in time"""
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(X, shift, axis=-1)

def add_noise(X, noise_level=0.05):
    """Add Gaussian noise"""
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def channel_dropout(X, p=0.1):
    """Randomly drop channels"""
    mask = np.random.binomial(1, 1-p, X.shape[1])
    X_aug = X.copy()
    X_aug[:, mask == 0, :] = 0
    return X_aug

def augment_batch(X, y):
    """Apply random augmentations"""
    X_aug = []
    for i in range(len(X)):
        x = X[i].copy()
        # Apply augmentations with 50% probability each
        if np.random.rand() > 0.5:
            x = temporal_shift(x)
        if np.random.rand() > 0.5:
            x = add_noise(x)
        if np.random.rand() > 0.5:
            x = channel_dropout(x)
        X_aug.append(x)
    return np.array(X_aug), y

# ----------------------------
# IMPROVED EEGNET
# ----------------------------
def EEGNet_Improved(nb_classes, Chans, Samples, dropoutRate=0.2, F1=16, D=4, F2=32):
    """
    Improved EEGNet with better regularization
    F1: Number of temporal filters
    D: Depth multiplier (spatial filters per temporal filter)
    F2: Number of pointwise filters
    """
    input_main = Input(shape=(Chans, Samples, 1))
    
    # Block 1: Temporal convolution + Spatial filtering  
    block1 = Conv2D(F1, (1, 64), padding='same', use_bias=False,
                    kernel_constraint=max_norm(2.))(input_main)  # YOUR WORKING KERNEL SIZE
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, 
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)
    
    # Block 2: Separable convolution
    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropoutRate)(block2)
    
    # Classification head - keep your original architecture that works!
    flatten = Flatten()(block2)
    dense1 = Dense(64, activation='elu', kernel_constraint=max_norm(0.25))(flatten)
    dense1 = Dropout(dropoutRate)(dense1)
    dense2 = Dense(nb_classes, kernel_constraint=max_norm(0.25))(dense1)
    softmax = Activation('softmax')(dense2)
    
    return Model(inputs=input_main, outputs=softmax)

# ----------------------------
# LOAD AND PREPARE DATA
# ----------------------------
print("Loading preprocessed data...")
all_X, all_y, all_subjects = [], [], []

for subj in range(1, 110):
    subj_str = f"S{str(subj).zfill(3)}"
    subj_folder = f"{base_path}/{subj_str}"
    
    if not os.path.exists(subj_folder):
        continue
        
    for run in wanted_runs:
        file = f"{subj_folder}/{subj_str}{run}_clean_raw.fif"
        
        if not os.path.exists(file):
            continue
            
        try:
            raw_clean = mne.io.read_raw_fif(file, preload=True, verbose=False)
            events, event_dict = mne.events_from_annotations(raw_clean, event_id=event_id)
            
            if len(events) == 0:
                continue
                
            epochs = mne.Epochs(raw_clean, events, event_id=event_id,
                                tmin=tmin, tmax=tmax, baseline=baseline,
                                preload=True, reject=reject_criteria, verbose=False)
            
            if len(epochs) < 2:  # Need at least 2 trials
                continue
            
            X = epochs.get_data()
            y = epochs.events[:, -1]
            
            # YOUR ORIGINAL NORMALIZATION (works better!)
            X_mean = X.mean(axis=(0, 2), keepdims=True)
            X_std = X.std(axis=(0, 2), keepdims=True) + 1e-6
            X = (X - X_mean) / X_std
            
            all_X.append(X)
            all_y.append(y)
            all_subjects.append(np.full(len(y), subj))
            
        except Exception as e:
            print(f"Error with {subj_str}{run}: {e}")
            continue

print(f"Loaded {len(all_X)} runs")

# Concatenate all data
X_all = np.concatenate(all_X, axis=0)
y_all = np.concatenate(all_y, axis=0)
subjects = np.concatenate(all_subjects)

# Add channel dimension
X_all = X_all[..., np.newaxis]

# Check class balance
unique, counts = np.unique(y_all, return_counts=True)
print(f"\nClass distribution: {dict(zip(unique, counts))}")
print(f"Total samples: {len(y_all)}")
print(f"Shape: {X_all.shape}")

# Compute class weights for imbalanced data
class_weights = compute_class_weight('balanced', classes=np.unique(y_all), y=y_all)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

# ----------------------------
# CROSS-VALIDATION TRAINING
# ----------------------------
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

fold_scores = []
best_models = []

print(f"\nStarting {n_folds}-fold cross-validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
    print(f"\n{'='*50}")
    print(f"FOLD {fold + 1}/{n_folds}")
    print(f"{'='*50}")
    
    # Split data
    X_train, X_val = X_all[train_idx], X_all[val_idx]
    y_train, y_val = y_all[train_idx], y_all[val_idx]
    
    # Convert to categorical
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_val_cat = to_categorical(y_val, num_classes=2)
    
    # Build model
    model = EEGNet_Improved(
        nb_classes=2,
        Chans=X_all.shape[1],
        Samples=X_all.shape[2],
        dropoutRate=0.5,
        F1=16,
        D=4,
        F2=32
    )
    
    # Compile with optimized settings
    # Use batch_size=32 for better convergence
    optimizer = Adam(learning_rate=1.0)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=20,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            f'best_model_fold_{fold}.keras',  # Use .keras format
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        )
    ]
    
    # Train
    history = model.fit(
        X_train, y_train_cat,
        batch_size=32,  # Use your original batch size
        epochs=300,
        validation_data=(X_val, y_val_cat),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
    fold_scores.append(val_acc)
    best_models.append(model)
    
    print(f"\nFold {fold + 1} Results:")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")

# ----------------------------
# FINAL RESULTS
# ----------------------------
print(f"\n{'='*50}")
print("CROSS-VALIDATION RESULTS")
print(f"{'='*50}")
print(f"Mean Accuracy: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
print(f"Best Fold Accuracy: {np.max(fold_scores):.4f}")
print(f"Worst Fold Accuracy: {np.min(fold_scores):.4f}")

# Save best overall model (use .keras format)
best_fold_idx = np.argmax(fold_scores)
best_models[best_fold_idx].save("best_eegnet_model.keras")
print(f"\nBest model (Fold {best_fold_idx + 1}) saved as 'best_eegnet_model.keras'")

# ----------------------------
# ENSEMBLE PREDICTION (OPTIONAL)
# ----------------------------
print("\nCreating ensemble predictions...")
ensemble_preds = []
for model in best_models:
    preds = model.predict(X_val, verbose=0)
    ensemble_preds.append(preds)

# Average predictions
ensemble_pred = np.mean(ensemble_preds, axis=0)
ensemble_acc = np.mean(np.argmax(ensemble_pred, axis=1) == y_val)
print(f"Ensemble Accuracy: {ensemble_acc:.4f}")