import numpy as np
import os
import tensorflow as tf
import mne
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.models import Model
from keras.layers import (Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
                          AveragePooling2D, Dense, Flatten, BatchNormalization,
                          Activation, SpatialDropout2D)
from keras.constraints import max_norm
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

mne.set_log_level('ERROR')
tf.config.set_visible_devices([], 'GPU')

# ----------------------------
# LOAD + SEGMENT + NORMALIZE (PER-TRIAL)
# ----------------------------
def load_all_preprocessed_data(save_base):
    wanted_runs = ["R03","R04","R05","R06","R07","R08","R09","R10","R11","R12"]
    sfreq = 256
    segment_length = 1.5
    segment_samples = int(sfreq * segment_length)

    label_map = {'T0': 0, 'T1': 1, 'T2': 1}

    all_X, all_y, all_subjects = [], [], []

    for subj_dir in sorted(os.listdir(save_base)):
        if not subj_dir.startswith('S'):
            continue
        
        subj_path = os.path.join(save_base, subj_dir)
        if not os.path.isdir(subj_path):
            continue
        
        print(f"Loading {subj_dir}...")

        for file in sorted(os.listdir(subj_path)):
            raw = mne.io.read_raw_fif(os.path.join(subj_path, file),
                                      preload=True, verbose=False)

            data = raw.get_data()

            for ann in raw.annotations:
                if ann['description'] not in label_map:
                    continue
                
                start_sample = int(ann['onset'] * sfreq)
                end_sample = start_sample + segment_samples
                
                if end_sample > data.shape[1]:
                    continue
                
                segment = data[:, start_sample:end_sample]

                # ✅ PER-TRIAL PER-CHANNEL NORMALIZATION (CRITICAL)
                segment = (segment - segment.mean(axis=1, keepdims=True)) / \
                          (segment.std(axis=1, keepdims=True) + 1e-6)

                all_X.append(segment)
                all_y.append(label_map[ann['description']])
                all_subjects.append(subj_dir)

    all_X = np.array(all_X)
    all_y = np.array(all_y)
    all_subjects = np.array(all_subjects)

    print(f"\nLoaded {len(all_X)} segments")
    print(f"Shape: {all_X.shape}")
    print(f"Class distribution: {np.bincount(all_y)}")

    return all_X, all_y, all_subjects

# ----------------------------
# EEGNET MODEL
# ----------------------------
def EEGNet(Chans, Samples, dropoutRate=0.2):
    inputs = Input(shape=(Chans, Samples, 1))

    x = Conv2D(16, (1, 64), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((Chans, 1), depth_multiplier=2,
                        use_bias=False,
                        depthwise_constraint=max_norm(1.))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)
    x = SpatialDropout2D(dropoutRate)(x)

    x = SeparableConv2D(32, (1, 16), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    x = SpatialDropout2D(dropoutRate)(x)

    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid',
                    kernel_constraint=max_norm(0.5))(x)

    return Model(inputs, outputs)

# ----------------------------
# FP-PENALIZED LOSS (ANTI-PHANTOM)
# ----------------------------
def weighted_bce(y_true, y_pred):
    fp_weight = 3.0   # punish phantom movement heavily
    fn_weight = 1.0
    
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    weights = y_true * fn_weight + (1 - y_true) * fp_weight
    return tf.reduce_mean(weights * bce)

# ----------------------------
# MAIN TRAINING PIPELINE
# ----------------------------
if __name__ == "__main__":
    
    save_base = "/Users/carterlawrence/Downloads/preprocessed_eeg_V4"
    print("Loading data...")
    X_all, y_all, subject_ids = load_all_preprocessed_data(save_base)

    # Subject-wise split
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, val_idx = next(gss.split(X_all, y_all, groups=subject_ids))

    X_train, X_val = X_all[train_idx], X_all[val_idx]
    y_train, y_val = y_all[train_idx], y_all[val_idx]

    print("\nTrain shape:", X_train.shape)
    print("Val shape:", X_val.shape)

    # Add channel dimension
    X_train_final = X_train[..., np.newaxis]
    X_val_final = X_val[..., np.newaxis]

    # Conservative class weights (movement downweighted)
    class_weight_dict = {0: 1.0, 1: 0.5}

    BATCH_SIZE = 32

    train_ds = tf.data.Dataset.from_tensor_slices((X_train_final, y_train))
    train_ds = train_ds.shuffle(2048).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val_final, y_val))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = EEGNet(
        Chans=X_train_final.shape[1],
        Samples=X_train_final.shape[2],
        dropoutRate=0.2
    )

    model.compile(
        optimizer=Adam(learning_rate=3e-4),
        loss=weighted_bce,
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    callbacks = [
        EarlyStopping(monitor='val_auc', patience=15, restore_best_weights=True, mode='max'),
        ReduceLROnPlateau(monitor='val_auc', patience=7, factor=0.5, min_lr=1e-6, mode='max')
    ]

    print("\nSTARTING TRAINING")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=150,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # 9. Threshold tuning
    val_probs = model.predict(X_val_final)
    
    print("\n" + "="*50)
    print("THRESHOLD TUNING")
    print("="*50)
    
    best_threshold = 0.5
    best_f1 = 0
    
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        preds = (val_probs > t).astype(int).flatten()
        
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        false_positives = ((preds == 1) & (y_val == 0)).sum()
        
        print(f"\nThreshold {t:.2f}:")
        print(f"  Accuracy:  {acc:.3f}")
        print(f"  Precision: {prec:.3f}")
        print(f"  Recall:    {rec:.3f}")
        print(f"  F1 Score:  {f1:.3f}")
        print(f"  Phantom Movements (FP): {false_positives}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    print(f"\n{'='*50}")
    print(f"RECOMMENDED THRESHOLD: {best_threshold:.2f}")
    print(f"{'='*50}")
    
    # 10. Save everything
    model.save("eegnet_rest_vs_movement_v4.h5")