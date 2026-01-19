"""
Emergency check - let's see what's actually in the data
"""

import numpy as np
import mne
from scipy import stats

base_path = "/Users/carterlawrence/Downloads/preprocessed_eeg"
event_id = {'T1': 0, 'T2': 1}
tmin, tmax = -0.25, 1.0
baseline = (0, 0)
mne.set_log_level('ERROR')

print("="*70)
print("EMERGENCY DATA CHECK")
print("="*70)

# Load ONE file to check
test_file = f"{base_path}/S002/S002R04_clean_raw.fif"

try:
    raw = mne.io.read_raw_fif(test_file, preload=True, verbose=False)
    
    print(f"\nFile loaded: {test_file}")
    print(f"Channels: {len(raw.ch_names)}")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1]:.1f} seconds")
    
    # Check events
    events, event_dict = mne.events_from_annotations(raw)
    print(f"\nALL available events: {event_dict}")
    
    # Try to get T1/T2
    try:
        events_t, event_dict_t = mne.events_from_annotations(raw, event_id=event_id)
        print(f"T1/T2 events found: {len(events_t)}")
        
        # Create epochs
        epochs = mne.Epochs(raw, events_t, event_id=event_id,
                           tmin=tmin, tmax=tmax, baseline=baseline,
                           preload=True, reject=None, verbose=False)
        
        print(f"Epochs created: {len(epochs)}")
        
        X = epochs.get_data()
        y = epochs.events[:, -1]
        
        print(f"\nData shape: {X.shape}")
        print(f"Class 0 (T1): {np.sum(y == 0)} epochs")
        print(f"Class 1 (T2): {np.sum(y == 1)} epochs")
        
        # Apply your normalization
        X_mean = X.mean(axis=(0, 2), keepdims=True)
        X_std = X.std(axis=(0, 2), keepdims=True) + 1e-6
        X_norm = (X - X_mean) / X_std
        
        print(f"\nRAW data statistics:")
        print(f"  Min: {X.min():.6e}")
        print(f"  Max: {X.max():.6e}")
        print(f"  Mean: {X.mean():.6e}")
        print(f"  Std: {X.std():.6e}")
        
        print(f"\nNORMALIZED data statistics:")
        print(f"  Min: {X_norm.min():.6f}")
        print(f"  Max: {X_norm.max():.6f}")
        print(f"  Mean: {X_norm.mean():.6f}")
        print(f"  Std: {X_norm.std():.6f}")
        
        # Check class separability
        X_class0 = X_norm[y == 0]
        X_class1 = X_norm[y == 1]
        
        print(f"\nClass separability:")
        print(f"  Class 0 mean: {X_class0.mean():.6f}")
        print(f"  Class 1 mean: {X_class1.mean():.6f}")
        print(f"  Difference: {abs(X_class0.mean() - X_class1.mean()):.6f}")
        
        # T-test
        flat_0 = X_class0.reshape(len(X_class0), -1).mean(axis=1)
        flat_1 = X_class1.reshape(len(X_class1), -1).mean(axis=1)
        t_stat, p_val = stats.ttest_ind(flat_0, flat_1)
        
        print(f"  T-test p-value: {p_val:.6f}")
        
        if p_val < 0.05:
            print("  ✓ Classes are statistically different")
        else:
            print("  ❌ Classes are NOT statistically different!")
            print("\nThis means T1 and T2 are IDENTICAL in this data!")
            print("Possible causes:")
            print("  1. Wrong event labels")
            print("  2. ASR removed all discriminative information")
            print("  3. These aren't motor imagery events")
        
        # Check what T1 and T2 actually are
        print(f"\n" + "="*70)
        print("WHAT ARE T1 AND T2?")
        print("="*70)
        print("\nIn PhysioNet EEG Motor Movement/Imagery Dataset:")
        print("  R04: Execution of motion (open/close left or right fist)")
        print("  R08: Imagery of motion (imagine opening/closing left or right fist)")
        print("  R12: Execution or imagery of motion (both fists or both feet)")
        print("\nEvent codes:")
        print("  T0: Rest")
        print("  T1: Onset of motion (real or imagined) involving left or right fist")
        print("  T2: Onset of motion (real or imagined) involving both fists or both feet")
        
        print(f"\n❌ CRITICAL ISSUE:")
        print("T1 and T2 are NOT left vs right hand!")
        print("T1 = left OR right fist (not specified which)")
        print("T2 = both fists or both feet")
        print("\nThis is why the model can't learn - you're mixing different tasks!")
        
        print(f"\n" + "="*70)
        print("SOLUTION")
        print("="*70)
        print("You need to use the ANNOTATIONS to separate left vs right:")
        print("\nFor R04 and R08, the actual events are:")
        print("  T1 (in combination with channel info) = Left or Right hand")
        print("  T2 = Both hands/feet")
        print("\nBut to properly classify left vs right, you need to:")
        print("1. Look at the detailed event descriptions in the raw file")
        print("2. OR use spatial patterns (C3 vs C4 activity) to separate them")
        print("3. OR check if there are other event markers that specify left/right")
        
    except KeyError as e:
        print(f"❌ Error: {e}")
        print("T1/T2 events not found with current event_id")
        
except Exception as e:
    print(f"❌ Fatal error: {e}")
    import traceback
    traceback.print_exc()