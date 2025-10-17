from tensorflow.keras.models import load_model
import numpy as np

model = load_model("eegnet_model_workingMVP.h5", compile=False)
print(model.output_shape)

# Test on some training samples
preds = model.predict(X_all[:10], verbose=0)
print(np.argmax(preds, axis=1))
