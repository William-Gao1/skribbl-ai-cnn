import tensorflow as tf

target_shape = (64, 64)

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(f"../quickdraw-{target_shape[0]}.model") # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open(f"../quickdraw-{target_shape[0]}.tflite", 'wb') as f:
  f.write(tflite_model)