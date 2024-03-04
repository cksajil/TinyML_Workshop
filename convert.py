import os
import tensorflow as tf


# Custom function to replace librosa.stft
def stft(y):
    return tf.signal.stft(y, frame_length=1024, frame_step=512, pad_end=False)


class Model(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def create_cnn_data(self, raw_data):
        Zxx = stft(raw_data)
        stft_sample = tf.abs(Zxx)
        return stft_sample


model = Model()

converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [model.create_cnn_data.get_concrete_function()], model
)

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()

with open("stft.tflite", "wb") as f:
    f.write(tflite_model)

# Print the signatures from the converted model
interpreter = tf.lite.Interpreter(model_content=tflite_model)
signatures = interpreter.get_signature_list()
# print(signatures)

MODEL_TFLITE = "stft.tflite"
MODEL_TFLITE_MICRO = "spectrogram.cc"
# xxd -i stft.tflite > spectrogram.cc
