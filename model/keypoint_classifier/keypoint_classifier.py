import numpy as np
import tensorflow as tf

class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):
        # Initialize the KeyPointClassifier object
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        # Allocate memory for model tensors
        self.interpreter.allocate_tensors()

        # Get details about the input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        # Get the index of the input tensor
        input_details_tensor_index = self.input_details[0]['index']

        # Set the input tensor with the provided landmark_list
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))

        # Run the model to perform inference
        self.interpreter.invoke()

        # Get the index of the output tensor
        output_details_tensor_index = self.output_details[0]['index']

        # Get the result from the output tensor
        result = self.interpreter.get_tensor(output_details_tensor_index)

        # Find the index with the highest value (prediction)
        result_index = np.argmax(np.squeeze(result))

        # Return the predicted index
        return result_index
