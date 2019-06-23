import numpy as np
import keras

from keras import backend as K
from skimage.transform import resize

#Code partially based on https://github.com/eclique/keras-gradcam/blob/master/grad_cam.py

def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)

def gradCAM(input_image, model, layer_name, output_idx, normalize_grad=False):
    """
        Performs Gradient Class Activation Map method saliency computation.
        
        Parameters
        ----------
        input_image: numpy array
            Image to compute saliency

        model: keras.models.Model
            Keras model to evaluate saliency

        layer_name: str
            Name of the (convolutional) layer to compute gradients

        output_idx: int
            Index of the output neuron to compute gradients
    """

    y_c = model.output[0, output_idx]
    conv_output = model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]

    # Normalize if necessary
    if normalize_grad:
        grads = normalize(grads)
 
    gradient_function = K.function([model.input, K.learning_phase()], [conv_output, grads])

    output, grads_val = gradient_function([input_image, 0])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    if K.image_data_format() == 'channels_first':
        mean_axis = (1, 2)
    else:
        mean_axis = (0, 1)

    weights = np.mean(grads_val, axis=mean_axis)
    cam = np.dot(output, weights)

    W, H = input_image.shape[1], input_image.shape[2]

    cam = resize(cam, (W, H))
    cam = np.maximum(cam, 0)

    cam_max = cam.max() 
    if cam_max != 0: 
        cam = cam / cam_max

    return cam
    
# Example code
if __name__ == "__main__":
    import sys
    from keras.applications.vgg16 import preprocess_input, decode_predictions
    from keras.preprocessing import image

    VGG_WIDTH, VGG_HEIGHT = 224, 224
    x = image.load_img(sys.argv[1], target_size=(VGG_WIDTH, VGG_HEIGHT))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    model = keras.applications.VGG16()
    preds = model.predict(x)[0]
    cls = np.argmax(preds)

    print("Model predicted class {}".format(cls))

    cam_heatmap = gradCAM(x, model, 'block5_conv3', cls)

    print("CAM Heatmap shape: {}".format(cam_heatmap.shape))

    import matplotlib.pyplot as plt

    plt.imshow(cam_heatmap)
    plt.show()