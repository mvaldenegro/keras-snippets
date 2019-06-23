import numpy as np

import keras
from keras import backend as K

import tensorflow as tf
from tensorflow.python.framework import ops

#Code partially based on https://github.com/eclique/keras-gradcam/blob/master/grad_cam.py

def make_guided_bp_model(model_fn):
    """
        Transform model to have overriden ReLU gradient for guided backpropagation.

        Parameters
        ----------
        model_fn: callable
            Function that builds and returns a model instance    
    """

    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)

    g = K.get_session().graph
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        gbp_model = model_fn()

    return gbp_model

def guided_bp(input_image, gbp_model, layer_name, output_idx=-1):
    """
        Performs guided backpropagation saliency computation.

        Parameters
        ----------
        input_image: numpy array
            Image to compute saliency

        gbp_model: keras.models.Model
            Keras model with overriden gradient, see make_guided_bp_model

        layer_name: str
            Name of the layer to compute gradients

        output_idx: int
            Index of the output neuron to compute gradients, defaults to use whole output vector
    """

    inp = gbp_model.input
    outp = gbp_model.get_layer(layer_name).output
    
    if output_idx > 0:
        outp = outp[:, output_idx]

    gradient = K.gradients(outp, inp)[0]
    grad_fn = K.function([inp, K.learning_phase()], [gradient])
    grads_val = grad_fn([input_image, 0])[0]

    return grads_val

def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    x = x.copy()
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_last':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


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

    gbp_model = make_guided_bp_model(lambda: keras.applications.VGG16())

    gbp_heatmap = guided_bp(x, gbp_model, 'block5_conv3')

    print("GBP Heatmap shape: {}".format(gbp_heatmap.shape))

    import matplotlib.pyplot as plt

    plt.imshow(deprocess_image(gbp_heatmap[0]))
    plt.show()