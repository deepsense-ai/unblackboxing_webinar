import numpy as np

from PIL import Image

from deepsense import neptune


def false_prediction_neptune_image(raw_image, index, epoch_number, prediction, actual):
    false_prediction_image = Image.fromarray(np.uint8(raw_image*255.))
    image_name = '(epoch {}) #{}'.format(epoch_number, index)
    image_description = 'Predicted: {}, actual: {}.'.format(prediction, actual)
    return neptune.Image(
            name=image_name,
            description=image_description,
            data=false_prediction_image)


TARGET_NAMES = {0:'Colin Powell', 
                1:'Donald Rumsfeld', 
                2:'George W Bush',
                3:'Gerhard Schroeder',
                4:'Tony Blair'}