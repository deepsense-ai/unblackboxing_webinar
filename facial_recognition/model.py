import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
import keras.backend as K

from experiment.callbacks import BatchEndCallback, EpochEndCallbackImage, ModelCheckpoint, TensorBoardCallback


class FaceClassifier:
    def __init__(self, input_shape, classes, model_save_filepath):
        self.model_save_filepath = model_save_filepath

        self.old_session = K.get_session()
        session = tf.Session('')
        K.set_session(session)
        K.set_learning_phase(1)

        face_input = Input(batch_shape=(None,) + (input_shape))

        pretrained_model = VGG16(input_tensor=face_input,
                                 weights='imagenet',
                                 include_top=False)
        x = pretrained_model.get_layer('block4_pool').output

        x = Flatten(name='flatten')(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu', name='fc1')(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu', name='fc2')(x)
        output = Dense(classes, activation='softmax', name='predictions')(x)

        self.facenet = Model(face_input, output)
        self.facenet.compile(optimizer='adam',
                             loss='categorical_crossentropy',
                             metrics=['categorical_accuracy'])
        self.facenet.summary()

        self.datagen = ImageDataGenerator(rotation_range=5,
                                          horizontal_flip=False,
                                          vertical_flip=True)

    def train(self, train, valid, batch_size, **kwargs):
        X_train, y_train = train
        X_valid, y_valid = valid
        steps = len(X_train) / batch_size

        tensorboard_callback = TensorBoardCallback(batch_size)
        checkpoint = ModelCheckpoint(filepath=self.model_save_filepath)
        batch_end_callback = BatchEndCallback()
        epoch_end_callback = EpochEndCallbackImage(image_model=self.facenet, test_data=(X_valid, y_valid))

        self.facenet.fit_generator(self.datagen.flow(X_train, y_train, batch_size),
                                   steps_per_epoch=steps,
                                   validation_data=[X_valid, y_valid],
                                   callbacks=[batch_end_callback, epoch_end_callback, tensorboard_callback,
                                              checkpoint],
                                   **kwargs)
        K.set_session(self.old_session)
