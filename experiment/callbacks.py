from PIL import Image
import numpy as np
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint
from deepsense import neptune

from experiment.utils import false_prediction_neptune_image, TARGET_NAMES

TENSORBOARD_LOGDIR = '/mnt/ml-team/homes/jakub.czakon/.unblackboxing_webinar_data/tensorboard_logs'


class BatchEndCallback(Callback):
    def __init__(self):
        self.ctx = neptune.Context()
        self.batch_id = 0

    def on_batch_end(self, batch, logs={}):
        self.batch_id += 1

        self.ctx.channel_send('batch_train_loss_channel', x=self.batch_id, y=float(logs.get('loss')))
        self.ctx.channel_send('batch_train_acc_channel', x=self.batch_id, y=float(logs.get('categorical_accuracy')))


class EpochEndCallback(Callback):
    def __init__(self, test_data=None):
        self.ctx = neptune.Context()
        self.test_data = test_data

        self.epoch_id = 0
        self.false_predictions = 0

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_id += 1

        self.ctx.channel_send('epoch_train_loss_channel', x=self.epoch_id, y=float(logs.get('loss')))
        self.ctx.channel_send('epoch_train_acc_channel', x=self.epoch_id, y=float(logs.get('categorical_accuracy')))
        self.ctx.channel_send('epoch_validation_loss_channel', x=self.epoch_id, y=float(logs.get('val_loss')))
        self.ctx.channel_send('epoch_validation_acc_channel', x=self.epoch_id, y=float(logs.get('val_categorical_accuracy')))


class EpochEndCallbackImage(EpochEndCallback):
    def __init__(self, image_model, test_data):
        super().__init__(test_data)
        self.image_model = image_model

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_id += 1

        self.ctx.channel_send('epoch_train_loss_channel', x=self.epoch_id, y=float(logs.get('loss')))
        self.ctx.channel_send('epoch_train_acc_channel', x=self.epoch_id, y=float(logs.get('categorical_accuracy')))
        self.ctx.channel_send('epoch_validation_loss_channel', x=self.epoch_id, y=float(logs.get('val_loss')))
        self.ctx.channel_send('epoch_validation_acc_channel', x=self.epoch_id, y=float(logs.get('val_categorical_accuracy')))

        self.send_to_image_channel()

    def send_to_image_channel(self):
        X_test, y_test = self.test_data
        y_pred = self.image_model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)

        for index, (prediction, actual) in enumerate(zip(y_pred, y_test)):
            if prediction != actual:
                self.false_predictions += 1
                pill_image = Image.fromarray(np.uint8(X_test[index] * 255.))

                self.ctx.channel_send('missclassification image channel', neptune.Image(
                    name='misclassification',
                    description="pred {} true {}".format(TARGET_NAMES[prediction], TARGET_NAMES[actual]),
                    data=pill_image))


def TensorBoardCallback(batch_size):
    return TensorBoard(log_dir=TENSORBOARD_LOGDIR,
                       histogram_freq=2,
                       batch_size=batch_size,
                       write_graph=True,
                       write_grads=False,
                       write_images=False,
                       embeddings_freq=0,
                       embeddings_layer_names=None,
                       embeddings_metadata=None)
