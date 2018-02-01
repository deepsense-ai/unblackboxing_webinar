import numpy as np
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint

from experiment.utils import false_prediction_neptune_image, TARGET_NAMES

TENSORBOARD_LOGDIR = '/mnt/ml-team/homes/jakub.czakon/.unblackboxing_webinar_data/tensorboard_logs'

class BatchEndCallback(Callback):
    def __init__(self, neptune_organizer=None):
        self.neptune_organizer = neptune_organizer
        
        self.batch_id = 0

    def on_batch_end(self, batch, logs={}):
        self.batch_id += 1
        
        if self.neptune_organizer:
            self.neptune_organizer.batch_train_loss_channel.send(x=self.batch_id, y=float(logs.get('loss')))
            self.neptune_organizer.batch_train_acc_channel.send(x=self.batch_id, y=float(logs.get('acc')))
        else:
            print(self.batch_id, float(logs.get('loss')))
            print(self.batch_id, float(logs.get('acc')))  
            

class EpochEndCallback(Callback):
    def __init__(self, neptune_organizer=None, image_model=None,test_data=None):
        self.neptune_organizer = neptune_organizer
        self.image_model = image_model
        self.test_data = test_data
        
        self.epoch_id = 0
        self.false_predictions = 0

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_id += 1

        if self.neptune_organizer:
            self.neptune_organizer.epoch_train_loss_channel.send(x=self.epoch_id, y=float(logs.get('loss')))
            self.neptune_organizer.epoch_train_acc_channel.send(x=self.epoch_id, y=float(logs.get('acc')))

            self.neptune_organizer.epoch_validation_loss_channel.send(x=self.epoch_id, y=float(logs.get('val_loss')))
            self.neptune_organizer.epoch_validation_acc_channel.send(x=self.epoch_id, y=float(logs.get('val_acc')))
            
            if self.image_model and self.neptune_organizer:       
                X_test, y_test = self.test_data
                y_pred = self.image_model.predict(X_test)
                y_pred = np.argmax(y_pred, axis=1)
                y_test = np.argmax(y_test, axis=1)               
                
                for index, (prediction, actual) in enumerate(zip(y_pred, y_test)):
                    if prediction != actual:
                        self.false_predictions += 1
                        false_prediction_image = false_prediction_neptune_image(
                            X_test[index], index, self.epoch_id, 
                            TARGET_NAMES[prediction], TARGET_NAMES[actual])
                        self.neptune_organizer.image_misclassification_channel.send(x=self.false_predictions,
                                                                                    y=false_prediction_image)
                        
            
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