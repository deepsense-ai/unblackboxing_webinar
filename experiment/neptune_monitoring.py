from deepsense import neptune


class NeptuneOrganizer:
    def __init__(self):
        self.ctx = neptune.Context()
        self.ctx.integrate_with_tensorflow()

    def create_channels(self):

        self.batch_train_loss_channel = self.ctx.job.create_channel(
            name='batch_train_loss',
            channel_type=neptune.ChannelType.NUMERIC)
        
        self.batch_train_acc_channel = self.ctx.job.create_channel(
            name='batch_train_acc',
            channel_type=neptune.ChannelType.NUMERIC)

        self.epoch_train_loss_channel = self.ctx.job.create_channel(
            name='epoch_train_loss',
            channel_type=neptune.ChannelType.NUMERIC)
        
        self.epoch_train_acc_channel = self.ctx.job.create_channel(
            name='epoch_train_acc',
            channel_type=neptune.ChannelType.NUMERIC)

        self.epoch_validation_loss_channel = self.ctx.job.create_channel(
            name='epoch_validation_loss',
            channel_type=neptune.ChannelType.NUMERIC)
        
        self.epoch_validation_acc_channel = self.ctx.job.create_channel(
            name='epoch_validation_acc',
            channel_type=neptune.ChannelType.NUMERIC)
        
        self.image_misclassification_channel= self.ctx.job.create_channel(
            name='false_predictions',
            channel_type=neptune.ChannelType.IMAGE)

        self.logging_channel = self.ctx.job.create_channel(
            name='logging_channel',
            channel_type=neptune.ChannelType.TEXT)

    def create_charts(self):

        self.ctx.job.create_chart(
            name='Batch training loss',
            series={
                'training loss': self.batch_train_loss_channel
            }
        )

        self.ctx.job.create_chart(
            name='Batch training accuracy',
            series={
                'training': self.batch_train_acc_channel
            }
        )

        self.ctx.job.create_chart(
            name='Epoch training and validation loss',
            series={
                'training': self.epoch_train_loss_channel,
                'validation': self.epoch_validation_loss_channel
            }
        )

        self.ctx.job.create_chart(
            name='Epoch training and validation accuracy',
            series={
                'training': self.epoch_train_acc_channel,
                'validation': self.epoch_validation_acc_channel
            }
        )