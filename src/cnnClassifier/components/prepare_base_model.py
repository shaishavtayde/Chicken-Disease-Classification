import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path

#the above code is used to prepare the configuration, setting up all the files and varibales needed to run the model
#this code is used to execute or prepare the base model
class PrepareBaseModel:
    def __init__(self,config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape = self.config.params_image_size,
            weights = self.config.params_weights,
            include_top = self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model = self.model)
    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
         # If freeze_all is True, make all layers of the model non-trainable
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        # If freeze_till is specified and greater than 0, freeze layers until the specified index
        elif(freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        # Flatten the output of the model
        flatten_in = tf.keras.layers.Flatten()(model.output)
        
        # Add a Dense layer for classification with softmax activation
        prediction = tf.keras.layers.Dense(
            units = classes,
            activation ='softmax'
        )(flatten_in)

        # This line creates a new Keras Model. It takes the input tensor (model.input) and the output tensor (prediction) and constructs a new model that connects these two tensors. In other words, it builds a model with the same input structure as the original model but with an additional classification layer.
        '''
    The original model (model) might have been pre-trained on a different task or dataset. By creating a new model (full_model) with a modified top (classification) layer, you are adapting the model to your specific classification task (in this case, distinguishing between coccidiosis and healthy fecal samples of chickens).

This process is common in transfer learning, where a pre-trained model is used as a starting point and fine-tuned for a specific task. The lower layers of the model, which capture general features, are often kept frozen, while the top layers are modified or replaced to suit the new task.

The new full_model retains the knowledge gained by the original model and adapts it to the target classification problem.

In summary, this line of code creates a new model tailored to the specific classification task by adding a classification layer on top of the existing pre-trained layers.'''
        
        full_model = tf.keras.models.Model(
            inputs = model.input,
            outputs = prediction
        )


        '''optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate):

This line specifies the optimizer used for training the neural network. Here, it's using Stochastic Gradient Descent (SGD) as the optimizer.
learning_rate=learning_rate sets the learning rate for the optimizer. The learning rate determines the step size taken during optimization and is a crucial hyperparameter.
loss=tf.keras.losses.CategoricalCrossentropy():

This line specifies the loss function used during training. The loss function measures the difference between the predicted output and the actual target (ground truth).
CategoricalCrossentropy() is commonly used for multi-class classification problems where each sample belongs to exactly one class. It is suitable for scenarios where the classes are exclusive, such as in this case with coccidiosis and healthy fecal samples.
metrics=["accuracy"]:

This line specifies the metrics used to evaluate the model during training. Here, it's using accuracy as the evaluation metric.
The accuracy metric measures the proportion of correctly classified samples over the total number of samples.'''

        full_model.compile(
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss = tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    # the compilation step is a crucial setup phase before training the model. It defines how the model will learn from the data (optimizer),
    # how it will measure its performance (loss function), and what metrics will be used to evaluate its accuracy during training.
            
        full_model.summary()

        return full_model
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model = self.model, #referenced first in get_base_model()
            classes = self.config.params_classes,
            freeze_all = True,
            freeze_till = None,
            learning_rate = self.config.params_learning_rate
        )

        self.save_model(path = self.config.updated_base_model_path, model = self.full_model)

    
    # This function is used to save the model to disk.
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
