from cnnClassifier.entity.config_entity import TrainingConfig
import tensorflow as tf
from pathlib import Path

class Training:
    def __init__(self, config: TrainingConfig):
        # Constructor to initialize the Training class with a provided configuration
        self.config = config

    def get_base_model(self):
        # Load the pre-trained base model from the specified path
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

    def train_valid_generate(self):
        # Data augmentation and preprocessing configuration
        datagenerator_kwargs = dict(
            rescale=1./255,  # Rescale pixel values to the range [0, 1]
            validation_split=0.20  # Split data for validation set (20%)
        )

        # Data flow configuration for training and validation
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # Target image size
            batch_size=self.config.params_batch_size,  # Batch size for training
            interpolation="bilinear"  # Interpolation method for resizing images
        )

        # Validation data generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        # Create validation data generator
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,  # Directory containing data
            subset="validation",  # Specify it's the validation set
            shuffle=False,  # Do not shuffle validation data
            **dataflow_kwargs
        )

        # Check if data augmentation is enabled
        if self.config.params_is_augmentation:
            # Training data generator with augmentation
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,  # Random rotation up to 40 degrees
                horizontal_flip=True,  # Random horizontal flipping
                width_shift_range=0.2,  # Random width shifting
                height_shift_range=0.2,  # Random height shifting
                shear_range=0.2,  # Shear transformations
                zoom_range=0.2,  # Random zooming
                **datagenerator_kwargs
            )
        else:
            # Training data generator without augmentation
            train_datagenerator = valid_datagenerator

        # Create training data generator
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,  # Directory containing data
            subset="training",  # Specify it's the training set
            shuffle=True,  # Shuffle training data
            **dataflow_kwargs
        )
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        # Save the trained model to the specified path
        model.save(path)

    def train(self, callback_list: list):
        # Calculate the number of steps per epoch for training and validation
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Train the model
        self.model.fit(
            self.train_generator,  # Training data generator
            epochs=self.config.params_epochs,  # Number of training epochs
            steps_per_epoch=self.steps_per_epoch,  # Number of steps per epoch
            validation_steps=self.validation_steps,  # Number of validation steps
            validation_data=self.valid_generator,  # Validation data generator
            callbacks=callback_list  # List of callbacks for training monitoring
        )

        # Save the trained model
        self.save_model(
            path=self.config.trained_model_path,  # Path to save the trained model
            model=self.model  # Trained model
        )
