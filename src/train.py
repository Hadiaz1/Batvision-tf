import yaml

import BatvisionV2_Dataset
import tensorflow as tf
from tensorflow import keras

from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.losses import MeanAbsoluteError
from tensorflow.keras.optimizers.experimental import AdamW

from models import simple_UNet

from utils import *

def get_model(params):
    if params["model"]["model_name"].lower() == "simple_UNet".lower():
        model = simple_UNet.get_scratch_model(params)

    return model

def get_optimizer(params):
    training_params = params["training"]

    if training_params["optimizer"].lower() == "Adam".lower():
        optimizer = Adam(learning_rate=training_params["initial_learning_rate"])
    elif training_params["optimizer"].lower() == "SGD".lower():
        optimizer = SGD(learning_rate=training_params["initial_learning_rate"])
    elif training_params["optimizer"].lower() == "AdamW".lower():
        optimizer = AdamW(learning_rate=training_params["initial_learning_rate"])
    else:
        optimizer = Adam(learning_rate=training_params["initial_learning_rate"])

    return optimizer

def get_loss(params):
    if params["training"]["loss"].lower() == "mae".lower():
        loss = MeanAbsoluteError()
    else:
        raise ValueError("The chosen loss is not implemented yet")

    return loss

def get_callbacks(params):
    callbacks = []

    # Learning rate scheduler
    if params["training"]["learning_rate_scheduler"].lower() == "ReduceLROnPlateau".lower():
        callbacks.append(ReduceLROnPlateau(monitor="val_loss", factor=0.05, patience=10, mode="min", min_lr=0.000001))
    else:
        raise ValueError("this learning rate scheduler is not implemented yet!")

    # EarlyStopping Callbacks
    callbacks.append((EarlyStopping(monitor="val_loss", mode="min", patience=30, restore_best_weights=True, verbose=2)))

    # Checkpoints Scheduler
    filepath = get_saving_path(params) + ".h5"
    callbacks.append(ModelCheckpoint(filepath=filepath,
                                     save_weights_only=False,
                                     monitor="val_loss",
                                     mode="min",
                                     save_best_only=True))

    return callbacks

def trainer(params, train_ds, val_ds):
    if params["training"]["load_checkpoint"]:
        model = tf.keras.models.load_model(params["training"]["load_checkpoint"])
    else:
        model = get_model(params)

    optimizer = get_optimizer(params)
    callbacks = get_callbacks(params)
    loss = get_loss(params)

    model.compile(optimizer=optimizer, loss=loss)
    history = model.fit(train_ds,
              validation_data=val_ds,
              callbacks=callbacks,
              epochs = params["training"]["epochs"])

    save_path = get_saving_path(params)
    model.save(save_path + ".h5")

    return history, model


if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))
    if params["dataset"]["name"] == "batvisionv2":
        train_ds = BatvisionV2_Dataset.load_batvisionv2_dataset(params, version="train")
        val_ds = BatvisionV2_Dataset.load_batvisionv2_dataset(params, version="val")
    else:
        raise ValueError("this batvision ds version is not implemented yet")

    history, model = trainer(params, train_ds, val_ds)




