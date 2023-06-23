print('\t \t Start of the script.')
print('---------------------------')
# Libraries and files and paths
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np

# For saving the weights
checkpoint_path_TRAINED = './WEIGHTS_SAVED/TRAINED_checkpoints'
checkpoint_path_SCRATCH = './WEIGHTS_SAVED/SCRATCH_checkpoints'

from data_prep import *
from models import *

# ----------------------------------------------------------------------
# CONTROL PARAMETERS
MODEL = 0 # Options: Pretrained - 0, Scratch - 1, Both - 2
READ_SAVE = False # To read the saved weights.

epochs = 2 # Number of epoch of training
BATCH_SIZE = 32

# HISTORY csv names
hist_file_TRAINED = './HISTORY_SAVED/history_TRAINED_model.npy'
hist_file_SCRATCH = './HISTORY_SAVED/history_SCRATCH_model.npy'

# ----------------------------------------------------------------------
# DATA PRE-PROCESSING
data_overview()
train_ds, val_ds = get_train_val_data(BATCH_SIZE)
num_classes = len(train_ds.class_names)

# ----------------------------------------------------------------------
# PRETRAINED MODEL CREATE
if MODEL == 0 or MODEL == 2:
    model = pretrained_model(num_classes=3)

    # TO READ SAVED WEIGHTS
    if READ_SAVE:
        model.load_weights(checkpoint_path_TRAINED)

    # COMPILE MODEL
    model.compile(optimizer='adam',
              loss=losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

    # TRAIN MODEL
    print('---------------------------')
    print('TRAINING')
    history_TRAINED = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    print('End training')
    print('---------------------------')


    # EVALUATE the model
    print('---------------------------')
    print('Evaluating model.')
    loss, acc = model.evaluate(val_ds, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


    # SAVE MODEL
    print('---------------------------')
    print('Storing model.')

    # Save the weights
    model.save_weights(checkpoint_path_TRAINED)

    # # Save history
    # hist_df = pd.DataFrame(history_TRAINED.history)
    # with open(hist_csv_file_TRAINED, mode='w') as f:
    #     hist_df.to_csv(f)

    np.save(hist_file_TRAINED, history_TRAINED.history)


# --------------------------------------------------------------------------------
# SCRATCH MODEL CREATE
if MODEL == 1 or MODEL == 2:
    model = scratch_model(num_classes=3)

    # TO READ SAVED WEIGHTS
    if READ_SAVE:
        model.load_weights(checkpoint_path_SCRATCH)

    # COMPILE MODEL
    model.compile(optimizer='adam',
                  loss=losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    # TRAIN MODEL
    print('---------------------------')
    print('TRAINING')
    history_SCRATCH = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=2,
        )
    print('End training')
    print('---------------------------')

    # EVALUATE the model
    print('---------------------------')
    print('Evaluating model.')
    loss, acc = model.evaluate(val_ds, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    # SAVE MODEL
    print('---------------------------')
    print('Storing model.')

    # Save the weights
    model.save_weights(checkpoint_path_SCRATCH)

    # # Save history
    # hist_df = pd.DataFrame(history_SCRATCH.history)
    # with open(hist_csv_file_SCRATCH, mode='w') as f:
    #     hist_df.to_csv(f)

    np.save(hist_file_SCRATCH, history_TRAINED.history)

print('---------------------------')
print('END SCRIPT')

