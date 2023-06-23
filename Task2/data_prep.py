import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import pathlib

def data_overview():
    print("Overview of data")
    print("\t Training + validation data")
    train_data_dir = pathlib.Path('./img/train')

    image_count = len(list(train_data_dir.glob('*/*.jpg')))
    print("\t \t ", image_count, "images total")

    marina = list(train_data_dir.glob('marina/*'))
    print("\t \t Marina images:", len(marina))

    animal = list(train_data_dir.glob('animal/*'))
    print("\t \t Animal images:", len(animal))

    flower = list(train_data_dir.glob('flower/*'))
    print("\t \t Flower images:", len(flower))

    print("\t Test data")
    test_data_dir = pathlib.Path('img/test')

    image_count = len(list(test_data_dir.glob('*/*.jpg')))
    print("\t \t ", image_count, "images total")

    marina = list(test_data_dir.glob('marina/*'))
    print("\t \t Marina images:", len(marina))

    animal = list(test_data_dir.glob('animal/*'))
    print("\t \t Animal images:", len(animal))

    flower = list(test_data_dir.glob('flower/*'))
    print("\t \t Flower images:", len(flower))

# Load training and validation data
def get_train_val_data(BATCH_SIZE):
    print("Loading training and validation data...")
    train_data_dir = pathlib.Path('./img/train')

    # Load our data and dividing it into training and validation data
    batch_size = BATCH_SIZE
    img_height = 224
    img_width = 224

    train_ds = image_dataset_from_directory(
        train_data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = image_dataset_from_directory(
        train_data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # Print shape of images to check loading went correctly
    for image_batch, labels_batch in train_ds:
        print("\t Shape of image batch", image_batch.shape)
        print("\t Shape of label batch", labels_batch.shape)
        break

    # Buffer TODO check because it does not work currently, "shuffle buffer failed" after 108/1000
    # AUTOTUNE = tf.data.AUTOTUNE
    # train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

# Load test data
def get_test_data():
    print("Loading test data...")
    test_data_dir = pathlib.Path('./img/test')

    batch_size = 32
    img_height = 224
    img_width = 224

    test_ds = image_dataset_from_directory(
    test_data_dir,
    validation_split=0.0,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    # Print shape of images to check loading went correctly
    for image_batch, labels_batch in train_ds:
        print("\t Shape of image batch", image_batch.shape)
        print("\t Shape of label batch", labels_batch.shape)
        break

    return test_ds
