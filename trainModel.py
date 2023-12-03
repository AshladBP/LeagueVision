import os
import sys
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if len(sys.argv) >= 5:
    continue_training = bool(int(sys.argv[6]))
else:
    continue_training = False
output_filename = sys.argv[4]
test_data_file = sys.argv[3]
train_data_file = sys.argv[2]
if len(sys.argv) >= 6:
    previous_file = sys.argv[7] #'./working_model.h5'
epochs = int(sys.argv[1])
remove =  sys.argv[5] #remove generated image files relicats

if sys.argv[1] == "help":
    print('Usage: python3 trainModel.py epochs train_data_file test_data output_filename remove continue_training previous_file')
    exit()


def remove_unwanted(str):
    if remove == "numbers":
        return ''.join([char for char in str if not char.isdigit()])
    else:
        return str.replace(remove, "")

def create_dataframe(dir_path):
    filenames = os.listdir(dir_path)
    labels = [remove_unwanted(fname.split('.')[0]) for fname in filenames]  
    return pd.DataFrame({'filename': filenames, 'label': labels})


train_data_dir = train_data_file
test_data_dir = test_data_file

def trainModel():
    train_df = create_dataframe(train_data_dir)
    test_df = create_dataframe(test_data_dir)

    # Image dimensions
    img_height, img_width = 72, 72  

    # Data generators
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2,
                                    height_shift_range=0.2, horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    # Data generators from DataFrames
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_data_dir,
        x_col='filename',
        y_col='label',
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical')

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=test_data_dir,
        x_col='filename',
        y_col='label',
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical')

    num_classes = len(train_generator.class_indices)

    model = Sequential([
    # Convolutional layer 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    # Convolutional layer 2
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    # Convolutional layer 3
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    # Convolutional layer 4
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    # Flattening the layers
    Flatten(),
    
    # Fully connected layer 1
    Dense(1024, activation='relu'),
    Dropout(0.5),
    
    # Fully connected layer 2
    Dense(512, activation='relu'),
    Dropout(0.5),
    
    # Output layer
    Dense(num_classes, activation='softmax')
])


    if continue_training:
        model = load_model(previous_file)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // 32,
        epochs=epochs, 
        validation_data=test_generator,
        validation_steps=test_generator.samples // 32
    )

    model.save(output_filename)

trainModel()
