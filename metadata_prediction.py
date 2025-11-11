"""
This script trains a multi-output CNN model (ResNet50) to predict
the 'State' and 'Species' of a pasture based on an image.

The trained model and the corresponding label encoders are saved to disk
for later use in the final biomass prediction pipeline.
"""
import pandas as pd
from pathlib import Path
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


def create_multi_output_generator(generator, df, data_path, batch_size, num_states, num_species, img_size=(224, 224)):
    """
    Creates a generator that yields batches of images and a dictionary of 
    one-hot encoded labels for a multi-output Keras model.
    """
    gen = generator.flow_from_dataframe(
        dataframe=df,
        directory=data_path,
        x_col='image_path',
        y_col=['state_encoded', 'species_encoded'],
        class_mode='raw',
        target_size=img_size,
        batch_size=batch_size
    )
    while True:
        X, y = next(gen)
        yield X, {
            'state_output': to_categorical(y[:, 0], num_classes=num_states),
            'species_output': to_categorical(y[:, 1], num_classes=num_species)
        }

def train_and_save_metadata_model():
    """
    Main function to load data, build, train, and save the metadata prediction model.
    """
    # 1. Setup Paths and Parameters
    data_path = Path('.')
    batch_size = 32
    img_size = (224, 224)
    
    # 2. Load and Preprocess Data
    print("Loading and preprocessing data...")
    train_df = pd.read_csv(data_path / 'train.csv')
    metadata_df = train_df[['image_path', 'State', 'Species']].drop_duplicates().reset_index(drop=True)

    # 3. Encode Labels
    state_encoder = LabelEncoder()
    species_encoder = LabelEncoder()
    metadata_df['state_encoded'] = state_encoder.fit_transform(metadata_df['State'])
    metadata_df['species_encoded'] = species_encoder.fit_transform(metadata_df['Species'])

    num_states = len(state_encoder.classes_)
    num_species = len(species_encoder.classes_)

    # 4. Split Data
    train_meta_df, val_meta_df = train_test_split(metadata_df, test_size=0.2, random_state=42)

    # 5. Build Model
    print("Building model...")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    state_output = Dense(num_states, activation='softmax', name='state_output')(x)
    species_output = Dense(num_species, activation='softmax', name='species_output')(x)

    model = Model(inputs=base_model.input, outputs=[state_output, species_output])

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer='adam', 
                  loss={'state_output': 'categorical_crossentropy', 'species_output': 'categorical_crossentropy'},
                  metrics={'state_output': 'accuracy', 'species_output': 'accuracy'})

    # 6. Create Data Generators
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = create_multi_output_generator(train_datagen, train_meta_df, data_path, batch_size, num_states, num_species, img_size)
    val_generator = create_multi_output_generator(val_datagen, val_meta_df, data_path, batch_size, num_states, num_species, img_size)

    # 7. Train the Model
    print("Starting model training...")
    model.fit(
        train_generator,
        steps_per_epoch=len(train_meta_df) // batch_size,
        epochs=10,
        validation_data=val_generator,
        validation_steps=len(val_meta_df) // batch_size
    )
    print("Model training finished.")

    # 8. Save Artifacts
    print("Saving model and encoders...")
    model.save(data_path / 'metadata_model.h5')
    with open(data_path / 'state_encoder.pkl', 'wb') as f:
        pickle.dump(state_encoder, f)
    with open(data_path / 'species_encoder.pkl', 'wb') as f:
        pickle.dump(species_encoder, f)

    print('Metadata prediction model trained and saved successfully!')


if __name__ == '__main__':
    train_and_save_metadata_model()