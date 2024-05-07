import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_patt_lite(input_shape=(224, 224, 3), weights=None):
    # Load and truncate MobileNetV1
    base_model = tf.keras.applications.MobileNet(input_shape=input_shape,
                                                 include_top=False,  # Do not include the classification layer.
                                                 weights='imagenet')
    # Create a new model that ends at block 9's output
    base_output = base_model.get_layer('conv_pw_9_relu').output

    # Patch Extraction Block
    patch_extraction = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(base_output)
    patch_extraction = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(patch_extraction)

    # Global Average Pooling
    gap = layers.GlobalAveragePooling2D()(patch_extraction)

    # Attention Mechanism
    attention_probs = layers.Dense(units=128, activation='softmax', name='attention_probs')(gap)
    attended_features = layers.Multiply()([gap, attention_probs])

    # Classifier
    dense_layer = layers.Dense(128, activation='relu')(attended_features)
    output = layers.Dense(7, activation='softmax')(dense_layer)  # Assuming 7 classes for FER

    # Create the model
    model = models.Model(inputs=base_model.input, outputs=output)

    # Load weights if specified
    if weights is not None:
        model.load_weights(weights)

    return model

def compile_and_train(model, train_data, validation_data, epochs=100):
    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Training the model
    history = model.fit(train_data,
                        epochs=epochs,
                        validation_data=validation_data)

    return history

def prepare_data(train_dir, val_dir, batch_size=32, target_size=(224, 224)):
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical')

    return train_generator, validation_generator

# Set up paths (These need to be changed to the actual paths where the dataset is stored)
train_dir = '/content/images/train'
val_dir = '/content/images/test'

# Model Creation with pre-trained weights
model = create_patt_lite(weights='patt_lite_weights.h5')

# Data Preparation
train_data, validation_data = prepare_data(train_dir, val_dir)

# Optionally, continue training the model or just perform inference
history = compile_and_train(model, train_data, validation_data, epochs=10)
