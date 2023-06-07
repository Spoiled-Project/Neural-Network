from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.optimizers import Adam

from keras.callbacks import EarlyStopping

num_classes = 3
img_dir = "fullImages"


def extract_label_from_filename(filename):
    label_map = {'avatar': 0, 'Nothing': 1, 'TheGoodPlace': 2}
    label = filename.split("\\")[1]
    return label_map[label]


# Load the ResNet50 model, excluding the final fully-connected layer
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Add a new Flatten layer
x = base_model.output
x = Flatten()(x)

# Add a dense layer with softmax activation for multi-class classification
predictions = Dense(num_classes, activation='softmax')(x)

# Create a new model with the added layers
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the model layers
for layer in base_model.layers:
    layer.trainable = False

# Define the directory path where your training data is stored
train_directory = img_dir

# Set the batch size for training
batch_size = 32

# Create an ImageDataGenerator for data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Generate the training data from the directory
train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)
learning_rate = 0.01
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(train_generator, epochs=15)

model.save('my_full_model.h5')