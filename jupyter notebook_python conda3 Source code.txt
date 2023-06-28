import zipfile
zip_ref= zipfile.ZipFile('dogs-vs-cats.zip', 'r')
zip_ref.extractall()
zip_ref.close()
!pip install opencv-python
! pip install tensorflow-gpu-=2.10.0
import tensorflow as tf
from tensorflow import keras
from keras import Sequential I
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
train_ds = keras.utils.image_dataset_from_directory(
directory = 'train',
labels='inferred',
label_mode = 'int',
batch_size=32,
image_size=(256,256)
)
validation_ds = keras.utils.image_dataset_from_directory(
directory = 'test',
labels='inferred',
label_mode 'int',
batch_size=32,
image_size=(256,256)
)
import tensorflow as tf
def process(image, label):
resized_image= tf.image.resize(image, (256, 256))
reshaped_image = tf.reshape(resized_image, [-1, 256, 256, 3])
transposed_image Itf.transpose(reshaped_image, perm=[0, 3, 1, 2])
processed_image = tf.cast(transposed_image/ 255., tf.float32)
return processed_image, label
train_ds = train_ds.map(process) validation_ds = validation_ds.map(process)
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
model = keras.Sequential([
Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(3, 256, 256), padding='same'), MaxPooling2D(pool_size=(2, 2), padding='same'),
Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
MaxPooling2D(pool_size=(2, 2), padding='same'),
Conv2D(64, kernel_size=(3, 3), padding='same', activation= 'relu'), MaxPooling2D(pool_size=(2, 2), padding='same')
Flatten(),
Dense (128, activation= 'relu'), Dense(1, activation= 'sigmoid')])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, epochs=10, validation_data=validation_ds)
def process_image(image_path):
image = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256)) image_array = tf.keras.preprocessing.image.img_to_array(image)
processed_image = tf.expand_dims (image_array, 0) processed_image = processed_image / 255.0
processed_image tf.transpose(processed_image, perm=[0, 3, 1, 2]) 
return processed_image
import matplotlib.pyplot as plt import cv2
test_img = cv2.imread('image.jpg') plt.imshow(test_img)
test_img.shape
test_input = process_image('image.jpg')
ans = model.predict(test_input)
prediction ans [0,0]
if prediction>= 0.5: print("Dog")
else:
print("Cat")
