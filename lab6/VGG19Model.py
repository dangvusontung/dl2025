import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = 64
NUM_CLASSES = 10
BATCH_SIZE = 32 
EPOCHS = 10    
LEARNING_RATE = 0.0001 

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = tf.image.resize(x_train, [IMG_SIZE, IMG_SIZE]).numpy()
x_test = tf.image.resize(x_test, [IMG_SIZE, IMG_SIZE]).numpy()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

x_train = x_train[:5000]  
y_train = y_train[:5000]
x_test = x_test[:1000]  
y_test = y_test[:1000]

print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")

class VGG19:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        model = models.Sequential()
        
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                               input_shape=(IMG_SIZE, IMG_SIZE, 3)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization()) 
        
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization())
        
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization())
        
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization())
        
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))  
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='relu')) 
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        self.model = model
        return model
        
    def compile(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def train(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=32):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy', 
                patience=3, 
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=2, 
                min_lr=1e-7
            )
        ]
        
        history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        return history
        
    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, verbose=0)
        
    def predict(self, x):
        return self.model.predict(x)

print("Building model...")
vgg = VGG19(num_classes=NUM_CLASSES)
vgg.build_model()
vgg.compile()
vgg.model.summary()

print("Starting training...")
history = vgg.train(x_train, y_train, x_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE)

test_loss, test_acc = vgg.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.grid(True)

plt.tight_layout()
plt.show()

class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
predictions = vgg.predict(x_test[:10])

plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[i])
    true_label = class_names[np.argmax(y_test[i])]
    pred_label = class_names[np.argmax(predictions[i])]
    confidence = np.max(predictions[i])
    plt.title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}')
    plt.axis('off')
plt.tight_layout()
plt.show()

print("Training finished!")
