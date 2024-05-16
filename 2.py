import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, verbose=1)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Точность на тестовом наборе: {test_acc * 100:.2f}%')
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.title('Кривая обучения')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.show()
