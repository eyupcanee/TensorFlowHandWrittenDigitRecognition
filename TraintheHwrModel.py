import tensorflow as tf

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = tf.keras.utils.normalize(x_train,axis = 1)
x_test = tf.keras.utils.normalize(x_test,axis = 1)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation = tf.nn.relu),
    tf.keras.layers.Dense(128,activation = tf.nn.relu),
    tf.keras.layers.Dense(10,activation = tf.nn.softmax)
])

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate= 0.001),
              loss = tf.keras.losses.sparse_categorical_crossentropy,
              metrics = [tf.keras.metrics.sparse_categorical_accuracy])

model.fit(x=x_train,y=y_train,epochs=10)
model.save("./model/hwrModel.h5")
model.save_weights("./model/hwrModelWeights.h5")

test_loss,test_precision =model.evaluate(x = x_test,y = y_test)


print("Test Precision : ",test_precision)
