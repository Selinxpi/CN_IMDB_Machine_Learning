from tabnanny import verbose
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size = batch_size,
    validation_split=0.2,
    subset="training",
    seed=seed
)

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size = batch_size,
    validation_split=0.2,
    subset="validation",
    seed=seed
)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/test',
    batch_size = batch_size,
)


max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)

train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text),label


train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

embedding_dim = 16

model = tf.keras.Sequential(
    [
        layers.Embedding(max_features +1,embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(1)
    ]
)

model.summary()

model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
optimizer="adam",metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

history = model.fit(train_ds,validation_data=val_ds,epochs=10,verbose=1)

loss, accuracy = model.evaluate(test_ds)

print("Loss.........: ", loss)
print("Accuracy....: ",accuracy)

export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam",
    metrics=["accuracy"]
)
export_model.save("imdb_classifier.model")