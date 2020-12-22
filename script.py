print('hello world')
import tensorflow as tf
import tensorflow_datasets as tfds
net = tf.keras.applications.InceptionResNetV2(
    include_top=True,
    weights=None,  # 'imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=2,  # 1000,
    classifier_activation='softmax'
)
def test_metric(y_true, y_pred):
    pass
net.compile(
    optimizer='ADAM',
    loss='sparse_categorical_crossentropy',
    metrics=[test_metric]
)

ds = tfds.folder_dataset.ImageFolder(root_dir='data')
net.fit(
    # x,y,
    ds.as_dataset(split='Training'),
    epochs=10,
    # verbose=self.VERBOSE_MODE,
    use_multiprocessing=True,
    workers=16,
    # steps_per_epoch=steps,
    batch_size=10,
    shuffle=False
)
net.evaluate(
    ds.as_dataset(split='Testing'),
    # verbose=self.VERBOSE_MODE
    # steps=steps,
    batch_size=10,
    use_multiprocessing=True,
    workers=16,
)
