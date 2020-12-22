print('hello world')
import tensorflow as tf
net = tf.keras.applications.InceptionResNetV2(
    include_top=True,
    weights=None,  # 'imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=2,  # 1000,
    classifier_activation='softmax'
)
