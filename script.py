import random

import os

import matplotlib.image as mpimg
import cv2
print('hello world')
import tensorflow as tf
HEIGHT_WIDTH = 299
BATCH_SIZE = 10
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
    # metrics=[test_metric]
)
class_map = {'dog': 0, 'cat': 1}
def preprocess(file):
    imdata = mpimg.imread(file)
    imdata = cv2.resize(imdata, dsize=(HEIGHT_WIDTH, HEIGHT_WIDTH), interpolation=cv2.INTER_LINEAR)
    imdata.shape = (HEIGHT_WIDTH, HEIGHT_WIDTH, 3)
    imdata /= 127.5
    imdata -= 1.
    return imdata, class_map[os.path.dirname(file)]


train_data = [f'data/Training/cat/{x}' for x in os.listdir('data/Training/cat')] + [f'data/Training/dog/{x}' for x in os.listdir('data/Training/dog')]
test_data = [f'data/Testing/cat/{x}' for x in os.listdir('data/Testing/cat')] + [f'data/Testing/dog/{x}' for x in os.listdir('data/Testing/dog')]
random.shuffle(train_data)
random.shuffle(test_data)


def get_gen(data):
    def gen():
        pairs = []
        i = 0
        for im_file in data:
            i += 1
            if i <= BATCH_SIZE:
                pairs += [preprocess(im_file)]
            if i == BATCH_SIZE:
                yield (
                    [pair[0] for pair in pairs],
                    [pair[1] for pair in pairs]
                )
                pairs.clear()
                i = 0
    return gen

def get_ds(data):
    return tf.data.Dataset.from_generator(
        get_gen(data),
        (tf.float32, tf.int64),
        output_shapes=(
            tf.TensorShape((BATCH_SIZE, HEIGHT_WIDTH, HEIGHT_WIDTH, 3)),
            tf.TensorShape(([BATCH_SIZE]))
        )
    )
net.fit(
    get_ds(train_data),
    epochs=5,
    # verbose=self.VERBOSE_MODE,
    use_multiprocessing=True,
    workers=16,
    # steps_per_epoch=steps,
    batch_size=BATCH_SIZE,
    shuffle=False
)
net.evaluate(
    get_ds(test_data),
    # verbose=self.VERBOSE_MODE
    # steps=steps,
    batch_size=BATCH_SIZE,
    use_multiprocessing=True,
    workers=16,
)
