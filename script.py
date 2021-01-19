from copy import deepcopy
import random
import os

import matplotlib.image as mpimg
import cv2
import tensorflow as tf

from inc_test import assemble_layers, CHANNEL_AXIS, INC_HW
HW = INC_HW

# from alex_test import assemble_layers, CHANNEL_AXIS, ALEX_HW
# HW = ALEX_HW


tf.compat.v1.enable_eager_execution()
HEIGHT_WIDTH = 299 #inc
# HEIGHT_WIDTH = 224
# HEIGHT_WIDTH = 227 #alex
BATCH_SIZE = 10
VERBOSE = 2

SANITY_SWITCH = False
SANITY_MIX = True

print('starting script')

# tf.keras.applications.vgg19
# net = tf.keras.applications.InceptionResNetV2(
#     # net = tf.keras.applications.InceptionV3(
#     # net = tf.keras.applications.VGG19(
#     include_top=True,
#     weights=None,  # 'imagenet' v,
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=2,  # 1000,
#     classifier_activation='softmax'
# )
#
# model = Sequential()
# for layer in net.layers[:-1]:  # go through until last layer
#     if "_bn" in layer.name or "batch_normalization" in layer.name:
#         print(f'skipping {layer.name}')
#     else:
#         print(f'adding {layer.name}')
#         model.add(layer)
# # model.add(Dense(3, activation='softmax'))
# model.summary()
# net = model
# # model.compile(optimizer='adam', loss='categorical_crossentropy')







# dims = [HW, HW, HW]
# dims[CHANNEL_AXIS - 1] = 3
# from tensorflow.python.keras import Input
# inputs = Input(tuple(dims))
#
# net = tf.python.keras.models.Model(
#     inputs=inputs,
#     outputs=assemble_layers(inputs)
#     # name=self.FULL_NAME.replace(' ', '_')
# )





from proko_inc import CustomInceptionResNetV2
net = CustomInceptionResNetV2(
    include_top=True,
    weights=None,  # 'imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1,  # 1000,2
    classifier_activation='sigmoid',
    layers=tf.keras.layers
)

print_output = True
def utility_metric(y_true, y_pred):
    global print_output
    if print_output:
        print(f'y_true:{y_true.numpy()}')
        print(f'y_pred:{y_pred.numpy()}')
        print_output = False
    return 0


net.compile(
    # optimizer='ADAM',
    optimizer='adam',
    # loss='sparse_categorical_crossentropy',
    loss='binary_crossentropy',
    # metrics=['accuracy', utility_metric]
    metrics=['accuracy']
)

net.run_eagerly = True

class_map = {'dog': 0, 'cat': 1}

def preprocess(file):
    imdata = mpimg.imread(file)




    # imdata = cv2.resize(imdata, dsize=(HEIGHT_WIDTH, HEIGHT_WIDTH), interpolation=cv2.INTER_LINEAR)
    # imdata.shape = (HEIGHT_WIDTH, HEIGHT_WIDTH, 3)
    # imdata /= 127.5
    # imdata -= 1.

    imdata = cv2.resize(imdata, dsize=(HEIGHT_WIDTH, HEIGHT_WIDTH), interpolation=cv2.INTER_LINEAR) * 255.0
    imdata = tf.keras.applications.inception_resnet_v2.preprocess_input(
        imdata, data_format=None
    )

    return imdata, class_map[os.path.basename(os.path.dirname(file))]



train_data = [f'data/Training/cat/{x}' for x in os.listdir('data/Training/cat')] + [f'data/Training/dog/{x}' for x in
                                                                                    os.listdir('data/Training/dog')]
test_data = [f'data/Testing/cat/{x}' for x in os.listdir('data/Testing/cat')] + [f'data/Testing/dog/{x}' for x in
                                                                                 os.listdir('data/Testing/dog')]

random.shuffle(train_data)
random.shuffle(test_data)

if SANITY_MIX:
    print('doing SANITY_MIX')
    mixed_data = train_data + test_data
    random.shuffle(mixed_data)
    train_data = mixed_data[:len(train_data)]
    test_data = mixed_data[len(train_data):]

if SANITY_SWITCH:
    print('doing SANITY_SWITCH')
    tmp_data = train_data
    train_data = test_data
    test_data = tmp_data


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
print('starting training')
net.fit(
    get_ds(train_data),
    epochs=50,
    verbose=VERBOSE,
    use_multiprocessing=True,
    workers=16,
    batch_size=BATCH_SIZE,
    shuffle=False,
    validation_data=get_ds(deepcopy(test_data))
)
print('starting testing')
print_output = True
net.evaluate(
    get_ds(test_data),
    verbose=VERBOSE,
    batch_size=BATCH_SIZE,
    use_multiprocessing=True,
    workers=16,
)
print('script complete')
