import math

from PIL import Image
from numpy import mean
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchnet as tnt

import random
import os
# import matplotlib.image as mpimg
# import cv2
# import tensorflow as tf

# from mlib.boot.mlog import err
from zhulf_inc import Inception_ResNetv2


HEIGHT_WIDTH = 299
BATCH_SIZE = 10
VERBOSE = 1

BASE_WEIGHT_URL = ('https://storage.googleapis.com/tensorflow/'
                   'keras-applications/inception_resnet_v2/')
layers = None

# err('not pretrained')


def train(model, epochs, num_ims_per_class):
    print('starting script')

    class_map = {'dog': 0, 'cat': 1}



    # IM_COUNT = 20 # I think it learned in like 12 epochs, not sure
    # another try:
    #     train: starting really getting better around e.27
    #     test: started really getting better around e.24
    #  ended with train and test accuracy both being 0.9!

    # IM_COUNT = 35
    # try 1: seemed to start learning at epoch 10, reaching .92 accuracy by 31, but then went back down to .50 at epoch 41 and never recovered
    # try 2: didnt really learn ever

    # overfitting?
    # look at both accuracy and val accuracy!


    # IM_COUNT = 50 # 0.5 until epoch 44, then started going up and down a bit until epoch 50. max 0.68, never below 0.5
    # another try:
    #     train: never above .52
    #     test: always .5


    class CustomDataset(Dataset):
        def __init__(self, X, length):
            self.X = X
            # self.Y = Y
            self.length = length
            # if len(self.X) != len(self.Y):
            #     raise Exception("The length of X does not match the length of Y")
            self.input_size = (299, 299, 3)
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
            self.input_range = [0, 1]
            self.scale = 0.875
            self.space = 'RGB'
            # self.filenames = glob.glob(os.path.join(self.data_dir, "*.JPEG"))
            # self.validation_synset_labels = self.get_label()
            # self.synset2label = self.synset2label()

        def transform(self, img):
            tfs = []
            tfs.append(transforms.Resize(int(math.floor(max(self.input_size) / self.scale))))
            tfs.append(transforms.CenterCrop(max(self.input_size)))
            tfs.append(transforms.ToTensor())
            tfs.append(transforms.Normalize(self.mean, self.std))
            tf = transforms.Compose(tfs)
            return tf


        def preprocess(self, file):
            # filename = self.filenames[idx]
            img = Image.open(file).convert(self.space)
            transforms = self.transform(img)
            tensor = transforms(img)
            # basename = os.path.basename(file).split('.')[0].split('_')[-1]
            # label = self.synset2label[self.validation_synset_labels[int(basename) - 1]]
            label = class_map[os.path.basename(os.path.dirname(file))]

            # imdata = mpimg.imread(file)
            # imdata = cv2.resize(imdata, dsize=(HEIGHT_WIDTH, HEIGHT_WIDTH), interpolation=cv2.INTER_LINEAR) * 255.0
            # imdata = tf.keras.applications.inception_resnet_v2.preprocess_input(
            #     imdata, data_format=None
            # )
            return tensor, label

        def __len__(self):
            # return len(self.X)
            return self.length

        def __getitem__(self, index):
            _x = self.X[index]
            return self.preprocess(_x)
            # _y = class_map[_x]
            # _y = self.Y[index]

            # return _x, _y



    train_data_cat = [f'data/Training/cat/{x}' for x in os.listdir('data/Training/cat')]
    train_data_dog = [f'data/Training/dog/{x}' for x in os.listdir('data/Training/dog')]
    random.shuffle(train_data_cat)
    random.shuffle(train_data_dog)
    train_data_cat = train_data_cat[0:num_ims_per_class]
    train_data_dog = train_data_dog[0:num_ims_per_class]
    train_data = train_data_cat + train_data_dog

    # my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])] # a list of numpy arrays
    # my_y = [np.array([4.]), np.array([2.])] # another list of numpy arrays (targets)
    #
    # tensor_x = torch.Tensor(my_x) # transform to torch tensor
    # tensor_y = torch.Tensor(my_y)
    #
    # my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset


    test_data_cat = [f'data/Testing/cat/{x}' for x in os.listdir('data/Testing/cat')]
    test_data_dog = [f'data/Testing/dog/{x}' for x in os.listdir('data/Testing/dog')]
    random.shuffle(test_data_cat)
    random.shuffle(test_data_dog)
    test_data_cat = test_data_cat[0:num_ims_per_class]
    test_data_dog = test_data_dog[0:num_ims_per_class]
    test_data = test_data_cat + test_data_dog

    random.shuffle(train_data)
    random.shuffle(test_data)

    # DEBUG
    # test_data = train_data


    train_dataset = CustomDataset(train_data, len(train_data))
    train_loader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=4
    )  # create your dataloader

    test_dataset = CustomDataset(test_data, len(test_data))
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=4
    )  # create your dataloader

    # def get_gen(data):
    #     def gen():
    #         pairs = []
    #         i = 0
    #         for im_file in data:
    #             i += 1
    #             if i <= BATCH_SIZE:
    #                 pairs += [preprocess(im_file)]
    #             if i == BATCH_SIZE:
    #                 yield (
    #                     [pair[0] for pair in pairs],
    #                     [pair[1] for pair in pairs]
    #                 )
    #                 pairs.clear()
    #                 i = 0
    #     return gen
    #
    # def get_ds(data):
    #     return tf.data.Dataset.from_generator(
    #         get_gen(data),
    #         (tf.float32, tf.int64),
    #         output_shapes=(
    #             tf.TensorShape((BATCH_SIZE, HEIGHT_WIDTH, HEIGHT_WIDTH, 3)),
    #             tf.TensorShape(([BATCH_SIZE]))
    #         )
    #     )


    # net.compile(
    #     optimizer='adam',
    #     loss='binary_crossentropy',
    #     metrics=['accuracy']
    # )

    optimizer = optim.Adam(model.parameters())
    loss_function = nn.CrossEntropyLoss()

    model.cuda()
    loss_function.cuda()

    print(f'starting training (num ims per class = {num_ims_per_class})')
    # history = net.fit(
    #     get_ds(train_data),
    #     epochs=epochs,
    #     verbose=VERBOSE,
    #     use_multiprocessing=False,
    #     shuffle=False,
    #     validation_data=get_ds(train_data)
    # )

    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = []
        top1 = tnt.meter.ClassErrorMeter()
        for batch_ix, (data, target) in enumerate(train_loader):
            # if args.cuda:
            data, target = data.cuda(), target.cuda()

            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            epoch_loss.append(loss.data.item())
            top1.add(output.data, target.data)
            optimizer.step()
            if batch_ix % 10 == 0 and batch_ix > 0:
                pass
                # import pdb; pdb.set_trace()
                # print('[Epoch %2d, batch %3d] training loss: %.4f' %
                #       (epoch, batch_ix, loss.data[0]))
                # print('[Epoch %2d, batch %3d] training loss: %.4f' %
                #       (epoch, batch_ix, loss.data.item()))
        print('[Epoch %2d] Average TRAIN loss: %.3f, accuracy: %.2f%%\n'
              % (epoch, float(mean(epoch_loss)), top1.value()[0]))
        history['loss'].append(mean(epoch_loss))
        history['accuracy'].append(top1.value()[0])

        print(f'eval on epoch {epoch}')
        model.eval()
        test_loss = tnt.meter.AverageValueMeter()
        top1 = tnt.meter.ClassErrorMeter()
        for data, target in test_loader:
            # if args.cuda:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            loss = loss_function(output, target)

            top1.add(output.data, target.data)
            # test_loss.add(loss.data[0])
            test_loss.add(loss.data.item())

        print('[Epoch %2d] Average TEST loss: %.3f, accuracy: %.2f%%\n'
              % (epoch, test_loss.value()[0], top1.value()[0]))
        history['val_loss'].append(test_loss.value()[0])
        history['val_accuracy'].append(top1.value()[0])

    # print('starting testing')
    # print_output = True
    # print(net.evaluate(
    #     get_ds(train_data),
    #     verbose=VERBOSE,
    #     use_multiprocessing=False
    # ))
    print('script complete')

    return history

data_result = []
USE_BN = True


def mkdirs(s):
    if not os.path.exists(s):
        os.makedirs(s)
import time



# bn = 'bn' if USE_BN else 'nobn'
# fold = f'data_result/pytorch_{bn}_{int(time.time())}'
fold = f'data_result/pytorch_zoo_{int(time.time())}'
mkdirs(fold)

import torchvision.models as models
models_to_test = {
    'resnet18'       : lambda: models.resnet18(pretrained=True),
    'alexnet'        : lambda: models.alexnet(pretrained=True),
    'squeezenet'     : lambda: models.squeezenet1_0(pretrained=True),
    'vgg16'          : lambda: models.vgg16(pretrained=True),
    'densenet'       : lambda: models.densenet161(pretrained=True),
    'inception'      : lambda: models.inception_v3(pretrained=True),
    'googlenet'      : lambda: models.googlenet(pretrained=True),
    'shufflenet'     : lambda: models.shufflenet_v2_x1_0(pretrained=True),
    'mobilenet'      : lambda: models.mobilenet_v2(pretrained=True),
    'resnext50_32x4d': lambda: models.resnext50_32x4d(pretrained=True),
    'wide_resnet50_2': lambda: models.wide_resnet50_2(pretrained=True),
    'mnasnet'        : lambda: models.mnasnet1_0(pretrained=True),
}

# for i in range(20, 102, 1):
for name,model in list(models_to_test.items()):
    num_epochs = 10
    num_ims = 100

    # inc = Inception_ResNetv2(
    #     use_bn=USE_BN,
    #     classes=2
    # )

    history = train(model(), num_epochs,num_ims)  # more epochs without BN is required to get to overfit
    # breakpoint()










    data_result.append({
        'model_name':name,
        'num_images': num_ims,
        'history'   : history
    })
    # import pdb; pdb.set_trace()

    with open(f'{fold}/data_result.json', 'w') as f:
        import json
        f.write(json.dumps(data_result))


# srun -n 1 --mem=50G --gres=gpu:1 --constraint=any-gpu -t 600 --pty bash
