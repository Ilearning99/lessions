# 参考 Getting started with 100+ flowers on TPU
# https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu

import math,re,os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from kaggle_datasets import KaggleDatasets
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# 查看tensorflow版本
print("Tensorflow version " + tf.__version__)

# 自动环境配置
AUTO = tf.data.experimental.AUTOTUNE
# 检测TPU环境，并返回kaggle的TPU集群环境
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # 需要设置环境变量TPU_NAME，在kaggle环境中已经设定好
    print('Running on TPU ', tpu.master())
except ValueError:
    print('There is an error in creating TF TPU environment')
    tpu = None
    
# 使用TPU集群环境，初始化python分布式策略
if tpu:
    tf.config.experimental_connect_to_cluster(tpu) # 连接到TPU集群
    tf.tpu.experimental.initialize_tpu_system(tpu) # 初始化TPU环境
    strategy = tf.distribute.experimental.TPUStrategy(tpu) # 初始化使用TPU的分布式方法
else:
    strategy = tf.distribute.get_strategy() # 或者使用tf默认的分布式方法，这里使用1CPU+1GPU

# 获取分布式策略可以使用的核心数，方便后续定义batch_size，批块的大小
print('REPLICAS: ', strategy.num_replicas_in_sync) 

# 数据在GCS(谷歌云存储上)，获取数据存储路径，通过add data添加对应数据
GCS_DS_PATH = KaggleDatasets().get_gcs_path()
print("GCS_DS_PATH:" + GCS_DS_PATH)

# 读取数据
IMAGE_SIZE = [192, 192] #归一化图片大小
EPOCHS = 1 #训练轮数
BATCH_SIZE = 128 * strategy.num_replicas_in_sync

GCS_PATH_SELECT = { # available image sizes 可以获取不同大小的图片
    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',
    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',
    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',
    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'
}
GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]
print("GCS PATH" + GCS_PATH)

# 获取训练、验证、测试数据
TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')
VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec')

# 数据集种类信息，一共103类
CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09
           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19
           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29
           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39
           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49
           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59
           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69
           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79
           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89
           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99
           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                                                                                                                                               # 100 - 102

# # 设置numpy显示参数
# np.set_printoptions(threshold=15, linewidth=80)

# def batch_to_numpy_images_and_labels(data):
#     images, labels = data
#     numpy_images = images.numpy()
#     numpy_labels = labels.numpy()
#     if numpy_labels.dtype == object: # 测试数据，label是图片序号
#         numpy_labels = [None for _  in enumerate(numpy_images)]
#     return numpy_images, numpy_labels

# # 展示数据，返回字符串和是否预测正确的标记
# def title_from_label_and_target(label, correct_label):
#     if correct_label is None: # 测试样本
#         return CLASSES[label], True
#     correct = (label == correct_label)
#     return "{}[{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', 
#                                u"\u2192" if not correct else '',
#                                CLASSES[correct_label] if not correct else ''), correct

# 解析数据
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0 #数据归一化到[0,1]
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) #转换数据大小
    return image

# 解析tfrecord数据中的图片和类别
def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "class": tf.io.FixedLenFeature([], tf.int64),
    } # 
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['class'], tf.int32)
    return image, label # returns 数据集(image, label)

# 读取不含label测试数据
def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "id": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['id']
    return image, idnum

# 获取文件夹中的样本数
def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))

# 从硬盘加载数据集
def load_dataset(filenames, labeled=True, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # 训练数据b不需要顺序读入数据
    # 自动并发从多个文件读入数据
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order) # 设置数据集类参数
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO) # 设置样本解析方法
    return dataset

# 训练样本在预读过程中进行数据增强，这部分操作在CPU上
def data_augment(image, label):
    image = tf.image.random_flip_left_right(image) # 随机左右翻转，翻转后仍是相同类别
    return image, label

# 获取训练数据集
def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # 训练数据需要重复多轮
    dataset = dataset.shuffle(2048) # 训练数据做打散
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # 多线程预获取下一个batch数据
    return dataset

# 获取验证数据集
def get_validation_dataset(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO)
    return dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset

# 在TPU策略上定义模型
# 使用keras接口
# modelA
with strategy.scope():
    # dense net 201
    pretrained_model = tf.keras.applications.DenseNet201(weights='imagenet', 
      include_top=False, input_shape=[*IMAGE_SIZE, 3])

    pretrained_model.trainable = True

    modelA = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
    
modelA.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)

# modelB
with strategy.scope(): 
    # Inception v3
    pretrained_model = tf.keras.applications.InceptionV3(weights='imagenet', 
      include_top=False, input_shape=[*IMAGE_SIZE, 3])
    
    pretrained_model.trainable = True

    modelB = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
    
modelB.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)

# 学习率变化函数
def lrfn(epoch):
    LR_START = 0.00001
    LR_MAX = 0.00005 * strategy.num_replicas_in_sync
    LR_MIN = 0.00001
    LR_RAMPUP_EPOCHS = 5
    LR_SUSTAIN_EPOCHS = 0
    LR_EXP_DECAY = .8
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr =  LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

print("begin the training of modelA")
history = modelA.fit(get_training_dataset(), 
  steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
  callbacks = [lr_callback],
  validation_data=get_validation_dataset())
print("the end of modelA training")

print("begin the training of modelB")
history = modelB.fit(get_training_dataset(), 
  steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
  callbacks = [lr_callback],
  validation_data=get_validation_dataset())
print("the end of modelB training")

# 模型集合
models = [modelA, modelB]

# 获取验证结果
validate_dataset = get_validation_dataset(ordered=True) # 顺序获取验证数据
validate_image_dataset = validate_dataset.map(lambda image, label: image)
validate_label_dataset = validate_dataset.map(lambda image, label: label).unbatch()
validate_labels = next(iter(validate_label_dataset.batch(NUM_VALIDATION_IMAGES))).numpy()
validate_pre_prob = np.average([model.predict(validate_image_dataset) for model in models],axis=0)
validate_pre_label = np.argmax(validate_pre_prob, axis=-1)
score = f1_score(validate_labels, validate_pre_label, 
  labels=range(len(CLASSES)), average='macro')
precision = precision_score(validate_labels, validate_pre_label,
  labels=range(len(CLASSES)), average='macro')
recall = recall_score(validate_labels, validate_pre_label,
  labels=range(len(CLASSES)), average='macro')
print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))

# 获取测试结果
test_dataset = get_test_dataset(ordered=True) # 顺序获取测试数据集
test_image_dataset = test_dataset.map(lambda image, idnum: image) # 只获取其中的图片
test_pre_prob = np.average([model.predict(test_image_dataset) for model in models],axis=0)# 预测评分
test_pre_label = np.argmax(test_pre_prob, axis=-1)

# 生成提交结果
test_id_dataset = test_dataset.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_id_dataset.batch(NUM_TEST_IMAGES))).numpy().astype('U') # id转为numpy数组
np.savetxt('submission.csv',np.rec.fromarrays([test_ids, test_pre_label]),
  fmt=['%s','%d'], delimiter=',', header='id,label', comments='')