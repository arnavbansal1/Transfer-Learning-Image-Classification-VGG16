#Arnav Bansal
import os
from os.path import isfile, isdir
from urllib.request import urlretrieve
from tqdm import tqdm
import tarfile
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils

dataset_folder_path = 'flower_photos'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile('flower_photos.tar.gz'):
    
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Flowers Dataset') as pbar:
        urlretrieve('http://download.tensorflow.org/example_images/flower_photos.tgz', 'flower_photos.tar.gz', pbar.hook)

if not isdir(dataset_folder_path):
    
    with tarfile.open('flower_photos.tar.gz') as tar:
        tar.extractall()
        tar.close()

data_dir = 'flower_photos/'
contents = os.listdir(data_dir)
classes = [each for each in contents if os.path.isdir(data_dir + each)]
    
with open('labels') as f:
    reader = csv.reader(f, delimiter='\n')
    labels = np.array([each for each in reader if len(each) > 0]).squeeze()
    
with open('codes') as f:
    codes = np.fromfile(f, dtype=np.float32)
    codes = codes.reshape((len(labels), -1))

batch_size = 10
codes_list = []
labels = []
batch = []

codes = None

with tf.Session() as sess:
    vgg = vgg16.Vgg16()
    input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
    
    with tf.name_scope('content_vgg'):
        vgg.build(input_)
        
    for each in classes:
        print("Starting {} images".format(each))
        class_path = data_dir + each
        files = os.listdir(class_path)
        
        for ii, file in enumerate(files, 1):
            img = utils.load_image(os.path.join(class_path, file))
            batch.append(img.reshape((1, 224, 224, 3)))
            labels.append(each)
            
            if ii % batch_size == 0 or ii == len(files):
                images = np.concatenate(batch)
                codes_batch = sess.run(vgg.relu6, feed_dict={input_:images})
                
                if codes is None:
                    codes = codes_batch
                else:
                    codes = np.concatenate((codes, codes_batch))
                
                batch = []
                print('{} images processed'.format(ii))

with open('codes', 'w') as f:
    codes.tofile(f)
    
with open('labels', 'w') as f:
    writer = csv.writer(f, delimiter='\n')
    writer.writerow(labels)
    
lb = LabelBinarizer()
labels_vecs = lb.fit_transform(labels)
sss = StratifiedShuffleSplit(1, 0.2)
train_index, val_index = next(sss.split(codes, labels))
val_train_split_index = len(val_index)//2
val_index, test_index = val_index[:val_train_split_index], val_index[val_train_split_index:]
train_x, train_y = codes[train_index], labels_vecs[train_index]
val_x, val_y = codes[val_index], labels_vecs[val_index]
test_x, test_y =  codes[test_index], labels_vecs[test_index]

inputs_ = tf.placeholder(tf.float32, shape=[None, codes.shape[1]])
labels_ = tf.placeholder(tf.int64, shape=[None, labels_vecs.shape[1]])

fc = tf.contrib.layers.fully_connected(inputs_, 256)
logits = tf.contrib.layers.fully_connected(fc, labels_vecs.shape[1], activation_fn=None) 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))

optimizer = tf.train.AdamOptimizer().minimize(cost)

predicted = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(labels_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def get_batches(x, y, n_batches=10):
    batch_size = len(x)//n_batches
    
    for ii in range(0, n_batches*batch_size, batch_size):
        
        if ii != (n_batches-1)*batch_size:
            X, Y = x[ii: ii+batch_size], y[ii: ii+batch_size] 
        else:
            X, Y = x[ii:], y[ii:]
        
        yield X, Y

epochs = 10
iteration = 0

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for e in range(epochs):
        
        for x, y in get_batches(train_x, train_y):
            feed_dict = {inputs_: x, labels_: y}
            loss, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            print('Epoch: {}/{}'.format(e+1, epochs), 'Iteration: {}'.format(iteration), 'Training loss: {:.5f}'.format(loss))
            iteration += 1
            
            if iteration%5 == 0:
                feed_dict = {inputs_: val_x, labels_: val_y}
                val_acc = sess.run(accuracy, feed_dict=feed_dict)
                print('Epoch: {}/{}'.format(e+1, epochs), 'Iteration: {}'.format(iteration), 'Validation Accuracy: {:.4f}'.format(val_acc))
    
    saver.save(sess, "checkpoints/flowers.ckpt")
    
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    feed = {inputs_: test_x, labels_: test_y}
    test_acc = sess.run(accuracy, feed_dict=feed)
    print("Test accuracy: {:.4f}".format(test_acc))

#If using Jupyter Notebook
%matplotlib inline

test_img_path = 'flower_photos/roses/10894627425_ec76bbc757_n.jpg'
test_img = imread(test_img_path)
plt.imshow(test_img)

if 'vgg' in globals():
    print('"vgg" object already exists.  Will not create again.')
else:
    
    with tf.Session() as sess:
        input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
        vgg = vgg16.Vgg16()
        vgg.build(input_)
         
with tf.Session() as sess:
    img = utils.load_image(test_img_path)
    img = img.reshape((1, 224, 224, 3))

    feed_dict = {input_: img}
    code = sess.run(vgg.relu6, feed_dict=feed_dict)
        
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    feed = {inputs_: code}
    prediction = sess.run(predicted, feed_dict=feed).squeeze()

plt.imshow(test_img)
plt.barh(np.arange(5), prediction)
_ = plt.yticks(np.arange(5), lb.classes_)