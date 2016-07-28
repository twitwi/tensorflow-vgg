import numpy as np
import tensorflow as tf

import vgg16
import utils
import time

start = time.time()
def tick():
    print(time.time() - start)

filenames = ["./test_data/tiger.jpeg", "./test_data/puzzle.jpeg", "./test_data/tiger.jpeg", "./test_data/puzzle.jpeg"]
images = [utils.load_image(f) for f in filenames]
batches = [im.reshape((1, 224, 224, 3)) for im in images]

batch = np.concatenate(batches, 0)

tick()

with tf.Session(
        config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
    images = tf.placeholder("float", [len(filenames), 224, 224, 3])
    feed_dict = {images: batch}

    tick()
    vgg = vgg16.Vgg16()
    with tf.name_scope("content_vgg"):
        vgg.build(images)
    tick()
    probs = sess.run(vgg.prob, feed_dict=feed_dict)
    tick()
    for pr in probs:
        utils.print_prob(pr, './synset.txt')

