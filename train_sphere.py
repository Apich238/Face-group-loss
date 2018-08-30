import argparse
import sys
import os
import random
import math
import importlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import evaluating.LFW as LFW
import evaluating.BLUFR as BLUFR
import nets.facenet_inception as net
import sphere_loss

from datetime import datetime
from timer import timer

def read_train_set(folder, minfilter=1):
    names = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
    names.sort()
    res = []
    img_formats = ['jpg', 'jpeg', 'png', 'bmp']
    i = 0
    for name in names:
        ls = [os.path.join(folder, name, fname) for fname in os.listdir(os.path.join(folder, name)) \
              if os.path.isfile(os.path.join(folder, name, fname)) and fname.split('.')[-1] in img_formats]
        if len(ls) >= minfilter:
            res.append((name, ls))
        # if i % 100 == 0:
        #     print('.', end='', flush=True)
        i = i + 1
    return res


def save_model(sess, saver, summary_writer, model_dir, step, global_step):
    # Save the model checkpoint
    print('Saving variables')
    checkpoint_path = os.path.join(model_dir, 'model.ckpt')
    saver.save(sess, checkpoint_path, global_step=global_step, write_meta_graph=False)
    metagraph_filename = os.path.join(model_dir, 'model.meta')
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        saver.export_meta_graph(metagraph_filename)


def augment(image, use_random_crop, add_noise, use_random_flip, image_sz):
    if use_random_crop:
        image = tf.random_crop(image, (image_sz, image_sz, 3))
    else:
        mx = (image.shape[1] - image_sz) // 2
        my = (image.shape[0] - image_sz) // 2
        image = tf.image.crop_to_bounding_box(image, my, mx, image_sz, image_sz)
    if add_noise:
        image = image + tf.random_uniform(tf.shape(image), -10, 10, dtype=tf.int32)
    if use_random_flip:
        image = tf.image.random_flip_left_right(image)
    image.set_shape((image_sz, image_sz, 3))
    return image


def sdiv(a, b):
    return int(math.ceil(a / b))


def select_images(sess, epoch_sz, train_data, enque_op, embeddings, loss, image_patches, labels_out,
                  persons_per_batch, images_per_person):
    '''
    :return: Список списков имён файлов изображений
    '''
    subset = random.sample(train_data, epoch_sz * persons_per_batch)
    subset = np.asarray([random.sample(p[1], images_per_person) for p in subset], dtype=str)
    return subset


def test(test_pairs, enque_op, embeddings, image_patches, labels, labels_out, batch_size,
         images_per_person, sess, embedding_size, phase_train_placeholder, persons_per_batch, summary_op,
         summary_writer, write_summary, global_step, log_dir, plot=False):
    print('Running LFW verification test protocol...')
    fnames = []
    issame = np.empty(shape=(len(test_pairs)), dtype=bool)
    for i, t in enumerate(test_pairs):
        fnames.append([t[0], t[1]])
        issame[i] = t[2]
    fnames = np.asarray(fnames)
    if_fnames_sz = batch_size * (fnames.size // batch_size + (1 if fnames.size % batch_size > 0 else 0))
    appendix = if_fnames_sz - fnames.size
    lin = np.reshape(fnames, (-1))
    if appendix > 0:
        lin = np.concatenate((lin, np.full((appendix), lin[0])))
    fnames_in = np.reshape(lin, (-1, images_per_person))
    labesl_in = np.arange(0, fnames_in.size, 1, int).reshape(fnames_in.shape)
    embs = np.full((lin.shape[0], embedding_size), -100.)
    fd = {
        image_patches: fnames_in,
        labels: labesl_in,
        phase_train_placeholder: False
    }
    sess.run(enque_op, feed_dict=fd)
    cntr = lin.shape[0]
    fd = {
        phase_train_placeholder: False}
    while cntr > 0:
        emb_vals, label_vals = sess.run([embeddings, labels_out], feed_dict=fd)
        embs[label_vals] = emb_vals
        cntr = max(0, cntr - len(label_vals))
    embs = embs[0:fnames.size, :]
    embs = embs.reshape((embs.shape[0] // 2, 2, embs.shape[1]))
    tpr, fpr, acc_mean, acc_std, val_mean, val_std, far = LFW.eval_accuracy(embs, issame, 0.001)
    print("accuracy = {:.3f}+-{:.3f}, validation rate = {:.3f}+-{:.3f} at FAR={}".format(acc_mean, acc_std, val_mean,
                                                                                         val_std, far))
    if write_summary:
        summary = tf.Summary()
        summary.value.add(tag='lfw/accuracy', simple_value=acc_mean)
        summary.value.add(tag='lfw/val_rate', simple_value=val_mean)
        summary_writer.add_summary(summary, sess.run(global_step))
        with open(os.path.join(log_dir, 'lfw_result.txt'), 'at') as f:
            f.write('%d\t%.5f\t%.5f\n' % (sess.run(global_step), acc_mean, val_mean))

    if plot:
        plt.gcf().clear()
        plt.interactive(False)
        plt.title('ROC on LFW')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % np.trapz(tpr, fpr))
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(os.path.join(log_dir,'LFW_ROC_gs=%d.png'%sess.run(global_step)))


def get_train_op(total_loss, learning_rate_ph, global_step, epoch_sz, lr_decay_epoch, learning_rate_decay_factor):
    decayed_learning_rate = tf.train.exponential_decay(learning_rate_ph, global_step, epoch_sz * lr_decay_epoch,
                                                       learning_rate_decay_factor)
    summ = tf.summary.scalar('learning_rate', decayed_learning_rate)
    opt = tf.train.GradientDescentOptimizer(decayed_learning_rate)#
    #opt=tf.train.AdadeltaOptimizer(decayed_learning_rate)#
    #opt=tf.train.AdagradOptimizer(decayed_learning_rate)#
    #opt=tf.train.AdamOptimizer(decayed_learning_rate)#
    #opt=tf.train.RMSPropOptimizer(decayed_learning_rate)#
    minop = opt.minimize(total_loss, global_step, tf.global_variables(), name='optimization')
    return minop, decayed_learning_rate


def get_total_loss(loss):
    regloss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([loss] + regloss, name='total_loss')
    return total_loss


def InitPipeline(image_sz, images_per_person, image_patches, labels, batch_size, keep_prob, phase_train_placeholder,
                 weight_decay, embedding_size, R, n, use_random_crop, add_noise, use_random_flip, network):
    with tf.name_scope('input_pipeline'):
        input_queue = tf.FIFOQueue(capacity=15000,
                                   dtypes=[tf.string, tf.int32],
                                   shapes=[(images_per_person,), (images_per_person,)],
                                   name='input_queue')
        enque_op = input_queue.enqueue_many(vals=[image_patches, labels])
        input_sz = input_queue.size()
        images_and_labels = []
        preprocess_threads = 4
        for _ in range(preprocess_threads):
            fnames, labels = input_queue.dequeue()
            images = []
            for fname in tf.unstack(fnames):
                file = tf.read_file(fname)
                img = tf.image.decode_image(file, channels=3)
                img = augment(img, use_random_crop, add_noise, use_random_flip, image_sz)
                img = tf.image.per_image_standardization(img)
                images.append(img)
            images_and_labels.append([images, labels])

        images_batch, labels_out = tf.train.batch_join(images_and_labels,
                                                       batch_size=batch_size,
                                                       shapes=[(image_sz, image_sz, 3), ()],
                                                       enqueue_many=True,
                                                       capacity=batch_size * preprocess_threads,
                                                       allow_smaller_final_batch=True)

    images_batch = tf.identity(images_batch, 'image_batch')
    labels_out = tf.identity(labels_out, 'label_batch')

    features = net.inference(images_batch, keep_prob,
                             phase_train=phase_train_placeholder,
                             bottleneck_layer_size=embedding_size,
                             weight_decay=weight_decay)
    features = tf.identity(features, 'features')
    mx, sd, centers, embeddings = sphere_loss.get_sphere_loss(features, images_per_person, R=R, n=n)  # m=m, l=l)
    loss = tf.reduce_mean(mx, name='my_loss')
    total_loss = get_total_loss(loss)
    return enque_op, loss, embeddings, labels_out, total_loss, input_sz


def train_epoch(sess, train_data, enque_op, embeddings, loss, total_loss, epoch, image_patches, labels_in, labels_out,
                phase_train_placeholder, learning_rate_placeholder, lr, batch_sz, train_op, epoch_sz,
                images_per_person, persons_per_batch, global_step, input_sz, summary_op, summary_writer):
    '''
                Одна эпоха обучения ИНС.
    '''
    train_images = select_images(sess, epoch_sz, train_data, enque_op, embeddings, loss, image_patches, labels_out,
                                 persons_per_batch, images_per_person)
    labels_inval = np.arange(0, train_images.size, 1, dtype=int).reshape(train_images.shape)
    fd = {image_patches: train_images,
          labels_in: labels_inval,
          phase_train_placeholder: True}

    sess.run(enque_op, feed_dict=fd)

    train_persons_count = train_images.shape[0]
    fd = {phase_train_placeholder: True,
          learning_rate_placeholder: lr}
    # i=0
    batches_per_epoch = train_persons_count // persons_per_batch + (
        1 if train_persons_count % persons_per_batch > 0 else 0)
    # while train_persons_count>0:
    for i in range(batches_per_epoch):
        with timer() as t:
            epoch_v, _, total_loss_v, loss_v, labels_v, embs_v, gs = sess.run(
                [epoch, train_op, total_loss, loss, labels_out, embeddings, global_step], feed_dict=fd)
            print(
                'epoch {:6d}, batch {:4d}/{}, global_step {:10d}, loss ={:10.3f}, total_loss ={:10.3f}, t={:7.3f}s'  # , in_sz={}, embs_v_mean_.max={:.8f}, embs_v_std_.min={:.8f}'
                    .format(epoch_v + 1, i + 1, batches_per_epoch, gs, loss_v,
                            total_loss_v, t.now()))  # , in_sz_v, embs_v.mean(0).max(), embs_v.std(0).min()))
            if i % 5 == 0:
                summary = tf.Summary()
                summary.value.add(tag='train/loss', simple_value=loss_v)
                summary.value.add(tag='train/total_loss', simple_value=total_loss_v)
                summary_writer.add_summary(summary, gs)
            i = i + 1
            train_persons_count = max(0, train_persons_count - persons_per_batch)


def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in args.__dict__.items():
            f.write('%s: %s\n' % (key, str(value)))


def eval_embeddings_and_save_for_BLUFR(file_list, enque_op, embeddings, image_patches, labels, labels_out, batch_size,
                                       images_per_person, sess, embedding_size, phase_train_placeholder, BLUFR_dir,
                                       global_step):
    print('evaluating embeddings for BLUFR')
    fnames = file_list
    fnames = np.asarray(fnames)
    if_fnames_sz = batch_size * (fnames.size // batch_size + (1 if fnames.size % batch_size > 0 else 0))
    appendix = if_fnames_sz - fnames.size
    lin = np.reshape(fnames, (-1))
    if appendix > 0:
        lin = np.concatenate((lin, np.full((appendix), lin[0])))
    fnames_in = np.reshape(lin, (-1, images_per_person))
    labesl_in = np.arange(0, fnames_in.size, 1, int).reshape(fnames_in.shape)
    embs = np.full((lin.shape[0], embedding_size), -100.)
    fd = {
        image_patches: fnames_in,
        labels: labesl_in,
        phase_train_placeholder: False
    }
    sess.run(enque_op, feed_dict=fd)
    cntr = lin.shape[0]
    fd = {
        phase_train_placeholder: False}
    while cntr > 0:
        emb_vals, label_vals = sess.run([embeddings, labels_out], feed_dict=fd)
        embs[label_vals] = emb_vals
        cntr = max(0, cntr - len(label_vals))
        # print('.',end='',flush=True)
    embs = embs[0:fnames.size, :]
    print('\nSaving embeddings...', end='')
    BLUFR.SaveEmbeddings(os.path.join(BLUFR_dir, str(format(sess.run(global_step)))), embs)
    print('done')


def main(args):
    # количественные параметры
    images_per_person = args.images_per_person
    persons_per_batch = args.persons_per_batch
    batch_size = persons_per_batch * images_per_person
    persons_per_epoch = args.persons_per_epoch
    epoch_sz = persons_per_epoch // persons_per_batch  # batches per epoch
    epoch_count = args.epoch_count  # 5000
    test_rate = args.test_rate

    validation = args.validation

    # размеры входа и выхода сети
    image_sz = args.image_sz
    embedding_size = args.embedding_size

    # аугментация
    use_random_flip = args.use_random_flip
    use_random_crop = args.use_random_crop
    add_noise = args.add_noise

    # параметры
    R = args.R
    n = args.n

    # регуляризация, скорость обучения
    lr_decay_epoch = 20
    learning_rate_decay_factor = 0.9
    learning_rate = 0.1
    keep_prob = 0.8
    weight_decay = 1e-4

    # директории
    # model_module = args.model_module
    train_set_dir = args.train_set_dir
    lfw_dir = args.lfw_dir
    lfw_pairs_file = args.lfw_pairs_file
    blufr_list_file = args.blufr_list_file

    continue_training =  r"D:\models\face_my_loss_GD-20180517-083111\model\model.ckpt-622650"
    if continue_training != '':
        dir = continue_training[:continue_training.rfind("\\")-6]
    else:
        subdir = datetime.strftime(datetime.now(), 'face_my_loss_GD-%Y%m%d-%H%M%S')
        dir = os.path.join(r'D:\models', subdir)
    print('model dir {}'.format(dir))
    gpu_memory_fraction = 0.9

    save_dir = os.path.join(dir, 'model')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    log_dir = os.path.join(dir, 'logs')
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    seed = 123
    random.seed(seed)
    np.random.seed(seed)

    write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))

    # build model
    with tf.Graph().as_default():
        # tf.set_random_seed(seed)
        # placeholders
        image_patches = tf.placeholder(dtype=tf.string, shape=[None, images_per_person], name='input_filenames')
        labels = tf.placeholder(dtype=tf.int32, shape=[None, images_per_person], name='labels')
        phase_train_placeholder = tf.placeholder(dtype=tf.bool, name='training')
        learning_rate_placeholder = tf.placeholder(dtype=tf.float32, name='learning_rate_placeholder')
        # vars
        global_step = tf.Variable(0, trainable=False, name='global_step')
        epoch = global_step // epoch_sz
        # pipeline
        # network =  importlib.import_module(model_module)
        enque_op, loss, embeddings, labels_out, total_loss, input_sz = InitPipeline(image_sz, images_per_person,
                                                                                    image_patches,
                                                                                    labels,
                                                                                    batch_size, keep_prob,
                                                                                    phase_train_placeholder,
                                                                                    weight_decay, embedding_size, R, n,
                                                                                    use_random_crop, add_noise,
                                                                                    use_random_flip, net)
        train_op, decayed_lr = get_train_op(total_loss, learning_rate_placeholder, global_step, epoch_sz,
                                                     lr_decay_epoch,
                                                     learning_rate_decay_factor)
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5)
        summary_op = tf.summary.merge_all()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)).as_default() as sess:
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

            with timer() as t:
                print('init variables...', end='')
                sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder: True})
                sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder: True})
                print('done in {} s'.format(t.now()))
                t.restart()
                print('start quere runners...', end='')
                tf.train.start_queue_runners(sess, coord=tf.train.Coordinator())
                print('done in {} s'.format(t.now()))
            with timer() as t:
                print('reading training data')
                train_data = read_train_set(train_set_dir, images_per_person)
                print('\ndone in {} s'.format(t.now()))
            with timer() as t:
                print('reading LFW pairs')
                lfw_pairs = LFW.read_pairs(lfw_dir, lfw_pairs_file)
                print('\ndone in {} s'.format(t.now()))
            # with timer() as t:
            #     print('reading BLUFR LFW list')
            #
            #     print('\ndone in {} s'.format(t.now()))
            if continue_training != '':
                print('Restoring pretrained model: {}'.format(continue_training))
                saver.restore(sess, continue_training)
                sess.run(global_step.assign(int(os.path.basename(continue_training).split('-')[-1])))
            if validation:
                test(lfw_pairs, enque_op, embeddings, image_patches, labels, labels_out, batch_size,
                     images_per_person, sess, embedding_size, phase_train_placeholder, persons_per_batch, summary_op,
                     summary_writer, False, global_step, log_dir, True)
                blufr_list = BLUFR.ReadLFWList(lfw_dir, blufr_list_file)
                eval_embeddings_and_save_for_BLUFR(blufr_list, enque_op, embeddings, image_patches, labels, labels_out,
                                                   batch_size, images_per_person, sess,
                                                   embedding_size, phase_train_placeholder, log_dir, global_step)
                return

            while sess.run(epoch) < epoch_count:
                # тест по протооклу LFW
                if sess.run(epoch) % test_rate == 0:
                    test(lfw_pairs, enque_op, embeddings, image_patches, labels, labels_out, batch_size,
                         images_per_person, sess, embedding_size, phase_train_placeholder, persons_per_batch,
                         summary_op,
                         summary_writer, True, global_step, log_dir,sess.run(epoch) % (10*test_rate)==0)
                # 1 эпоха обучения сети
                train_epoch(sess, train_data, enque_op, embeddings, loss, total_loss, epoch, image_patches, labels,
                            labels_out, phase_train_placeholder, learning_rate_placeholder, learning_rate, batch_size,
                            train_op, epoch_sz, images_per_person, persons_per_batch, global_step, input_sz, summary_op,
                            summary_writer)
                save_model(sess, saver, summary_writer, save_dir, epoch, global_step)
            # сохранение отображений для тестирования по протоколу BLUFR
            blufr_list = BLUFR.ReadLFWList(lfw_dir, blufr_list_file)
            eval_embeddings_and_save_for_BLUFR(blufr_list, enque_op, embeddings, image_patches, labels, labels_out,
                                               batch_size, images_per_person, sess,
                                               embedding_size, phase_train_placeholder, log_dir, global_step)
            test(lfw_pairs, enque_op, embeddings, image_patches, labels, labels_out, batch_size,
                 images_per_person, sess, embedding_size, phase_train_placeholder, persons_per_batch, summary_op,
                 summary_writer, True, global_step, log_dir, True)
            summary_writer.close()
    print('done')


def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--validation', type=bool, default=False)  # only evaluation

    parser.add_argument('--images_per_person', type=int, default=5)
    parser.add_argument('--persons_per_batch', type=int, default=20)
    parser.add_argument('--persons_per_epoch', type=int, default=10500)
    parser.add_argument('--epoch_count', type=int, default=1500)
    parser.add_argument('--test_rate', type=int, default=1)

    parser.add_argument('--image_sz', type=int, default=120)
    parser.add_argument('--embedding_size', type=int, default=128)

    parser.add_argument('--use_random_flip', type=bool, default=True)
    parser.add_argument('--use_random_crop', type=bool, default=True)
    parser.add_argument('--add_noise', type=bool, default=False)

    parser.add_argument('--R', type=float, default=3)
    parser.add_argument('--n', type=float, default=6)

    parser.add_argument('--lr_decay_epoch', type=int, default=20)
    parser.add_argument('--learning_rate_decay_factor', type=float, default=0.9)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--keep_prob', type=float, default=0.8)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # folders
    parser.add_argument('--train_set_dir', type=str, default=r'D:\datasets\verticalized\CASIA-120')
    parser.add_argument('--lfw_dir', type=str, default=r'D:\datasets\verticalized\LFW-120')
    parser.add_argument('--lfw_pairs_file', type=str, default=r'D:\datasets\lfw\view2\pairs.txt')
    parser.add_argument('--blufr_list_file', type=str,
                        default=r'D:\Dropbox\diss\src\BLUFR\list\lfw\image_list.txt')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
