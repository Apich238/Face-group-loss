import tensorflow as tf


def angle(a, b):
    return tf.acos(tf.clip_by_value(tf.reduce_sum(tf.multiply(a, b), axis=0), -1., 1.))


def angles(a, b):
    return tf.acos(tf.clip_by_value(tf.reduce_sum(tf.multiply(a, b), axis=-1), -1., 1.))  # axis=2


def get_centers(features, images_per_class):
    # вычисляем центры для каждой персоны
    with tf.variable_scope('centers_evaluation'):
        features = tf.reshape(features, (-1, images_per_class, features.shape[-1]))
        centers = tf.reduce_mean(features, axis=1, keepdims=False)
        centers = tf.nn.l2_normalize(centers, name='centers')
    return centers


def get_dists(fs, centers):
    # расчитываем расстояния каждого оторбажения до центра соотв. класса
    with tf.variable_scope('distances_to_centers'):
        cs = tf.expand_dims(centers, 1)
        dsts = angles(tf.nn.l2_normalize(fs), cs)
    return dsts


def get_sd(dists):
    '''
    вычисляет стандартные отклонения по известным расстояниям с помощью несмещённой оценки
    :param dists:
    :return:
    '''
    with tf.variable_scope('standart_deviations'):
        d = tf.sqrt(tf.to_float(tf.subtract(tf.shape(dists)[1], 1)))
        return tf.sqrt(tf.reduce_sum(tf.square(dists), axis=1)) / d


def SphereIntersection(sd1, C1, sd2, C2, R, m):
    i = tf.nn.relu(R * (sd1 + sd2) + m - angle(C1, C2))
    return i  # tf.square(i)


def get_intersection_matrix(sd, C, R, m, l):
    with tf.variable_scope('intersection_matrix'):
        # для составления матрицы пересечений определим её элемент
        m_el = lambda i, j: tf.cond(tf.equal(i, j),
                                    true_fn=lambda: l * sd[i],  # tf.constant(0.,dtype=tf.float32),
                                    false_fn=lambda: SphereIntersection(sd[i], C[i], sd[j], C[j], R, m))
        # индексы в квадратной матрице
        indices = tf.range(0, tf.shape(C)[0], dtype=tf.int32, name='indices')
        # строка матрицы пересечений в зависимости от индекса
        m_row = lambda i: tf.map_fn(fn=lambda j: m_el(i, j), elems=indices, dtype=tf.float32)
        return tf.map_fn(fn=lambda i: m_row(i), elems=indices, dtype=tf.float32)


def SphereIntersections(sd1, cs1, sd2, cs2, R, m):
    return tf.square(tf.nn.relu(m - angles(cs1, cs2))) +R* ( (sd1 + sd2) / angles(cs1, cs2))
    # + 0.1*tf.square(sd1)  # + sd2)


def get_intersection_by_pairs(sd, centers, R, m, l):
    sdp = tf.reshape(sd, [-1, 2])
    sd1, sd2 = tf.split(sdp, 2, 1)
    sd1 = tf.reshape(sd1, [-1], name='sdA')
    sd2 = tf.reshape(sd2, [-1], name='sdB')

    csp = tf.reshape(centers, [-1, 2, centers.shape[-1]])
    cs1, cs2 = tf.split(csp, 2, 1)
    cs1 = tf.reshape(cs1, [-1, centers.shape[-1]], name='mA')
    cs2 = tf.reshape(cs2, [-1, centers.shape[-1]], name='mB')
    return SphereIntersections(sd1, cs1, sd2, cs2, R, m)


def get_sphere_loss(features, images_per_class, R=3., m=0.1, l=0.1):
    with tf.variable_scope('my_loss_evaluation'):
        embeddings = tf.nn.l2_normalize(features, axis=-1, name='embeddings')
        # разбираем отображения по персонам
        embs_rs = tf.reshape(embeddings, (-1, images_per_class, embeddings.shape[-1]), name='embeddings_grouped')
        centers = get_centers(features, images_per_class)
        dists = get_dists(embs_rs, centers)
        # стандартное отклонение (корень из дисперсии)
        sd = get_sd(dists)
        # определение матрицы пересечений
        # mx = get_intersection_matrix(sd, centers, R, m, l)
        mx = get_intersection_by_pairs(sd, centers, R, m, l)
    return mx, sd, centers, embeddings
