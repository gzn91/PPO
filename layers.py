import tensorflow as tf, numpy as np
import tensorflow.contrib as tc


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init


def batch_to_seq(xb, nbatch, nsteps, flat=False):
    if flat:
        xb = tf.reshape(xb, [nbatch, nsteps])
    else:
        xb = tf.reshape(xb, [nbatch, nsteps, -1])
    xs = [tf.squeeze(x, axis=1) for x in tf.split(value=xb, num_or_size_splits=nsteps, axis=1)]
    return xs


def seq_to_batch(h, flat=False):
    shape = h[0].get_shape().as_list()
    if not flat:
        assert (len(shape) > 1)
        nh = h[0].get_shape()[-1].value
        return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
    else:
        return tf.reshape(tf.stack(values=h, axis=1), [-1])


def lstm(xs, ms, s, scope, nh, use_ln=True, reuse=False):
    nbatch, nin = xs[0].get_shape().as_list()

    """nh is times 3 due to the same weight matrix is used for the forget, output and update gate"""
    with tf.variable_scope(scope):
        wx = tf.get_variable("wx", [nin, nh * 4], initializer=ortho_init())
        wh = tf.get_variable("wh", [nh, nh * 4], initializer=ortho_init())
        b = tf.get_variable("b", [nh * 4], initializer=tf.constant_initializer(0.0))

    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    for idx, (x, m) in enumerate(zip(xs, ms)):
        c = c * (1 - m)  # mask is 1 if terminal state, removes terminal states from training
        h = h * (1 - m)

        # z holds the forget, update and output gate
        if use_ln:
            z = tc.layers.layer_norm(tf.matmul(x, wx), center=True, scale=True, reuse=reuse,
                                      scope='ln1') + tc.layers.layer_norm(
                tf.matmul(h, wh), center=True, scale=True, reuse=reuse, scope='ln2') + b
        else:
            z = tf.matmul(x, wx) + tf.matmul(h, wh) + b


        # This is why nh is nh*3, splits z into forget, output and possible new cell states
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)

        i = tf.nn.sigmoid(i) #ignore
        f = tf.nn.sigmoid(f) #forget
        o = tf.nn.sigmoid(o) #output
        u = tf.tanh(u) # update
        c = f * c + i * u  # forget and update cells
        h = o * tf.tanh(c)  # output new hidden states
        xs[idx] = h
    s = tf.concat(axis=1, values=[c, h])  # concat cell and hidden states for ease of use
    return xs, s


# Gated Recurrent Unit implemented with a single neural net for the input and the merged hidden/cell state
def gru(xs, ms, h, scope, nh):
    nbatch = xs[0].get_shape().as_list()[0]
    nin = xs[0].get_shape().as_list()[1] + h.get_shape().as_list()[1]

    """nh is times 2 due to the output and update gate share weight matrix"""
    with tf.variable_scope(scope):
        w_hx = tf.get_variable("ws", [nin, nh * 2], initializer=ortho_init())
        w_h = tf.get_variable("wh", [nin, nh], initializer=ortho_init())
        b_hx = tf.get_variable("bs", [nh * 2], initializer=tf.constant_initializer(0.0))
        b_h = tf.get_variable("bh", [nh], initializer=tf.constant_initializer(0.0))

    for idx, (x, m) in enumerate(zip(xs, ms)):
        hx = tf.concat([h, x], axis=-1)  # concatenate hidden states and new inputs so we can use a single weight matrix
        hx = hx * (1 - m)  # mask is 1 if terminal state, removes terminal states from training

        zr = tf.matmul(hx, w_hx) + b_hx
        z, r = tf.split(axis=1, num_or_size_splits=2, value=zr)  # This is why nh is nh*2
        z = tf.nn.sigmoid(z)  # update gate / output gate
        r = tf.nn.sigmoid(r)  # part of h(t-1) to be used in updating h

        rhx = tf.concat([r * h, x], axis=-1)
        h_bar = tf.tanh(tf.matmul(rhx, w_h) + b_h)  # new possible hidden states
        h = (1 - z) * h + z * h_bar  # update the hidden state
        xs[idx] = h
    return xs


# Gated Recurrent Unit implemented with a seperate neural net for input and the merged hidden/cell state
def gru_(xs, ms, h, scope, nh, use_ln=True, reuse=False):
    nbatch, nin = xs[0].get_shape().as_list()
    # print(nin,nh)

    """nh is times 2 due to the output and update gate share weight matrix"""
    with tf.variable_scope(scope):
        # Weight matrix for z and r gates
        wx = tf.get_variable("wx", [nin, nh * 2], initializer=ortho_init())
        wh = tf.get_variable("wh", [nh, nh * 2], initializer=ortho_init())
        b = tf.get_variable("b", [nh * 2], initializer=tf.constant_initializer(0.0))

        # Weight matrix for new cell-state/output
        wx_ = tf.get_variable("wx_", [nin, nh], initializer=ortho_init())
        whr_ = tf.get_variable("wh_", [nh, nh], initializer=ortho_init())
        b_ = tf.get_variable("b_", [nh], initializer=tf.constant_initializer(0.0))

    for idx, (x, m) in enumerate(zip(xs, ms)):
        # Zero terminal states with mask
        x = x * (1 - m)
        h = h * (1 - m)

        if use_ln:
            zr = tc.layers.layer_norm(tf.matmul(x, wx), center=True, scale=True, reuse=reuse,
                                      scope='ln1') + tc.layers.layer_norm(
                tf.matmul(h, wh), center=True, scale=True, reuse=reuse, scope='ln2') + b
        else:
            zr = tf.matmul(x, wx) + tf.matmul(h, wh) + b

        z, r = tf.split(axis=1, num_or_size_splits=2, value=zr)  # This is why nh is nh*2
        z = tf.nn.sigmoid(z)  # update gate / output gate
        r = tf.nn.sigmoid(r)  # part of h(t-1) to be used in updating h
        h_bar = tf.tanh(tf.matmul(x, wx_) + tf.matmul(h * r, whr_) + b_)
        h = (1 - z) * h + z * h_bar
        xs[idx] = h
    return xs


def fc(x, units, scope, activate=True, gain=1.):
    with tf.variable_scope(scope):
        x = tf.layers.dense(x, units, kernel_initializer=ortho_init(scale=gain))
        if activate:
            # x = tc.layers.layer_norm(x, scale=True, center=True)
            x = tf.tanh(x)
        return x


def conv2d(x, scope, nfilters, nkernel, stride, pad='VALID', activate=True):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(x, filters=nfilters, kernel_size=nkernel, strides=stride, padding=pad,
                             kernel_initializer=ortho_init(scale=np.sqrt(2)))
        if activate:
            x = lrelu(x)
        return x


def max_pool2d(x, scope, nkernel, stride, pad='SAME'):
    with tf.variable_scope(scope):
        return tf.nn.max_pool(x, ksize=[1, nkernel, nkernel, 1], strides=[1, stride, stride, 1], padding=pad)


def prelu(x, scope='prelu'):
    with tf.name_scope(scope):
        alphas = tf.get_variable('alphas', shape=x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.01), dtype=tf.float32)
        return tf.maximum(x, 0.) + alphas * tf.minimum(x, 0.)


def lrelu(x, alpha=0.01):
    return tf.nn.leaky_relu(x, alpha)


def conv2d_block(x):

    h1 = conv2d(x, 'conv1', nfilters=32, nkernel=8, stride=4, pad='VALID')
    h2 = conv2d(h1, 'conv2', nfilters=64, nkernel=4, stride=2, pad='VALID')
    h3 = conv2d(h2, 'conv3', nfilters=64, nkernel=3, stride=1, pad='VALID')
    h3_flat = tf.layers.flatten(h3)

    return lrelu(
            fc(h3_flat, 512, 'fc1', activate=False, gain=np.sqrt(2)))


def normalize(x, stats=None, eps=1e-8):
    if stats is None:
        return (x - x.mean()) / (x.std() + eps)
    else:
        return (x - stats.mean) / stats.std
