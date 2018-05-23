import keras.backend as K
from keras.optimizers import Optimizer

import six
import warnings
import functools

def generate_legacy_interface(allowed_positional_args=None,
                              conversions=None,
                              preprocessor=None,
                              value_conversions=None,
                              object_type='class'):
    if allowed_positional_args is None:
        check_positional_args = False
    else:
        check_positional_args = True
    allowed_positional_args = allowed_positional_args or []
    conversions = conversions or []
    value_conversions = value_conversions or []

    def legacy_support(func):
        @six.wraps(func)
        def wrapper(*args, **kwargs):
            if object_type == 'class':
                object_name = args[0].__class__.__name__
            else:
                object_name = func.__name__
            if preprocessor:
                args, kwargs, converted = preprocessor(args, kwargs)
            else:
                converted = []
            if check_positional_args:
                if len(args) > len(allowed_positional_args) + 1:
                    raise TypeError('`' + object_name +
                                    '` can accept only ' +
                                    str(len(allowed_positional_args)) +
                                    ' positional arguments ' +
                                    str(tuple(allowed_positional_args)) +
                                    ', but you passed the following '
                                    'positional arguments: ' +
                                    str(list(args[1:])))
            for key in value_conversions:
                if key in kwargs:
                    old_value = kwargs[key]
                    if old_value in value_conversions[key]:
                        kwargs[key] = value_conversions[key][old_value]
            for old_name, new_name in conversions:
                if old_name in kwargs:
                    value = kwargs.pop(old_name)
                    if new_name in kwargs:
                        raise_duplicate_arg_error(old_name, new_name)
                    kwargs[new_name] = value
                    converted.append((new_name, old_name))
            if converted:
                signature = '`' + object_name + '('
                for i, value in enumerate(args[1:]):
                    if isinstance(value, six.string_types):
                        signature += '"' + value + '"'
                    else:
                        if isinstance(value, np.ndarray):
                            str_val = 'array'
                        else:
                            str_val = str(value)
                        if len(str_val) > 10:
                            str_val = str_val[:10] + '...'
                        signature += str_val
                    if i < len(args[1:]) - 1 or kwargs:
                        signature += ', '
                for i, (name, value) in enumerate(kwargs.items()):
                    signature += name + '='
                    if isinstance(value, six.string_types):
                        signature += '"' + value + '"'
                    else:
                        if isinstance(value, np.ndarray):
                            str_val = 'array'
                        else:
                            str_val = str(value)
                        if len(str_val) > 10:
                            str_val = str_val[:10] + '...'
                        signature += str_val
                    if i < len(kwargs) - 1:
                        signature += ', '
                signature += ')`'
                warnings.warn('Update your `' + object_name +
                              '` call to the Keras 2 API: ' + signature, stacklevel=2)
            return func(*args, **kwargs)
        wrapper._original_function = func
        return wrapper
    return legacy_support

generate_legacy_method_interface = functools.partial(generate_legacy_interface,
                                                     object_type='method')

def get_updates_arg_preprocessing(args, kwargs):
    # Old interface: (params, constraints, loss)
    # New interface: (loss, params)
    if len(args) > 4:
        raise TypeError('`get_update` call received more arguments '
                        'than expected.')
    elif len(args) == 4:
        # Assuming old interface.
        opt, params, _, loss = args
        kwargs['loss'] = loss
        kwargs['params'] = params
        return [opt], kwargs, []
    elif len(args) == 3:
        if isinstance(args[1], (list, tuple)):
            assert isinstance(args[2], dict)
            assert 'loss' in kwargs
            opt, params, _ = args
            kwargs['params'] = params
            return [opt], kwargs, []
    return args, kwargs, []

legacy_get_updates_support = generate_legacy_interface(
    allowed_positional_args=None,
    conversions=[],
    preprocessor=get_updates_arg_preprocessing)



class Eve(Optimizer):
    '''Eve optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2/beta_3: floats, 0 < beta < 1. Generally close to 1.
        small_k/big_K: floats
        epsilon: float >= 0. Fuzz factor.

    # References
        - [Improving Stochastic Gradient Descent With FeedBack](http://arxiv.org/abs/1611.01505v1.pdf)
    '''

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 beta_3=0.999, small_k=0.1, big_K=10,
                 epsilon=1e-8, decay=0., **kwargs):
        super(Eve, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0)
        self.lr = K.variable(lr)
        self.beta_1 = K.variable(beta_1)
        self.beta_2 = K.variable(beta_2)
        self.beta_3 = K.variable(beta_3)
        self.small_k = K.variable(small_k)
        self.big_K = K.variable(big_K)
        self.decay = K.variable(decay)
        self.inital_decay = decay

    @legacy_get_updates_support
    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.inital_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1
        lr_t = lr * K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t))

        shapes = [K.get_variable_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]
        f = K.variable(0)
        d = K.variable(1)
        self.weights = [self.iterations] + ms + vs + [f, d]

        cond = K.greater(t, K.variable(1))
        small_delta_t = K.switch(K.greater(loss, f), self.small_k + 1, 1. / (self.big_K + 1))
        big_delta_t = K.switch(K.greater(loss, f), self.big_K + 1, 1. / (self.small_k + 1))

        c_t = K.minimum(K.maximum(small_delta_t, loss / (f + self.epsilon)), big_delta_t)
        f_t = c_t * f
        r_t = K.abs(f_t - f) / (K.minimum(f_t, f))
        d_t = self.beta_3 * d + (1 - self.beta_3) * r_t

        f_t = K.switch(cond, f_t, loss)
        d_t = K.switch(cond, d_t, K.variable(1.))

        self.updates.append(K.update(f, f_t))
        self.updates.append(K.update(d, d_t))

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            p_t = p - lr_t * m_t / (d_t * K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))

            new_p = p_t
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'beta_3': float(K.get_value(self.beta_3)),
                  'small_k': float(K.get_value(self.small_k)),
                  'big_K': float(K.get_value(self.big_K)),
                  'epsilon': self.epsilon}
        base_config = super(Eve, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
