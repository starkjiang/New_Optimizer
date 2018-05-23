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


class AEMVSGD(Optimizer):
    """AEMVSGD optimizer.


    Default parameters follow those provided in the following paper:
    Adam: A method for stochastic optimization
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta_1 <1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If 'None', defaults to 'K.epsilon()'.
        decay: float >= 0. Learning rate decay over each update.
    """
    def __init__(self, lr=0.001, beta_1=0.1, beta_2=0.999, beta_3=0.999, epsilon=None,
                 decay=0., **kwargs):
        super(AEMVSGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.beta_3 = K.variable(beta_3, name='beta_3')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay

    @legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations,1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t)))
        # lr_t = lr * (1. - K.pow(self.beta_1, t)) / (1. - K.pow(self.beta_2, t))
        # lr_t = lr
        shapes = [K.get_variable_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]
        vhats = [K.zeros(shape) for shape in shapes]
        ls = [K.zeros(shape) for shape in shapes]
        #ws = [K.zeros(shape) for shape in shapes]
        #whats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs + vhats + ls

        for p, g, m, v, vhat, l in zip(params, grads, ms, vs, vhats, ls):
            m_t = g - v
            v_t = v + self.beta_1 * m_t
            vhat_t = self.beta_2 * vhat + (1 - self.beta_2) * K.square(m_t)
            #l_t = K.square(v_t + K.sqrt(vhat_t)) - w
            #w_t = w + self.beta_2 * l_t
            #l_t = (self.beta_3 * l) + (1. - self.beta_3) * K.square(K.square(v_t + K.pow((1 - self.beta_3), t) * K.sqrt(vhat_t)))
            l_t = (self.beta_3 * l) + (1. - self.beta_3) * K.square(g)
            #w_t = (self.beta_3 * w) + (1. - self.beta_3) * p
            #what_t = (1-self.beta_2) * (what + self.beta_2 * K.square(l_t))
            #p_t = p - lr_t * (v_t) / (K.sqrt(w_t) + K.sqrt(K.sqrt(what_t)))
            p_t = p - lr_t * (v_t + K.pow((1 - self.beta_2), t) * K.sqrt(vhat_t)) / (K.sqrt(l_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            self.updates.append(K.update(vhat, vhat_t))
            self.updates.append(K.update(l, l_t))
            #self.updates.append(K.update(w, w_t))
            #self.updates.append(K.update(what, what_t))

            new_p = p_t

            # Apply constraint
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))

        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'beta_3': float(K.get_value(self.beta_3)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(AEMVSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
