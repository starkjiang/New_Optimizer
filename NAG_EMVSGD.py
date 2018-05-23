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


class EMVNAG(Optimizer):
    """Stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr=0.01, beta_1=0.1, beta_2=0.999, momentum=0.95, decay=0.,
                 nesterov=True, **kwargs):
        super(EMVNAG, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov

    @legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        
        # momentum
        shapes = [K.get_variable_shape(p) for p in params]
        ws = [K.zeros(shape) for shape in shapes]
        ls = [K.zeros(shape) for shape in shapes]
        lhats = [K.zeros(shape) for shape in shapes]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ws + ls + lhats + moments
        for p, g, w, l, lhat, m in zip(params, grads, ws, ls, lhats, moments):
            w_t = g - l
            l_t = l + self.beta_1 * w_t
            lhat_t = self.beta_2 * lhat + (1 - self.beta_2) * K.square(w_t)
            v = self.momentum * m - lr * (l_t + K.pow((1 - self.beta_2), t) * K.sqrt(lhat_t))  # velocity
            # v = self.momentum * m - lr * l_t / lhat_t
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * (l_t + K.pow((1 - self.beta_2), t) * K.sqrt(lhat_t))
                # new_p = p + self.momentum * v - lr * l_t / lhat_t
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
            self.updates.append(K.update(w, w_t))
            self.updates.append(K.update(l, l_t))
            self.updates.append(K.update(lhat, lhat_t))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(EMVNAG, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
