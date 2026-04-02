import tensorflow as tf
from typing import Literal
class Conv1DPCNLayer:
    is_clamped : tf.Variable # bool
    fix_wts : tf.Variable # bool
    num_units : int
    prev_layer: object
    next_layers: list
    wts : tf.Variable # tf.Tensor
    output_shape : tuple
    kernel_size: int
    activation : str
    state : tf.Variable # tf.Tensor
    learning_rate:float

    def __init__(self, num_units:int, kernel_size:int, learning_rate:float, activation:Literal['linear', 'relu']='linear'):
        self.is_clamped = tf.Variable(False, trainable=False)
        self.fix_wts_b = tf.Variable(False, trainable=False)
        self.num_units = num_units
        self.prev_layer = None
        self.next_layers = []
        self.wts = None
        self.output_shape = None
        self.state = None
        self.activation = activation
        self.learning_rate = learning_rate
        self.kernel_size = kernel_size

    def init_params(self, input_shape:tuple):
        # print(self.get_kaiming_gain()/tf.sqrt(float(input_shape[-1])))
        self.wts = tf.Variable(tf.random.normal((self.kernel_size, input_shape[-1], self.num_units), 
                                                stddev=self.get_kaiming_gain()/tf.sqrt(float(self.kernel_size*input_shape[-1]))), trainable=False)
    
    def predict_prev(self):
        return tf.nn.conv1d_transpose(self.state, self.wts, padding='SAME', strides=1, output_shape=(*self.output_shape[:2], self.wts.shape[1]))
    
    def predict_next(self):
        return self.state
    
    def d_gelu(self, x:tf.Tensor):
        return 0.5*(1+tf.math.erf(x/tf.sqrt(2.))) + x/tf.sqrt(2*tf.acos(-1.))*tf.exp(-tf.square(x)/2)
    
    def update_state(self):
        pass #TBD

    def update_wts(self):
        pass #TBD

    def update_b(self):
        pass # there is no bias

        
    def init_state(self):
        self.state = None

    def init_wts_b(self):
        self.wts = None

    def __call__(self, x : tf.Tensor):
        if len(x.shape)==2:
            x = x[:, :, None]

        if len(x.shape)!=3:
            raise ValueError("Input shape of wrong dimensions")

        if self.wts is None:
            self.init_params(x.shape)

        net_out = tf.nn.conv1d(x, self.wts, padding='SAME', stride=1)

        if self.activation == 'relu':
            net_act = tf.nn.relu(net_out)
        else:
            net_act = net_out

        if self.state is None:
            self.state = tf.Variable(net_act, trainable=False)
            self.output_shape = net_act.shape

        return net_act

        

    def get_kaiming_gain(self):
        if self.activation == 'relu':
            return tf.sqrt(2.)
        else:
            return 1

    def clamp(self, set_clamp:bool):
        self.is_clamped.assign(set_clamp)

    def set_fix_wts_b(self, fix_wts_b:bool):
        self.fix_wts_b.assign(fix_wts_b)