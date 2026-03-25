import tensorflow as tf
from typing import Literal
class DensePCNLayer:
    is_clamped : tf.Variable
    fix_wts_b : tf.Variable
    num_units : int
    prev_layer: DensePCNLayer
    next_layers: list[DensePCNLayer]
    wts : tf.Tensor
    output_shape : tuple
    activation : str
    state : tf.Tensor
    learning_rate:float
    def __init__(self, num_units:int, learning_rate:float, activation:Literal['linear', 'relu']='linear', ):
        self.is_clamped = tf.Variable(False, trainable=False)
        self.fix_wts_b = tf.Variable(False, trainable=False)
        self.num_units = num_units
        self.prev_layer = None
        self.next_layers = []
        self.wts = None
        self.b = None
        self.output_shape = None
        self.state = None
        self.activation = activation
        self.learning_rate = learning_rate

    def init_params(self, input_shape:tuple):
        self.wts = tf.Variable(tf.random.normal((input_shape[-1], self.num_units), 
                                                stddev=self.get_kaiming_gain()/tf.sqrt(float(input_shape[-1]))), trainable=False)
        self.b = tf.Variable(tf.zeros(self.num_units), trainable=False)


    def predict_prev(self):
        return (self.state - self.b) @ tf.transpose(self.wts)
    
    def predict_next(self):
        return self.state
    
    def d_gelu(self, x:tf.Tensor):
        return 0.5*(1+tf.math.erf(x/tf.sqrt(2.))) + x/tf.sqrt(2*tf.acos(-1.))*tf.exp(-tf.square(x)/2)
        

    # pred_err = state - pred
    # 1/2*(state - pred)^2
    def update_state(self):
        if not self.is_clamped:
            # pred next layer & pred from next layer
            average_d_pred = tf.zeros_like(self.state)
            average_d_state = tf.zeros_like(self.state)
            num_next_layers = 0 
            for layer in self.next_layers:
                if layer.is_clamped:
                    continue
                num_next_layers += 1
                pred = layer(self.predict_next())
                pred_state = layer.predict_prev()
                if layer.activation == 'relu':
                    average_d_pred += -(layer.predict_next()-pred)*self.d_gelu(self.predict_next() @ self.wts + self.b) @ tf.transpose(layer.wts)
                    average_d_state += (tf.nn.relu(self.predict_next()) - tf.nn.relu(pred_state))
                else:
                    average_d_pred += -(layer.predict_next()-pred) @ tf.transpose(layer.wts)
                    average_d_state += (self.predict_next() - pred_state)
            if num_next_layers!=0:
                self.state.assign_sub(self.learning_rate * ((average_d_pred+average_d_state)/(2*num_next_layers)))

            # pred prev layer & pred from prev layer
            average_d_pred = tf.zeros_like(self.state)
            average_d_state = tf.zeros_like(self.state)
            layer = self.prev_layer
            if self.activation == 'relu':
                average_d_pred += -(tf.nn.relu(layer.predict_next()) - tf.nn.relu(self.predict_prev())) @ tf.transpose(self.wts)
            else:
                average_d_pred += -(layer.predict_next() - self.predict_prev()) @ tf.transpose(self.wts)
            if layer.is_clamped:
                average_d_pred*=2
            else:
                average_d_state += (self.predict_next() - layer.predict_next())
            self.state.assign_sub(self.learning_rate * ((average_d_pred+average_d_state)/2))

    def update_wts(self):
        if not self.fix_wts_b:
            pass
        pass

    def update_b(self):
        if not self.fix_wts_b:
            pass
        pass

    def init_state(self):
        self.state = None

    def __call__(self, x : tf.Tensor):
        if self.wts is None:
            self.init_params(x.shape)
        net_out = x @ self.wts + self.b
        if self.activation == 'relu':
            net_act = tf.nn.relu(net_out)
        else:
            net_act = net_out
        if self.state is None:
            self.state = tf.Variable(net_act, trainable=False)
        return net_act

    def get_kaiming_gain(self):
        if self.activation == 'relu':
            return tf.sqrt(2.)
        else:
            return 1

    def clamp(self, set_clamp:bool):
        self.is_clamped.assign(set_clamp)

    def set_fix_wts_b(self, fix_wts_b:bool):
        self.fix_wts_b.assugn(fix_wts_b)


