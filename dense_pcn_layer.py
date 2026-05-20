import tensorflow as tf
from typing import Literal
class DensePCNLayer:
    is_clamped : tf.Variable # bool
    fix_wts_b : tf.Variable # bool
    num_units : int
    prev_layer: object
    next_layers: list
    wts : tf.Variable # tf.Tensor
    b : tf.Variable # tf.Tensor
    output_shape : tuple
    activation : str
    state : tf.Variable # tf.Tensor
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
        # print(self.get_kaiming_gain()/tf.sqrt(float(input_shape[-1])))
        self.wts = tf.Variable(tf.random.normal((input_shape[-1], self.num_units), 
                                                stddev=self.get_kaiming_gain()/tf.sqrt(float(input_shape[-1]))), trainable=False)
        self.b = tf.Variable(tf.zeros(self.num_units), trainable=False)


    def predict_prev(self):
        return (self.state - self.b) @ tf.transpose(self.wts)
    
    def predict_next(self):
        return self.state
    
    def d_gelu(self, x:tf.Tensor):
        return 0.5*(1+tf.math.erf(x/tf.sqrt(2.))) + x/tf.sqrt(2*tf.acos(-1.))*tf.exp(-tf.square(x)/2)
    
    def pred_loss_d_input(self, x:tf.Tensor):
        if self.activation == 'relu':
            return -(self.predict_next()-self(x))*self.d_gelu(self.net_in(x)) @ tf.transpose(self.wts)
        else:
            return -(self.predict_next()-self(x)) @ tf.transpose(self.wts)

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
                # print(layer)
                state = self.predict_next()
                pred_state = layer.predict_prev()

                if layer.activation == 'relu':
                    state = tf.nn.relu(state)
                    pred_state = tf.nn.relu(pred_state)

                average_d_pred += layer.pred_loss_d_input(self.predict_next())
                average_d_state += (state - pred_state)
                
            if num_next_layers!=0:
                self.state.assign_sub(self.learning_rate * ((average_d_pred+average_d_state)/(2*num_next_layers)))

            # pred prev layer & pred from prev layer
            if self.prev_layer is not None:
                d_pred = tf.zeros_like(self.state)
                d_state = tf.zeros_like(self.state)
                layer = self.prev_layer
                if self.activation == 'relu':
                    d_pred += -(1+int(layer.is_clamped))*(tf.nn.relu(layer.predict_next()) - tf.nn.relu(self.predict_prev())) @ self.wts
                else:
                    d_pred += -(1+int(layer.is_clamped))*(layer.predict_next() - self.predict_prev()) @ self.wts
                if not layer.is_clamped:
                    d_state += (self.predict_next() - self(layer.predict_next()))
                self.state.assign_sub(self.learning_rate * ((d_pred+d_state)/2))
    # pred_err = state - pred
    # 1/2*(state - pred)^2 = 1/2*(state - act(x@wts+b))^2
    # x.t @ ((state - act(x@wts+b))*act'(x@wts+b))
    # (B, N) (B, M)
    # 1/2*(state - pred)^2 = 1/2*( self.prev_layer.predict_next - self.predict_prev)^2
    # self.predict_prev  = (self.state-self.b) @ tf.transpose(self.wts)
    # ( self.prev_layer.predict_next - self.predict_prev) @ (self.state-self.b)
    # (B, N) (B, M)
    def update_wts(self):
        d_state = tf.zeros_like(self.wts)
        d_pred = tf.zeros_like(self.wts)
        if not self.fix_wts_b and self.prev_layer is not None:
            if not self.prev_layer.is_clamped:
                if self.activation == 'relu':
                    d_state += tf.transpose(self.prev_layer.predict_next()) @ (
                        -(self.predict_next() - self(self.prev_layer.predict_next()))*(self.d_gelu(self.net_in(self.prev_layer.predict_next()))))
                else:
                    d_state += tf.transpose(self.prev_layer.predict_next()) @ -(self.predict_next() - self(self.prev_layer.predict_next()))
            if not self.is_clamped:
                if self.activation == 'relu':
                    d_pred += tf.transpose(tf.nn.relu(self.prev_layer.predict_next()) - tf.nn.relu(self.predict_prev())) @ -(self.predict_next()-self.b)
                else:
                    d_pred += tf.transpose(self.prev_layer.predict_next() - self.predict_prev()) @ -(self.predict_next()-self.b)
            if not self.is_clamped or not self.prev_layer.is_clamped:
                self.wts.assign_sub(self.learning_rate*(d_state+d_pred)/(int(not self.is_clamped)+int(not self.prev_layer.is_clamped)))

    # pred_err = state - pred
    # 1/2*(state - pred)^2 = 1/2*(state - act(x@wts+b))^2
    # ((state - act(x@wts+b))*act'(x@wts+b))
    # (B, N) (B, M)
    # 1/2*(state - pred)^2 = 1/2*( self.prev_layer.predict_next - self.predict_prev)^2
    # self.predict_prev  = (self.state-self.b) @ tf.transpose(self.wts)
    # ( self.prev_layer.predict_next - self.predict_prev) @ (self.state-self.b)
    # (B, N) (B, M)
    def update_b(self):
        d_state = tf.zeros_like(self.b)
        d_pred = tf.zeros_like(self.b)
        if not self.fix_wts_b and self.prev_layer is not None:
            if not self.prev_layer.is_clamped:
                if self.activation == 'relu':
                    d_state += tf.reduce_mean((-(self.predict_next() - self(self.prev_layer.predict_next()))*(self.d_gelu(self.net_in(self.prev_layer.predict_next())))), axis=0)
                else:
                    d_state += tf.reduce_mean(-(self.predict_next() - self(self.prev_layer.predict_next())), axis=0)
            if not self.is_clamped:
                if self.activation == 'relu':
                    d_pred += tf.reduce_mean((tf.nn.relu(self.prev_layer.predict_next()) - tf.nn.relu(self.predict_prev())) @ self.wts, axis=0)
                else:
                    d_pred += tf.reduce_mean((self.prev_layer.predict_next() - self.predict_prev()) @ self.wts, axis=0)
            if not self.is_clamped or not self.prev_layer.is_clamped:
                self.b.assign_sub(self.learning_rate*(d_state+d_pred)/(int(not self.is_clamped)+int(not self.prev_layer.is_clamped)))


    def init_state(self):
        self.state = None

    def init_wts_b(self):
        self.wts = None
        self.b = None

    def net_in(self, x:tf.Tensor):
        if self.wts is None:
            self.init_params(x.shape)
        return x @ self.wts + self.b

    def __call__(self, x : tf.Tensor):
        net_in = self.net_in(x)
        if self.activation == 'relu':
            net_act = tf.nn.relu(net_in)
        else:
            net_act = net_in

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


