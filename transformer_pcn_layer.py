import tensorflow as tf
from dense_pcn_layer import DensePCNLayer

# TBD
class AttentionPCNLayer:
    is_clamped : tf.Variable # bool
    fix_wts_b : tf.Variable # bool
    next_layers: list
    q : DensePCNLayer
    k : DensePCNLayer
    v : DensePCNLayer
    output_shape : tuple
    state : tf.Variable # tf.Tensor
    def __init__(self, q: DensePCNLayer, k: DensePCNLayer, v: DensePCNLayer):
        self.is_clamped = tf.Variable(True, trainable=False)
        self.fix_wts_b = tf.Variable(True, trainable=False)
        self.next_layers = []
        self.q = q
        self.k = k
        self.v = v
        self.output_shape = None
        self.state = None
        
    
    def __call__(self):
        q_state = self.q.predict_next()
        q_state = tf.reshape(q_state, (q_state.shape[0], 1, q_state.shape[1]))
        k_state = self.k.predict_next()
        k_state = tf.reshape(k_state, (k_state.shape[0], q_state.shape[1], 1))
        v_state = self.v.predict_next()
        net_out = (q_state@tf.transpose()/tf.sqrt(float(self.k.predict_next().shape[-1]))) @ self.v.predict_next()

        if self.state is None:
            self.state = tf.Variable(net_out, trainable=False)
