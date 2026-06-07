import tensorflow as tf
from conv_pcn_layer import Conv2DPCNLayer, MaxPool2DPCNLayer
from dense_pcn_layer import DensePCNLayer
from transformer_pcn_layer import TransformerPCNLayer, PositionalEncodingLayer, AttentionPCNLayer, AddNormalizePCNLayer
class InputPCNLayer:
    is_clamped : tf.Variable # bool
    fix_wts_b : tf.Variable # bool
    next_layers: list
    output_shape : tuple
    state : tf.Variable # tf.Tensor
    learning_rate:float
    def __init__(self, learning_rate: float, next_layers:list=None):
        self.is_clamped = tf.Variable(True, trainable=False)
        self.fix_wts_b = tf.Variable(True, trainable=False)
        self.next_layers = [] if next_layers is None else next_layers
        self.output_shape = None
        self.state = None
        self.learning_rate = learning_rate
    
    def update_state(self):
        if not self.is_clamped:
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

    def update_wts(self):
        pass # there is no wts

    def update_b(self):
        pass # there is no bias
    
    def predict_next(self):
        return self.state
        
    def init_state(self):
        self.state = None

    def set_state(self, x:tf.Tensor):
        self.state = tf.Variable(x, trainable=False)
        self.output_shape = x.shape

class TransposePCNLayer:
    is_clamped : tf.Variable # bool
    fix_wts_b : tf.Variable # bool
    prev_layer : DensePCNLayer
    next_layers: list
    output_shape : tuple
    def __init__(self, prev_layer:object, next_layers:list=None):
        self.is_clamped = tf.Variable(False, trainable=False)
        self.fix_wts_b = tf.Variable(True, trainable=False)
        self.prev_layer = prev_layer
        self.next_layers = [] if next_layers is None else next_layers
        self.output_shape = None

    def __call__(self, x:tf.Tensor):
        self.output_shape = x.shape
        self.output_shape[-1], self.output_shape[-2] = self.output_shape[-2], self.output_shape[-1]
        return tf.transpose(x, perm = list(range(tf.rank(x)-2))+[tf.rank(x)-1, tf.rank(x)-2])
    
    def predict_next(self):
        return self(self.prev_layer.predict_next())
    
    # assume 1 next layer
    def predict_prev(self):
        return self(self.next_layers[0].predict_prev())
    
    def pred_loss_d_input(self):
        return 1.

class EncoderEncoderPCN:
    trainable_layers : list
    learning_rate : float
    img_input : object
    txt_input : object
    def __init__(self, learning_rate : float, mask: tf.Tensor=None):
        self.trainable_layers = []
        self.learning_rate = learning_rate
        self.img_input = InputPCNLayer(learning_rate)
        self.trainable_layers.append(self.img_input)
        conv1 = Conv2DPCNLayer(64, (3, 3), learning_rate, 'relu', self.img_input)
        self.trainable_layers.append(conv1)
        self.img_input.next_layers = [conv1]
        conv2 = Conv2DPCNLayer(64, (3, 3), learning_rate, 'relu', conv1)
        self.trainable_layers.append(conv2)
        conv1.next_layers = [conv2]
        mp1 = MaxPool2DPCNLayer((2, 2), conv2)
        conv2.next_layers = [mp1]
        conv3 = Conv2DPCNLayer(128, (3, 3), learning_rate, 'relu', mp1)
        self.trainable_layers.append(conv3)
        mp1.next_layers = [conv3]
        conv4 = Conv2DPCNLayer(128, (3, 3), learning_rate, 'relu', conv3)
        self.trainable_layers.append(conv4)
        conv3.next_layers = [conv4]
        mp2 = MaxPool2DPCNLayer((2, 2), conv4)
        conv4.next_layers = [mp2]
        conv5 = Conv2DPCNLayer(256, (3, 3), learning_rate, 'relu', mp2)
        self.trainable_layers.append(conv5)
        mp2.next_layers = [conv5]
        conv6 = Conv2DPCNLayer(256, (3, 3), learning_rate, 'relu', conv5)
        self.trainable_layers.append(conv6)
        conv5.next_layers = [conv6]
        mp3 = MaxPool2DPCNLayer((2, 2), conv6)
        conv6.next_layers = [mp3]
        conv7 = Conv2DPCNLayer(512, (3, 3), learning_rate, 'relu', mp3)
        self.trainable_layers.append(conv7)
        mp3.next_layers = [conv7]
        conv8 = Conv2DPCNLayer(512, (3, 3), learning_rate, 'relu', conv7)
        self.trainable_layers.append(conv8)
        conv7.next_layers = [conv8]
        mp4 = MaxPool2DPCNLayer((2, 2), conv8)
        conv8.next_layers = [mp4]
        conv9 = Conv2DPCNLayer(1024, (3, 3), learning_rate, 'relu', mp4)
        self.trainable_layers.append(conv9)
        mp4.next_layers = [conv9]

        self.txt_input = InputPCNLayer(learning_rate)
        self.trainable_layers.append(self.txt_input)
        txt_embedding = DensePCNLayer(512, learning_rate, 'linear', self.txt_input)
        self.trainable_layers.append(txt_embedding)
        self.txt_input.next_layers = [txt_embedding]
        pos_encoding = PositionalEncodingLayer(512, txt_embedding)
        txt_embedding.next_layers = [pos_encoding]
        transformer1 = TransformerPCNLayer(3, 512, 8, learning_rate, pos_encoding, mask=mask)
        transformer1_layers = transformer1.get_layers()
        pos_encoding.next_layers=[transformer1_layers[0]]
        self.trainable_layers += transformer1_layers
        transformer2 = TransformerPCNLayer(3, 512, 8, learning_rate, self.trainable_layers[-1], mask=mask)
        transformer2_layers = transformer2.get_layers()
        self.trainable_layers[-1].next_layers = [transformer2_layers[0]]
        self.trainable_layers += transformer2_layers
        transformer3 = TransformerPCNLayer(3, 512, 8, learning_rate, self.trainable_layers[-1], mask=mask)
        transformer3_layers = transformer3.get_layers()
        self.trainable_layers[-1].next_layers = [transformer3_layers[0]]
        self.trainable_layers += transformer3_layers
        linear_1 = DensePCNLayer(1024, learning_rate, 'linear', self.trainable_layers[-1])
        self.trainable_layers[-1].next_layers = [linear_1]
        self.trainable_layers.append(linear_1)
        tp1 = TransposePCNLayer(linear_1)
        linear_1.next_layers = [tp1]
        linear_2 = DensePCNLayer(48, learning_rate, 'linear', tp1)
        tp1.next_layers = [linear_2]
        self.trainable_layers.append(linear_2)
        tp2 = TransposePCNLayer(linear_2)
        linear_2.next_layers = [tp2]
        # mask_bool = (mask==0)
        # new_mask_bool_48 = (mask_bool @ tf.abs(linear_2.wts)) > 0
        # new_mask_48 = tf.where(new_mask_bool_48, -1e9, 0.0) 
        transformer4 = TransformerPCNLayer(3, 1024, 8, learning_rate, tp2)
        transformer4_layers = transformer4.get_layers()
        tp2.next_layers = [transformer4_layers[0]]
        self.trainable_layers += transformer4_layers
        transformer5 = TransformerPCNLayer(3, 1024, 8, learning_rate, self.trainable_layers[-1])
        transformer5_layers = transformer5.get_layers()
        self.trainable_layers[-1].next_layers = [transformer5_layers[0]]
        self.trainable_layers += transformer5_layers
        transformer6 = TransformerPCNLayer(3, 1024, 8, learning_rate, self.trainable_layers[-1])
        transformer6_layers = transformer6.get_layers()
        self.trainable_layers[-1].next_layers = [transformer6_layers[0]]
        self.trainable_layers += transformer6_layers
        linear_3 = DensePCNLayer(2048, learning_rate, 'linear', self.trainable_layers[-1])
        self.trainable_layers[-1].next_layers = [linear_3]
        self.trainable_layers.append(linear_3)
        tp3 = TransposePCNLayer(linear_3)
        linear_3.next_layers = [tp3]
        linear_4 = DensePCNLayer(12, learning_rate, 'linear', tp3)
        tp3.next_layers = [linear_4]
        self.trainable_layers.append(linear_4)
        tp4 = TransposePCNLayer(linear_4)
        linear_4.next_layers = [tp4]
        # new_mask_bool_12 = (new_mask_bool_48 @ tf.abs(linear_4.wts)) > 0
        # new_mask_12 = tf.where(new_mask_bool_12, -1e9, 0.0) 
        transformer7 = TransformerPCNLayer(3, 2048, 8, learning_rate, tp4)
        transformer7_layers = transformer7.get_layers()
        tp4.next_layers = [transformer7_layers[0]]
        self.trainable_layers += transformer7_layers
        transformer8 = TransformerPCNLayer(3, 2048, 8, learning_rate, self.trainable_layers[-1])
        transformer8_layers = transformer8.get_layers()
        self.trainable_layers[-1].next_layers = [transformer8_layers[0]]
        self.trainable_layers += transformer8_layers
        transformer9 = TransformerPCNLayer(3, 2048, 8, learning_rate, self.trainable_layers[-1])
        transformer9_layers = transformer9.get_layers()
        self.trainable_layers[-1].next_layers = [transformer9_layers[0]]
        self.trainable_layers += transformer9_layers
        linear_5 = DensePCNLayer(4096, learning_rate, 'linear', self.trainable_layers[-1])
        self.trainable_layers[-1].next_layers = [linear_5]
        self.trainable_layers.append(linear_5)
        tp5 = TransposePCNLayer(linear_5)
        linear_5.next_layers = [tp5]
        linear_6 = DensePCNLayer(3, learning_rate, 'linear', tp5)
        tp5.next_layers = [linear_6]
        self.trainable_layers.append(linear_6)
        tp6 = TransposePCNLayer(linear_6)
        linear_6.next_layers = [tp6]
        # new_mask_bool_3 = (new_mask_bool_12 @ tf.abs(linear_6.wts)) > 0
        # new_mask_3 = tf.where(new_mask_bool_3, -1e9, 0.0)
        transformer10 = TransformerPCNLayer(3, 4096, 8, learning_rate, tp6)
        transformer10_layers = transformer10.get_layers()
        tp6.next_layers = [transformer10_layers[0]]
        self.trainable_layers+=transformer10_layers
        transformer11 = TransformerPCNLayer(3, 4096, 8, learning_rate, self.trainable_layers[-1])
        transformer11_layers = transformer11.get_layers()
        self.trainable_layers[-1].next_layers = [transformer11_layers[0]]
        self.trainable_layers += transformer11_layers
        transformer12 = TransformerPCNLayer(3, 4096, 8, learning_rate, self.trainable_layers[-1])
        transformer12_layers = transformer12.get_layers()
        self.trainable_layers[-1].next_layers = [transformer12_layers[0]]
        self.trainable_layers += transformer12_layers
        transformer13 = TransformerPCNLayer(3, 4096, 8, learning_rate, self.trainable_layers[-1])
        transformer13_layers = transformer13.get_layers()
        self.trainable_layers[-1].next_layers = [transformer13_layers[0]]
        self.trainable_layers += transformer13_layers
        transformer14 = TransformerPCNLayer(3, 4096, 8, learning_rate, self.trainable_layers[-1])
        transformer14_layers = transformer14.get_layers()
        self.trainable_layers[-1].next_layers = [transformer14_layers[0]]
        self.trainable_layers += transformer14_layers
        transformer15 = TransformerPCNLayer(3, 4096, 8, learning_rate, self.trainable_layers[-1])
        transformer15_layers = transformer15.get_layers()
        self.trainable_layers[-1].next_layers = [transformer15_layers[0]]
        self.trainable_layers += transformer15_layers
        transformer16 = TransformerPCNLayer(3, 4096, 8, learning_rate, self.trainable_layers[-1])
        transformer16_layers = transformer16.get_layers()
        self.trainable_layers[-1].next_layers = [transformer16_layers[0]]
        self.trainable_layers += transformer16_layers
        transformer17 = TransformerPCNLayer(3, 4096, 8, learning_rate, self.trainable_layers[-1])
        transformer17_layers = transformer17.get_layers()
        self.trainable_layers[-1].next_layers = [transformer17_layers[0]]
        self.trainable_layers += transformer17_layers

    def pass_next(self, prev_layer, layer):
        if isinstance(layer, AddNormalizePCNLayer):
            new_output = layer(layer.prev_layers[0].predict_next(), layer.prev_layers[1].predict_next())
        else:
            new_output = layer(prev_layer.predict_next())
        if layer.next_layers is not None:
            for next_layer in layer.next_layers:
                self.pass_next(layer, next_layer)
        else:
            print(new_output.shape)


    def pass_through(self, img_tensor, txt_tensor):
        self.img_input.set_state(img_tensor)
        self.pass_next(self.img_input, self.img_input.next_layers[0])
        self.txt_input.set_state(txt_tensor)
        self.pass_next(self.txt_input, self.txt_input.next_layers[0])






        



        



