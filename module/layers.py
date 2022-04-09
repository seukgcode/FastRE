from paddle import fluid
import numpy as np


class Conv1D(fluid.dygraph.Layer):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, act: str = None, dilation_rate: int = 1):
        super(Conv1D, self).__init__(None)
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.rec_field = self.kernel_size + (self.kernel_size - 1) * (self.dilation_rate - 1)
        self.pad = self.rec_field // 2
        self.conv1d = fluid.dygraph.Conv2D(num_channels=1, num_filters=output_dim, filter_size=(3, input_dim),
                                           padding=(self.pad, 0), dilation=(dilation_rate, 1), act=act)

    def forward(self, seq):
        h = fluid.layers.unsqueeze(seq, axes=[1])
        h = self.conv1d(h)
        h = fluid.layers.squeeze(h, axes=[3])
        h = fluid.layers.transpose(h, perm=[0, 2, 1])
        return h


class GatedDilatedResidualConv1D(fluid.dygraph.Layer):
    def __init__(self, dim: int, dilation_rate: int):
        super(GatedDilatedResidualConv1D, self).__init__(None)
        self.dim = dim
        self.conv1d = Conv1D(input_dim=self.dim, output_dim=2 * self.dim, kernel_size=3, dilation_rate=dilation_rate)

    def forward(self, seq, mask):
        c = self.conv1d(seq)

        def _gate(x):
            dropout_rate = 0.1
            s, h = x
            g, h = h[:, :, :self.dim], h[:, :, self.dim:]
            g = fluid.layers.dropout(g, dropout_rate, dropout_implementation="upscale_in_train")
            g = fluid.layers.sigmoid(g)
            return g * s + (1 - g) * h

        seq = _gate([seq, c])
        seq = seq * mask
        return seq


class MultiHeadSelfAttention(fluid.dygraph.Layer):
    def __init__(self, input_size: int, num_heads: int, head_size: int):
        super(MultiHeadSelfAttention, self).__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.hidden_size = num_heads * head_size
        self.w_q = fluid.dygraph.Linear(self.input_size, self.hidden_size)
        self.w_k = fluid.dygraph.Linear(self.input_size, self.hidden_size)
        self.w_v = fluid.dygraph.Linear(self.input_size, self.hidden_size)
        self.scale = fluid.layers.sqrt(fluid.dygraph.to_variable(np.array([self.head_size], "float32")))

    def forward(self, q_inputs, k_inputs, v_inputs, mask):
        batch_size = mask.shape[0]
        q = self.w_q(q_inputs)
        k = self.w_k(k_inputs)
        v = self.w_v(v_inputs)
        q = fluid.layers.reshape(q, shape=[batch_size, -1, self.num_heads, self.head_size])
        q = fluid.layers.transpose(q, perm=[0, 2, 1, 3])
        k = fluid.layers.reshape(k, shape=[batch_size, -1, self.num_heads, self.head_size])
        k = fluid.layers.transpose(k, perm=[0, 2, 1, 3])
        v = fluid.layers.reshape(v, shape=[batch_size, -1, self.num_heads, self.head_size])
        v = fluid.layers.transpose(v, perm=[0, 2, 1, 3])
        energy = fluid.layers.matmul(q, fluid.layers.transpose(k, perm=[0, 1, 3, 2])) / self.scale
        energy = fluid.layers.transpose(energy, perm=[0, 3, 2, 1])
        mask = fluid.layers.unsqueeze(mask, 3)
        energy = energy - (1 - mask) * 1e30
        energy = fluid.layers.transpose(energy, perm=[0, 3, 2, 1])
        attention = fluid.layers.softmax(energy, axis=-1)
        x = fluid.layers.matmul(attention, v)
        x = fluid.layers.transpose(x, perm=[0, 2, 1, 3])
        x = fluid.layers.reshape(x, shape=[batch_size, -1, self.hidden_size])
        return x
