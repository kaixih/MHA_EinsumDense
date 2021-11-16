import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import EinsumDense
from mha import MultiHeadAttention

class MultiHeadAttentionEinsum(layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttentionEinsum, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.Q = EinsumDense('BFD,DNH->BFNH',
                         output_shape=(None, self.num_heads, self.depth))
    self.K = EinsumDense('BTD,DNH->BTNH',
                         output_shape=(None, self.num_heads, self.depth))
    self.V = EinsumDense('BTD,DNH->BTNH',
                         output_shape=(None, self.num_heads, self.depth))

    self.dense = EinsumDense('BFNH,NHD->BFD',
                             output_shape=(None, self.d_model))

  def call(self, v_input, k_input, q_input, mask):
    q = self.Q(q_input) # (batch_size, seq_len_q, num_heads, depth)
    v = self.V(v_input) # (batch_size, seq_len_k, num_heads, depth)
    k = self.K(k_input) # (batch_size, seq_len_v, num_heads, depth)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    matmul_qk = tf.einsum('BFNH,BTNH->BNFT', q, k)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
      scaled_attention_logits += (mask * -1e9)

    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    # scaled_attention.shape == (batch_size, seq_len_q, num_heads, depth)
    scaled_attention = tf.einsum('BNFT,BTNH->BFNH', attention_weights, v)

    output = self.dense(scaled_attention)
    return output, attention_weights

# Test
D = 512
N = 8
H = int(D / N)
rand_weight = np.random.normal(size=(D, D))

mha = MultiHeadAttention(d_model=D, num_heads=8)
mha_einsum = MultiHeadAttentionEinsum(d_model=512, num_heads=8)
y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
_ = mha(y, k=y, q=y, mask=None)
mha.wq.set_weights([rand_weight])
mha.wk.set_weights([rand_weight])
mha.wv.set_weights([rand_weight])
mha.dense.set_weights([rand_weight])
out, attn = mha(y, k=y, q=y, mask=None)

rand_weight_einsum = rand_weight.reshape(D, N, H)
_ = mha_einsum(y, y, y, mask=None)
mha_einsum.Q.set_weights([rand_weight_einsum])
mha_einsum.K.set_weights([rand_weight_einsum])
mha_einsum.V.set_weights([rand_weight_einsum])
# Need to swap the input and output dims
rand_weight_einsum1 = rand_weight.reshape(N, H, D)
mha_einsum.dense.set_weights([rand_weight_einsum1])
out1, attn1 = mha_einsum(y, y, y, mask=None)

if (tf.math.reduce_all(tf.math.equal(out, out1)).numpy() and
    tf.math.reduce_all(tf.math.equal(attn, attn1))):
  print("Test passes.")
else:
  print("Test fails.")
