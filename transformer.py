import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
      super().__init__()
    
        # Multihead Attention layer
      self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # Feed forward layer
      self.ffn = keras.Sequential(
          [layers.Dense(ff_dim, activation='relu'), layers.Dense(embed_dim),])
    
        # The two add and norm layers
      self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
      self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Dropouts to control overfitting
      self.dropout1 = layers.Dropout(rate)
      self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        # Pass the input to the multihead attention layer
      attn_output = self.att(inputs, inputs)
      attn_output = self.dropout1(attn_output, training = training)
        # Add and norm the attention layer output and input 
      out1 = self.layernorm1(inputs + attn_output)
        # Feed to feedforward network
      ffn_output = self.ffn(out1)
      ffn_output = self.dropout2(ffn_output, training=training)
        # Add and norm with output of feedforward network and output of
        # multihead attention layer.
      return self.layernorm2(out1 + ffn_output)

# Token and position embedding which goes to the output network.
class TokenAndPositionEmbedding(layers.Layer):
  def __init__(self, maxlen, vocab_size, embed_dim):
    super().__init__()
    
    # Token embedding
    self.token_emb = layers.Embedding(input_dim = vocab_size, output_dim = embed_dim)
    # Position embedding. Basically the positions just go from 1 to maxlen
    self.pos_emb = layers.Embedding(input_dim = maxlen, output_dim = embed_dim)

  def call(self, x):
    maxlen = tf.shape(x)[-1]
    
    # Create the position embedding. Just goes from 1 to maxlen-1.
    positions = tf.range(start=0, limit=maxlen, delta=1)
    positions = self.pos_emb(positions)
    
    # Embedded the input tokens
    x = self.token_emb(x)
    
    # Return addition of token positions and embeddings
    return x + positions

vocab_size = 20000
maxlen = 200

# Load the IMDB database. Pad to fixed length.
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words = vocab_size)
print(len(x_train), " Training Sequences")
print(len(x_val), " Testing Sequences")
x_train = keras.utils.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.utils.pad_sequences(x_val, maxlen=maxlen)


embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

# Create the transformer with multiple blocks.
inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
transformer_block2 = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x, training=True)
x = transformer_block2(x, training=True)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(2, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
history = model.fit(
    x_train, y_train, batch_size=32, epochs=5, validation_data=(x_val, y_val)
)