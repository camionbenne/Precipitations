import tensorflow as tf 


"""This part of code originaly provides from https://github.com/AlirezaFBabaei/CBAM-Convolutional-Block-Attention-Module.git"""

def Channel_Attention_Module(input, ratio=16):

  b, _, _, channel = input.shape

  # Shared MLP
  l1 = tf.keras.layers.Dense(channel//ratio, activation='relu', use_bias=False)
  l2 = tf.keras.layers.Dense(channel, use_bias=False)

  # Global Average Pooling
  avepool = tf.keras.layers.GlobalAveragePooling2D()(input)
  a = l1(avepool)
  a = l2(a)

  # Global Max Pooling
  maxpool = tf.keras.layers.GlobalMaxPooling2D()(input)
  m = l1(maxpool)
  m = l2(m)

  # Add Average and Max Pooling
  concat = a + m
  concat = tf.keras.layers.Activation('sigmoid')(concat)

  output = tf.keras.layers.Multiply()([input, concat])

  return output



def Spatial_Attention_Module(input):

  # Average Pooling
  avepool = tf.reduce_mean(input, axis=-1)
  avepool = tf.expand_dims(input, axis=-1)

  # Max Pooling
  maxpool = tf.reduce_max(input, axis=-1)
  maxpool = tf.expand_dims(input, axis=-1)

  # Concatenate Average and Max Pooling
  concat = tf.keras.layersConcatenate()([avepool, maxpool])

  conv = tf.keras.layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)

  output = tf.keras.layers.Multiply()([input, conv])

  return output

def CBAM(input):
  attention = Channel_Attention_Module(input)
  attention = Spatial_Attention_Module(attention)

  return attention