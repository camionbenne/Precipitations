import tensorflow as tf
from cbam import CBAM


"""This is my own code"""

class EPD(tf.keras.Model):
  def __init__(self, kernel_size, filter):
    super(EPD, self).__init__(name='')
    filter_3D = filter
    """---------------Encoder Block 1----------------------"""
    self.conv2a = tf.keras.layers.Conv2D(filter_3D, kernel_size)
    self.bn2a = tf.keras.layers.BatchNormalization()
    self.relua = tf.keras.layers.ReLU()
   
    """---------------Encoder Block 2----------------------"""
    self.cbam = CBAM(2)

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)
