import tensorflow as tf
from cbam import CBAM


"""This is my own code"""

class EPD(tf.keras.Model):
  def __init__(self, kernel3D_size,kernel2D_size ,filter = (64,1)):
    super(EPD, self).__init__(name='')
    filter_3D,filter_2D  = filter
    
    """---------------Encoder Block 1----------------------"""
    self.conv3a = tf.keras.layers.Conv3D(filter_3D, kernel3D_size, data_format = "channels_first")
    self.bn2a = tf.keras.layers.BatchNormalization()
    self.relua = tf.keras.layers.ReLU()
   
    """---------------Encoder Block 2----------------------"""
    self.conv2a = tf.keras.layers.Conv2D(filter_2D,kernel2D_size)
    self.cbam = CBAM
    self.maxpool = tf.keras.layers.MaxPool2D()

    """---------------Decoder Block 1----------------------"""
    self.pixel_shuffle = tf.nn.depth_to_space() #(args are input, block_size, data_format='NHWC', name=None)

  def call(self, input_tensor, training=False):
    """Fonction feedforward model
    input shape (B, T=8, C=11, 66, 61)

    Args:
        input_tensor (_tensor_): 
        training (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    """-------------------------Encoder-------------------------------"""

    x = self.conv3a(input_tensor) # output (B,T = 64 , C = , H = 66, W = 61)
    x = self.bn2a(x, training=training) # don't change shape ig
    x = tf.nn.relu(x)#don't change either

    """-------------------------Encoder-------------------------------"""

    x = x + self.cbam(self.conv2a(x),ratio = 1)  # à remplir
    x = self.pixel_shuffle(x,block_size = , data_format = "NCWH")
    return x

block = EPD((8,3,3))
block.summary()