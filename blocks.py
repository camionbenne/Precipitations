import tensorflow as tf
from cbam import CBAM
from Static_fields import StaticFields

"""This is my own code"""

class EPD(tf.keras.Model):
  def __init__(self, kernel3D_size,kernel2D_size,kernel1D_size,filter = (64,1)):
    super(EPD, self).__init__(name='')
    filter_3D,filter_2D  = filter
    
    """---------------Encoder Block 1----------------------"""
    self.conv3a = tf.keras.layers.Conv3D(filter_3D, kernel3D_size, data_format = "channels_first")
    self.bn2a = tf.keras.layers.BatchNormalization()
    self.relua = tf.keras.layers.ReLU()
   
    """---------------Encoder Block 2----------------------"""
    self.conv2a = tf.keras.layers.Conv2D(filter_2D,kernel2D_size, data_format= "channels_first")
    self.cbam = CBAM
    self.maxpool = tf.keras.layers.MaxPool2D()

    """---------------Decoder Block 1----------------------"""
    self.pixel_shuffle = tf.nn.depth_to_space() #(args are input, block_size, data_format='NHWC', name=None)

    """---------------Decoder Block 2----------------------"""
    self.conv1Da = tf.keras.layers.Conv1D(256, kernel_size = kernel1D_size)

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

    x = self.conv3a(input_tensor) # output (B,C = 64, H = 66, W = 61)
    print("Encoder conv3D : "+ str(x.shape))
    x = self.bn2a(x, training=training) # don't change shape ig
    print("Encoder b2na : "+ str(x.shape))
    x = tf.nn.relu(x)#don't change either
    print("Encoder ReLu : "+ str(x.shape))
    """-------------------------Decoder-------------------------------"""
    #Decoder input = (B,C = 64, H = 66, W = 61)

    x = x + self.cbam(self.conv2a(x),ratio = 1)  # à remplir / no shape change
    print("Decoder cbam : "+ str(x.shape))
    x += StaticFields() #ajout de 3 chanels à voir comment les concaténer 
    print("Decoder concat: "+ str(x.shape))
    x = self.conv1Da()
    print("Decoder conv1D : "+ str(x.shape))
    x = self.pixel_shuffle(x,block_size =x.shape[1]**0.5 , data_format = "NCWH")
    print("Decoder pixel shuffle : "+ str(x.shape))
    return x

block = EPD((8,3,3))
block.summary()