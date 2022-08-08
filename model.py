from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import Activation, MaxPool2D, Concatenate, Dense, Flatten

# Unidad base de la red U-net:
#https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
#https://github.com/bnsreenu/python_for_image_processing_APEER/blob/master/tutorial119_multiclass_semantic_segmentation.ipynb

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

#Bloque del Encoder
def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p   

#Bloque del Decoder
#Skip features vienen de las entradas del encoder para ser concatenados

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

# Construir la red Unet
def build_unet(input_shape, n_classes):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    #Se realiza un cambio en la arquitectura para aprovechar la fase del encoding en la tarea de clasificación de la image
    flat_1 = Flatten(name='flat_1')(p4)
    img = Dense(3, activation='softmax',name='img')(flat_1)

    b1 = conv_block(p4, 1024) #Puente

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    seg = Conv2D(n_classes, 1, padding="same", activation='softmax',name='seg')(d4)

    model = Model(inputs, [img, seg], name="U-Net")
    return model