import tensorflow as tf
from tensorflow.keras.layers import Input, MaxPooling2D, UpSampling2D, concatenate, Conv2D, BatchNormalization


#Multi-Attention Mechanism
def multi_attention_block(input_feature, path_feature):
    g = tf.keras.layers.Conv2D(filters=input_feature.shape[-1], kernel_size=1)(input_feature)

    # Upsample g to have the same spatial dimensions as path_feature
    g = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(g)

    x = tf.keras.layers.Conv2D(filters=input_feature.shape[-1], kernel_size=1)(path_feature)
    psi = tf.keras.activations.relu(g + x, alpha=0.0)
    psi = tf.keras.layers.Conv2D(1, kernel_size=1)(psi)
    psi = tf.keras.activations.sigmoid(psi)
    return path_feature * psi

#Deptwise Convolution Block
def depthwise_conv_block(tensor, n_filters):
    tensor = tf.keras.layers.SeparableConv2D(filters=n_filters, kernel_size=(3, 3), padding='same')(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = tf.keras.activations.relu(tensor)

    tensor = tf.keras.layers.SeparableConv2D(filters=n_filters, kernel_size=(3, 3), padding='same')(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = tf.keras.activations.relu(tensor)
    return tensor

#Unet3+ 
def unetpp(input_shape=(256, 256, 3), num_classes=3):
    inputs = Input(shape=input_shape)

    # Contracting/downsampling path
    c1 = depthwise_conv_block(inputs, 64)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = depthwise_conv_block(p1, 128)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = depthwise_conv_block(p2, 256)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = depthwise_conv_block(p3, 512)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = depthwise_conv_block(p4, 1024)
    c5 = multi_attention_block(c5, c4)  # Added attention here

    # Nested skip pathways
    z1_1 = concatenate([c5, c4])
    z1_1 = depthwise_conv_block(z1_1, 512)
    z1_1 = multi_attention_block(z1_1, c3)  # Added attention here

    #u7 = UpSampling2D((2, 2))(z1_1)
    z2_1 = concatenate([z1_1, c3])
    z2_1 = depthwise_conv_block(z2_1, 256)
    z2_1 = multi_attention_block(z2_1, c2)  # Added attention here

    #u8 = UpSampling2D((2, 2))(z2_1)
    z3_1 = concatenate([z2_1, c2])
    z3_1 = depthwise_conv_block(z3_1, 128)

    # More nested skip pathways
    upsamp_c3 = UpSampling2D((2, 2))(c3)
    z2_2 = concatenate([z2_1, upsamp_c3, z3_1])
    z2_2 = depthwise_conv_block(z2_2, 256)

    u10 = UpSampling2D((2, 2))(z2_2)
    upsamp_c2 = UpSampling2D((2, 2))(c2)
    upsamp_z3_1 = UpSampling2D((2, 2))(z3_1)
    z3_2 = concatenate([u10, upsamp_c2, upsamp_z3_1])
    z3_2 = depthwise_conv_block(z3_2, 128)

    # Yet more nested skip pathways
    upsamp_c4 = UpSampling2D((2, 2))(c4)
    mp_z2_2 = MaxPooling2D((2, 2))(z2_2)
    z1_2 = concatenate([z1_1, upsamp_c4, mp_z2_2])
    z1_2 = depthwise_conv_block(z1_2, 512)


    mp_z3_2 = MaxPooling2D((4, 4))(z3_2)
    mp_z2_2_1 = MaxPooling2D((2, 2))(z2_2)
    z2_3 = concatenate([z1_2, c3, mp_z2_2_1, mp_z3_2])
    z2_3 = depthwise_conv_block(z2_3, 256)

    u13 = UpSampling2D((2, 2))(z2_3)
    mp_z3_2 = MaxPooling2D((2, 2))(z3_2)
    z3_3 = concatenate([u13, c2, mp_z3_2, z3_1])
    z3_3 = depthwise_conv_block(z3_3, 128)
    z3_3 = UpSampling2D((2, 2))(z3_3)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(z3_3)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    return model
