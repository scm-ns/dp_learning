# Copy/Rewrite of fast.ai tutorials

from __future__ import division, print_function
from theano.sandbox import cuda

#from utils import *
import os

path = "data/dogscats/"
model_path = path + "models/"

if not os.path.exists(model_path):
    os.mkdir(model_path)

batch_size = 64



model = vgg_ft(2)


model.load_weights(model_path + "finetine3.h5")


layers = model.layers

"""
    Dropout -> underfitting on the training set as we randomly setting the weights to zero
    Try removing dropout to get better training accuracy. 
    Droput applied to full connected network
    So freeze rest of the network with the weights that were pretrained.
            ie . convolution layer weights are not updated.
    Then retrain on the fully connected networks without the dropout  
"""


# Get all the convolution layers and then get the last layer
last_conv_idx = [index for index,layer in enumerate(layers) if type(layer) is Convolutional2D][-1]

print(last_conv_idx)


layers[last_conv_idx]

conv_layers = layers[:last_conv_idx + 1]
conv_model = Sequential(conv_layers) # Create seperate model from only the conv 2d layers

fc_layers = layers[last_conv_idx + 1:] # Create fc starting from the end of the conv2d layers


batches = get_batches(path + "train" , shuffle = False , batch_size = batch_size)
val_batches = get_batches(path + "valid" , shuffle = False , batch_size = batch_size)

print(batches)

# Get the number of classes ? What is this ? 
val_classes = val_batches.classes
trn_classes = branches.classes

print(val_classes)


# Create a vector of classes ? 
val_labels = onehot(val_classes)
trn_labels = onehot(trn_classes)

print(val_labels)

# Run the batch through the conv net to prodcue outputs. 
val_features = conv_model.predict_generator(val_batches , val_batches.nb_sample)
trn_features = conv_model.predict_generator(batches , batches.nb_sample)

# save the outputs 

save_array(model_path + "train_convlayer_features.bc", trn_features)
save_array(model_path + "valid_convlayer_features.bc", val_features)



trn_features = load_array(model_path + "train_convlayer_features.bc")
val_features = load_array(model_path + "valid_convlayer_features.bc")


print(trn_features.shape)


# On adding dropout, say in general a 0.5 % application will mean that half of the weights
# do not work, so the other neurons will overcompensate for this and acqure 2x the weight.
# So when we remove dropuout and want to use the weights in a similar architecture, we have to 
# divide it by half

def weight_remove_dropout(layer):
    return [o/2 for o in layer.get_weights()]


# SInce most of the weights have been trained, we need to have every small learning rates.
opt = RMSprop(lr = 0.000001 , rho = 0.7)


def get_fn_model():
    model = Sequential([
        MaxPooling2D(input_shape = conv_layers[-1].output_shape[1:]), 
        Flatten(),
        Dense(4096 , activation = "relu"),
        Dropout(0.)_,
        Dense(4096 , activation - "relu"),
        Dropout(0.),
        Dense(2 , activation = "softmax")
        ])

    for l1 , l2 in zip(model.layers , fc_layers): # move the weights from pretrained layer to current layer
        l1.set_weights(weight_remove_dropout(l2) 


    model.compile(optimizer = opt, loss = "categorical_crossentropy" , metrics = ["accuracy"])
    return model 

fc_model = get_fc_model() # get model same as vgg, but with the dropout values removed.

# fit the model in the usual way
fc_model.fit(trn_features , trn_labels , nb_epoch = 8 , 
    batch_size = batch_size , validation = (val_features , val_labels))

fc_model.save_weights(model_path + "no_dropout.h5")
fc_model.load_weights(model_path + "no_dropout.h5")

gen = image.ImageDataGenerator(rotation_range = 10 , width_shift_range = 0.15 , zoom_range = 0.1 ,
    channel_shift_range = 10. , horizontal_flip = True , dim_ordering = "tf")

# Create a batch og single image
img = np.expand_dims(ndimage.imread("cat.jpg") , 0)

aug_iter = gen.flow(img)

aug_imgs = [next(aug_iter)[0].astype(np.uint8) for i in range(8)]

plt.imshow(img[0])

plots(aug_imgs, (20 , 7) , 2)

K.set_image_dim_ordering("th")

gen = image.ImageDataGenerator(rotation_range =15 , widht_shift_range =0.1 , 
            height_shift_range = 0.1 , zoom_range = 0.1 , horizontal_slip = True)

batches = get_batches(path + "train" , gen , batch_size = batch_size)
val_batches = get_batches(path + "valid" , shuffle=False , batch_size = batch_size)

# The generator applies some random transformation with the bounds that we specify to it. 
# What this guarentees is that each image moved through the network is always going to be unique.

fc_model =  get_fc_model()

for layer in conv_model.layers: 
    layer.trainable = False

conv_model.add(fc_model) # Combine the dropout removed layers and the conv model to create a single model 



conv_model.compile(optimizer = opt , loss = "categorical_crossentropy" , metrics = ["accuracy"])

conv_model.fit_generator(batches , samples_per_epoch = batches.nb_sample , nb_epoch = 8 , 
        validation_data=val_batches , nb_val_samples = val_batches.nb_sample)

conv_model.fit_generator(batches , samples_per_epoch = batches.nb_sample , nb_epoch = 3 ,
        validation_data = val_batches , nb_val_sample = val_batches.nb_sample)



conv_layers[-1].output_shape[1:]

def get_bn_layers(p):
    return [
            MaxPooling2D(input_shape = conv_layers[-1].output_shape[1:]),
            Flatten(),
            Dense(4096 , activation = "relu"),
            Dropout(p),
            BatchNormallization(),
            Dense(4096 , activation = "relu"),
            Dropout(p),
            BatchNormalization(),
            Dense(1000, activation = "softmax")
            ]

p=0.6

bn_model = Sequential(get_bn_layers(p))

bn_model.load_weights("/data/ILSVRC2012_img/bn_do3_1.h5")


def proc_weights(layer , prev_p, new_p):
    scal = (1 - prev_p) / ( 1 - new_p)
    return [o * scal for o in layer.get_weights()]

for i in bn_model.layers:
    if type(i) == Dense:
        i.set_weights(proc_weights(1 , 0.3 , 0.6)) # if dense change the weights back ? 

bn_model.pop()
for layer in bn_model.layers: # Train no layers ? # May be we are adding a layer at the end ?
    layers.trainable = False

bn_model.add(Dense(2 , activation = "softmax"))

bn_model.compile(Adam() , "categorical_crossentropy" , metrics=["accuracy"])

bn_model.fit(trn_features , trn_labels , nb_epoch = 8 , validation_data = (val_features , val_labels))

bn_model.save_weights(model_path + "bn.h5")
bn_model.load_weights(model_path + "bn.h5")

bn_layers = get_bn_layers(0.6)
bn_layers.pop()
bn_layers.append(Dense(2,activation="softmax"))

final_model = Sequential(conv_layers)

for layer in final_model.layers:
    layer.trainable = False
    
for layer in bn_layers: # Combine  the conv layers and the batch norm layers
    final_model.add(layer)

for l1 , l2 in zip(bn_model.layers ,bn_layers):  # move weights from the bn for 1000 classes to bn layers for 2 classe
    l2.set_weights(l1.get_weights())

final_model.compile(optimizer=Adam() , loss= "categorical_crossentropy" , metrics=["accuracry"])

final_model.fit_generator(batches , samples_per_epoch = batches.nb_sample , nb_epoch = 1,
        validation_data = val_batches , nb_val_samples = val_batches.nb_sample)

final_model.save_weights(model_path + "final1.h5")

final_model.fit_generator(batches , samples_per_epoch = batches.nb_sample , nb_epoch = 1,
        validation_data = val_batches , nb_val_samples = val_batches.nb_sample)


final_model.save_weights(model_path + "final2.h5")

final_model.optimizer.lr = 0.001

final_model.fit_generator(batches , samples_per_epoch = batches.nb_sample , nb_epoch = 4 , 
        validation_data = val_batches , nb_val_samples = val_batches.nb_sample)

bn_model.save_weights(model_path + "final3.h5")


