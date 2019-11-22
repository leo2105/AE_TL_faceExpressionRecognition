import keras
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Reshape, Conv2DTranspose, Concatenate
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
    
#size_jaffe = (256, 256, 1)
#size_FER = (48, 48, 1)

def ShowImage(img):
    img = cv2.imread(img)
    print(img.shape)
    plt.imshow(img)
    plt.show()
    
    
def Main(dataset_source, dataset_target, dataset_fusion):

    results_source = []
    results_target = []
    
    # Get data
    data_source = Data_supervised(dataset_source)
    data_target = Data_supervised(dataset_target)
    data_fusion = Data_supervised(dataset_fusion)
    
    # Results in source dataset
    ## Results baseline
    model_source = Model_()
    model_source_trained = Train(model_source, data_source)
    loss_source, acc_source = Prediction(model_source_trained, dataset_source)
    loss_target, acc_target = Prediction(model_source_trained, dataset_target)
    results_source.append(acc_source)
    results_target.append(acc_target)

    
    ## Results Fully-Connected layers
    model_target = Transfer_learning(model_source, 'FC')
    model_target_trained = Train(model_target, data_target)
    loss_source, acc_source = Prediction(model_target_trained, dataset_source)
    loss_target, acc_target = Prediction(model_target_trained, dataset_target)
    results_source.append(acc_source)
    results_target.append(acc_target)

    ## Results Fully-Connected and Convolutional layers
    model_target = Transfer_learning(model_source, 'FC_CL')
    model_target_trained = Train(model_target, data_target)
    loss_source, acc_source = Prediction(model_target_trained, dataset_source)
    loss_target, acc_target = Prediction(model_target_trained, dataset_target)
    results_source.append(acc_source)
    results_target.append(acc_target)
    
    ## Results Retraining
    model_target = Transfer_learning(model_source, 'RE')
    model_target_trained = Train(model_target, data_target)
    loss_source, acc_source = Prediction(model_target_trained, dataset_source)
    loss_target, acc_target = Prediction(model_target_trained, dataset_target)
    results_source.append(acc_source)
    results_target.append(acc_target)
    
    ## Results Fusion Dataset
    model_fusion_trained = Train(model_source, data_fusion)
    loss_source, acc_source = Prediction(model_fusion_trained, dataset_source)
    loss_target, acc_target = Prediction(model_fusion_trained, dataset_target)
    results_source.append(acc_source)
    results_target.append(acc_target)
    
    return [results_source, results_target]
    
def Main_2(dataset_source, dataset_target):

    print("HOLA1")
    # Get data source
    data_source = Data_supervised(dataset_source)
    data_source_unsup = Data_unsupervised(dataset_source)
    
    print("HOLA2")
    # Get data source
    data_target = Data_supervised(dataset_target)
    data_target_unsup = Data_unsupervised(dataset_target)

    print("HOLA3")
    # Train supervised model
    model_source = Model_()
    model_source_trained = Train(model_source, data_source)
    
    print("HOLA4")
    # Train unsupervised model
    model_AECNN = AECNN_model()
    model_AECNN_trained = Fit_AECNN(model_AECNN, data_source_unsup)
    
    # Get feature vectors
    latent_vector_model = Get_latent_space_model(model_AECNN_trained)
    print("latent_vector got")
    feature_vector_model = Get_feature_vector_model(model_source_trained)
    print("feature_vector got")

    # Pass source model, AECNN model, generator of data_source and generator of data_source_unsupervised.
    #print("starting processing")
    #Save_processing(latent_vector_model, feature_vector_model, data_source, data_source_unsup)
    #print("finish processing...")
    #del data_source, data_source_unsup
    
    # Get generators transfer learning
    transfer_learning_gen_train = Combine_both_generator_train(data_target)
    print("transfer learning gen train finished")
    transfer_learning_gen_valid = Combine_both_generator_valid(data_target)
    print("transfer learning gen valid finished")

    # Fit transfer learning
    model_finetunned_FC = Fit_transfer_learning_AECNN(latent_vector_model, feature_vector_model, transfer_learning_gen_train, transfer_learning_gen_valid)

    # Get generators prediction on data target
    #test_gen_train = Combine_both_generator_train(data_source)
    #print("test gen train finished")
    #test_gen_valid = Combine_both_generator_valid(data_source)
    #print("test gen valid finished")

    # Make predictions on data target
    loss, acc = Prediction(model_finetunned_FC, "FER")
    print("Accuracy of FER: ", str(acc))
    loss, acc = Prediction(model_finetunned_FC, "JAFFE")
    print("Accuracy of JAFFE: ", str(acc))
    
def Data_supervised(dataset_name):
    path_train = './' + dataset_name + '/Training/'
    path_valid = './' + dataset_name + '/Validation/'

    train_datagen  = ImageDataGenerator()
    valid_datagen = ImageDataGenerator()
    print(path_train)
    train_generator = train_datagen.flow_from_directory(path_train, target_size=(48, 48), batch_size=64, color_mode='grayscale', shuffle=True, seed=10101)

    valid_generator = valid_datagen.flow_from_directory(path_valid, target_size=(48, 48), batch_size=64, color_mode='grayscale', shuffle=True, seed=10101)
    
    print(train_generator.class_indices)
    #showImage(path_train + '/' + np.random.choice(train_generator.filenames))
    return [train_generator, valid_generator]
    
def Data_unsupervised(dataset_name):
    path_train = './' + dataset_name + '/Training/'
    path_valid = './' + dataset_name + '/Validation/'

    train_datagen  = ImageDataGenerator()
    valid_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(path_train, target_size=(48, 48), batch_size=64, color_mode='grayscale', shuffle=True, class_mode="input", seed=10101)

    valid_generator = valid_datagen.flow_from_directory(path_valid, target_size=(48, 48), batch_size=64, color_mode='grayscale', shuffle=True, class_mode="input", seed=10101)

    print(train_generator.class_indices)
    #showImage(path_train + '/' + np.random.choice(train_generator.filenames))
    return [train_generator, valid_generator]


def Train(model, data):

    print("CHAU1")
    # get train and valid data
    [train_gen, valid_gen] = data
    
    print("CHAU2")
    # model training parameter
    #file_path = 'AEC_featExtrac_model.h5'
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    #checkpoint = tf.keras.callbacks.ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [earlyStop]
    #callbacks_list = []
    
    # compile and fit the model
    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(1e-5))
    model.fit_generator( train_gen, validation_data=valid_gen,
            steps_per_epoch=100,
            epochs=100,
            validation_steps=50,
            callbacks=callbacks_list)
    return model

def Transfer_learning(model, option):

    if option == 'FC':
        for layer in model.layers[:9]:
            layer.trainable=False
        for layer in model.layers[9:]:
            layer.trainable=True
        
    if option == 'FC_CL':
        for layer in model.layers[:6]:
            layer.trainable=False
        for layer in model.layers[6:]:
            layer.trainable=True
            
    if option == 'RE':
        for layer in model.layers:
            layer.trainable=True
        
    return model
          
def Prediction(model, dataset_name):
    path_test = dataset_name + '/Test'
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(
            path_test,
            target_size=(48, 48),
            color_mode='grayscale',
            batch_size=1)
    
    filenames = test_generator.filenames
    nb_samples = len(filenames)
    
    # for new model
    test_generator = Combine_both_generator_test(test_generator)
    
    loss, acc = model.evaluate_generator(test_generator, steps=nb_samples)
    return loss, acc
 

def Model_():

    input= Input(shape=(48, 48, 1))

    #model = Lambda(lambda image: tf.image.resize(image, target_size))(input)
    model = Conv2D(64, (2, 2), strides=1, activation='relu', padding='valid')(input)
    model = Conv2D(64, (2, 2), strides=1, activation='relu', padding='valid')(model)
    model = MaxPooling2D((2, 2), strides=2, padding='valid')(model)
    model = Conv2D(128, (3, 3), strides=1, activation='relu', padding='valid')(model)
    model = Conv2D(128, (3, 3), strides=1, activation='relu', padding='valid')(model)
    model = Conv2D(128, (3, 3), strides=1, activation='relu', padding='valid')(model)
    model = MaxPooling2D((2, 2), strides=2, padding='valid')(model)
    model = Flatten()(model)
    model = Dense(2048, activation="relu")(model)
    model = Dense(1024, activation="relu")(model)
    model = Dropout(0.5)(model)
    out = Dense(7, activation="softmax")(model)

    model = Model(input, out)
    model.summary()

    return model


def AECNN_model():

    filters = [16, 32, 64, 128, 7]

    input_img_AECNN = Input(shape=(48, 48, 1))

    # Encoder
    x_AECNN = Conv2D(filters[0], (3, 3), strides=2, activation='relu', padding='same')( input_img_AECNN)
    x_AECNN = Conv2D(filters[1], (3, 3), strides=2, activation='relu', padding='same')(x_AECNN)
    x_AECNN = Conv2D(filters[2], (3, 3), strides=2, activation='relu', padding='same')(x_AECNN)
    x_AECNN = Conv2D(filters[3], (3, 3), strides=2, activation='relu', padding='same')(x_AECNN)

    #x_AECNN = Flatten()(x_AECNN)

    #x_AECNN = Dense(units=filters[3] * int(48/(2**(len(filters)-1))) * int(48/(2**(len(filters)-1))), activation='relu')(x_AECNN) # 12500

    #x_AECNN = Reshape((int(48/(2**(len(filters)-1))), int(48/(2**(len(filters)-1))), filters[-2]))(x_AECNN)

    # Decoder
    x_AECNN = Conv2DTranspose(filters[2], (3, 3), strides=2, activation='relu', padding='same')(x_AECNN)
    x_AECNN = Conv2DTranspose(filters[1], (3, 3), strides=2, activation='relu', padding='same')(x_AECNN)
    x_AECNN = Conv2DTranspose(filters[0], (3, 3), strides=2, activation='relu', padding='same')(x_AECNN)
    decoded_AECNN = Conv2DTranspose(1, (3, 3), strides=2, activation='sigmoid', padding='same')(x_AECNN)


    model = Model(input_img_AECNN, decoded_AECNN)
    print(model.summary)
    
    return model
    
    
def Fit_AECNN(model, data):
    [train_gen, valid_gen] = data
    
    #AECNN.compile(optimizer='adadelta', loss='binary_crossentropy')
    model.compile(optimizer='adadelta', loss='mse')

    #file_path = 'AEC_featExtrac_model.h5'
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    #checkpoint = tf.keras.callbacks.ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [earlyStop]

    model.fit_generator(train_gen, validation_data=valid_gen,
              epochs=100,
              steps_per_epoch=100,
              validation_steps=50,
              callbacks=callbacks_list)

    return model
              
def Get_latent_space_model(model_AECNN):
    
    #[X_train, X_valid] = X
    filters = [16, 32, 64, 128, 7]
    input = Input(shape=(48, 48, 1))
    
    model = Conv2D(filters[0], (3, 3), strides=2, activation='relu', padding='same', weights=model_AECNN.layers[1].get_weights())(input)
    model = Conv2D(filters[1], (3, 3), strides=2, activation='relu', padding='same', weights=model_AECNN.layers[2].get_weights())(model)
    model = Conv2D(filters[2], (3, 3), strides=2, activation='relu', padding='same', weights=model_AECNN.layers[3].get_weights())(model)
    model = Conv2D(filters[3], (3, 3), strides=2, activation='relu', padding='same', weights=model_AECNN.layers[4].get_weights())(model)

    out = Flatten()(model)

    #featuremodel_AECNN.add(Dense(units=filters[-1], weights=AECNN.layers[6].get_weights()))
    #out = Dense(units=filters[3] * int(size_to_resize[0]/(2**(len(filters)-1))) * int(size_to_resize[0]/(2**(len(filters)-1))), weights=AECNN.layers[6].get_weights())(out)

    model = Model(input, out)
    
    for layer in model.layers[:]:
        layer.trainable=False
    
    return model


def Get_feature_vector_model(model_base):

    input= Input(shape=(48, 48, 1))

    model = Conv2D(64, (2, 2), strides=1, activation='relu', padding='valid', weights=model_base.layers[1].get_weights())(input)
    model = Conv2D(64, (2, 2), strides=1, activation='relu', padding='valid', weights=model_base.layers[2].get_weights())(model)
    model = MaxPooling2D((2, 2), strides=2, padding='valid', weights=model_base.layers[3].get_weights())(model)
    model = Conv2D(128, (3, 3), strides=1, activation='relu', padding='valid', weights=model_base.layers[4].get_weights())(model)
    model = Conv2D(128, (3, 3), strides=1, activation='relu', padding='valid', weights=model_base.layers[5].get_weights())(model)
    model = Conv2D(128, (3, 3), strides=1, activation='relu', padding='valid', weights=model_base.layers[6].get_weights())(model)
    model = MaxPooling2D((2, 2), strides=2, padding='valid', weights=model_base.layers[7].get_weights())(model)
    out = Flatten()(model)
    
    model = Model(input, out)
        
    for layer in model.layers[:]:
        layer.trainable=False

    return model

def Combine_both_generator_train(data_gen):

    while True:
        X1i = data_gen[0].next()
        #X2i = data_gen[0].next()
        yield [X1i[0], X1i[0]], X1i[1]  #Yield both images and their mutual label
            
            
def Combine_both_generator_valid(data_gen):

    while True:
        X1i = data_gen[1].next()
        #X2i = data_gen[1].next()
        yield [X1i[0], X1i[0]], X1i[1]  #Yield both images and their mutual label

def Combine_both_generator_test(data_gen):

    while True:
        X1i = data_gen.next()
        #X2i = data_gen[.next()
        yield [X1i[0], X1i[0]], X1i[1]  #Yield both images and their mutual label


# Fix this for training and validation and train
def Fit_transfer_learning_AECNN(model_latent, model_feature, gen_train, gen_valid):
    
    combined = Concatenate()([model_latent.output, model_feature.output])
    model = Dense(2048, activation="relu")(combined)
    model = Dense(1024, activation="relu")(model)
    model = Dropout(0.5)(model)
    out = Dense(7, activation="softmax")(model)
    
    model = Model(inputs=[model_latent.input, model_feature.input], outputs=out)
    print(model.summary)
    
    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(1e-5))

    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    #checkpoint = tf.keras.callbacks.ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [earlyStop]
        
    H=model.fit_generator(gen_train,
                           steps_per_epoch=100,
                           epochs = 100,
                           validation_data = gen_valid,
                           validation_steps = 50,
                           shuffle=False)
    #plot_model(model, show_shapes=True, show_layer_names=True)
    """
    loss, acc = model.evaluate_generator(generator=datagen.flow([vect1_train, vect2_train], [vect1_valid, vect2_valid]))
    """
    #loss, acc = model.predict([vect1_valid, vect2_valid])
    print("Accuracy: ", H.history)

    return model


opcion = 'new_model'
if opcion == 'replicate':
    rpta = Main('FER', 'JAFFE', 'FER_JAFFE')
    print('BL, FC, FC+CL, RE, FU')
    print(rpta[0])
    print(rpta[1])
elif opcion == 'new_model':
    Main_2('FER', 'JAFFE')
else:
    print("Choose an option.")


