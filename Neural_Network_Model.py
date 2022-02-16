import scipy
from keras import Sequential
import numpy as np
from Parameters import *
from keras import Sequential
from keras.layers import Dense
from keras.activations import relu
from keras.activations import sigmoid
from keras.layers import BatchNormalization
# from keras.optimizers import SGD
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Conv1D, Conv2D
from keras.layers import MaxPooling1D, MaxPooling2D
from keras.layers import Flatten
from keras.callbacks import EarlyStopping, TerminateOnNaN
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras.losses import mean_squared_error, mean_absolute_percentage_error
from keras import backend as K
from tensorflow import keras
from keras.optimizers import *
from Parameters import *
import random
from keras.layers import LeakyReLU

class Neural_Network_Model:

    # attributes


    # methods
    def __init__(self):
        self.nn_model                           = None
        self.history                            = None

    def combined_loss(self, params):
        energy_difference, drive_cycle_id_t0_np, drive_cycle_id_t100_np, lam = params
        #if the drive cycles are different the energy difference will be ignored
        energy_difference_np = np.array(energy_difference, dtype=np.float32)
        for item_idx in range(len(energy_difference_np)):
            if drive_cycle_id_t0_np[item_idx] != drive_cycle_id_t100_np[item_idx]:
                 energy_difference_np[item_idx] = 0
        def loss(y_true, y_pred):
            # return mean_squared_error(y_true, y_pred) + lam * K.mean(K.relu(energy_difference_np))
            return mean_absolute_percentage_error(y_true, y_pred) + lam * K.mean(K.relu(energy_difference_np))

        return loss

    def compute_energy(self, soc_prediction, voltage_v):
        # E=Q*V*Delta_soc --> to simplify calculation Delta_z = 0
        voltage_v=voltage_v[:,-1,:]
        voltage_v = np.array(voltage_v).reshape((-1,1))
        soc_min  = 0
        delta_soc = soc_prediction - soc_min
        energy_Wh = np.multiply(RATED_CAPACITY*np.abs(voltage_v), delta_soc)
        # energy_Wh = RATED_CAPACITY * voltage_v * delta_soc

        return energy_Wh

    def batch_generator_one_dim_cnn(self,time_window_size, data_x_np, data_y_np, batch_size):
        # TODO: to be implemented

        number_of_features = NUMBER_OF_FEATURES
        total_number_of_samples = data_x_np.shape[0]
        assert (data_x_np.shape[0] == data_y_np.shape[0]), "Inputs and Targets do not have the same size!"
        assert (data_x_np.shape[1] == NUMBER_OF_FEATURES), "Invalid number of features!"

        x_batch = np.zeros((batch_size, time_window_size * number_of_features, 1))
        y_batch = np.zeros((batch_size, 1))

        start_idx = 0

        while (True):

            for batch_index in range(batch_size):

                start_idx = random.randint(0, total_number_of_samples - time_window_size-1)

                x_batch[batch_index] = data_x_np[start_idx:start_idx + time_window_size,:].reshape(-1, 1) #flattening --> 1D CNN
                y_batch[batch_index] = data_y_np[start_idx + time_window_size]

            yield x_batch, y_batch


    def batch_generator_two_dim_cnn(self,time_window_size, data_x_np, data_y_np, batch_size):
        # TODO: to be implemented

        number_of_features = NUMBER_OF_FEATURES
        total_number_of_samples = data_x_np.shape[0]
        assert (data_x_np.shape[0] == data_y_np.shape[0]), "Inputs and Targets do not have the same size!"
        assert (data_x_np.shape[1] == NUMBER_OF_FEATURES), "Invalid number of features!"

        x_batch = np.zeros((batch_size, time_window_size, number_of_features, 1))
        y_batch = np.zeros((batch_size, 1))

        start_idx = random.randint(0, total_number_of_samples - time_window_size-1)

        while (True):

            for batch_index in range(batch_size):

                # start_idx = random.randint(0, total_number_of_samples - time_window_size-1)

                x_batch[batch_index] = data_x_np[start_idx:start_idx + time_window_size,:].reshape((MODEL_INPUT_TIME_STEPS, NUMBER_OF_FEATURES,1)) #flattening --> 1D CNN
                y_batch[batch_index] = data_y_np[start_idx + time_window_size]

                start_idx += time_window_size
                if start_idx >= total_number_of_samples - time_window_size:
                    start_idx = 0

            yield x_batch, y_batch


    def batch_generator_two_dim_cnn_v2(self,time_window_size, data_x_np, data_y_np, batch_size):
        # in this version we slide by one sample between matrices contained by the batch
        # NOT USEFUL

        number_of_features = NUMBER_OF_FEATURES
        total_number_of_samples = data_x_np.shape[0]
        assert (data_x_np.shape[0] == data_y_np.shape[0]), "Inputs and Targets do not have the same size!"
        assert (data_x_np.shape[1] == NUMBER_OF_FEATURES), "Invalid number of features!"

        x_batch = np.zeros((batch_size, time_window_size, number_of_features, 1))
        y_batch = np.zeros((batch_size, 1))

        start_idx = 0

        while (True):

            for batch_index in range(batch_size):

                # start_idx = random.randint(0, total_number_of_samples - time_window_size-1)

                x_batch[batch_index] = data_x_np[start_idx:start_idx + time_window_size,:].reshape((MODEL_INPUT_TIME_STEPS, NUMBER_OF_FEATURES,1)) #flattening --> 1D CNN
                y_batch[batch_index] = data_y_np[start_idx + time_window_size]

                start_idx += 1 # sliding by 1 row
                if start_idx >= total_number_of_samples - time_window_size:
                    start_idx = 0

            yield x_batch, y_batch



    def create_model(self, input_at_t0_np, input_at_t100_np, regularization_param_lambda, optimizer_value):
        # self.nn_model = Sequential()
        # self.nn_model.add(Dense(NUMBER_OF_FEATURES, 'relu'))
        # self.nn_model.add(BatchNormalization())
        # self.nn_model.add(Dense(5, 'relu'))
        # self.nn_model.add(BatchNormalization())
        # self.nn_model.add(Dense(5, 'relu'))
        # self.nn_model.add(BatchNormalization())
        # self.nn_model.add(Dense(5, 'relu'))
        # self.nn_model.add(BatchNormalization())
        # self.nn_model.add(Dense(5, 'relu'))
        # self.nn_model.add(BatchNormalization())
        # self.nn_model.add(Dense(5, 'relu'))
        # self.nn_model.add(BatchNormalization())
        # self.nn_model.add(Dense(5, 'relu'))
        # self.nn_model.add(BatchNormalization())
        # self.nn_model.add(Dense(5, 'relu'))
        # self.nn_model.add(BatchNormalization())
        # self.nn_model.add(Dense(1, 'linear'))

        ################################### 1D CNN
        # self.nn_model = Sequential()
        # self.nn_model.add(Conv1D(filters=5, kernel_size=5, padding='same', activation='relu', input_shape=(NUMBER_OF_FEATURES*MODEL_INPUT_TIME_STEPS, 1)))
        # self.nn_model.add(MaxPooling1D(pool_size=3))
        # # self.nn_model.add(Dropout(0.3))
        # self.nn_model.add(BatchNormalization())
        # self.nn_model.add(Flatten())
        # self.nn_model.add(Dense(5, 'relu'))
        # self.nn_model.add(Dense(1, 'sigmoid'))
        #
        # #physics based regularization
        # input_vector_t0 = K.constant(value=input_at_t0_np[:,1:].reshape((-1,NUMBER_OF_FEATURES * MODEL_INPUT_TIME_STEPS,1)))
        # input_vector_t100 = K.constant(value=input_at_t100_np[:,1:].reshape((-1,NUMBER_OF_FEATURES * MODEL_INPUT_TIME_STEPS,1)))
        # lam             = K.constant(value=regularization_param_lambda)
        #
        # output_vector_t0   = self.nn_model(input_vector_t0) #vector of calculated SOC
        # output_vector_t100 = self.nn_model(input_vector_t100)
        #
        # energy_computed_at_t0 = self.compute_energy(output_vector_t0, input_vector_t0[:,LABELLED_DATA_WITH_ID_VOLTAGE_INDEX])
        # energy_computed_at_t100 = self.compute_energy(output_vector_t100, input_vector_t100[:,LABELLED_DATA_WITH_ID_VOLTAGE_INDEX])
        #
        # drive_cycle_id_t0_np   = input_at_t0_np[:,LABELLED_DATA_WITH_ID_DRIVE_INDEX]
        # drive_cycle_id_t100_np = input_at_t100_np[:,LABELLED_DATA_WITH_ID_DRIVE_INDEX]
        #
        # energy_difference = energy_computed_at_t100 - energy_computed_at_t0
        # total_loss = self.combined_loss([energy_difference, drive_cycle_id_t0_np, drive_cycle_id_t100_np, lam])
        #
        #
        # self.nn_model.compile(loss=total_loss,
        #                       optimizer=optimizer_value,
        #                       metrics=['mean_absolute_percentage_error'])




        ###################### 2D CNN
        self.nn_model = Sequential()
        self.nn_model.add(Conv2D(filters=5, kernel_size=(3,3), padding='same', input_shape=(MODEL_INPUT_TIME_STEPS, NUMBER_OF_FEATURES, 1)))
        self.nn_model.add(LeakyReLU(alpha=0.3))

        # self.nn_model.add(Dropout(0.3))
        self.nn_model.add(MaxPooling2D((3,3), padding='same'))
        self.nn_model.add(LeakyReLU(alpha=0.2))
        self.nn_model.add(BatchNormalization())

        self.nn_model.add(Conv2D(filters=5, kernel_size=(3,3), padding='same', input_shape=(MODEL_INPUT_TIME_STEPS, NUMBER_OF_FEATURES, 1)))
        self.nn_model.add(LeakyReLU(alpha=0.1))
        self.nn_model.add(BatchNormalization())

        # self.nn_model.add(Dropout(0.3))
        self.nn_model.add(MaxPooling2D((3,3), padding='same'))
        self.nn_model.add(LeakyReLU(alpha=0.5))

        # self.nn_model.add(LeakyReLU(alpha=0.5))
        self.nn_model.add(BatchNormalization())
        self.nn_model.add(Flatten())
        self.nn_model.add(Dense(15))
        # self.nn_model.add(BatchNormalization())
        self.nn_model.add(Dense(1, 'sigmoid'))

        #physics based regularization
        input_vector_t0 = K.constant(value=input_at_t0_np[:,1:].reshape((-1,MODEL_INPUT_TIME_STEPS, NUMBER_OF_FEATURES,1)))
        input_vector_t100 = K.constant(value=input_at_t100_np[:,1:].reshape((-1, MODEL_INPUT_TIME_STEPS, NUMBER_OF_FEATURES,1)))
        lam             = K.constant(value=regularization_param_lambda)

        output_vector_t0   = self.nn_model(input_vector_t0) #vector of calculated SOC
        output_vector_t100 = self.nn_model(input_vector_t100)

        debug1 = input_vector_t0[:,:,LABELLED_DATA_WITH_ID_VOLTAGE_INDEX,:]

        energy_computed_at_t0 = self.compute_energy(output_vector_t0, input_vector_t0[:,:,LABELLED_DATA_WITH_ID_VOLTAGE_INDEX,:])
        energy_computed_at_t100 = self.compute_energy(output_vector_t100, input_vector_t100[:,:,LABELLED_DATA_WITH_ID_VOLTAGE_INDEX,:])

        drive_cycle_id_t0_np   = input_at_t0_np[:,LABELLED_DATA_WITH_ID_DRIVE_INDEX]
        drive_cycle_id_t100_np = input_at_t100_np[:,LABELLED_DATA_WITH_ID_DRIVE_INDEX]

        energy_difference = energy_computed_at_t100 - energy_computed_at_t0
        total_loss = self.combined_loss([energy_difference, drive_cycle_id_t0_np, drive_cycle_id_t100_np, lam])

        self.nn_model.compile(loss=total_loss,
                              optimizer=optimizer_value,
                              metrics=['mean_absolute_percentage_error'])


        return self.nn_model






    def fit_model(self, train_X_np, valid_X_np, train_Y_np, valid_Y_np,num_epochs, patience_value):

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience_value, mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=PATIENCE_NUM_EPOCHS, verbose=1, epsilon=0.01,
                                           mode='min')

        # self.history = self.nn_model.fit(train_X_np, train_Y_np,
        #                       batch_size=batch_size,
        #                       epochs=num_epochs,
        #                       validation_split=validation_rate,
        #                       callbacks=[early_stopping, reduce_lr_loss])

        # train_gen = self.batch_generator_one_dim_cnn(100,train_X_np,train_Y_np,300)
        # valid_gen = self.batch_generator_one_dim_cnn(100, valid_X_np, valid_Y_np,300)

        train_gen = self.batch_generator_two_dim_cnn(MODEL_INPUT_TIME_STEPS,train_X_np,train_Y_np,DATA_GENERATOR_BATCH_SIZE)
        valid_gen = self.batch_generator_two_dim_cnn(MODEL_INPUT_TIME_STEPS, valid_X_np, valid_Y_np,DATA_GENERATOR_BATCH_SIZE)

        # self.history = self.nn_model.fit_generator(generator=train_gen, validation_data=valid_gen,
        #                       epochs=num_epochs,
        #                       callbacks=[early_stopping, reduce_lr_loss])

        self.history = self.nn_model.fit_generator(generator=train_gen, validation_data=valid_gen,
                              epochs=num_epochs,
                               steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=100,
                              callbacks=[early_stopping, reduce_lr_loss],
                              shuffle=False)


        print(self.history.history.keys())

        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        return self.history

    def test_model(self, test_X_np, test_Y_np):
        # test_batch_x, test_batch_y = self.batch_generator_two_dim_cnn(MODEL_INPUT_TIME_STEPS,test_X_np,test_Y_np,DATA_GENERATOR_BATCH_SIZE)
        gen = self.batch_generator_two_dim_cnn(MODEL_INPUT_TIME_STEPS, test_X_np, test_Y_np,
                                                                     DATA_GENERATOR_BATCH_SIZE)

        test_batch_x, test_batch_y = next(gen)

        y_pred = self.nn_model.predict_on_batch(x=test_batch_x)
        print(y_pred)
        print("############################################################################################3")
        print(test_batch_y)

        filtered_pred_np= scipy.signal.medfilt(y_pred, 11)


        plt.plot(y_pred)
        plt.plot(test_batch_y)
        plt.plot(filtered_pred_np)
        plt.show()


    def plot_trainig_history(self):
        # list all data in history
        print(self.history.history.keys())
        # summarize history for accuracy
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()



