import scipy.signal
from mat4py import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.signal import medfilt

from Parameters import *
import random
import matplotlib.pyplot as plt
import copy


#TODO: add sample index adn drive cycle ID to data



class Data_Processor:

    def __init__(self):
        pass

    def read_mat_data_file(self, path_str, drive_cycle_id, chamber_temperature):

        mat_file_dict = loadmat(path_str)['meas']
        keys = mat_file_dict.keys()
        labels_list = list(keys)

        # we extract all the columns from mat_file_dict dictionary
        time_stamp_np           = np.array(mat_file_dict['TimeStamp'])
        voltage_v_np            = np.array(mat_file_dict['Voltage'])
        current_ampere_np       = np.array(mat_file_dict['Current'])
        capacity_ah_np          = np.array(mat_file_dict['Ah'])
        energy_wh_np            = np.array(mat_file_dict['Wh'])
        power_w_np              = np.array(mat_file_dict['Power'])
        battery_temp_degc_np    = np.array(mat_file_dict['Battery_Temp_degC'])
        time_np                 = np.array(mat_file_dict['Time'])
        chamber_temp_degc_np    = np.array(mat_file_dict['Chamber_Temp_degC'])

        number_of_samples = chamber_temp_degc_np.shape[0]
        drive_cycle_id_np = np.zeros((number_of_samples,1))
        chamber_temperature_np = np.zeros((number_of_samples, 1))

        for data_point_idx in range(0,drive_cycle_id_np.shape[0]):
            drive_cycle_id_np[data_point_idx] = drive_cycle_id
            chamber_temperature_np[data_point_idx] = chamber_temperature


        # plt.plot(current_ampere_np)
        # plt.plot(capacity_ah_np)
        # plt.show()

        # raw_data_np = np.concatenate((drive_cycle_id_np,
        #                             voltage_v_np,
        #                             current_ampere_np,
        #                             capacity_ah_np,
        #                             energy_wh_np,
        #                             power_w_np,
        #                             battery_temp_degc_np),
        #                             axis=1)

        raw_data_np = np.concatenate((drive_cycle_id_np,
                                    chamber_temperature_np,
                                    voltage_v_np,
                                    current_ampere_np,
                                    battery_temp_degc_np,
                                    capacity_ah_np),
                                    axis=1)

        return labels_list, raw_data_np

    def read_all_files(self, all_data_files_array, all_data_paths_array, debug_level):

        labels = None
        dataset_np = np.zeros((1, 6))
        drive_cycle_index = 0

        for name in all_data_files_array:
            for path in all_data_paths_array:
                if os.path.exists(path + '/' + name):
                    labels, drive_cycle_data_np = self.read_mat_data_file(path + '/' + name,
                                                                          drive_cycle_index,
                                                                          CHAMBER_TEMPERATURE[drive_cycle_index])
                    dataset_np = np.concatenate((dataset_np, drive_cycle_data_np))
                    drive_cycle_index += 1
                    print(path + '/' + name)

                    if debug_level == Debug_Levels.DEBUG_LEVEL_ALL:
                        print("Reading files ... "+ str(dataset_np.shape[0]) +" lines are read")

                    if debug_level == Debug_Levels.DEBUG_LEVEL_ALL:
                        print('Reading  drive cycle ID =  '+str(drive_cycle_index))
                else:
                    print(path + '/' + name + "does not exist")

        if debug_level == Debug_Levels.DEBUG_LEVEL_ALL:
            print('All data are read from .mat files')


        return dataset_np

    def save_batches(self, debug_level, dataset_np):
        # this version includes chamber temperature as feature
        batch_size = 100
        saved_dataset_np = np.zeros((0, 6))
        line_counter = 0
        number_of_lines = dataset_np.shape[0]
        reached_line = 0

        # read the last saved data if it exists
        if os.path.exists('clean_drive_cycles_dataset_unlabelled.csv'):
            # clean_dataset_np = np.genfromtxt('clean_drive_cycles_dataset.csv', delimiter=',')
            try:
                saved_dataset_np = np.loadtxt("clean_drive_cycles_dataset_unlabelled.csv", delimiter=',', dtype=str)
                saved_dataset_np = saved_dataset_np.reshape((-1, 6))
                reached_line = saved_dataset_np.shape[0]
            except UserWarning:
                print("File exists but empty, ignored ...")

        for line_index in range(reached_line, number_of_lines):
            # if  (self.raw_data_np[line_index][0] != 'nan') and (self.raw_data_np[line_index][1] != 'nan') and \
            #     (self.raw_data_np[line_index][3] != 'nan') and (self.raw_data_np[line_index][4] != 'nan') and \
            #     (self.raw_data_np[line_index][5] != 'nan') and (self.raw_data_np[line_index][6] != 'nan'):

            line_counter += 1
            line = np.array(dataset_np[line_index])
            line = line.reshape((1, 6))
            saved_dataset_np = np.concatenate((saved_dataset_np, line))

            if line_index % batch_size == 0:
                # save the batch ever 100 lines
                np.savetxt("clean_drive_cycles_dataset_unlabelled.csv", saved_dataset_np, fmt="%s", delimiter=",")
                if debug_level == Debug_Levels.DEBUG_LEVEL_ALL:
                    print('Batch saved ! total number of data lines is ' + str(saved_dataset_np.shape[0]))

            if debug_level == Debug_Levels.DEBUG_LEVEL_ALL:
                print('number of  lines = ' + str(line_counter))

    def read_labelled_dataset_from_csv_file(self, debug_level):
        labelled_dataset_np = np.loadtxt("clean_drive_cycles_dataset_labelled.csv", delimiter=',', dtype=str)
        if debug_level == Debug_Levels.DEBUG_LEVEL_ALL:
            print("Loaded Clean Dataset Shape = " + str(labelled_dataset_np))
        return labelled_dataset_np

    def read_unlabelled_dataset_from_csv_file(self, debug_level):
        unlabelled_dataset_np = np.loadtxt("clean_drive_cycles_dataset_unlabelled.csv", delimiter=',', dtype=str)
        if debug_level == Debug_Levels.DEBUG_LEVEL_ALL:
            print("Loaded Clean Dataset Shape = " + str(unlabelled_dataset_np))
        return unlabelled_dataset_np

    def label_raw_dataset(self, debug_level):
        unlabelled_dataset_np = self.read_unlabelled_dataset_from_csv_file(Debug_Levels.DEBUG_LEVEL_ALL)
        labelled_dataset_np = unlabelled_dataset_np

        #change the last column with the SOC value
        for line_idx in range(0, unlabelled_dataset_np.shape[0]):
            if float(labelled_dataset_np[line_idx][-1]) <= 0:
                labelled_dataset_np[line_idx][-1] = (RATED_CAPACITY + float(labelled_dataset_np[line_idx][-1]))/RATED_CAPACITY
            else:
                labelled_dataset_np[line_idx][-1] = (RATED_CAPACITY - float(labelled_dataset_np[line_idx][-1]))/RATED_CAPACITY
            if debug_level == Debug_Levels.DEBUG_LEVEL_ALL:
                print(str(line_idx) + ' are labelled !')


        np.savetxt("clean_drive_cycles_dataset_labelled.csv", labelled_dataset_np, fmt="%s", delimiter=",")
        if debug_level == Debug_Levels.DEBUG_LEVEL_ALL:
            print('Labelled dateset is saved ! total number of data lines is ' + str(labelled_dataset_np.shape[0]))

    def split_data(self, clean_dataset_np, test_rate):

        #TODO: change this function , this function will split the data to only training and test

        # training and validation rates should be less or equal 1
        assert (test_rate < 0.5), "Invalid test rate"


        number_of_samples_all = clean_dataset_np.shape[0]
        number_of_samples_test = int(number_of_samples_all * test_rate)
        number_of_samples_training = int(number_of_samples_all-number_of_samples_test)

        training_data_np = clean_dataset_np[:number_of_samples_training]
        test_data_np    = clean_dataset_np[number_of_samples_training:]

        return training_data_np, test_data_np

    def smart_data_split(self, clean_data_np, validation_rate, test_rate, time_steps):
        #TODO: implement this method

        number_of_samples_all           = clean_data_np.shape[0]
        number_of_samples_test          = int(number_of_samples_all * test_rate)
        number_of_samples_validation    = int(number_of_samples_all*validation_rate)
        number_of_samples_training      = int(number_of_samples_all - number_of_samples_test-number_of_samples_validation)

        if number_of_samples_training % time_steps != 0:
            number_of_samples_training -= number_of_samples_training % time_steps

        if number_of_samples_validation % time_steps != 0:
            number_of_samples_validation -= number_of_samples_validation % time_steps

        if number_of_samples_test % time_steps != 0:
            number_of_samples_test -= number_of_samples_test % time_steps


        training_data_np        = np.zeros((number_of_samples_training,clean_data_np.shape[1]))
        validation_data_np      = np.zeros((number_of_samples_validation, clean_data_np.shape[1]))
        test_data_np            = np.zeros((number_of_samples_test, clean_data_np.shape[1]))

        training_index      = 0
        validation_index    = 0
        test_index          = 0

        for samples_index in range(0,number_of_samples_all, time_steps):
            if training_index < number_of_samples_training:
                training_data_np[samples_index:samples_index+time_steps,:] = clean_data_np[samples_index:samples_index+time_steps, :]
                training_index += time_steps
            if validation_index < number_of_samples_validation:
                validation_data_np[samples_index:samples_index+time_steps,:] = clean_data_np[samples_index+time_steps:samples_index + time_steps*2, :]
                validation_index += time_steps
            if test_index < number_of_samples_test:
                test_data_np[samples_index:samples_index + time_steps, :] = clean_data_np[samples_index + time_steps*2:samples_index + time_steps *3, :]
                test_index += time_steps

        return training_data_np, validation_data_np, test_data_np


    # def label_data(self):
    #      #add soc to clean dataset


    def batch_generator_features_vector(self,data_x, data_y, batch_size, number_of_features):

        x_batch = np.zeros((batch_size, number_of_features, 1))
        y_batch = np.zeros((batch_size, 1))
        number_of_samples = x_batch.shape[0]

        # while (True):
        #
        #     for batch_index in range(batch_size):
        #
        #         feature_vector_index = random.randint(0, number_of_samples)
        #
        #         x_batch[batch_index] = data_x[feature_vector_index].reshape(number_of_features, 1)
        #         y_batch[batch_index] = data_y[feature_vector_index]
        #         start_idx += time_window_size
        #         if (start_idx >= labelled_data_x.shape[0] - time_window_size):
        #             start_idx = 0
        #
        #     yield x_batch, y_batch

    def delete_first_line(self, data_np):
        return  data_np[1:-1]

    def correct_data_types(self, dataset_np):
        # dataset_corrected_np = dataset_np
        # for line_index in range(0, dataset_np.shape[0]):
        #     for feature_index in range(0, NUMBER_OF_FEATURES+1+1): # number_of_features + drive cycle id + label
        #         dataset_corrected_np[line_index][feature_index] = float(dataset_np[line_index][feature_index])
        dataset_corrected_np =dataset_np.astype(np.float)
        return dataset_corrected_np

    def generate_physics_regularization_data(self, number_of_rows, dataset_np, delta_sample):
        start_index = random.randint(0, dataset_np.shape[0] - number_of_rows - delta_sample)
        physics_regularization_input_at_t0 = dataset_np[start_index:start_index + number_of_rows, :-1]# -2 --> remove the label
        physics_regularization_input_at_t100 = dataset_np[start_index + delta_sample:start_index + delta_sample + number_of_rows, :-1]
        return physics_regularization_input_at_t0, physics_regularization_input_at_t100


    def apply_median_filter_to_all_input_features(self, data_np, window_length):
        filtered_data_np = copy.copy(data_np)

        filtered_data_np[:,LABELLED_DATA_WITH_ID_VOLTAGE_INDEX] =\
            scipy.signal.medfilt(filtered_data_np[:,LABELLED_DATA_WITH_ID_VOLTAGE_INDEX], window_length)

        filtered_data_np[:,LABELLED_DATA_WITH_ID_CURRENT_INDEX] =\
            scipy.signal.medfilt(filtered_data_np[:,LABELLED_DATA_WITH_ID_CURRENT_INDEX], window_length)

        filtered_data_np[:,LABELLED_DATA_WITH_ID_TEMPERATURE_INDEX] =\
            scipy.signal.medfilt(filtered_data_np[:,LABELLED_DATA_WITH_ID_TEMPERATURE_INDEX], window_length)

        return filtered_data_np

    def shuffle_data(self, data_np, time_steps, shuffler_iterations):
        number_of_rows = data_np.shape[0]
        number_of_columns = data_np.shape[1]
        assert (number_of_rows % time_steps == 0), "The number of rows should be divisible by the time steps"
        number_of_steps = int(number_of_rows / time_steps)
        shuffled_data_np = np.zeros((number_of_rows, number_of_columns))

        for shuffle_index in range(shuffler_iterations):
            print("shuffle index = " + str(shuffle_index))
            for step_index in range(0,number_of_rows, time_steps):
                batch_start_index = random.randint(0, number_of_rows - time_steps)
                batch = data_np[batch_start_index:batch_start_index+time_steps, :]
                shuffled_data_np[step_index:step_index+time_steps,:] = batch






        return shuffled_data_np













