import keras.optimizers

from Data_Processor import Data_Processor
from Parameters import *
import matplotlib.pyplot as plt
from Neural_Network_Model import *
from keras.optimizers import *
from sklearn import preprocessing


# run_mode = Program_Modes.PROG_MODE_SAVE_RAW_DATA
run_mode = Program_Modes.PROG_MODE_LABEL_DATA
# run_mode = Program_Modes.PROG_MODE_CLEAN_DATA

debug_level = Debug_Levels.DEBUG_LEVEL_ALL


if __name__ == '__main__':

    processor = Data_Processor()

    # if run_mode == Program_Modes.PROG_MODE_SAVE_RAW_DATA:
    #     # read all mat files, delete the lines in which important features contain nan,
    #     # and save data batch by batch
    #     dataset_np_unlabelled = processor.read_all_files(DATASET_ALL_FILES, DRIVE_CYCLE_PATHS, Debug_Levels.DEBUG_LEVEL_ALL)
    #     processor.save_batches(Debug_Levels.DEBUG_LEVEL_ALL, dataset_np_unlabelled)
    #
    #     #go to the next step
    #     run_mode = Program_Modes.PROG_MODE_LABEL_DATA

    if run_mode == Program_Modes.PROG_MODE_LABEL_DATA:
        processor.label_raw_dataset(Debug_Levels.DEBUG_LEVEL_ALL)

        run_mode = Program_Modes.PROG_MODE_CLEAN_DATA

    if run_mode == Program_Modes.PROG_MODE_CLEAN_DATA:
        dataset_labelled_np = processor.read_labelled_dataset_from_csv_file(Debug_Levels.DEBUG_LEVEL_ALL)
        #delete first line (null vector)
        dataset_labelled_np = processor.delete_first_line(dataset_labelled_np)
        dataset_labelled_correct_data_types_np = processor.correct_data_types(dataset_labelled_np)


        # applying low pass filter to eliminate sensors noise
        dataset_labelled_filtered_np = processor.apply_median_filter_to_all_input_features(dataset_labelled_correct_data_types_np,
                                                                                 MEDIAN_FILTER_WINDOW_SIZE)


        # plt.plot(dataset_labelled_filtered_np[:,LABELLED_DATA_WITH_ID_VOLTAGE_INDEX])
        # plt.plot(dataset_labelled_correct_data_types_np[:, LABELLED_DATA_WITH_ID_VOLTAGE_INDEX])
        # plt.show()


        physics_regularization_input_at_t0, \
        physics_regularization_input_at_t100 = processor.generate_physics_regularization_data(PHYSICS_LOSS_DATA_LENGTH,
                                                                                               dataset_labelled_filtered_np,
                                                                                               delta_sample=DELTA_SAMPLES)

        #remove drive id from dataset (drive id is only needed for physics based loss calculation)
        dataset_labelled_clean_np = dataset_labelled_filtered_np[:,1:]

        # go to the next step
        run_mode = Program_Modes.PROG_MODE_SPLIT_CLEAN_DATA

    if run_mode == Program_Modes.PROG_MODE_SPLIT_CLEAN_DATA:
        training_data_labelled_np, \
        test_data_labelled_np = processor.split_data(dataset_labelled_clean_np, TEST_DATA_RATE) #data without id but with label


        #go to the next step
        run_mode = Program_Modes.PROG_MODE_CREATE_AND_COMPILING_MODEL

    if run_mode == Program_Modes.PROG_MODE_CREATE_AND_COMPILING_MODEL:
        physics_guided_nn = Neural_Network_Model()
        opt = keras.optimizers.Adam(learning_rate=0.01)
        nn_model = physics_guided_nn.create_model(physics_regularization_input_at_t0,   #drive cycle included label removed
                                                  physics_regularization_input_at_t100,
                                                  PHYSICS_LAMBDA,
                                                  opt)
        print(nn_model.summary())

        run_mode = Program_Modes.PROG_MODE_DATA_NORMALIZATION

    if run_mode == Program_Modes.PROG_MODE_DATA_NORMALIZATION:

        train_X_np = training_data_labelled_np[:, :-1].reshape((-1,3))
        train_Y_np = training_data_labelled_np[:, -1]

        valid_X_np = training_data_labelled_np[:, :-1].reshape((-1, 3))
        valid_Y_np = training_data_labelled_np[:, -1]

        # scaler = preprocessing.MinMaxScaler().fit(train_X_np)
        scaler = preprocessing.StandardScaler().fit(training_data_labelled_np[:,:-1])
        train_X_np = scaler.transform(train_X_np)
        valid_X_np = scaler.transform(valid_X_np)



        run_mode = Program_Modes.PROG_MODE_TRAIN_MODEL

    if run_mode == Program_Modes.PROG_MODE_TRAIN_MODEL:


        history = physics_guided_nn.fit_model(train_X_np=train_X_np,
                                              valid_X_np=valid_X_np,
                                              train_Y_np=train_Y_np,
                                              valid_Y_np=valid_Y_np,
                                              num_epochs=EPOCHS,
                                              patience_value=PATIENCE_NUM_EPOCHS)


