from sktime.datatypes import check_raise, convert_to
import pandas as pd
import pickle, os, glob
import structlog
import os
from datetime import datetime

log = structlog.get_logger()

def load_individual_instance(filename, needed_columns):
    df = pd.read_csv(filename)
    for col in needed_columns:
        if not (col in df.columns):
            df[col] = 0.0
            # Ensure all the columns are in the correct order!
    return df[needed_columns]

def read_data(results_directory, mfile):
    data_files = list(map(os.path.basename, sorted(glob.glob(results_directory + "/*Test*"))))
    metrics_orig = pd.read_csv(mfile)
    # Ensure NaN column is removed
    metrics = metrics_orig.dropna(axis=1, how="all")
    return data_files, metrics

def create_combined_data(base_dir, filenames, needed_columns):
    combined_data_m = map(lambda file: load_individual_instance(base_dir + "/" + file, needed_columns), filenames)
    combined_data = list(combined_data_m)
    print("Check data: ",check_raise(combined_data, mtype="df-list"))
    return combined_data

def load_predictor_from_file(trained_predictor_filename):
    """Load the predictor from the given files"""
    file = open(trained_predictor_filename, mode="rb")
    trained_predictor = pickle.load(file)
    file.close()
    log.debug(f"Loaded the predictor from " + str(trained_predictor_filename))
    return trained_predictor

def save_predictor_to_file(trained_predictor_filename, predictor):
    """Load the predictor from the given files"""
    file = open(trained_predictor_filename, mode="wb")
    pickle.dump(predictor, file)
    file.close()
    log.debug(f"Saved the predictor to " + str(trained_predictor_filename))

def create_directory_for_results(directory_name, append_date=True, base_directory="results"):
    # Get current date and time
    current_time = datetime.now()

    # Format directory_name with YYYYMMDD_HHMMSS string appended
    if append_date:
        target_directory_name = directory_name + "_" + current_time.strftime("%Y_%m_%d_%H_%M_%S")
    else:
        target_directory_name = directory_name

    # Define the path for the new directory (e.g., in 'logs' folder)
    base_path = os.path.join(os.getcwd(), base_directory)
    new_directory = os.path.join(base_path, target_directory_name)

    try:
        # Create the directory with parents if needed
        os.makedirs(new_directory, exist_ok=True)
        log.info(f"Directory created successfully: {new_directory}")
        return new_directory
    except OSError as e:
        log.info(f"Error creating directory: {e}")
        raise e
