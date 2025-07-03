from sktime.datatypes import check_raise, convert_to
import pandas as pd
import pickle
import structlog

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
    metrics = pd.read_csv(mfile)
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
    log.debug(f"Loaded the predictor from #{trained_predictor_filename}")
    return trained_predictor

def save_predictor_to_file(trained_predictor_filename, predictor):
    """Load the predictor from the given files"""
    file = open(trained_predictor_filename, mode="wb")
    pickle.dump(predictor, trained_predictor_filename)
    file.close()
    log.debug(f"Saved the predictor to #{trained_predictor_filename}")
