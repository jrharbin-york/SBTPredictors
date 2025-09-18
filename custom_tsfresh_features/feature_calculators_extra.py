
@set_property("fctype", "combiner")
def jrh_null_feature(x, param):
    features = [(None, 0.0)]
    #return [(convert_to_output_format(config), features) for config in param]
    return features

def get_numbered_features_for_window_range(numpy_s, window_starts, window_length, feature_count_start):
    features = []
    feature_count = feature_count_start 
    
    for ws in window_starts:
        if True:
            window_end = ws + window_length
            window = numpy_s[ws:window_end]
            feature = np.max(window)
            feature_tag = feature_count
            features.append((feature_tag, feature))
            feature_count += 1

    return features

@set_property("fctype", "combiner")
def jrh_windowed_features_calculation_fixedsize(x, param):
    """
    Short description of your feature (should be a one liner as we parse the first line of the description)

    Long detailed description, add somme equations, add some references, what kind of statistics is the feature
    capturing? When should you use it? When not?

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param c: the time series name
    :type c: str
    :param param: contains dictionaries {"p1": x, "p2": y, ...} with p1 float, p2 int ...
    :type param: list
    :return: list of tuples (s, f) where s are the parameters, serialized as a string,
             and f the respective feature value as bool, int or float
    :return type: pandas.Series
    """
    # Do some pre-processing if needed for all parameters
    # f is a function that calculates the feature value for each single parameter combination

    DEBUG_PRINT = False

    numpy_s = x

    if DEBUG_PRINT:
        print("param=", str(param))
        print("param len=", str(len(param)))

    param_fixed = param[0]

    # Only using a fixed size here from first element
    window_size_secs = param_fixed["windowsize"]
    resolution_samples_per_second = param_fixed["resolution_samples_per_second"]

    series_length_samples = len(x)
    series_length_time = series_length_samples / resolution_samples_per_second

    if DEBUG_PRINT:
        print("windowsize = %.2f, resolution_samples_per_second=%f" % (window_size_secs, resolution_samples_per_second))
        print("series_length_samples = %u, series_length_time=%f" % (series_length_samples, series_length_time))

    feature_count_base = 0
    features = []

    window_length = int(series_length_samples * window_size_secs / series_length_time)
    window_count = int(series_length_time / window_size_secs)

    if DEBUG_PRINT:
        print("window_length=", str(window_length))

    window_starts_f = np.linspace(0, (series_length_samples-window_length), window_count)
    window_starts = list(map(lambda n: int(n), window_starts_f))

    w_features = get_numbered_features_for_window_range(numpy_s, window_starts, window_length, feature_count_base)
    features = features + w_features
    feature_count_base += len(w_features)
    return features

@set_property("fctype", "combiner")
def jrh_windowed_features_calculation(x, param):
    """
    Short description of your feature (should be a one liner as we parse the first line of the description)

    Long detailed description, add somme equations, add some references, what kind of statistics is the feature
    capturing? When should you use it? When not?

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :param c: the time series name
    :type c: str
    :param param: contains dictionaries {"p1": x, "p2": y, ...} with p1 float, p2 int ...
    :type param: list
    :return: list of tuples (s, f) where s are the parameters, serialized as a string,
             and f the respective feature value as bool, int or float
    :return type: pandas.Series
    """
    numpy_s = x

    feature_count_base = 0
    features = []

    all_window_counts = [100,50,25,5]
  
    
    for window_count in all_window_counts:
        # convert floats to int
        #print("Producing window_count = " + str(window_count))

        #print("window_starts = " + str(window_starts))
        #print("window_length = " + str(window_length))

        window_length = int(series_length_samples / window_count)
        # Need to subtract window_length since we are doing starts
        window_starts_f = np.linspace(0, (series_length_samples-window_length), window_count)
        window_starts = list(map(lambda n: int(n), window_starts_f))
        
        w_features = get_numbered_features_for_window_range(numpy_s, window_starts, window_length, feature_count_base)
        features = features + w_features
        feature_count_base += len(w_features)
        
    return features

