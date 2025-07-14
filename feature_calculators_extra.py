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
    Splits the series into windows and returns the max value of each as a feature

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
