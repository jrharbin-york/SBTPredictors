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
             and f the respecti
