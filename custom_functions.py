import numpy as np

# ------------------Defining the outlier detectors---------------
def tucky_method(array: np.array, indecies= True) -> np.array:
    """
    This function works with any list-like numerical object
    (don't work with pandas series's) and returns the indexes
    of the found outliers in the array.
    
    :Params: Takes only the series.
    :Returns: S list of the outliers indexes.
    """
    
    Q3 = np.quantile(array, 0.75)
    Q1 = np.quantile(array, 0.25)
    IQR = Q3 - Q1
    
    upper_range = Q3 + (IQR * 1.5)
    lower_range = Q1 - (IQR * 1.5)
    
    outliers = [x for x in array if ((x < lower_range) | (x > upper_range))]
    print(f"Found {len(outliers)} outliers from {len(array)} length series!")
    
    return outliers


def z_score(array, indecies= True) -> np.array:
    """
    This function uses Z-score outlier detection method
    to detect outliers and return them into array.

    :Params: array: list-like numerical object
    :Returns: outliers: np.array (array of the outliers)
    """
    
    std: float = array.std()
    mean: float = array.mean()
    
    upper_limit = mean + (3 * std)
    lower_limit = mean - (3 * std)

    outliers = [value for value in array if 
        (value > upper_limit) | (value < lower_limit)]

    print(f"Found {len(outliers)} outliers from {len(array)} length series!")

    return outliers
# ---------------------------------------------------------------