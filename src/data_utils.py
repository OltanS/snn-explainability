import numpy as np

def combine_polarities(frames):
    # frames are of shape time_frame, polarity, height, width
    # we can concat both polarities (there might be some overlap, but that's ok!)
    # negative polarities are turned into negative numbers
    polarity_zero = -frames[:, 0, :, :]
    polarity_one = frames[:, 1, :, :]
    # Take the value with the largest magnitude
    mask = np.abs(polarity_zero) > np.abs(polarity_one)
    combined = np.where(mask, polarity_zero, polarity_one)
    
    # print(combined.shape)
    return combined
