import numpy as np
import scipy as sp

def SNR_bin(data, min_SNR):
    '''A functions that iterates over the data, binning the spatial pixels annularly 
    until a minimum SNR (min_SNR) is reached. 'data' must be in the format returned by
    prepare_data function. '''

    # Find how many points are in each spectrum.
    spec_len = len(data[0]['spec'])
    # Find how many data points there are in total (i.e. number of pixels).
    num_points = len(data)

    # Create empty output.
    binned_data = []

    i = 0
    # Iterate through all data points binning up.
    while i < num_points:

        # After a set of data is made into a single bin, start again.
        SNR = 0.
        rad = 0.
        spec = np.zeros(spec_len)
        err_sq = np.zeros(spec_len)
        I_mod = 0
        S_mod = 0
        j = 0

        while SNR < min_SNR:

            # Break out if it reaches end of data whilst in loop.
            if i+j == num_points:
                break
            
            # Add signal linearly, error in quadrature.
            spec += data[i+j]['spec']
            err_sq += (data[i+j]['spec_err'])**2
            
            # Also sum the radii and theoretical values for each point.
            rad += data[i+j]['rad']
            I_mod += data[i+j]['I_mod']
            S_mod += data[i+j]['S_mod']

            # Calculate SNR after adding latest data point.
            SNR = np.sum(spec / np.sqrt(err_sq)) / spec_len

            j += 1

        err = np.sqrt(err_sq)

        # Take average over the bin for radius and dispersion
        rad /= j
        S_mod /= j

        # Make the data for each annulus into a new dictionary.
        data_dict = {}
        data_dict['rad'] = rad
        data_dict['spec'] = spec
        data_dict['spec_err'] = err
        data_dict['SNR'] = SNR
        data_dict['I_mod'] = I_mod
        data_dict['S_mod'] = S_mod

        # j is the number of pixels in the annulus
        data_dict['j'] = j

        # Append onto the list of binned data points.
        binned_data.append(data_dict)

        i += j

    # The final data point does not reach the min SNR.
    del binned_data[-1]

    print 'Binning complete'

    return binned_data

def reduce(data, num):
    '''A function that reduces the input data set to a smaller number of points,
    that are sampled evenly in radius.
        Input:
            num : the number of points of the output data set.
    '''

    max_rad = data[-1]['rad']
    rad_array = np.linspace(0, max_rad, num=num)

    # Make a list of the radii of the data points.
    rad_list = []
    for i in range(len(data)):
        rad_list.append(data[i]['rad'])

    reduced_data = []

    for rad in rad_array:
        j = next(i for i,v in enumerate(rad_list) if v >= rad)
        reduced_data.append(data[j])
        reduced_data[-1]['j'] = 0

    return reduced_data