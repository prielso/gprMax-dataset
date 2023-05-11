import numpy as np
from random import random as rd
import h5py
import re
from scipy import ndimage
import subprocess
from skimage.filters import gaussian
import os
import json


def get_rd3_data(path, file_name, num_samples):
    """num_samples - num of samples per singel Ascan (singel coloumn).
       rd3 is a file containing 16bit samples of GPR Bscan raw data"""
    data = open(path + file_name + '.rd3', 'rb')
    data = np.transpose(np.fromfile(data, dtype=np.int16).reshape(-1, num_samples))
    return data


def get_output_data(filename, rxnumber, rxcomponent):
    """Gets B-scan output data from a synthetic model.
    Args:
        filename (string): Filename (including path) of output file.
        rxnumber (int): Receiver output number.
        rxcomponent (str): Receiver output field/current component.
    Returns:
        outputdata (array): Array of A-scans, i.e. B-scan data.
        dt (float): Temporal resolution of the model.
    """
    # Open output file and read some attributes
    f = h5py.File(filename, 'r')
    nrx = f.attrs['nrx']
    dt = f.attrs['dt']
    # Check there are any receivers
    # if nrx == 0:
    #     raise CmdInputError('No receivers found in {}'.format(filename))
    path = '/rxs/rx' + str(rxnumber) + '/'
    availableoutputs = list(f[path].keys())
    # Check if requested output is in file
    # if rxcomponent not in availableoutputs:
    #     raise CmdInputError('{} output requested to plot, but the available output for receiver 1 is
    #     {}'.format(rxcomponent, ', '.join(availableoutputs)))
    outputdata = f[path + '/' + rxcomponent]
    outputdata = np.array(outputdata)
    f.close()
    return outputdata, dt


def gprmax_inputfile_metadata(path, name):
    """
    this function exracts all this important details from an gprMax input files:
    domain -

    :param path: the input file path
    :param name: the input file name
    :return: domain(The entire scene sizes),
    """
    data = open(path + name + ".in", 'r')
    data = data.read()

    # domain details:
    domain_line = re.findall("#domain:.*", data)[0]
    domain_list = re.findall(r"\d+\.?\d*", domain_line)
    domain_x = float(domain_list[0])
    domain_y = float(domain_list[1])
    resultion = float(domain_list[2])
    domain = [domain_x, domain_y, resultion]
    c = 299792458  # light speed
    time_step = 1 / (c * np.sqrt(2 / (resultion ** 2)))
    # radar details:
    time_window_line = re.findall("#time_window:.*", data)[0]
    time_window = float(re.findall(r"\d+\.?\d*e?[+-]?\d*", time_window_line)[0])
    frequency_line = re.findall("#waveform:.*", data)[0]
    frequency = float(re.findall(r"\d+\.?\d*e?[+-]?\d*", frequency_line)[1])
    tx_start_line = re.findall("#hertzian_dipole:.*", data)[0]
    tx_start_x = float(re.findall(r"\d+\.?\d*e?[+-]?\d*", tx_start_line)[0])
    tx_start_y = float(re.findall(r"\d+\.?\d*e?[+-]?\d*", tx_start_line)[1])
    tx_start_z = float(re.findall(r"\d+\.?\d*e?[+-]?\d*", tx_start_line)[2])
    tx_start = [tx_start_x, tx_start_y, tx_start_z]
    rx_start_line = re.findall("#rx:.*", data)[0]
    rx_start_x = float(re.findall(r"\d+\.?\d*e?[+-]?\d*", rx_start_line)[0])
    rx_start_y = float(re.findall(r"\d+\.?\d*e?[+-]?\d*", rx_start_line)[1])
    rx_start_z = float(re.findall(r"\d+\.?\d*e?[+-]?\d*", rx_start_line)[2])
    rx_start = [rx_start_x, rx_start_y, rx_start_z]
    rxtx_step_line = re.findall("#src_steps:.*", data)[0]
    rxtx_step = float(re.findall(r"\d+\.?\d*e?[+-]?\d*", rxtx_step_line)[0])

    # ground details:
    ground_soil_line = re.findall("#soil_peplinski:.*", data)[0]
    ground_line = re.findall("#fractal_box:.*", data)[0]
    ground_height = float(re.findall(r"\d+\.?\d*e?[+-]?\d*", ground_line)[4])

    # object lines:
    burried_object_lines = re.findall("## square hollow object.*\n.*\n.*\n.*\n.*\n.*\n.*", data)
    objects_list = []
    for i in range(int(len(burried_object_lines) / 2)):
        burried_object_depth = ground_height - float(
            re.findall(r"\d+\.?\d*e?[+-]?\d*", burried_object_lines[2 * i])[10])
        burried_object_x_center = (float(re.findall(r"\d+\.?\d*e?[+-]?\d*", burried_object_lines[2 * i])[3]) + float(
            re.findall(r"\d+\.?\d*e?[+-]?\d*", burried_object_lines[2 * i])[0])) / 2
        burried_object_width = float(re.findall(r"\d+\.?\d*e?[+-]?\d*", burried_object_lines[2 * i])[3]) - float(
            re.findall(r"\d+\.?\d*e?[+-]?\d*", burried_object_lines[2 * i])[0])
        burried_object_height = float(re.findall(r"\d+\.?\d*e?[+-]?\d*", burried_object_lines[2 * i])[10]) - float(
            re.findall(r"\d+\.?\d*e?[+-]?\d*", burried_object_lines[2 * i])[7])
        objects_list.append(
            (burried_object_width, burried_object_height, burried_object_x_center, burried_object_depth))

    return domain, time_step, time_window, frequency, tx_start, rx_start, rxtx_step, ground_height, objects_list


def get_real_metadata(path, file_name):
    # reading the meta data of the recording like: number of vertical samples, horizontal distance, time-window of the
    # signal. meaning for how long we sampled the returning signal.

    meta_data = open(path + file_name + '.rad', 'r+')
    meta_data = meta_data.readlines()
    samples = ''.join(meta_data[0])
    num_samples = int(re.findall("\d+", samples)[0])
    time_window = ''.join(meta_data[18])
    num_time_window = float(re.findall("\d+", time_window)[0])
    distance = ''.join(meta_data[23])
    num_distance = float(re.findall("\d+", distance)[0])
    return num_samples, num_time_window, num_distance


def lpf(data, frequency_bounds):
    data_fft = np.fft.fftshift(np.fft.fft2(data))
    new_data_fft = data_fft
    rows = data_fft.shape[0]
    cols = data_fft.shape[1]
    new_data_fft[:int(rows / 2) - frequency_bounds[0], :] = 0
    new_data_fft[int(rows / 2) + frequency_bounds[0]:, :] = 0
    new_data_fft[:, :int(cols / 2) - frequency_bounds[1]] = 0
    new_data_fft[:, int(cols / 2) + frequency_bounds[1]:] = 0
    filtered_data = np.fft.ifft2(new_data_fft)
    return abs(filtered_data)


def decay_correction(data, std, amp, air_gap):
    gained_data = data
    for i in list(range(air_gap, data.shape[0])):
        gained_data[i, :] = data[i, :] * amp * np.exp(i / std)
    return gained_data


def time_radius_correction(data):
    pixel_cnt = np.zeros(data.shape)
    new_data = np.zeros(data.shape)
    for row in range(data.shape[0]):  # row is the Time Dimension.
        for col in range(data.shape[1]):  # col is the number of the Horizontal location on the surface.
            for side_col in range(row + 1):
                row_in_radius = int(np.sqrt(row ** 2 - side_col ** 2))
                if (col - side_col) in range(data.shape[1]):
                    new_data[row_in_radius, col - side_col] += data[row, col]
                    pixel_cnt[row_in_radius, col - side_col] += 1
                if ((col + side_col) in range(data.shape[1])) and (side_col != 0):
                    new_data[row_in_radius, col + side_col] += data[row, col]
                    pixel_cnt[row_in_radius, col + side_col] += 1
    return new_data, pixel_cnt


def hyperbola_draw(a0, a1, a2, dim):
    drawing = np.zeros(dim)
    drawing_dy = np.zeros(dim)
    for x in range(dim[1]):
        y = a0 + a1 * (x - a2) ** 2
        dy = 2 * a1 * (x - a2)
        if int(y) < dim[0]:
            drawing[int(y), x] = 100
            drawing_dy[int(y), x] = dy
    # drawing = gaussian(drawing, sigma=2)
    return drawing, drawing_dy


def find_surface(data, estimated_surface_row, est_offset, smoothing_size, median_size):
    ground_offset = est_offset
    # eps = 1e3
    data_smoothed = gaussian(data, sigma=[1, smoothing_size])
    dy = ndimage.sobel(data_smoothed, axis=0, mode='constant')
    dx = ndimage.sobel(data_smoothed, axis=1, mode='constant')
    data_high_diff_high_val = abs(dx) + abs(data_smoothed)  # * np.sqrt(abs(dy) / (abs(dx) + eps))
    surface_rows = np.zeros((data.shape[1]))
    for col in range(data.shape[1]):
        ascan = data_high_diff_high_val[:, col]
        max_row = (np.where(ascan == np.amax(
            ascan[(estimated_surface_row - ground_offset):(estimated_surface_row + ground_offset)]))[0][0])
        surface_rows[col] = max_row

        # mean_val = np.mean(abs(data_smoothed[:25, 100*col]))
        # # first_rise = np.all(mean_val < abs(data_smoothed[100:, 100*col]))
        # plt.axvline(x=200, color='g')
        # plt.plot(abs(data_smoothed[:, 100*col]))
        # plt.plot(mean_val*np.ones(data.shape[0]))
        # plt.title(str(col*100) + ' rise:' + str(200))
        # plt.show()
    surface_median = ndimage.median_filter(surface_rows, size=median_size)
    # f, plots = plt.subplots(2, 1)
    # plots[0].plot(surface_rows)
    # plots[1].plot(surface_median)
    # plt.show()
    return surface_median.astype(int)


def create_Bscan(input_fullpaths, num_traces, conda_path, batfile_path):
    """
    To use this function one must have:
    1. gprMax(http://docs.gprmax.com/en/latest/index.html) install and cloned.
    2. Conda or miniconda installed.
    3. gprMax enviroment installed.
    4. The batch file "gprMax_input2bscan" (get it here -> https://drive.google.com/file/d/1jEDKuHtJlFQZTdNwbzOBBHM7fz09zyt_/view?usp=sharing).
    using the open source gprMax(http://docs.gprmax.com/en/latest/index.html) this function create Bscans data for a
    list of geometric scenes, defined by a specific input files for gprMax.
    :param input_fullpaths: A list of full paths for gprMax input files not including the type '.in'.
           e.g. ( ['c:/users/input_file1' , 'c:/users/input_file2']
    :param num_traces: the number of Ascans that make up each Bscan.
    :param conda_path: the path of the installed conda. (e.g. 'C:/Users/admin/Miniconda3/')
    :param batfile_path: the path for batch file "gprMax_input2bscan".
    """
    for idx, input_file in enumerate(input_fullpaths):
        if type(num_traces) == list:
            number_of_ascans = num_traces[idx]
        else:
            number_of_ascans = num_traces
        subprocess.call([batfile_path + 'gprMax_input2bscan.bat', conda_path,
                         input_fullpaths[idx], str(number_of_ascans)], shell=True)


def create_sim_file(path, file_name, simulation_dict):
    """
    This function creates the inputfile (.in) for the gprMax open source software (http://docs.gprmax.com/en/latest/include_readme.html)
    to create the simulation, specifications need to be defined (e.g. the space size, the radar frequency and more).
    :param path: The path for the .in file to be created.
    :param file_name: The name of the .in file to created.
    :param simulation_dict: A python dict object including the simulation details as detailed below:
    1. 'domain' - [domain x size, domain y size, domain z size]. the simulation x,y,z size (e.g. [20, 15, 0.01]).
                the z size is also the voxel resolution. the simulation is a 2D space.
    2. 'time window' - is the time for every radar signal collection (seconds).
    3. 'frequency' - is the centre frequency of the waveform (Hertz).
    4. 'ground height' - is the height of the ground in the y dimension.
    5. 'radar height' - is the height of the radar above the ground.
    6. 'radar steps' - the distance between following A-scans.
    7.  'objects' - is a list in which every element is a dict() of an object to be put in the simulation.
        each object dict will contain the keys:
        'type' - this key will contain the type of object.
        'material' - the material of the objects "walls". inside it's just free space.
        'depth' - the distance between the top part of the box and the ground.
        'center x' - the lower left x coordinate of the box.
        'walls thickness' - the objects walls thickness with the chosen material, inside is free-space.
                            if equal to 'None' no free-space will be put inside the object.
        There are different types of objects, each include it specific keys:
        7.1 Box - type key will contain the string 'box':
            'width' - the x size of the box.
            'height' - the y size of the box.
            'corner radius' - the box corners can be rounded by making this larger than 0.
        7.2 Cylinder - type key will contain the string 'cylinder':
            'radius' - the radius of the cylinder.
    8. 'random seed' - an integer number for seeding the randomness of the ground surface, material and grass(if present).
    ** note: all the dictionaries have to filled with all the needed details above. **
    """
    new_path = path + file_name + '_files/'
    os.makedirs(new_path, exist_ok=True)

    ## saving a json file with all the scene details:
    with open(new_path + file_name + '.json', "w") as write_file:
        json.dump(simulation_dict, write_file, indent=4)

    seed = simulation_dict.get('random seed')
    dim = simulation_dict.get('domain')
    tx_frequency = simulation_dict.get('frequency')
    time_window = simulation_dict.get('time window')
    ground_height = simulation_dict.get('ground height')
    height = simulation_dict.get('radar height')
    txrx_step = simulation_dict.get('radar steps')
    res = dim[2]
    txrx_start = [res * 12, ground_height + height, 0]
    border = 10 * res  # the border of the domain which cannot be calculated
    distance_rx2tx = 2 * res

    file = open(new_path + file_name + '.in', "w")
    file.write("#title: simulation scene for B-scan\n")
    file.write("#domain: {} {} {}\n".format(dim[0], dim[1], res))
    file.write("#dx_dy_dz: {} {} {}\n".format(res, res, res))
    file.write("#time_window: {}\n".format(time_window))

    ## materials for the Scene: used to introduce different materials into the model
    file.write("\n#soil_peplinski: 0.5 0.5 2.0 2.66 0.001 0.01 my_soil\n")
    file.write("#material: 7.42 20.6e-3 1 0 Concrete \n")
    file.write("#add_dispersion_debye: 1 5.42 0.611e-9 Concrete \n")

    ## TX/RX properties:
    file.write("\n#waveform: ricker 1 {} signal\n".format(tx_frequency))
    file.write("#hertzian_dipole: z {} {} {} signal\n".format(txrx_start[0], txrx_start[1], txrx_start[2], ".2f"))
    file.write("#rx: {} {} {}\n".format(txrx_start[0] + distance_rx2tx, txrx_start[1], txrx_start[2]))
    file.write("#src_steps: {} 0 0\n".format(txrx_step))
    file.write("#rx_steps: {} 0 0\n".format(txrx_step))

    ## Creating the Scene: used to build geometric shapes with different constitutive parameters
    file.write(
        "\n#fractal_box: 0 0 0 {} {} {} 2 1.5 0.3 1 50 my_soil ground {}\n".format(dim[0], ground_height, res, seed))
    file.write("#add_surface_roughness: 0 {} 0 {} {} {} 1.5 0.5 0.5 {} {} ground {}\n".format(ground_height, dim[0],
                                                                                              ground_height, res,
                                                                                              ground_height * 0.96,
                                                                                              ground_height, seed))
    file.write(
        "##add_grass: 0 {} 0 {} {} {} 0.5 {} {} 100 ground {}\n".format(ground_height, dim[0], ground_height, res,
                                                                        ground_height * 1.1,
                                                                        ground_height * 1.15, seed))
    # adding objects to the scene:
    insert_obj(file, simulation_dict)

    file.write("\n##geometry_view: 0 0 0 {} {} {} {} {} {} {} n".format(dim[0], dim[1], res, res, res, res,
                                                                        file_name + '_geo'))
    file.close()


def insert_obj(file, simulation_dict):
    """
    (see function create_sim_file)
    This function is writing text in the given file object that will add objects to the simulation scene.
    the objects & scene details will be read from a given python dict.
    :param file: an open file object that is been writen currently by the 'create_sim_file' function
    :param simulation_dict: A python dict object including the simulation details as detailed below:
    1.  'domain' - [domain x size, domain y size, domain z size]. the simulation x,y,z size (e.g. [20, 15, 0.01]).
                the z size is also the voxel resolution. the simulation is a 2D space.
    2.  'time window' - is the time for every radar signal collection (seconds).
    3.  'frequency' - is the centre frequency of the waveform (Hertz).
    4.  'ground height' - is the height of the ground in the y dimension.
    5.  'radar height' - is the height of the radar above the ground.
    6.  'radar steps' - the distance between following A-scans.
    7.  'objects' - is a list in which every element is a dict() of an object to be put in the simulation.
        each object dict will contain the keys:
        'type' - this key will contain the type of object.
        'material' - the material of the objects "walls". inside it's just free space.
        'depth' - the distance between the top part of the box and the ground.
        'center x' - the lower left x coordinate of the box.
        'walls thickness' - the objects walls thickness with the chosen material, inside is free-space.
                            if equal to 'None' no free-space will be put inside the object.
        There are different types of objects, each include it specific keys:
        7.1 Box - type key will contain the string 'box':
            'width' - the x size of the box.
            'height' - the y size of the box.
            'corner radius' - the box corners can be rounded by making this larger than 0.
        7.2 Cylinder - type key will contain the string 'cylinder':
            'radius' - the radius of the cylinder.
    8. 'random seed' - an integer number for seeding the randomness of ground surface, material and grass(if present).
    ** note: all the dictionaries have to filled with all the needed details above. **
    """
    objects_list = simulation_dict.get('objects')
    if objects_list is None:
        return
    res = simulation_dict['domain'][2]
    ground_height = simulation_dict['ground height']
    # inserting the objects in the list:
    for object_dict in objects_list:
        material = object_dict['material']
        depth = object_dict['depth']
        center_x = object_dict['center x']
        walls_wdth = object_dict['walls thickness']
        if object_dict['type'] == 'box':
            # creating a box shaped object with rounded corners
            height = object_dict['height']
            width = object_dict['width']
            lower_left_x = center_x - np.float16(width / 2)
            lower_left_y = ground_height - depth - height
            corner_radius = object_dict['corner radius']
            if corner_radius > np.float16((height - 2 * walls_wdth) / 2) or corner_radius > np.float16(
                    (width - 2 * walls_wdth) / 2):
                print("\n*** Corners Radius is too big! ***\n")
                continue
            if corner_radius == 0:
                file.write("\n## box shape hollow object with rounded corners:\n")
                file.write(
                    f"#box: {lower_left_x} {lower_left_y} 0 {lower_left_x + width} {lower_left_y + height} {res} {material}\n")
                if min(height, width) <= 2 * walls_wdth:
                    continue
                file.write(f"#box: {lower_left_x + walls_wdth} {lower_left_y + walls_wdth} 0 "
                           f"{lower_left_x + width - walls_wdth} {lower_left_y + height - walls_wdth} {res} {'free_space'}\n")
            else:
                file.write("\n## box shape hollow object with rounded corners:\n")
                file.write(f"#box: {lower_left_x} {lower_left_y + corner_radius} 0 "
                           f"{lower_left_x + width} {lower_left_y + height - corner_radius} {res} {material}\n")
                file.write(f"#box: {lower_left_x + corner_radius} {lower_left_y} 0 "
                           f"{lower_left_x + width - corner_radius} {lower_left_y + height} {res} {material}\n")
                # create the rounded corners:
                file.write("#cylinder: {} {} 0 {} {} {} {} {}\n".format(lower_left_x + corner_radius,
                                                                        lower_left_y + corner_radius,
                                                                        lower_left_x + corner_radius,
                                                                        lower_left_y + corner_radius,
                                                                        res, corner_radius, material))
                file.write("#cylinder: {} {} 0 {} {} {} {} {}\n".format(lower_left_x - corner_radius + width,
                                                                        lower_left_y + corner_radius,
                                                                        lower_left_x - corner_radius + width,
                                                                        lower_left_y + corner_radius,
                                                                        res, corner_radius, material))
                file.write("#cylinder: {} {} 0 {} {} {} {} {}\n".format(lower_left_x + corner_radius,
                                                                        lower_left_y - corner_radius + height,
                                                                        lower_left_x + corner_radius,
                                                                        lower_left_y - corner_radius + height,
                                                                        res, corner_radius, material))
                file.write("#cylinder: {} {} 0 {} {} {} {} {}\n".format(lower_left_x - corner_radius + width,
                                                                        lower_left_y - corner_radius + height,
                                                                        lower_left_x - corner_radius + width,
                                                                        lower_left_y - corner_radius + height,
                                                                        res, corner_radius, material))
                if min(height, width) <= 2 * walls_wdth:
                    continue
                ## creating the free-space inside the box:
                file.write("\n## the box's free space:\n")
                file.write(f"#box: {lower_left_x + walls_wdth} {lower_left_y + corner_radius + walls_wdth} 0 "
                           f"{lower_left_x + width - walls_wdth} {lower_left_y + height - corner_radius - walls_wdth} "
                           f"{res} {'free_space'}\n")
                file.write(f"#box: {lower_left_x + corner_radius + walls_wdth} {lower_left_y + walls_wdth} 0 "
                           f"{lower_left_x + width - corner_radius - walls_wdth} {lower_left_y + height - walls_wdth} "
                           f"{res} {'free_space'}\n")
                # create the rounded corners:
                file.write("#cylinder: {} {} 0 {} {} {} {} {}\n".format(lower_left_x + corner_radius + walls_wdth,
                                                                        lower_left_y + corner_radius + walls_wdth,
                                                                        lower_left_x + corner_radius + walls_wdth,
                                                                        lower_left_y + corner_radius + walls_wdth,
                                                                        res, corner_radius, 'free_space'))
                file.write(
                    "#cylinder: {} {} 0 {} {} {} {} {}\n".format(lower_left_x - corner_radius + width - walls_wdth,
                                                                 lower_left_y + corner_radius + walls_wdth,
                                                                 lower_left_x - corner_radius + width - walls_wdth,
                                                                 lower_left_y + corner_radius + walls_wdth,
                                                                 res, corner_radius, 'free_space'))
                file.write("#cylinder: {} {} 0 {} {} {} {} {}\n".format(lower_left_x + corner_radius + walls_wdth,
                                                                        lower_left_y - corner_radius + height - walls_wdth,
                                                                        lower_left_x + corner_radius + walls_wdth,
                                                                        lower_left_y - corner_radius + height - walls_wdth,
                                                                        res, corner_radius, 'free_space'))
                file.write(
                    "#cylinder: {} {} 0 {} {} {} {} {}\n".format(lower_left_x - corner_radius + width - walls_wdth,
                                                                 lower_left_y - corner_radius + height - walls_wdth,
                                                                 lower_left_x - corner_radius + width - walls_wdth,
                                                                 lower_left_y - corner_radius + height - walls_wdth,
                                                                 res, corner_radius, 'free_space'))
        if object_dict['type'] == 'cylinder':
            # creating a box shaped object with rounded corners
            radius = object_dict['radius']
            center_x = object_dict['center x']
            center_y = ground_height - depth - radius
            if radius - walls_wdth <= 0:
                file.write("\n##add cylinder object to the scene:\n")
                file.write(f"#cylinder: {center_x} {center_y} 0 {center_x} {center_y} {res} {radius} {material}\n")
            else:
                file.write("\n##add cylinder object to the scene:\n")
                file.write(f"#cylinder: {center_x} {center_y} 0 {center_x} {center_y} {res} {radius} {material}\n")
                file.write(
                    f"#cylinder: {center_x} {center_y} 0 {center_x} {center_y} {res} {radius - walls_wdth} {'free_space'}\n")
