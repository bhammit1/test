from classes import *

"""
/*******************************************************************
Combination of all functions used to process and analyze NDS CF Data,
FHWA Work Zone Project Data, and Data from Sky Sample.

Functions are separated in the following categories:
1. Basic Functions
2. Car Following - Specific Functions
3. Genetic Algorithm Functions

This script is intended to be a reference file for all functions, which
can be pulled out and used in "main" files - as needed.

Author: Britton Hammit
E-mail: bhammit1@gmail.com
********************************************************************/
"""

"""
Basic Functions
"""

def import_trip_ids(path,filename):
    trip_num = list()
    file = open(os.path.join(path, filename), 'r')
    for line in file:
        trip_num.append(line.strip().split(','))
    file.close()
    return trip_num


def import_nds_stac(stac_file_name, path):
    stac_file = open(os.path.join(path,stac_file_name), 'r')  # Open NDSfile
    stac_data = np.genfromtxt(stac_file, delimiter=',',skip_header=1,missing_values=np.nan)  # Import data such that blank entries are given 'nan' values
    stac_file.close()

    return stac_data


def import_wy_nds(nds_file_name,path):
    # Import NDS Data as DATA POINTS
    nds_file = open(os.path.join(path,nds_file_name), 'r')  # Open NDSfile
    nds_data = np.genfromtxt(nds_file, delimiter=',',skip_header=1,missing_values=np.nan)  # Import data such that blank entries are given 'nan' values
    nds_file.close()

    data_points = list()
    for i in range(len(nds_data)):
        data_points.append(DataPoint(nds_data[i]))
    del nds_data
    return PointCollection(data_points)


def import_wy_nds_stac(nds_file_name,nds_path,stac_file_name,stac_path):

    def time_interpollation(time,value,nds_time):
        new_value = [np.nan for i in range(len(nds_time))]

        j = 0
        for i in range(len(nds_time)):
            while time[j]<nds_time[i] and j<len(time)-1:
                if time[j]<nds_time[i] and time[j+1]>nds_time[i]:
                    new_value[i] = (value[j+1]-value[j])/(time[j+1]-time[j])*(nds_time[i]-time[j])+value[j]
                j+=1

        # Get rid of non values by averaging existing values
        for i in range(len(new_value)-2):
            if np.isfinite(new_value[i+1]) == False:
                new_value[i+1] = np.mean([new_value[i],new_value[i+2]])
        return new_value

    nds_datapoints = import_wy_nds(nds_file_name,nds_path)  # Point Collection!
    stac_data = import_nds_stac(stac_file_name, stac_path)

    new_time_nds = list()
    for i in range(nds_datapoints.point_count()):
        new_time_nds.append(nds_datapoints[i].vtti_time_stamp)

    time_stac = list()
    for i in range(len(stac_data)):
        time_stac.append(stac_data[i][0])

    value_stac_dV = list()
    for i in range(len(stac_data)):
        value_stac_dV.append(stac_data[i][6])

    # Separate stac based on track
    track_list = [1,2,3,4,5,6,7,8]
    track_value_list_stac = [list() for i in range(len(track_list))]
    track_time_list_stac = [list() for i in range(len(track_list))]

    for i in range(len(stac_data)):
        for j in range(len(track_list)):
            if track_list[j]==stac_data[i][1]:
                track_value_list_stac [j].append(stac_data[i][10])
                track_time_list_stac[j].append(stac_data[i][0])

    track_value_new_list = [list() for i in range(len(track_list))]
    for i in range(len(track_list)):
        if len(track_time_list_stac[i]) != 0:  # Accounting for tracks with no lead vehicles
            track_value_new_list[i] = time_interpollation(track_time_list_stac[i],track_value_list_stac[i],new_time_nds)

    for i in range(len(track_value_new_list)):
        if len(track_value_new_list[i]) != 0:  # Accounting for tracks with no lead vehicles
            if i+1 == 1:
                for j in range(nds_datapoints.point_count()):
                    nds_datapoints[j].track1_x_vel_processed = track_value_new_list[i][j]
            elif i+1 == 2:
                for j in range(nds_datapoints.point_count()):
                    nds_datapoints[j].track2_x_vel_processed = track_value_new_list[i][j]
            elif i+1 == 3:
                for j in range(nds_datapoints.point_count()):
                    nds_datapoints[j].track3_x_vel_processed = track_value_new_list[i][j]
            elif i+1 == 4:
                for j in range(nds_datapoints.point_count()):
                    nds_datapoints[j].track4_x_vel_processed = track_value_new_list[i][j]
            elif i+1 == 5:
                for j in range(nds_datapoints.point_count()):
                    nds_datapoints[j].track5_x_vel_processed = track_value_new_list[i][j]
            elif i+1 == 6:
                for j in range(nds_datapoints.point_count()):
                    nds_datapoints[j].track6_x_vel_processed = track_value_new_list[i][j]
            elif i+1 == 7:
                for j in range(nds_datapoints.point_count()):
                    nds_datapoints[j].track7_x_vel_processed = track_value_new_list[i][j]
            elif i+1 == 8:
                for j in range(nds_datapoints.point_count()):
                    nds_datapoints[j].track8_x_vel_processed = track_value_new_list[i][j]

    return nds_datapoints


def compile_wy_nds_stac_CF_events(nds_file_name,nds_path,stac_file_name,stac_path,save_path,timestamp):
    # Output is Processed Data Class - Combining all of the CF instanes in one file to one dataset

    processed_data_output_filename = '{}_nds_processed_data.csv'.format(timestamp)
    processed_data_all = ProcessedData('blank',list())

    point_collection_1 = import_wy_nds_stac(nds_file_name,nds_path,stac_file_name,stac_path)
    list_of_collections = generate_car_following_collections(point_collection_1,min_cf_time=40,max_cf_dist=60)
    processed_data_combined = processed_data(list_of_collections)

    processed_data_all.merge(processed_data_combined)
    #processed_data_all.plot_dV_X_original()
    processed_data_all.original_export(processed_data_output_filename,save_path)

    return processed_data_all


def import_stac_radardata(file_name,path):
    # Imports STAC Radar Data
    file = open(os.path.join(path,file_name), 'r')  # Open Vehicle Data file
    for i in range(1):  # Number of header lines
        file.next()  # Header Line
    data = []
    for line in file:
        data.append(line.strip().split(','))
    file.close()

    # Convert data from Float to Integer
    for i in range(len(data)):
        for j in range(len(data[i])):
            try:
                data[i][j] = float(data[i][j])
            except ValueError:
                data[i][j] = np.nan

    return data


def import_fhwa_irv(irv_file_name,path):
    # Imports FHWA IRV data - to be directly entered in as "Processed Data Class"
    file = open(os.path.join(path,irv_file_name), 'r')  # Open Vehicle Data file
    for i in range(14):  # Number of header lines
        file.next()  # Header Line
    data = []
    for line in file:
        data.append(line.strip().split(','))
    file.close()

    # Convert data from Float to Integer
    for i in range(len(data)):
        for j in range(len(data[i])):
            try:
                data[i][j] = float(data[i][j])
            except ValueError:
                data[i][j] = np.nan

    return data


def import_synthetic_data(file_name,path):
    # Imports generated synthetic data - to be directly entered in as "Processed Data Class"
    file = open(os.path.join(path,file_name), 'r')  # Open Vehicle Data file
    for i in range(4):  # Number of header lines
        file.next()  # Header Line
    data = []
    for line in file:
        data.append(line.strip().split(','))
    file.close()

    # Convert data from Float to Integer
    for i in range(len(data)):
        for j in range(len(data[i])):
            try:
                data[i][j] = float(data[i][j])
            except ValueError:
                data[i][j] = np.nan
    return data


def initialize_save_path(save_path):
    # If save_path doesn't already exist, it will be created
    if not os.path.exists(save_path):
        os.makedirs(save_path)


def import_variable_preferences(path,variable_file):
    # Import variable preferences
    file = open(os.path.join(path,variable_file), 'r')  # Open Variable Preference file
    file.next()  # Header Line
    temp = []
    for line in file:
        temp.append(line.strip().split(','))
    file.close()
    variable_index = [[0 for i in range(len(temp[0]))] for i in range(len(temp))]
    for i in range(len(temp)):
        variable_index[i] = [temp[i][0], int(float(temp[i][1])), []]  # Array with variables of interest
    return variable_index


def check_files_exist(trip_numbers,open_nds_data_path):
    # Check validity of each trip number before starting analysis
    file_error_count = 0
    for i in range(len(trip_numbers)):
        try:
            file = open(os.path.join(open_nds_data_path, 'Event_ID_{}.csv'.format(trip_numbers[i][0])), 'r')
            file.close()
        except IOError:
            print "ERROR: Event_ID_{}".format(trip_numbers[i][0])
            #log_file.write("ERROR: Event_ID_{}".format(trip_numbers[i][0]))
            #log_file.write("\n")
            file_error_count += 1
    if file_error_count > 0:  # causes the program to come to a stop IF one trip number is invalid
        #log_file.write("System Error Raised")
        raise SystemError  # Stop running script if error detected.


def import_cf_data(input_source,input_filename,input_path,log_path=None,timestamp=None,stac_filename=None,stac_path=None):
    if input_source == 'fhwa_irv':
        data = import_fhwa_irv(input_filename,input_path)
        processed_data = ProcessedData(input_source,data)
    elif input_source == 'wy_nds':
        data = import_wy_nds(input_filename,input_path)
        # todo STAC
    elif input_source == 'wy_nds_stac':
        processed_data = compile_wy_nds_stac_CF_events(input_filename,input_path,stac_filename,stac_path,log_path,timestamp)
    elif input_source == 'synthetic':
        data = import_synthetic_data(input_filename,input_path)
        processed_data = ProcessedData(input_source,data)

    return processed_data


"""
Car Following Functions
"""

def generate_collection_from_timestamps(collection_initial,start,stop):
    """
    :param collection_initial: Original PointCollection datapoint
    :param start: starting VTTI time stamp value
    :param stop: stopping VTTI time stamp value
    :return: PointCollection with DataPoints between the start and stop time stamp values
    """
    generated_collection = PointCollection()
    for i in range(collection_initial.point_count()):
        if collection_initial[i].vtti_time_stamp >= start and collection_initial[i].vtti_time_stamp <= stop:
            generated_collection.point_append(collection_initial[i])
    return generated_collection


def generate_car_following_collections(collection_initial, min_cf_time=60, max_cf_dist=80, min_speed=1,event_id = None, figure_save_path = None):
    # Create initial Arrays for aggregating CF Instances
    lead_targets = collection_initial.list_of_lead_targets()
    list_of_collections = [PointCollection() for row in range(len(lead_targets))]

    # Sort Data Points in CF behavior based on target ID into DataCollections
    for i in range(collection_initial.point_count()):
        for j in range(len(lead_targets)):
            if collection_initial[i].lead_target_id() == lead_targets[j]:
                try:
                    if collection_initial[i].vtti_time_stamp < list_of_collections[j][list_of_collections[j].point_count()-1].vtti_time_stamp+3000:
                        list_of_collections[j].point_append(collection_initial[i])
                except IndexError:
                    list_of_collections[j].point_append(collection_initial[i])

    # Checking distance requirements
    temp = list()  # For collecting single instances
    temp_list = list()  # For collecting all instances
    for i in range(len(list_of_collections)):
        for j in range(list_of_collections[i].point_count()):
            if list_of_collections[i][j].lead_target_dist() < max_cf_dist:
                temp.append(list_of_collections[i][j])
            else:
                if len(temp) > 0:
                    temp_list.append(PointCollection(temp))
                    temp = list()
        if len(temp) > 0:
            temp_list.append(PointCollection(temp))
            temp = list()

    list_of_collections = deepcopy(temp_list)

    # Check the subject vehicle's velocity requirements - for freeway driving
    temp = list()  # For collecting single instances
    temp_list = list()  # For collecting all instances
    for i in range(len(list_of_collections)):
        for j in range(list_of_collections[i].point_count()):
            if list_of_collections[i][j].vtti_speed_network > min_speed:
                temp.append(list_of_collections[i][j])
            else:
                if len(temp) > 0:
                    temp_list.append(PointCollection(temp))
                    temp = list()
        if len(temp) > 0:
            temp_list.append(PointCollection(temp))
            temp = list()

    list_of_collections = deepcopy(temp_list)

    # Checking time requirements
    temp_list = list()
    for i in range(len(list_of_collections)):
        if list_of_collections[i].time_elapsed() >= min_cf_time:
            temp_list.append(list_of_collections[i])
    list_of_collections = temp_list

    # If required information is provided, produce and save Dist - Time plots for CF instances
    if figure_save_path is not None and event_id is not None:
        for i in range(len(list_of_collections)):
            temp = list_of_collections[i].plot_t_X('{}_{}_t-X fig2'.format(event_id,i+1))
            temp.savefig(os.path.join(figure_save_path,'{}_{}_t-X fig2'.format(event_id,i+1)))
        for i in range(len(list_of_collections)):
            temp = list_of_collections[i].plot_dV_X('{}_{}_dV-X fig2'.format(event_id,i+1))
            temp.savefig(os.path.join(figure_save_path,'{}_{}_dV-X fig2'.format(event_id,i+1)))
        for i in range(len(list_of_collections)):
            temp = list_of_collections[i].plot_t_dV('{}_{}_t-dV fig2'.format(event_id,i+1))
            temp.savefig(os.path.join(figure_save_path,'{}_{}_t-dV fig2'.format(event_id,i+1)))

    return list_of_collections


def collection_summary(collection_initial,run_time,summary_save_path = None, event_id = None, summary_file = 'collection_summary.csv',log_file = 'runtime_log.csv'):
    print "----------------------------------------------------------------------------"
    print "Collection Length: {} min".format(round(collection_initial.time_elapsed()/float(60)),3)
    print "Number of Data Points: {}".format(collection_initial.point_count())
    print "Approximate Distance Traveled: {} km".format(collection_initial.dist_traveled())
    print "Trip VTTI Timestamps: {} to {}".format(collection_initial.start_stop_vtti_timestamp()[0],
                                                  collection_initial.start_stop_vtti_timestamp()[1])
    print "Percent Time Vehicle in Car Following: {}%".format(collection_initial.percent_car_following()*100)
    print "Number of Total Targets Identified: {}".format(len(collection_initial.list_of_target_ids()))
    print "Number of Lead Targets Identified: {}".format(len(collection_initial.list_of_lead_targets()))

    if summary_save_path is not None and event_id is not None:
        target = open(os.path.join(summary_save_path,log_file), 'a')
        target.write("----------------------------------------------------------------------------")
        target.write("\n")
        target.write("Event ID: {} ---- Time: {}".format(event_id,run_time))
        target.write("\n")
        target.write("\n")
        target.write("Collection Length: {} min".format(round(collection_initial.time_elapsed()/float(60)),3))
        target.write("\n")
        target.write("Number of Data Points: {}".format(collection_initial.point_count()))
        target.write("\n")
        target.write("Approximate Distance Traveled: {} km".format(round(collection_initial.dist_traveled()),3))
        target.write("\n")
        target.write("Trip VTTI Timestamps: {} to {}".format(collection_initial.start_stop_vtti_timestamp()[0], collection_initial.start_stop_vtti_timestamp()[1]))
        target.write("\n")
        target.write("Percent Time Vehicle in Car Following: {}%".format(collection_initial.percent_car_following()*100))
        target.write("\n")
        target.write("Number of Total Targets Identified: {}".format(len(collection_initial.list_of_target_ids())))
        target.write("\n")
        target.write("Number of Lead Targets Identified: {}".format(len(collection_initial.list_of_lead_targets())))
        target.write("\n")
        target.close()

        if not os.path.isfile(os.path.join(summary_save_path,summary_file)):
            target2 = open(os.path.join(summary_save_path,summary_file), 'a')
            target2.write("Event Id,Run Time,Collection Length [min],No. Points,Distance Traveled [km],VTTI Start Timestamp,"
                          "VTTI Stop Timestamp, Percent Time Vehicle Car Following,Number Total Targets Identified,"
                          "Number Lead Targets Identified, Mean Speed, Maximum Deceleration, Percent Time Wipers Active")
            target2.write("\n")
        else:
            target2 = open(os.path.join(summary_save_path,summary_file), 'a')
        target2.write("{},{},{},{},{},{},{},{},{},{},{},{},{}".format(event_id,run_time,round(collection_initial.time_elapsed()/float(60),3),
                     collection_initial.point_count(),round(collection_initial.dist_traveled(),3),collection_initial.start_stop_vtti_timestamp()[0],
                     collection_initial.start_stop_vtti_timestamp()[1], collection_initial.percent_car_following()*100,
                     len(collection_initial.list_of_target_ids()),len(collection_initial.list_of_lead_targets()),collection_initial.mean_speed(),
                     collection_initial.max_deceleration(),collection_initial.percent_wipers_active()))
        target2.write("\n")
        target2.close()


def car_following_collection_summary(list_car_following_collections, run_time, min_cf_time, max_cf_dist,min_speed,summary_save_path = None, event_id = None, log_file = 'runtime_log.csv'):
    print "----------------------------------------------------------------------------"
    print "CF Instances Identified greater than {} seconds and less than {} meters: {}".format(min_cf_time,max_cf_dist,len(list_car_following_collections))

    if summary_save_path is not None and event_id is not None:
        target = open(os.path.join(summary_save_path,log_file), 'a')
        target.write("----------------------------------------------------------------------------")
        target.write("\n")
        target.write("Event ID: {} ---- Time: {}".format(event_id,run_time))
        target.write("\n")
        target.write("\n")
        target.write("CF Instances Identified greater than {} seconds and less than {} meters: {}".format(min_cf_time,max_cf_dist,len(list_car_following_collections)))
        target.write("\n")

        summary_file = '{}_cf_collections_summary.csv'.format(event_id)
        target2 = open(os.path.join(summary_save_path,summary_file), 'w')
        target2.write("Event ID: {}".format(event_id))
        target2.write("\n")
        target2.write("Run Time: {}".format(run_time))
        target2.write("\n")
        target2.write("Min Following Time: {}".format(min_cf_time))
        target2.write("\n")
        target2.write("Max Following Dist: {}".format(max_cf_dist))
        target2.write("\n")
        target2.write("\n")
        target2.write("Instance No.,Length [sec],VTTI Start Timestamp, VTTI Stop Timestamp, Mean Speed [km/hr], Max Deceleration [m/s2], "
                      "% Active Wipers, Mean Distance to Lead Target [m], Mean Headway to Lead Target [s], Lead Target ID No.")
        target2.write("\n")

    for i in range(len(list_car_following_collections)):
        print "  Length of Instance {}: {} sec, {} - {}".format(i+1,list_car_following_collections[i].time_elapsed(),
                                                        list_car_following_collections[i].start_stop_vtti_timestamp()[0],
                                                        list_car_following_collections[i].start_stop_vtti_timestamp()[1])

        if summary_save_path is not None and event_id is not None:
            target.write(" ")
            target.write("  Length of Instance {}: {} sec, {} - {}".format(i+1,list_car_following_collections[i].time_elapsed(),
                                                        list_car_following_collections[i].start_stop_vtti_timestamp()[0],
                                                        list_car_following_collections[i].start_stop_vtti_timestamp()[1]))
            target.write("\n")


            target2.write("{},{},{},{},{},{},{},{},{},{}".format(i+1,list_car_following_collections[i].time_elapsed(),
                                                        list_car_following_collections[i].start_stop_vtti_timestamp()[0],
                                                        list_car_following_collections[i].start_stop_vtti_timestamp()[1],
                                                        list_car_following_collections[i].mean_speed(),
                                                        list_car_following_collections[i].max_deceleration(),
                                                        list_car_following_collections[i].percent_wipers_active(),
                                                        list_car_following_collections[i].mean_dist_lead_target(),
                                                        list_car_following_collections[i].mean_headway_lead_target(),
                                                        list_car_following_collections[i].cf_instance_lead_target_id()))
            target2.write("\n")
    if summary_save_path is not None and event_id is not None:
        target.close()
        target2.close()




    # todo create two new arrays with the removed outliers and the remaining dV and dX so both can be plotted in the same plot and color coordinated


def processed_data(list_car_following_collections, export_save_path = None, event_id = None,log_file = 'runtime_log.csv'):
    processed_data_list = list()
    for i in range(len(list_car_following_collections)):
        temp_processed_data = list_car_following_collections[i].generate_processed_data_class()
        temp_processed_data.moving_average(16)

        if export_save_path is not None and event_id is not None:
            temp_fig = temp_processed_data.plot_t_dV_kalman()
            temp_fig.savefig(os.path.join(export_save_path,'{}_{}_t-dV_filter'.format(event_id,i+1)))
            temp_fig = temp_processed_data.plot_dV_X_kalman()
            temp_fig.savefig(os.path.join(export_save_path,'{}_{}_X-dV_filter'.format(event_id,i+1)))

        temp_processed_data.replace_original()
        processed_data_list.append(temp_processed_data)  # List of Processed Data Classes
        if export_save_path is not None and event_id is not None:
            temp_fig = temp_processed_data.plot_t_dV_original()
            temp_fig.savefig(os.path.join(export_save_path,'{}_{}_t-dV_new'.format(event_id,i+1)))
            temp_fig = temp_processed_data.plot_dV_X_original()
            temp_fig.savefig(os.path.join(export_save_path,'{}_{}_X-dV_new'.format(event_id,i+1)))

    processed_data_combined = ProcessedData('list',processed_data_list)

    if export_save_path is not None and event_id is not None:
        temp_fig = processed_data_combined.plot_dV_X_original()
        temp_fig.savefig(os.path.join(export_save_path,'{}_{}_ALL_Events'.format(event_id,i+1)))

    return processed_data_combined


def smooth_with_kalman_filter(processed_data_list, export_save_path = None, event_id = None, log_file = 'runtime_log.csv'):
    # todo - generate plots, .csv file with dV dX updated
    kalman_processed_data_list = list()
    for i in range(len(processed_data_list)):
        dataset = processed_data_list[i]
        dataset.kalman_filter(process_variance=.0001, estimated_measurement_variance=.01)
        if export_save_path is not None and event_id is not None:
            # Save Updated Data Output
            dataset.update_export('{}_{}_updated_output.csv'.format(event_id,i+1),export_save_path)
            # Save Plots
            temp = dataset.plot_t_v_target_kalman()
            temp.savefig(os.path.join(export_save_path,'{}_{}_plot_t_target_v_kalman'.format(event_id,i+1)))
            temp = dataset.plot_dV_X_kalman()
            temp.savefig(os.path.join(export_save_path,'{}_{}_plot_dV_X_kalman'.format(event_id,i+1)))
            temp = dataset.plot_t_dV_kalman()
            temp.savefig(os.path.join(export_save_path,'{}_{}_plot_t_dV_kalman'.format(event_id,i+1)))

        kalman_processed_data_list.append(dataset)
    return kalman_processed_data_list


"""
FHWA Driver (CF) Model Genetic Algorithm
"""

def create_aggregate_plots(plot_content,plot_evaluation,vehicle_data,file = None):
    """
    :param plot_content: the vehicle data values to be used in the plot - "acceleration, velocity"
    :param plot_evaluation: the metric by which the plots will be generated - "mean, stdev, frequency"
    :param vehicle_data:
    :param file:
    :return: Saves .csv file with aggregated plots
    """

    # Print csv file with average acceleration values for viewing
    matrix_bounds_dV = [-10, -9.5, -9, -8.5, -8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1,
                        -.5, 0, .5,
                        1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
    matrix_bounds_dX = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77,
                        76, 75, 74, 73, 72, 71,
                        70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47,
                        46, 45, 44, 43, 42, 41,
                        40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
                        16, 15, 14, 13, 12, 11,
                        10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    data_matrix = [[list() for i in range(len(matrix_bounds_dV)-1)] for j in
                       range(len(matrix_bounds_dX)-1)]
    processed_data_matrix = [[np.nan for i in range(len(matrix_bounds_dV))] for j in
                       range(len(matrix_bounds_dX))]

    if plot_content == 'acceleration':
        for i in range(len(vehicle_data.a_following)):
            for j in range(len(data_matrix)):  #dX
                if vehicle_data.dX[i] < matrix_bounds_dX[j] and vehicle_data.dX[i] >= matrix_bounds_dX[j+1]:
                    for k in range(len(data_matrix[0])):  #dV
                        if vehicle_data.dV[i] > matrix_bounds_dV[k] and vehicle_data.dV[i] <= matrix_bounds_dV[k+1]:
                            data_matrix[j][k].append(vehicle_data.a_following[i])
                            break

    elif plot_content == 'velocity':
        for i in range(len(vehicle_data.v_following)):
            for j in range(len(data_matrix)):
                if vehicle_data.dX[i] < matrix_bounds_dX[j] and vehicle_data.dX[i] >= matrix_bounds_dX[j+1]:
                    for k in range(len(data_matrix[0])):
                        if vehicle_data.dV[i] > matrix_bounds_dV[k] and vehicle_data.dV[i] <= matrix_bounds_dV[k+1]:
                            data_matrix[j][k].append(vehicle_data.v_following[i])
                            break

    else:
        data_matrix = None

    if plot_evaluation == 'mean':
        # This is where we can update the output to be exactly what andy's is!
        #   Minimum of 2 points && Variance under 1m/s2
        for i in range(len(data_matrix)):
            for j in range(len(data_matrix[0])):
                if len(data_matrix[i][j]) > 2:  # Requirement from the frequency of points
                    if np.nanstd(data_matrix[i][j]) < np.sqrt(1):  # Requirement from the stdev of the points
                        processed_data_matrix[i+1][j+1] = np.nanmean(data_matrix[i][j])
            processed_data_matrix[i+1][0] = matrix_bounds_dX[i+1]
        for i in range(len(processed_data_matrix[0])):
            processed_data_matrix[0][i] = matrix_bounds_dV[i]

    else:
        processed_data_matrix = None

    # If Target file provided, print to file.
    if file is not None:
        file.write('\n')
        for i in range(len(processed_data_matrix)):
            file.write('{},'.format(matrix_bounds_dX[i]))
            for j in range(len(processed_data_matrix[0])):
                if j != len(processed_data_matrix[0]) - 1:
                    file.write('{},'.format(processed_data_matrix[i][j]))
                else:
                    file.write('{}'.format(processed_data_matrix[i][j]))
            file.write('\n')
        file.write('dX/dV,')
        for i in range(len(matrix_bounds_dV)):
            if i != len(matrix_bounds_dV) - 1:
                file.write('{},'.format(matrix_bounds_dV[i]))
            else:
                file.write('{}'.format(matrix_bounds_dV[i]))
        file.close()

    return processed_data_matrix

    

"""
Gipps CFM Genetic Algorithm
"""
