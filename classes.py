from statsmodels import robust
import os
import numpy as np
import warnings
import scipy.stats as sps
import matplotlib.pyplot as plt
from copy import deepcopy

"""
/*******************************************************************
Utility for Interpreting NDS Data - WY SHRP2 Data (as provided by VTTI).
Class Definitions for creating:
 (1) a data point (all data from collected from one timeseries point)
 (2) a collection - set of data points to describe (consecutive) time series
 points.

Author: Britton Hammit
E-mail: bhammit1@gmail.com
********************************************************************/
"""

class DataPoint:
    """ Data from Single Timestamp """  # Returned when __doc__() is called

    # Instantiation
    def __init__(self,data):
        self.index = len(data)
        self.data = data
        """
        self.system_time_stamp = data[0]
        """
        self.vtti_time_stamp = data[1]
        """
        self.vtti_file_id = data[2]
        """
        self.vtti_speed_network = data[3]
        """
        self.vtti_speed_gps = data[4]
        """
        self.vtti_accel_x = data[5]
        """
        self.vtti_accel_y = data[6]
        self.vtti_pedal_brake_state = data[7]
        self.vtti_pedal_gas_position = data[8]
        self.vtti_abs = data[9]
        self.vtti_traction_control_state = data[10]
        self.vtti_esc = data[11]
        self.vtti_lane_distance_off_center = data[12]
        self.vtti_left_line_right_distance = data[13]
        self.vtti_right_line_left_distance = data[14]
        self.vtti_left_marker_probability = data[15]
        self.vtti_right_marker_probability = data[16]
        self.vtti_light_level = data[17]
        self.vtti_gyro_z = data[18]
        """
        self.vtti_wiper = data[19]
        """
        self.vtti_latitude = data[20]
        self.vtti_longitude = data[21]
        self.vtti_steering_angle = data[22]
        self.vtti_steering_wheel_position = data[23]
        self.vtti_turn_signal = data[24]
        self.vtti_head_confidence = data[25]
        self.vtti_head_position_x = data[26]
        self.vtti_head_position_x_baseline = data[27]
        self.vtti_head_position_y = data[28]
        self.vtti_head_position_y_baseline = data[29]
        self.vtti_head_position_z = data[30]
        self.vtti_head_position_z_baseline = data[31]
        self.vtti_head_rotation_x = data[32]
        self.vtti_head_rotation_x_baseline = data[33]
        self.vtti_head_rotation_y = data[34]
        self.vtti_head_rotation_y_baseline = data[35]
        self.vtti_head_rotation_z = data[36]
        self.vtti_head_rotation_z_baseline = data[37]
        self.computed_time_bin = data[38]
        self.computed_day_of_month = data[39]
        self.vtti_month_gps = data[40]
        self.vtti_year_gps = data[41]
        self.vtti_eye_glance_location = data[42]
        self.vtti_alcohol_interior = data[43]
        self.vtti_airbag_driver = data[44]
        self.vtti_engine_rpm_instant = data[45]
        self.vtti_odometer = data[46]
        self.vtti_prndl = data[47]
        self.vtti_seatbelt_driver = data[48]
        self.vtti_temperature_interior = data[49]
        self.vtti_heading_gps = data[50]
        self.vtti_headlight = data[51]
        self.vtti_lane_width = data[52]
        self.vtti_object_id_t0 = data[53]
        self.vtti_object_id_t1 = data[54]
        self.vtti_object_id_t2 = data[55]
        self.vtti_object_id_t3 = data[56]
        self.vtti_object_id_t4 = data[57]
        self.vtti_object_id_t5 = data[58]
        self.vtti_object_id_t6 = data[59]
        self.vtti_object_id_t7 = data[60]
        self.vtti_range_rate_x_t0 = data[61]
        self.vtti_range_rate_x_t1 = data[62]
        self.vtti_range_rate_x_t2 = data[63]
        self.vtti_range_rate_x_t3 = data[64]
        self.vtti_range_rate_x_t4 = data[65]
        self.vtti_range_rate_x_t5 = data[66]
        self.vtti_range_rate_x_t6 = data[67]
        self.vtti_range_rate_x_t7 = data[68]
        self.vtti_range_rate_y_t0 = data[69]
        self.vtti_range_rate_y_t1 = data[70]
        self.vtti_range_rate_y_t2 = data[71]
        self.vtti_range_rate_y_t3 = data[72]
        self.vtti_range_rate_y_t4 = data[73]
        self.vtti_range_rate_y_t5 = data[74]
        self.vtti_range_rate_y_t6 = data[75]
        self.vtti_range_rate_y_t7 = data[76]
        self.vtti_range_x_t0 = data[77]
        self.vtti_range_x_t1 = data[78]
        self.vtti_range_x_t2 = data[79]
        self.vtti_range_x_t3 = data[80]
        self.vtti_range_x_t4 = data[81]
        self.vtti_range_x_t5 = data[82]
        self.vtti_range_x_t6 = data[83]
        self.vtti_range_x_t7 = data[84]
        self.vtti_range_y_t0 = data[85]
        self.vtti_range_y_t1 = data[86]
        self.vtti_range_y_t2 = data[87]
        self.vtti_range_y_t3 = data[88]
        self.vtti_range_y_t4 = data[89]
        self.vtti_range_y_t5 = data[90]
        self.vtti_range_y_t6 = data[91]
        self.vtti_range_y_t7 = data[92]
        self.vtti_headway_to_lead_vehicle = data[93]
        self.vtti_video_frame = data[94]
        self.track1_target_travel_direction = data[95]
        self.track2_target_travel_direction = data[96]
        self.track3_target_travel_direction = data[97]
        self.track4_target_travel_direction = data[98]
        self.track5_target_travel_direction = data[99]
        self.track6_target_travel_direction = data[100]
        self.track7_target_travel_direction = data[101]
        self.track8_target_travel_direction = data[102]
        """
        self.track1_x_acc_estimated = data[103]
        self.track2_x_acc_estimated = data[104]
        self.track3_x_acc_estimated = data[105]
        self.track4_x_acc_estimated = data[106]
        self.track5_x_acc_estimated = data[107]
        self.track6_x_acc_estimated = data[108]
        self.track7_x_acc_estimated = data[109]
        self.track8_x_acc_estimated = data[110]
        self.track1_headway = data[111]
        self.track2_headway = data[112]
        self.track3_headway = data[113]
        self.track4_headway = data[114]
        self.track5_headway = data[115]
        self.track6_headway = data[116]
        self.track7_headway = data[117]
        self.track8_headway = data[118]
        """
        self.track1_lane = data[119]
        self.track2_lane = data[120]
        self.track3_lane = data[121]
        self.track4_lane = data[122]
        self.track5_lane = data[123]
        self.track6_lane = data[124]
        self.track7_lane = data[125]
        self.track8_lane = data[126]
        """
        self.track1_is_lead_vehicle = data[127]
        self.track2_is_lead_vehicle = data[128]
        self.track3_is_lead_vehicle = data[129]
        self.track4_is_lead_vehicle = data[130]
        self.track5_is_lead_vehicle = data[131]
        self.track6_is_lead_vehicle = data[132]
        self.track7_is_lead_vehicle = data[133]
        self.track8_is_lead_vehicle = data[134]
        self.track1_x_pos_processed = data[135]
        self.track2_x_pos_processed = data[136]
        self.track3_x_pos_processed = data[137]
        self.track4_x_pos_processed = data[138]
        self.track5_x_pos_processed = data[139]
        self.track6_x_pos_processed = data[140]
        self.track7_x_pos_processed = data[141]
        self.track8_x_pos_processed = data[142]
        """
        self.track1_y_pos_processed = data[143]
        self.track2_y_pos_processed = data[144]
        self.track3_y_pos_processed = data[145]
        self.track4_y_pos_processed = data[146]
        self.track5_y_pos_processed = data[147]
        self.track6_y_pos_processed = data[148]
        self.track7_y_pos_processed = data[149]
        self.track8_y_pos_processed = data[150]
        """
        self.track1_target_id = data[151]
        self.track2_target_id = data[152]
        self.track3_target_id = data[153]
        self.track4_target_id = data[154]
        self.track5_target_id = data[155]
        self.track6_target_id = data[156]
        self.track7_target_id = data[157]
        self.track8_target_id = data[158]

        # To be added from Corresponding STAC Data - Must have STAC data for this!
        self.track1_x_vel_processed = np.nan
        self.track2_x_vel_processed = np.nan
        self.track3_x_vel_processed = np.nan
        self.track4_x_vel_processed = np.nan
        self.track5_x_vel_processed = np.nan
        self.track6_x_vel_processed = np.nan
        self.track7_x_vel_processed = np.nan
        self.track8_x_vel_processed = np.nan

    # Iterator
    def __iter__(self):
        return self

    # Next
    def next(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.data[self.index]

    # Getitem
    def __getitem__(self,index):
        return self.data[index]

    # Detect Car Following
    def is_car_following(self):
        """
        Method used to determine if a vehicle is in "car-following" by detecting if a lead vehicle exists in any track
        :return: Returns True if vehicle is in car-following and False if vehicle is not in
        """
        if str(float(self.track1_target_id)) != 'nan' and self.track1_is_lead_vehicle == 1:
            return True
        elif str(float(self.track2_target_id)) != 'nan' and self.track2_is_lead_vehicle == 1:
            return True
        elif str(float(self.track3_target_id)) != 'nan' and self.track3_is_lead_vehicle == 1:
            return True
        elif str(float(self.track4_target_id)) != 'nan' and self.track4_is_lead_vehicle == 1:
            return True
        elif str(float(self.track5_target_id)) != 'nan' and self.track5_is_lead_vehicle == 1:
            return True
        elif str(float(self.track6_target_id)) != 'nan' and self.track6_is_lead_vehicle == 1:
            return True
        elif str(float(self.track7_target_id)) != 'nan' and self.track7_is_lead_vehicle == 1:
            return True
        elif str(float(self.track8_target_id)) != 'nan' and self.track8_is_lead_vehicle == 1:
            return True
        else:
            return False

    # Detect Current Targets
    def current_targets(self):
        """
        :return: Returns the list of all targets detected; if no targets detected, empty list is returned
        """
        targets = list()
        if np.isnan(self.track1_target_id) != True:
            targets.append(int(self.track1_target_id))
        if np.isnan(self.track2_target_id) != True:
            targets.append(int(self.track2_target_id))
        if np.isnan(self.track3_target_id) != True:
            targets.append(int(self.track3_target_id))
        if np.isnan(self.track4_target_id) != True:
            targets.append(int(self.track4_target_id))
        if np.isnan(self.track5_target_id) != True:
            targets.append(int(self.track5_target_id))
        if np.isnan(self.track6_target_id) != True:
            targets.append(int(self.track6_target_id))
        if np.isnan(self.track7_target_id) != True:
            targets.append(int(self.track7_target_id))
        if np.isnan(self.track8_target_id) != True:
            targets.append(int(self.track8_target_id))
        return targets

    # Detect Current Lead Vehicle Target Value
    def lead_target_id(self):
        """
        :return: Returns the lead vehicle target id; if no lead vehicle - returns "False"
        """
        if self.is_car_following() is True:
            if np.isnan(self.track1_target_id) is not True and self.track1_is_lead_vehicle == 1:
                return int(self.track1_target_id)
            elif np.isnan(self.track2_target_id) is not True and self.track2_is_lead_vehicle == 1:
                return int(self.track2_target_id)
            elif np.isnan(self.track3_target_id) is not True and self.track3_is_lead_vehicle == 1:
                return int(self.track3_target_id)
            elif np.isnan(self.track4_target_id) is not True and self.track4_is_lead_vehicle == 1:
                return int(self.track4_target_id)
            elif np.isnan(self.track5_target_id) is not True and self.track5_is_lead_vehicle == 1:
                return int(self.track5_target_id)
            elif np.isnan(self.track6_target_id) is not True and self.track6_is_lead_vehicle == 1:
                return int(self.track6_target_id)
            elif np.isnan(self.track7_target_id) is not True and self.track7_is_lead_vehicle == 1:
                return int(self.track7_target_id)
            elif np.isnan(self.track8_target_id) is not True and self.track8_is_lead_vehicle == 1:
                return int(self.track8_target_id)
            else:
                return np.inf

    # Detect Current Lead Vehicle Target Track
    def lead_target_track(self):
        """
        :return: Returns the lead vehicle target track; if no lead vehicle - returns "False"
        """
        if self.is_car_following() is True:
            if str(float(self.track1_target_id)) != 'nan' and self.track1_is_lead_vehicle == 1:
                return 1
            elif str(float(self.track2_target_id)) != 'nan' and self.track2_is_lead_vehicle == 1:
                return 2
            elif str(float(self.track3_target_id)) != 'nan' and self.track3_is_lead_vehicle == 1:
                return 3
            elif str(float(self.track4_target_id)) != 'nan' and self.track4_is_lead_vehicle == 1:
                return 4
            elif str(float(self.track5_target_id)) != 'nan' and self.track5_is_lead_vehicle == 1:
                return 5
            elif str(float(self.track6_target_id)) != 'nan' and self.track6_is_lead_vehicle == 1:
                return 6
            elif str(float(self.track7_target_id)) != 'nan' and self.track7_is_lead_vehicle == 1:
                return 7
            elif str(float(self.track8_target_id)) != 'nan' and self.track8_is_lead_vehicle == 1:
                return 8
            else:
                return np.inf

    # Detect Distance to Current Lead Target
    def lead_target_dist(self):
        """
        :return: Returns the lead vehicle target id; if no lead vehicle - returns "False"
        """
        if self.is_car_following() is True:
            if str(float(self.track1_target_id)) != 'nan' and self.track1_is_lead_vehicle == 1:
                return round(self.track1_x_pos_processed,3)
            elif str(float(self.track2_target_id)) != 'nan' and self.track2_is_lead_vehicle == 1:
                return round(self.track2_x_pos_processed,3)
            elif str(float(self.track3_target_id)) != 'nan' and self.track3_is_lead_vehicle == 1:
                return round(self.track3_x_pos_processed,3)
            elif str(float(self.track4_target_id)) != 'nan' and self.track4_is_lead_vehicle == 1:
                return round(self.track4_x_pos_processed,3)
            elif str(float(self.track5_target_id)) != 'nan' and self.track5_is_lead_vehicle == 1:
                return round(self.track5_x_pos_processed,3)
            elif str(float(self.track6_target_id)) != 'nan' and self.track6_is_lead_vehicle == 1:
                return round(self.track6_x_pos_processed,3)
            elif str(float(self.track7_target_id)) != 'nan' and self.track7_is_lead_vehicle == 1:
                return round(self.track7_x_pos_processed,3)
            elif str(float(self.track8_target_id)) != 'nan' and self.track8_is_lead_vehicle == 1:
                return round(self.track8_x_pos_processed,3)
            else:
                return np.inf
        else:
            return np.inf

    # Detect Distance to Current Lead Target
    def lead_target_headway(self):
        """
        :return: Returns the lead vehicle target id; if no lead vehicle - returns "False"
        """
        if self.is_car_following() is True:
            if str(float(self.track1_target_id)) != 'nan' and self.track1_is_lead_vehicle == 1:
                return round(self.track1_headway,3)
            elif str(float(self.track2_target_id)) != 'nan' and self.track2_is_lead_vehicle == 1:
                return round(self.track2_headway,3)
            elif str(float(self.track3_target_id)) != 'nan' and self.track3_is_lead_vehicle == 1:
                return round(self.track3_headway,3)
            elif str(float(self.track4_target_id)) != 'nan' and self.track4_is_lead_vehicle == 1:
                return round(self.track4_headway,3)
            elif str(float(self.track5_target_id)) != 'nan' and self.track5_is_lead_vehicle == 1:
                return round(self.track5_headway,3)
            elif str(float(self.track6_target_id)) != 'nan' and self.track6_is_lead_vehicle == 1:
                return round(self.track6_headway,3)
            elif str(float(self.track7_target_id)) != 'nan' and self.track7_is_lead_vehicle == 1:
                return round(self.track7_headway,3)
            elif str(float(self.track8_target_id)) != 'nan' and self.track8_is_lead_vehicle == 1:
                return round(self.track8_headway,3)
            else:
                return np.inf

    def lead_target_acc(self):
        """
        :return: Returns the lead vehicle target id; if no lead vehicle - returns "False"
        """
        if self.is_car_following() is True:
            if str(float(self.track1_target_id)) != 'nan' and self.track1_is_lead_vehicle == 1:
                return round(self.track1_x_acc_estimated,3)
            elif str(float(self.track2_target_id)) != 'nan' and self.track2_is_lead_vehicle == 1:
                return round(self.track2_x_acc_estimated,3)
            elif str(float(self.track3_target_id)) != 'nan' and self.track3_is_lead_vehicle == 1:
                return round(self.track3_x_acc_estimated,3)
            elif str(float(self.track4_target_id)) != 'nan' and self.track4_is_lead_vehicle == 1:
                return round(self.track4_x_acc_estimated,3)
            elif str(float(self.track5_target_id)) != 'nan' and self.track5_is_lead_vehicle == 1:
                return round(self.track5_x_acc_estimated,3)
            elif str(float(self.track6_target_id)) != 'nan' and self.track6_is_lead_vehicle == 1:
                return round(self.track6_x_acc_estimated,3)
            elif str(float(self.track7_target_id)) != 'nan' and self.track7_is_lead_vehicle == 1:
                return round(self.track7_x_acc_estimated,3)
            elif str(float(self.track8_target_id)) != 'nan' and self.track8_is_lead_vehicle == 1:
                return round(self.track8_x_acc_estimated,3)
            else:
                return np.inf

    def lead_relative_velocity(self):
        # Relative Velocity with respect to following vehicle
        if self.is_car_following() is True:  #todo remove this statement
            if str(float(self.track1_target_id)) != np.nan and self.track1_is_lead_vehicle == 1:
                return self.track1_x_vel_processed
            elif str(float(self.track2_target_id)) != np.nan and self.track2_is_lead_vehicle == 1:
                return self.track2_x_vel_processed
            elif str(float(self.track3_target_id)) != np.nan and self.track3_is_lead_vehicle == 1:
                return self.track3_x_vel_processed
            elif str(float(self.track4_target_id)) != np.nan and self.track4_is_lead_vehicle == 1:
                return self.track4_x_vel_processed
            elif str(float(self.track5_target_id)) != np.nan and self.track5_is_lead_vehicle == 1:
                return self.track5_x_vel_processed
            elif str(float(self.track6_target_id)) != np.nan and self.track6_is_lead_vehicle == 1:
                return self.track6_x_vel_processed
            elif str(float(self.track7_target_id)) != np.nan and self.track7_is_lead_vehicle == 1:
                return self.track7_x_vel_processed
            elif str(float(self.track8_target_id)) != np.nan and self.track8_is_lead_vehicle == 1:
                return self.track8_x_vel_processed
            else:
                return np.NINF  # Negative infinity
        else:
             return np.NINF

    # Return when print(instance) is used
    def __str__(self):
        return "Summary of Important Variables/Features of Data Point"


class PointCollection:
    """ Summary of Collected Data Points """

    # Instantiation
    def __init__(self,input_data = None):
        """
        :param list_of_data_points: List of DataPoints to be provided
        :return: no return - initializes the collection object
        """
        if input_data is None:
            self.list_of_data_points = list()
            self.index = 0
        else:
            if isinstance(input_data,list) is True:
                if isinstance(input_data[0],DataPoint) is True:
                    self.list_of_data_points = input_data
            elif isinstance(input_data,DataPoint) is True:
                self.list_of_data_points = list()
                self.list_of_data_points.append(input_data)
            else:
                raise TypeError('Must input DataPoint objects into PointCollections')
            self.index = len(self.list_of_data_points)

    # Iterator
    def __iter__(self):
        return self

    # Next
    def next(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.list_of_data_points[self.index]

    # Getitem
    def __getitem__(self,index):
        return self.list_of_data_points[index]

    # Append Data Point to Collection
    def point_append(self,added_points):
        """
        :param added_points: DataPoints to be added to Collection - single point or list of points
        :return: no return - updates the collection to include the DataPoints specified
        """
        if isinstance(added_points,list) is True:
            if isinstance(added_points[0],DataPoint) is True:
                temp_list = list()
                if len(self.list_of_data_points) > 0:
                    for i in range(len(self.list_of_data_points)):
                        temp_list.append(self.list_of_data_points[i])
                for i in range(len(added_points)):
                    temp_list.append(added_points[i])

                self.list_of_data_points = temp_list
                del temp_list
                self.index = len(self.list_of_data_points)

        elif isinstance(added_points,DataPoint) is True:
            temp_list = list()
            if len(self.list_of_data_points) > 0:
                for i in range(len(self.list_of_data_points)):
                    temp_list.append(self.list_of_data_points[i])

            temp_list.append(added_points)

            self.list_of_data_points = temp_list
            del temp_list
            self.index = len(self.list_of_data_points)

        else:
            raise TypeError('List of DataPoints or single DataPoint required')

    # Number of Points in Collection
    def point_count(self):
        return len(self.list_of_data_points)

    # Median
    def median(self):
        if self.point_count % 2 == 0:
            return np.mean(self.list_of_data_points[self.point_count()/2],self.list_of_data_points[(self.point_count()/2)-1])
        else:
            return self.list_of_data_points[(self.point_count()-1)/2]

    # Driving Time covered by Collection [sec]
    def time_elapsed(self):
        return self.point_count()/float(10)

    # List of Target IDs for trip
    def list_of_target_ids(self):
        """
        :return: Return a list of all targets identified within the Collection
        """
        unique_target_ids = list()
        temp_targets = list()
        for i in range(len(self.list_of_data_points)):
            temp_targets.append(self.list_of_data_points[i].current_targets())
        for i in range(len(temp_targets)):
            if len(temp_targets[i]) != 0:
                for j in range(len(temp_targets[i])):
                    if temp_targets[i][j] not in unique_target_ids:
                        unique_target_ids.append(temp_targets[i][j])
        return unique_target_ids

    # List of Lead Target Ids
    def list_of_lead_targets(self):
        """
        :return: Return a list of all identified LEAD targets within the Collection
        """
        unique_lead_target_ids = list()
        temp_targets = list()
        for i in range(len(self.list_of_data_points)):
            temp_targets.append(self.list_of_data_points[i].lead_target_id())
        for i in range(len(temp_targets)):
            if temp_targets[i] is not None:
                if temp_targets[i] not in unique_lead_target_ids:
                    unique_lead_target_ids.append(int(temp_targets[i]))
        return unique_lead_target_ids

    # Distance Travelled (assuming consecutive points) in Collection
    def dist_traveled(self):
        temp_dist = 0
        for i in range(len(self.list_of_data_points)):
            speed_temp = self.list_of_data_points[i].vtti_speed_network
            try:
                int(speed_temp)
                dist = speed_temp / float(36000)
                temp_dist += dist
            except ValueError:
                continue
        return round(temp_dist,3)

    # Location start (assuming consecutive points) in collection
    def lat_long_start(self):
        pass

    # Location end (assuming consecutive points) in collection
    def lat_long_end(self):
        pass

    # Average Speed
    def mean_speed(self):
        temp = list()
        for i in range(len(self.list_of_data_points)):
            temp.append(self.list_of_data_points[i].vtti_speed_network)
        return round(np.nanmean(temp),3)

    # Max Deceleration
    def max_deceleration(self):
        temp = list()
        for i in range(len(self.list_of_data_points)):
            temp.append(self.list_of_data_points[i].vtti_accel_x)
        return round(np.nanmin(temp),3)

    # Percent Wipers
    def percent_wipers_active(self):
        temp = 0
        for i in range(len(self.list_of_data_points)):
            wiper_status = self.list_of_data_points[i].vtti_wiper
            if wiper_status == 1 or wiper_status == 2 or wiper_status == 3:
                temp += 1
        return round(float(temp)/len(self.list_of_data_points),3)

    # Percent of Time Car Following
    def percent_car_following(self):
        temp = 0
        for i in range(len(self.list_of_data_points)):
            if self.list_of_data_points[i].is_car_following() is True:
                temp += 1
        return round(float(temp)/len(self.list_of_data_points),3)

    # Summary Statistics of Variable Availability
    def summary_variable_availability(self,variable_index,save_path,output_file):
        count_available = [0 for col in range(len(variable_index))]
        for i in range(len(self.list_of_data_points)):
            for j in range(len(variable_index)):
                temp = self.list_of_data_points[i][variable_index[j][1]]
                if str(float(self.list_of_data_points[i][variable_index[j][1]])) != 'nan':
                    count_available[j] += 1

        percent_available = [0 for col in range(len(variable_index))]
        for i in range(len(count_available)):
            percent_available[i] = count_available[i]/float(len(self.list_of_data_points))

        target = open(os.path.join(save_path, output_file),'w')  # Output file
        target.write("Variable Name, Percent, Count")
        target.write("\n")
        for i in range(len(count_available)):
            target.write("{},{},{}".format(variable_index[i][0],percent_available[i],count_available[i]))
            target.write("\n")
        target.close()

        print "Success - Data Availability File Generated"

    # Summary of Relevant Statistics
    def summary_statistics(self,variable_index,save_path,output_file):
        # Iteration assigning values from the NDS data file
        for i in range(len(variable_index)):
            index = variable_index[i][1]
            values_temp = ['nan' for k in range(len(self.list_of_data_points))]  # defaults to 'nan' if value missed
            for j in range(len(self.list_of_data_points)):
                values_temp[j] = self.list_of_data_points[j][index]
            variable_index[i][2] = values_temp

        target = open(os.path.join(save_path, output_file),'w')  # Output file

        # Print Headers to file
        target.write("File: {}".format(output_file))
        target.write("\n")
        target.write("Variables,")
        for i in range(1):      # Write variable names to file
            for j in range(len(variable_index)):
                # Formatting adjustment for final value in a row
                if j == len(variable_index) - 1:
                    target.write("{}".format(variable_index[j][0]))
                else:
                    target.write("{},".format(variable_index[j][0]))
        target.write("\n")

        stats_operations_names = ['Mean','Max','Min','Median','Percentile85','Stdev','Variance','Coeff of Variation']

        # Statistics Calculations
        warnings.filterwarnings('ignore') # Warnings regarding all "nan" values ignored
        for i in range(len(stats_operations_names)):
            target.write(stats_operations_names[i]+',')
            for j in range(len(variable_index)):
                variable_array = variable_index[j][2] # Array of values for statistical operation
                if i == 0:  # Mean
                    # Formatting adjustment for final value in a row
                    if j == len(variable_index) - 1:
                        mean = np.nanmean(variable_array)
                        target.write("{}".format(mean))
                    else:
                        mean = np.nanmean(variable_array)
                        target.write("{},".format(mean))
                elif i == 1:  # Maximum
                    # Formatting adjustment for final value in a row
                    if j == len(variable_index) - 1:
                        max = np.nanmax(variable_array)
                        target.write("{}".format(max))
                    else:
                        max = np.nanmax(variable_array)
                        target.write("{},".format(max))
                elif i == 2:  # Minimum
                    # Formatting adjustment for final value in a row
                    if j == len(variable_index) - 1:
                        min = np.nanmin(variable_array)
                        target.write("{}".format(min))
                    else:
                        min = np.nanmin(variable_array)
                        target.write("{},".format(min))
                elif i == 3:  # Median
                    # Formatting adjustment for final value in a row
                    if j == len(variable_index) - 1:
                        median = np.nanmedian(variable_array)
                        target.write("{}".format(median))
                    else:
                        median = np.nanmedian(variable_array)
                        target.write("{},".format(median))
                elif i == 4:  # 85th Percentile
                    # Formatting adjustment for final value in a row
                    if j == len(variable_index) - 1:
                        percentile85 = np.nanpercentile(variable_array,85)
                        target.write("{}".format(percentile85))
                    else:
                        percentile85 = np.nanpercentile(variable_array,85)
                        target.write("{},".format(percentile85))
                elif i == 5:  # Standard Deviation
                    # Formatting adjustment for final value in a row
                    if j == len(variable_index) - 1:
                        std = np.nanstd(variable_array)
                        target.write("{}".format(std))
                    else:
                        std = np.nanstd(variable_array)
                        target.write("{},".format(std))
                elif i == 6:  # Variance
                    # Formatting adjustment for final value in a row
                    if j == len(variable_index) - 1:
                        var = np.nanvar(variable_array)
                        target.write("{}".format(var))
                    else:
                        var = np.nanvar(variable_array)
                        target.write("{},".format(var))
                elif i == 7:  # Coefficient of Variation
                    # Formatting adjustment for final value in a row
                    if j == len(variable_index) - 1:
                        coeff_of_var = sps.variation(variable_array)
                        target.write("{}".format(coeff_of_var))
                    else:
                        coeff_of_var = sps.variation(variable_array)
                        target.write("{},".format(coeff_of_var))
            target.write("\n")
        target.close()
        print "Success - Statistics Generated"

    def start_stop_vtti_timestamp(self):
        try: start = int(self.list_of_data_points[0].vtti_time_stamp)
        except ValueError:
            try: start = int(self.list_of_data_points[1].vtti_time_stamp)
            except ValueError:
                try: start = int(self.list_of_data_points[2].vtti_time_stamp)
                except ValueError:
                    try: start = int(self.list_of_data_points[3].vtti_time_stamp)
                    except ValueError:
                        try: start = int(self.list_of_data_points[4].vtti_time_stamp)
                        except ValueError:
                            raise ValueError('Numeric Value not within first five points of collection')

        stop = int(self.list_of_data_points[len(self.list_of_data_points)-1].vtti_time_stamp)
        return [start,stop]

    def v_following(self):
        # Using Network Speed - should check with GPS Speed
        v_following = list()
        for i in range(len(self.list_of_data_points)-1):
            v_following.append(self.list_of_data_points[i].vtti_speed_network*1000/3600)  # m/s

        return v_following  # m/s

    def dV(self):
        dV = list()
        for i in range(len(self.list_of_data_points)-1):
            # relative velocity between following vehicle and target
            dV_temp = self.list_of_data_points[i+1].lead_relative_velocity()
            dV.append(dV_temp)
        return dV  #m/s

    def v_target(self):
        v_target = list()
        v_following = self.v_following()
        dV = self.dV()
        for i in range(len(v_following)):
            if np.isinf(dV[i]) is True:  # Works for both positive and negative infinity!
                v_target.append(np.inf)
            else:
                v_target.append(np.subtract(v_following[i],dV[i]))
        return v_target  # m/s

    def dX(self):
        dX = list()
        for i in range(len(self.list_of_data_points)-1):
            # separating distance between following vehicle and target
            x_sep = self.list_of_data_points[i+1].lead_target_dist()
            dX.append(x_sep)
        return dX

    def dT(self):
        dT = list()
        t_temp = 0
        for i in range(len(self.list_of_data_points)-1):
            # seconds elapsed since last timestep
            dt_temp = (self.list_of_data_points[i+1].vtti_time_stamp-self.list_of_data_points[i].vtti_time_stamp)/1000  # sec
            # accumulative timestamp (starting at 0)
            t_temp += dt_temp
            dT.append(t_temp)
        return dT

    def dT_vtti(self):
        dT = list()
        for i in range(len(self.list_of_data_points)-1):
            dT.append(self.list_of_data_points[i].vtti_time_stamp)
        return dT

    def a_target(self):
        a_target = list()
        for i in range(len(self.list_of_data_points)):
            a_target.append(self.list_of_data_points[i].lead_target_acc())
        return a_target

    def a_following(self):
        a_following = list()
        for i in range(len(self.list_of_data_points)):
            a_following.append(self.list_of_data_points[i].vtti_accel_x)
        return a_following

    def plot_t_X(self,title = 't-X'):
        fig = plt.figure(figsize=(16,12))  # Size of figure
        fig.suptitle(title,fontsize=18, fontweight='bold')
        target_distances = self.dX()
        timestamp_values = self.dT_vtti()
        plt.scatter(timestamp_values,target_distances)
        plt.ylabel("Distance to Lead Vehicle")
        plt.xlabel("VTTI Timestamp")
        plt.close()
        return fig

    def plot_dV_X(self, title = 'dV-X'):
        fig = plt.figure(figsize=(16,12))  # Size of figure
        fig.suptitle(title,fontsize=18, fontweight='bold')
        dV = self.dV()
        dX = self.dX()
        plt.scatter(dV,dX)
       # plt.xlim([-3.5,3.5])
        plt.ylabel("Distance to Target, X")
        plt.xlabel("Change in Velocity, dV")

        # plt.ylim([0,60])  # Distance to lead vehicle
        dV_abs = list()
        for i in range(len(dV)):
            dV_abs.append(abs(dV[i]))
        plt.xlim([-max(dV_abs),max(dV_abs)])
        plt.close()
        return fig

    def plot_t_dV(self, title = 't-dV'):
        fig = plt.figure(figsize=(16,12))  # Size of figure
        fig.suptitle(title,fontsize=18, fontweight='bold')
        dV = self.dV()
        dt = self.dT()
        plt.scatter(dt,dV)
        plt.xlabel("Time")
        plt.ylabel("Change in Velocity, dV")
        plt.close()
        return fig

    def outlier_separation(self,mad_count=5,rolling_increment=20):
        dV = list()
        dX = list()
        dt = list()
        V_lead = list()
        t_temp = 0
        for i in range(len(self.list_of_data_points)-1):
            dt_temp = (self.list_of_data_points[i+1].vtti_time_stamp-self.list_of_data_points[i].vtti_time_stamp)/1000  #[sec]
            d_travelled_temp = self.list_of_data_points[i].vtti_speed_network*1000/3600*dt_temp
            dX_temp = self.list_of_data_points[i+1].lead_target_dist()+d_travelled_temp-self.list_of_data_points[i].lead_target_dist()
            X_temp = self.list_of_data_points[i+1].lead_target_dist()
            V_target_temp = dX_temp/dt_temp - self.list_of_data_points[i+1].lead_target_acc()*dt_temp
            dV_temp = self.list_of_data_points[i].vtti_speed_network*1000/3600 - V_target_temp  # Convert to m/s
            t_temp += dt_temp
            dX.append(X_temp)
            dV.append(dV_temp)
            dt.append(t_temp)
            V_lead.append(V_target_temp)
        index_list = list()
        for i in range(len(dV)):
            index_list.append(i)

        rolling_increment = rolling_increment*10
        outlier_index = list()
        for i in range(0,len(dV),rolling_increment):
            temp_dV = list()
            temp_index_list = list()
            for j in range(rolling_increment):
                index = i+j
                try:
                    temp_dV.append(dV[index])
                    temp_index_list.append(index_list[index])
                except IndexError:
                    break

            median = np.median(temp_dV)
            temp_diff_list = list()
            for i in range(len(temp_dV)):
                temp_diff_list.append(abs(temp_dV[i]-median))
            median_absolute_deviation = np.median(temp_diff_list)
            print "{} : {}".format(robust.mad(temp_dV),median_absolute_deviation)
            median_absolute_deviation = robust.mad(temp_dV)
            temp_outlier_index = list()
            for i in range(len(temp_diff_list)):
                if temp_diff_list[i] >= mad_count*median_absolute_deviation:
                    temp_outlier_index.append(i)

            for i in temp_outlier_index:
                outlier_index.append(temp_index_list[i])

        new_dV = list()
        new_X = list()
        new_t = list()
        outlier_dV = list()
        outlier_X = list()
        outlier_t = list()
        for i in range(len(dV)):
            if i in outlier_index:
                outlier_dV.append(dV[i])
                outlier_X.append(dX[i])
                outlier_t.append(dt[i])
            else:
                new_dV.append(dV[i])
                new_X.append(dX[i])
                new_t.append(dt[i])

        fig = plt.figure(figsize=(16,12))  # Size of figure
        fig.suptitle('title',fontsize=18, fontweight='bold')
        plt.scatter(dt,V_lead,c='b')
        #plt.scatter(outlier_t,outlier_dV,c='r')
        plt.xlabel("Time, T")
        plt.ylabel("Change in Velocity, dV")
        plt.close()
        return fig

    def kalman_filter(self):
        dV = self.dV()
        dX = self.dX()
        dt = self.dT()
        V_lead = self.v_target()

        """
        Kalman Filter
        """
        # http://scottlobdell.me/2014/08/kalman-filtering-python-reading-sensor-input/
        # http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf

        # Initial parameters
        iteration_count = len(V_lead)
        noisy_measurement = V_lead
        process_variance = .0001
        estimated_measurement_variance = .01 # estimate of measurement var, change to see effect

        # Initial Guess
        posteri_estimate = 0.0
        posteri_error_estimate = 1.0

        posteri_estimate_for_graphing = []

        for iteration in range(1,iteration_count):
            # time update
            priori_estimate = posteri_estimate
            priori_error_estimate = posteri_error_estimate + process_variance

            # measurement update
            blending_factor = priori_error_estimate / (priori_error_estimate + estimated_measurement_variance)
            posteri_estimate = priori_estimate + blending_factor * (noisy_measurement[iteration] - priori_estimate)
            posteri_error_estimate = (1 - blending_factor) * priori_error_estimate
            posteri_estimate_for_graphing.append(posteri_estimate)

        noisy_measurement_graph = noisy_measurement
        noisy_measurement_graph.remove(noisy_measurement[0])

        dt_graph = dt
        dt_graph.remove(dt[0])

        dX_graph = dX
        dX_graph.remove(dX[0])

        for i in range(len(self.list_of_data_points)-2):
            V_target_temp = posteri_estimate_for_graphing[i]
            dV_temp = self.list_of_data_points[i].vtti_speed_network*1000/3600 - V_target_temp  # Convert to m/s
            dV.append(dV_temp)

        print len(noisy_measurement)
        print len(dV)
        print len(dX_graph)

        fig = plt.figure(figsize=(16,12))  # Size of figure
        fig.suptitle('sample',fontsize=18, fontweight='bold')
        #plt.scatter(dt_graph,noisy_measurement_graph,c='r')
        #plt.plot(dt_graph,posteri_estimate_for_graphing, c='b')
        plt.scatter(dV,dX_graph)
        #plt.plot(actual_values, c='g')
        #plt.scatter(dt,V_lead,c='b')
        #plt.scatter(outlier_t,outlier_dV,c='r')
        plt.xlabel("Time, T")
        plt.ylabel("Change in Velocity, dV")
        plt.close()
        return fig

    def mean_dist_lead_target(self):
        temp = list()
        for i in range(len(self.list_of_data_points)):
            temp.append(self.list_of_data_points[i].lead_target_dist())
        return np.nanmin(temp)

    def mean_headway_lead_target(self):
        temp = list()
        for i in range(len(self.list_of_data_points)):
            temp.append(self.list_of_data_points[i].lead_target_headway())
        return np.nanmin(temp)

    def cf_instance_lead_target_id(self):
        temp = self.list_of_data_points[0].lead_target_id()
        for i in range(len(self.list_of_data_points)):
            if temp == self.list_of_data_points[i].lead_target_id():
                continue
            else:
                temp = False
        return temp

    def generate_processed_data_class(self):
        data = list()
        dV = self.dV()
        dX = self.dX()
        dT = self.dT()
        dT_vtti = self.dT_vtti()
        v_target = self.v_target()
        v_following = self.v_following()
        a_target = self.a_target()
        a_following = self.a_following()

        for i in range(len(self.dV())):
            data.append([dV[i],dX[i],dT[i],dT_vtti[i],v_target[i],v_following[i],a_target[i],a_following[i]])
        return ProcessedData('wy_nds',data)


class ProcessedData:
    """ Processed Car Following Data dV dX """

    # Instantiation
    def __init__(self,data_source,data):

        self.data = data
        self.data_source = data_source

        if self.data_source == 'list':
            self.index = 0
            for i in range(len(self.data)):
                self.index += len(self.data[i].dV)
        else:
            self.index = len(self.data)

        if self.data_source == 'wy_nds':
            self.dV = list()
            self.dX = list()
            self.T = list()
            self.T_vtti = list()
            self.v_target = list()
            self.v_following = list()
            self.a_target = list()
            self.a_following = list()
            for i in range(len(self.data)):
                self.dV.append(self.data[i][0])
                self.dX.append(self.data[i][1])
                self.T.append(self.data[i][2])
                self.T_vtti.append(self.data[i][3])
                self.v_target.append(self.data[i][4])
                self.v_following.append(self.data[i][5])
                self.a_target.append(self.data[i][6])
                self.a_following.append(self.data[i][7])
            # Create empty lists because size of list changes with update
            self.v_target_update = None
            self.dV_update = None
            self.v_target_original_graph = None
            self.dV_original_graph = None
            self.dX_graph = None
            self.T_graph = None
            self.T_vtti_graph = None
            self.v_following_graph = None
            self.a_target_graph = None
            self.a_following_graph = None

        elif self.data_source == 'blank':
            # To create an empty assignment Processed Data Array (for combinations)
            self.dV = list()
            self.dX = list()
            self.T = list()
            self.T_vtti = list()
            self.v_target = list()
            self.v_following = list()
            self.a_target = list()
            self.a_following = list()

            # Create empty lists because size of list changes with update
            self.v_target_update = None
            self.dV_update = None
            self.v_target_original_graph = None
            self.dV_original_graph = None
            self.dX_graph = None
            self.T_graph = None
            self.T_vtti_graph = None
            self.v_following_graph = None
            self.a_target_graph = None
            self.a_following_graph = None

        elif self.data_source == 'list':
            # To create an empty assignment Processed Data Array (for combinations)
            self.dV = list()
            self.dX = list()
            self.T = list()
            self.T_vtti = list()
            self.v_target = list()
            self.v_following = list()
            self.a_target = list()
            self.a_following = list()
            for i in range(len(self.data)):
                self.dV.extend(data[i].dV)
                self.dX.extend(data[i].dX)
                self.T.extend(data[i].T)
                self.T_vtti.extend(data[i].T_vtti)
                self.v_target.extend(data[i].v_target)
                self.v_following.extend(data[i].v_following)
                self.a_target.extend(data[i].a_target)
                self.a_following.extend(data[i].a_following)
            # Create empty lists because size of list changes with update
            self.v_target_update = None
            self.dV_update = None
            self.v_target_original_graph = None
            self.dV_original_graph = None
            self.dX_graph = None
            self.T_graph = None
            self.T_vtti_graph = None
            self.v_following_graph = None
            self.a_target_graph = None
            self.a_following_graph = None

        elif self.data_source == 'fhwa_irv':
            # Headers- 0: driver_id,1: unique_inst,2: instance_id,3: timestamp,4: leader_vel,5: leader_accel,
            # 6: follower_vel,7: follower_accel,8: delta_dist,9: delta_vel
            self.dV = list()
            self.dX = list()
            self.T = list()
            self.T_vtti = list()
            self.v_target = list()
            self.v_following = list()
            self.a_target = list()
            self.a_following = list()
            for i in range(len(self.data)):
                self.dV.append(self.data[i][9])
                self.dX.append(self.data[i][8])
                self.T.append(self.data[i][3])
                self.v_target.append(self.data[i][4])
                self.v_following.append(self.data[i][6])
                self.a_target.append(self.data[i][5])
                self.a_following.append(self.data[i][7])
            # Create empty lists because size of list changes with update
            self.v_target_update = None
            self.dV_update = None
            self.v_target_original_graph = None
            self.dV_original_graph = None
            self.dX_graph = None
            self.T_graph = None
            self.T_vtti_graph = None
            self.v_following_graph = None
            self.a_target_graph = None
            self.a_following_graph = None

        elif data_source == 'stac_nds':
            self.dV = list()
            self.dX = list()
            self.T = list()
            self.T_vtti = list()
            self.v_target = list()
            self.v_following = list()
            self.a_target = list()
            self.a_following = list()
            for i in range(len(self.data)):
                self.T_vtti.append(self.data[i][0])
                self.dV.append(self.data[i][10])
                self.dX.append(self.data[i][8])
                self.T.append(self.data[i][0])
                #self.v_target.append(self.data[i][4])
                #self.v_following.append(self.data[i][6])
                self.a_target.append(self.data[i][12])
                #self.a_following.append(self.data[i][7])
            # Create empty lists because size of list changes with update
            self.v_target_update = None
            self.dV_update = None
            self.v_target_original_graph = None
            self.dV_original_graph = None
            self.dX_graph = None
            self.T_graph = None
            self.T_vtti_graph = None
            self.v_following_graph = None
            self.a_target_graph = None
            self.a_following_graph = None

        elif data_source == "synthetic":
            # Headers- 0: Timestamp, 1: lead velocity, 2: lead acceleration, 3: dX, 4: following velocity,
            # 5: following acceleration, 6: dV
            self.dV = list()
            self.dX = list()
            self.T = list()
            self.T_vtti = list()
            self.v_target = list()
            self.v_following = list()
            self.a_target = list()
            self.a_following = list()
            for i in range(len(self.data)):
                self.dV.append(self.data[i][6])
                self.dX.append(self.data[i][3])
                self.T.append(self.data[i][0])
                self.v_target.append(self.data[i][1])
                self.v_following.append(self.data[i][4])
                self.a_target.append(self.data[i][2])
                self.a_following.append(self.data[i][5])
            # Create empty lists because size of list changes with update
            self.v_target_update = None
            self.dV_update = None
            self.v_target_original_graph = None
            self.dV_original_graph = None
            self.dX_graph = None
            self.T_graph = None
            self.T_vtti_graph = None
            self.v_following_graph = None
            self.a_target_graph = None
            self.a_following_graph = None

        else:
            raise ValueError("Incorrect Data Source Provided")

    def next(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index - 1
        return self.data[self.index]

    def __getitem__(self,index):
        return self.data[index]

    def merge(self,processed_data_object):
        # Right now this is only merging original findings (ignoring updated)
        self.dV.extend(processed_data_object.dV)
        self.dX.extend(processed_data_object.dX)
        self.T.extend(processed_data_object.T)
        self.T_vtti.extend(processed_data_object.T_vtti)
        self.v_target.extend(processed_data_object.v_target)
        self.v_following.extend(processed_data_object.v_following)
        self.a_target.extend(processed_data_object.a_target)
        self.a_following.extend(processed_data_object.a_following)

        self.index += len(processed_data_object.dV)

    def replace_original(self):
        if len(self.dV_update) != 0:
            self.dV = deepcopy(self.dV_update)
            self.dX = deepcopy(self.dX_graph)
            self.T = deepcopy(self.T_graph)
            self.T_vtti = deepcopy(self.T_vtti_graph)
            self.v_target = deepcopy(self.v_target_update)
            self.v_following = deepcopy(self.v_following_graph)
            self.a_target = deepcopy(self.a_target_graph)
            self.a_following = deepcopy(self.a_following_graph)

    def original_export(self,filename,path):
        target = open(os.path.join(path,filename),'w')
        target.write("{}".format(filename))
        target.write("\n")
        target.write("\n")
        target.write("dV [m/s], dX [m], dT [s], dT_vtti, v_target [m/s], v_following [m/s], a_target [m/s2], a_following [m/s2]")
        target.write("\n")

        for i in range(len(self.dV)):
            target.write("{},{},{},{},{},{},{},{}".format(self.dV[i],self.dX[i],self.T[i],self.T_vtti[i],self.v_target[i],
                                                          self.v_following[i],self.a_target[i],self.a_following[i]))
            target.write('\n')
        target.close()

    def update_export(self,filename,path):
        target = open(os.path.join(path,filename),'w')
        target.write("{}".format(filename))
        target.write("\n")
        target.write("\n")
        target.write("Update dV [m/s], dX [m], dT [s], dT_vtti, Update v_target [m/s], v_following [m/s], a_target [m/s2], a_following [m/s2]")
        target.write("\n")
        for i in range(len(self.dV_update)):
            target.write("{},{},{},{},{},{},{},{}".format(self.dV_update[i],self.dX_graph[i],self.T_graph[i],self.T_vtti_graph[i],
                                                          self.v_target_update[i],self.v_following_graph[i],
                                                          self.a_target_graph[i],self.a_following_graph[i]))
            target.write('\n')
        target.close()
        target2 = open(os.path.join(path,'00-updated_exports_combined.csv'),'a')
        for i in range(len(self.dV_update)):
            target2.write("{},{},{},{},{},{},{},{}".format(self.dV_update[i],self.dX_graph[i],self.T_graph[i],self.T_vtti_graph[i],
                                                          self.v_target_update[i],self.v_following_graph[i],
                                                          self.a_target_graph[i],self.a_following_graph[i]))
            target2.write('\n')
        target2.close()

    def reset(self):
        self.v_target_update = None
        self.dV_update = None
        self.v_target_original_graph = None
        self.dV_original_graph = None
        self.dX_graph = None
        self.T_graph = None
        self.T_vtti_graph = None
        self.v_following_graph = None

    def kalman_filter(self, process_variance = .0001):
        """
        Kalman Filter
        -----------------------------------------------------------------------------------------------
        # http://scottlobdell.me/2014/08/kalman-filtering-python-reading-sensor-input/
        # http://scottlobdell.me/2017/01/gps-accelerometer-sensor-fusion-kalman-filter-practical-walkthrough/
        # http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf

        This function filters the derived velocity of the target vehicle, which was calculated as
        a function of the distance travelled by the following vehicle, the separating distance at two consecutive time
        steps, and the acceleration estimation from the radar. In order to account for measurement errors from the
        radar data indicated in the calculation of the target velocity, the Kalman filter is used to predict and correct
        the target velocity.
        """

        # Input Relative Velocity - Noisy measurement to be filtered
        dV = self.dV

        diff_list = list()
        for i in range(len(dV)-1):
            diff_list.append(dV[i+1]-dV[i])
        stdev_dV = np.nanstd(diff_list)
        estimated_measurement_variance = .01  # Radar Documentation NDS
        print estimated_measurement_variance

        # Initial Guess
        posteri_estimate = 0.0
        posteri_error_estimate = 1.0

        # Updated relative velocity values
        self.dV_update = []

        # Loop for each velocity reading
        for i in range(len(dV)-1):
            # time update
            priori_estimate = posteri_estimate
            priori_error_estimate = posteri_error_estimate + process_variance

            # measurement update
            blending_factor = priori_error_estimate / (priori_error_estimate + estimated_measurement_variance)
            posteri_estimate = priori_estimate + blending_factor * (dV[i+1] - priori_estimate)
            posteri_error_estimate = (1 - blending_factor) * priori_error_estimate
            self.dV_update.append(posteri_estimate)

        print self.dV_update

        # Create Updated Variables - removing the first row due to change in size of kalman updated data
        self.v_target_original_graph = self.v_target
        self.v_target_original_graph.remove(self.v_target[0])

        self.dV_original_graph = self.dV
        self.dV_original_graph.remove(self.dV[0])

        self.v_target_update = self.v_target
        # self.v_target_update.remove(self.v_target[0])

        self.T_graph = self.T
        self.T_graph.remove(self.T[0])

        self.dX_graph = self.dX
        self.dX_graph.remove(self.dX[0])

        self.T_vtti_graph = self.T_vtti
        self.T_vtti_graph.remove(self.T_vtti[0])

        self.a_target_graph = self.a_target
        self.a_target_graph.remove(self.a_target[0])

        self.a_following_graph = self.a_following
        self.a_following_graph.remove(self.a_following[0])

        self.v_following_graph = self.v_following
        self.v_following_graph.remove(self.v_following[0])

    def moving_average(self, resolution='automatically_defined'):
        # Assumes consecutive timestamps
        if resolution == 'automatically_defined':
            resolution = int(len(self.dV)*0.01)
        #print resolution

        X = self.dV
        Y = [np.nan for i in range(len(X)-resolution/2)]
        #print X
        for i in range(resolution/2, len(X)-resolution/2):
            index = i
            #print index
            Y[index] = 0
            counter = 0
            for j in range(-resolution/2,resolution/2):
                if np.isnan(X[index-j])== False:
                    Y[index]=Y[index]+X[index-j]
                    counter += 1
            if np.isnan(Y[index]) == True:
                print 'nan'
                Y[index] = Y[index-1]
            else:
                Y[index] = Y[index]/counter

        self.dV_update = deepcopy(Y)
        #print self.dV_update

        # Create Updated Variables - removing the first row due to change in size of updated data
        self.v_target_original_graph = deepcopy(self.v_target)
        self.dV_original_graph = deepcopy(self.dV)
        self.v_target_update = deepcopy(self.v_target)
        self.T_graph = deepcopy(self.T)
        self.dX_graph = deepcopy(self.dX)
        self.T_vtti_graph = deepcopy(self.T_vtti)
        self.a_target_graph = deepcopy(self.a_target)
        self.a_following_graph = deepcopy(self.a_following)
        self.v_following_graph = deepcopy(self.v_following)

        for i in range(resolution/2):
            self.v_target_original_graph.pop(-1)
            self.dV_original_graph.pop(-1)
            self.v_target_update.pop(-1)
            self.T_graph.pop(-1)
            self.dX_graph.pop(-1)
            self.T_vtti_graph.pop(-1)
            self.a_target_graph.pop(-1)
            self.a_following_graph.pop(-1)
            self.v_following_graph.pop(-1)

    def plot_t_X_original(self,title = 'Original: Time vs. Separation Distance'):
        fig = plt.figure(figsize=(16,12))  # Size of figure
        fig.suptitle(title, fontsize=18, fontweight='bold')
        plt.scatter(self.T,self.dX,c='b')
        plt.ylabel("Separation Distance [m]")
        plt.xlabel("Normalized Time Stamp [s]")
        plt.close()
        return fig

    def plot_dV_X_original(self, title = 'Original: Relative Velocity vs. Separation Distance'):
        fig = plt.figure(figsize=(16,12))  # Size of figure
        fig.suptitle(title,fontsize=18, fontweight='bold')
        plt.scatter(self.dV,self.dX,c='b')
        #plt.xlim([-10,10]); plt.ylim([0,120])
        plt.ylabel("Separation Distance [m]")
        plt.xlabel("Relative Velocity [m/s]: Following Vehicle - Lead Vehicle")
        plt.legend()

        plt.close()

        return fig

    def plot_t_v_host_original(self, title = 'Original: Host Vehicle Velocity over CF Instance'):
        fig = plt.figure(figsize=(16,12))  # Size of figure
        fig.suptitle(title,fontsize=18, fontweight='bold')
        plt.scatter(self.T,self.v_following,c='b')
        plt.xlabel("Normalized Time Stamp [s]")
        plt.ylabel("Change in Velocity, dV [m/s]")
        plt.close()
        return fig

    def plot_t_dV_original(self, title = 'Original: Relative Velocity over CF Instance'):
        fig = plt.figure(figsize=(16,12))  # Size of figure
        fig.suptitle(title,fontsize=18, fontweight='bold')
        plt.scatter(self.T,self.dV,c='b')
        plt.xlabel("Normalized Time Stamp [s]")
        plt.ylabel("Relative Velocity [m/s]: Following Vehicle - Lead Vehicle")
        plt.close()
        return fig

    def plot_t_v_target_kalman(self, title = 'Kalman: Target Vehicle Velocity over CF Instance'):
        fig = plt.figure(figsize=(16,12))  # Size of figure
        fig.suptitle(title,fontsize=18, fontweight='bold')
        plt.scatter(self.T_graph,self.v_target_original_graph,c='b',label='original')
        plt.plot(self.T_graph,self.v_target_update, c='r',label='update')
        plt.xlabel("Normalized Time Stamp [s]")
        plt.ylabel("Target Velocity [m/s]")
        plt.legend()
        plt.close()
        return fig

    def plot_dV_X_kalman(self, title = 'Kalman: Relative Velocity vs. Separation Distance'):
        fig = plt.figure(figsize=(16,12))  # Size of figure
        fig.suptitle(title,fontsize=18, fontweight='bold')
        plt.scatter(self.dV_original_graph,self.dX_graph,c='b',label='original')
        plt.scatter(self.dV_update,self.dX_graph,c='r',label='update')
        plt.ylabel("Separation Distance [m]")
        plt.xlabel("Relative Velocity [m/s]: Following Vehicle - Lead Vehicle")
        plt.legend()
        plt.close()
        return fig

    def plot_t_dV_kalman(self, title = 'Kalman: Relative Velocity over CF Instance'):
        #fig = plt.figure(figsize=(16,12))  # Size of figure
        fig = plt.figure(figsize=(16,8))
        fig.suptitle(title,fontsize=18, fontweight='bold')
        plt.scatter(self.T_graph,self.dV_original_graph,c='b',label='original')
        plt.plot(self.T_graph,self.dV_update,c='r',label='update')
        #plt.xlim([0,600])
        plt.xlabel("Normalized Time Stamp [s]")
        plt.ylabel("Relative Velocity [m/s]: Following Vehicle - Lead Vehicle")
        plt.legend()
        plt.close()
        return fig

    def plot_t_X_update(self,title = 'Updated: Separation Distance over CF Instance'):
        fig = plt.figure(figsize=(16,12))  # Size of figure
        fig.suptitle(title,fontsize=18, fontweight='bold')
        target_distances = self.dX_graph
        timestamp_values = self.T
        plt.scatter(timestamp_values,target_distances)
        plt.ylabel("Separation Distance [m]")
        plt.xlabel("Normalized Time Stamp [s]")
        plt.legend()
        plt.close()
        return fig

    def plot_dV_X_update(self, title = 'Updated: Relative Velocity vs. Separation Distance'):
        fig = plt.figure(figsize=(16,12))  # Size of figure
        fig.suptitle(title,fontsize=18, fontweight='bold')
        dV = self.dV_update
        dX = self.dX_graph
        plt.scatter(dV,dX)
        plt.ylabel("Distance to Lead Vehicle [m]")
        plt.xlabel("Relative Velocity [m/s]: Following Vehicle - Lead Vehicle")
        # plt.ylim([0,60])  # Distance to lead vehicle
        dV_abs = list()
        for i in range(len(dV)):
            dV_abs.append(abs(dV[i]))
        plt.xlim([-max(dV_abs),max(dV_abs)])
        plt.close()
        return fig

    def plot_t_dV_update(self, title = 'Updated: Relative Velocity over CF Instance'):
        fig = plt.figure(figsize=(16,12))  # Size of figure
        fig.suptitle(title,fontsize=18, fontweight='bold')
        dV = self.dV_update
        t = self.T_vtti_graph
        plt.scatter(t,dV)
        plt.xlabel("Normalized Time Stamp [s]")
        plt.ylabel("Relative Velocity [m/s]: Following Vehicle - Lead Vehicle")
        plt.close()
        return fig


def main():
    """
    CHANGE THESE NAMES/ LOCATIONS IF NECESSARY.
    """
    # Input Variables:
    event_id = 'test'
    test_path = r'C:\Users\Britton\Dropbox\00-Education\Research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\03-Python Script Files\TEST_IndividualFunctions'  # location where test inputs are located
    NDSfile_name = 'Event_ID_{}.csv'.format(event_id)  # Original NDS data file
    variable_file = 'variable_preference_full.csv'
    #############################################################################

    # Import NDS Data
    NDSfile = open(os.path.join(test_path,NDSfile_name), 'r')  # Open NDSfile
    NDSdata = np.genfromtxt(NDSfile, delimiter=',')  # Import data such that blank entries are given 'nan' values

    # If save_path doesn't already exist, it will be created
    save_path = test_path
    output_file = '{}_D_output.csv'.format(event_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Import variable preferences
    file = open(os.path.join(test_path,variable_file), 'r')  # Open Variable Preference file
    file.next()  # Header Line
    temp = []
    for line in file:
        temp.append(line.strip().split(','))
    file.close()
    variable_index = [[0 for i in range(len(temp[0]))] for i in range(len(temp))]
    for i in range(len(temp)):
        variable_index[i] = [temp[i][0], int(float(temp[i][1])), []]  # Array with variables of interest



    #############################################################################

    # Testing Class Functionalities
    data_points = list()
    for i in range(len(NDSdata)):
        data_points.append(DataPoint(NDSdata[i]))

    print data_points[2]

    data_points_sample = list()
    for i in range(100):
        data_points_sample.append(data_points[i])

    collection_test = PointCollection(data_points)

    print collection_test.point_count()

    #collection_test.summary_statistics(variable_index,save_path,'example.csv')

    collection_test.summary_variable_availability(variable_index,save_path,'availability_full.csv')


##############################################################
# Run main() if script is being run directly
if __name__ == "__main__":
    main()
