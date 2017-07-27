from __future__ import division
import time
import os
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
from copy import deepcopy
from functions import import_cf_data


"""
/*******************************************************************
Utility for calibrating the Gipps CFM Parameters using microscopic data.
Constructed optimization algorithm as a Genetic Algorithm.

Author: Britton Hammit
E-mail: bhammit1@gmail.com
********************************************************************/
"""


def run_GA(pop_size, vehicle_data, degrade, restart, p_retain, random_select, random_mutate, log_file,
           fig_file_pop_score=None):
    """
    Executes the genetic algorithm with the selected inputs.
    - The GA is set up to minimize the scores - therefore, the "goal" is implicitly defined to be "0"
    - The "percents" are not exact percentages - the retain, random, and mutation rates are all generated
    randomly; for example, a random float is generated between [0,1] and if that value is less than
    or equal to the p_retain/random_select/random_mutate, that individual or parameter is selected for the action.

    :param pop_size: Integer indicating the number of individuals to be included in the population
    :param vehicle_data: Instance of Processed Data Class with vehicle trajectory data
    :param degrade: Integer indicating the allowable number of generations with worsening population
    scores before restart
    :param restart: Integer indicating the allowable number of restarts with the best population (the
    population with the lowest score)
    :param p_retain: Float indicating the exact percent of best individuals (lowest scores) retained in each
    population as parents
    :param random_select: Float indicating the desired percent of random individuals (from the remaining individuals)
    selected as parents for diversity
    :param random_mutate: Float indicating the desired percent of individual parameters selected for mutation
    :param log_file: CSV File variable with filename.csv and save path = open(os.path.join(filename.csv,path),'w')
    :param fig_file_pop_score: File variable with filename and save path = os.path.join(filename,path)
    :return: ga_results = [low_pop, run_time, generation_counter] where:
    low_pop is the population (list of individuals) with the lowest population score
    run_time is the algorithm process run time
    generation_counter is the number of generations executed
    """

    time_start = time.clock()  # Record Start Time

    # Initialize Population
    # ----------------------------------------------------------------------------------------
    population = create_population(pop_size)

    # Initialize "while loop" variables
    # ----------------------------------------------------------------------------------------
    pop_score_history = list()
    degrade_count = 0  # Counter indicating the generations with consecutive degradation
    restart_count = 0  # Counter indicating the number of restarts
    generation_counter = 1  # Counter indicating the current generation - starting at generation 1
    low_pop_score = np.inf  # starting point for the lowest population score
    low_pop = None  # initialize the lowest population variable
    termination_condition = False  # When "True" loop will stop

    while termination_condition is False:
        log_file.write('\n')
        log_file.write('\n')
        log_file.write("Generation: {}".format(generation_counter))
        log_file.write('\n')

        # Send population through evolution process:
        # ----------------------------------------------------------------------------------------
        population = evolve(population=population, vehicle_data=vehicle_data, retain=p_retain,
                            random_select=random_select, random_mutate=random_mutate, log_file=log_file)
        pop_score = get_population_score(population=population, vehicle_data=vehicle_data)
        pop_score_history.append(pop_score)
        print "Generation {} Pop Score: {}".format(generation_counter, round(pop_score, 4))
        log_file.write("Generation {} Pop Score: {}".format(generation_counter, round(pop_score, 4)))
        # Determine if population score improved or degraded:
        if pop_score >= low_pop_score:
            degrade_count += 1
        elif pop_score < low_pop_score:
            degrade_count = 0  # Ensure XX-consecutive degrades...
            low_pop_score = deepcopy(pop_score)
            low_pop = deepcopy(population)
        print "Current Degrade: {} & Current Restart: {} & Low Pop Score: {}".format(degrade_count, restart_count,
                                                                                     round(low_pop_score, 4))
        print " "
        log_file.write("\n")
        log_file.write("Current Degrade: {} & Current Restart: {} & Low Pop Score: {}".format(degrade_count, restart_count,
                                                                                   round(low_pop_score, 4)))
        generation_counter += 1

        # Termination Conditions:
        # ----------------------------------------------------------------------------------------
        # Determine if the number of consecutive generations with degrading scores has surpassed degrade threshold
        if degrade_count == degrade:
            population = deepcopy(low_pop)  # Restart search at the best population
            restart_count += 1
            degrade_count = 0
        # Determine if the number of restarts has surpassed the degrade threshold
        if restart_count == restart:
            termination_condition = True

    # Indicate score of lowest/best population:
    # ----------------------------------------------------------------------------------------
    print "Low Population Score: {}".format(round(low_pop_score, 4))
    print " "  # Formatting
    log_file.write('\n'); log_file.write('\n')
    log_file.write("Low Population Score: {}".format(round(low_pop_score, 4)))

    # Produce GA-related plots:
    # ----------------------------------------------------------------------------------------
    if fig_file_pop_score is not None:
        plot_GA(plot_name='pop_score_history', pop_score_history=pop_score_history, fig_file=fig_file_pop_score)

    # Compute algorithm run time:
    # ----------------------------------------------------------------------------------------
    time_end = time.clock()  # Record ending time
    run_time = round((time_end - time_start) / 60)  # Record algorithm run time
    print "Total Run Time: {} min".format(run_time)
    log_file.write('\n');
    log_file.write("Total Run Time: {} min".format(run_time))

    # Create return array:
    # ----------------------------------------------------------------------------------------
    ga_results = [low_pop, run_time, generation_counter]
    return ga_results


def create_individual():
    """
    Creates a member of the population
    :return: Returns an "individual" in the format of an array: [t_rxn, V_des, a_des, d_des, d_lead, g_min]
    """

    # Papathanasopoulou Papers
    t_rxn = round((random.random() * (3 - 0.4) + 0.4), 1)
    V_des = round((random.random() * (40 - 10) + 10), 1)  # Adjusted from 29.6 to 40 and 10.4 to 10
    a_des = round((random.random() * (2.6 - 0.8) + 0.8), 1)
    d_des = round((random.random() * -(5.2 - 1.6) - 1.6), 1)
    d_lead = round((random.random() * -(4.5 - 3) - 3), 1)
    g_min = round((random.random() * (4 - 2) + 2), 1)


    individual = [t_rxn, V_des, a_des, d_des, d_lead, g_min]

    return individual


def create_population(pop_size):
    """
    Creates a population of individuals
    :param pop_size: Integer indicating the number of individuals to be included in the population
    :return: List of individuals comprising a population
    """

    population = list()
    for i in range(pop_size):
        population.append(create_individual())

    return population


def predict_v_f(individual,v_foll,v_lead,dX):
    """
    Predict the following vehicle's velocity
    :param individual: Array of model parameters: [t_rxn, V_des, a_des, d_des, d_lead, g_min]
    :param v_foll: Float of the following vehicle's velocity [m/s]
    :param v_lead: Float of the lead vehicle's velocity [m/s]
    :param dX: Float of the separation distance between the lead and following vehicle [m]
    :return: v_f_next_p: Predicted following velocity for next time stamp
    """

    t_rxn,V_f,a_f,b_f,b_l,g_min = individual

    # Solve for v_f_next
    v_f_acc_next = v_foll+2.5*a_f*t_rxn*(1-v_foll/V_f)*math.sqrt((0.025+v_foll/V_f))
    try:
        v_f_dec_next = b_f*t_rxn+math.sqrt(b_f**2*t_rxn**2-(b_f*(2*(dX+g_min)-v_foll*t_rxn-((v_lead**2)/b_l))))
    except TypeError:
        v_f_dec_next = np.inf  # This occurs if no lead vehicle is present and will force the first equation to be used

    v_f_next_p = min(v_f_acc_next,v_f_dec_next)

    return v_f_next_p


def calc_RMSE(pred_list, act_list):
    """
    Calculate the root mean square error between a list of predicted points and actual points.
    :param pred_list: List of predicted points
    :param act_list: List of actual points
    :return: RMSE: Float of the RMSE describing the two input data lists
    """

    diff_square = list()
    nan_count = 0
    for i in range(len(pred_list)):
        temp_diff_square = (pred_list[i] - act_list[i])**2
        if np.isnan(temp_diff_square) is True:
            nan_count += 1
        diff_square.append(temp_diff_square)
    sum_diff_square = np.nansum(diff_square)
    RMSE = np.sqrt(sum_diff_square/(len(pred_list)-nan_count))

    return RMSE


def calc_RMSpE(pred_list, act_list):
    """
    Calculate the root mean square percent error.
    :param pred_list: List of predicted points
    :param act_list: List of actual points
    :return: RMSE: Float of the RMSpE describing the two input data lists
    """

    # Initialize Input Variables
    RMSE = calc_RMSE(pred_list,act_list)
    RMSpE = RMSE/np.mean(act_list)*100

    return RMSpE


def get_individual_score(individual,vehicle_data,log_file = None):
    """
    Generate the score for an individual.
        According to literature, it is best to calibrate a model based on the RMSE of spacing;
        therefore, the RMSE_dX was chosen as the fitness function used to evaluate the score
        of each individual.
    :param individual: Array of model parameters: [t_rxn, V_des, a_des, d_des, d_lead, g_min]
    :param vehicle_data: Instance of Processed Data Class with vehicle trajectory data
    :param log_file: (optional) CSV File variable with filename.csv and save path
    = open(os.path.join(filename.csv,path),'w') & If included will provide descriptive information about
    each individual's parameter configuration.
    :return: Individual Score = RMSE_dX: Float value indicating the individual score
    """

    t_rxn,V_f,a_f,b_f,b_l,g_min = individual

    # Calculate the number of points ahead that will be predicted (data collected at 10Hz):
    data_collection_rate = 0.1
    prediction_steps = int(t_rxn/data_collection_rate)

    # Create lists of predicted and actual following vehicle velocity and vehicle separation distance:
    v_f_pred_list = list(); v_f_act_list = list()
    dX_pred_list = list(); dX_act_list = list()
    for i in range(len(vehicle_data.dX)-prediction_steps):
        # Initialize Variables for Clarity
        v_following_this = vehicle_data.v_following[i]
        v_following_next = vehicle_data.v_following[i+prediction_steps]
        v_lead_this = vehicle_data.v_target[i]
        v_lead_next = vehicle_data.v_target[i+prediction_steps]
        dX_this = vehicle_data.dX[i]
        dX_next = vehicle_data.dX[i+prediction_steps]

        # Predict Following Velocity
        v_f_pred_temp = predict_v_f(individual,v_following_this,v_lead_this,dX_this)
        v_f_pred_list.append(v_f_pred_temp)
        v_f_act_list.append(v_following_next)

        # Calculate New/Predicted Spacing
        acc_lead_temp = (v_lead_next-v_lead_this)/t_rxn
        dist_lead_temp = v_lead_this*t_rxn+0.5*acc_lead_temp*t_rxn**2
        acc_foll_temp = (v_f_pred_temp-v_following_this)/t_rxn
        dist_foll_temp = v_following_this*t_rxn+0.5*acc_foll_temp*t_rxn**2
        dX_pred_temp = round(dX_this - dist_foll_temp + dist_lead_temp,3)

        dX_pred_list.append(dX_pred_temp)
        dX_act_list.append(dX_next)

    RMSE_v = calc_RMSE(v_f_pred_list, v_f_act_list)
    RMSpE_v = calc_RMSpE(v_f_pred_list, v_f_act_list)

    RMSE_dx = calc_RMSE(dX_pred_list, dX_act_list)
    RMSpE_dx = calc_RMSpE(dX_pred_list, dX_act_list)

    if log_file is not None:
        log_file.write("\n"); log_file.write("\n")
        log_file.write("{}".format(individual))
        log_file.write("\n")
        log_file.write("RMSE_v: {} m/s & RMSpE_v: {}%".format(round(RMSE_v,2), round(RMSpE_v,2)))
        log_file.write("\n")
        log_file.write("RMSE_dx: {} m & RMSpE_dx: {}%".format(round(RMSE_dx,2), round(RMSpE_dx,2)))
        log_file.write("\n")
        print individual
        print "RMSE_v: {} m/s & RMSpE_v: {}%".format(round(RMSE_v,2), round(RMSpE_v,2))
        print "RMSE_dx: {} m & RMSpE_dx: {}%".format(round(RMSE_dx,2), round(RMSpE_dx,2))
        print " "

    return RMSE_dx


def get_population_score(population, vehicle_data):
    """
    Generates the score of a population by collecting all of the individual scores.
    Population score = MEAN of Individual Scores.

    :param population: List of individuals
    :param vehicle_data: Instance of Processed Data Class with vehicle trajectory data
    :return: Float value indicating the population score
    """

    individual_scores = list()
    for i in range(len(population)):
        individual_scores.append(get_individual_score(population[i], vehicle_data))
    population_score = np.mean(individual_scores)

    return population_score


def evolve(population, vehicle_data, retain, random_select, random_mutate, log_file=None):
    """
    Evolution process transforming a population for one generation.
    The evolution process for one generation:
        1. Mutation
        2. Parent Selection (elitism and diversity)
        3. Cross Over (multi-point and arithmetic recombination)
        4. Survivor Selection (elitism)
    :param population: List of individuals
    :param vehicle_data: Instance of Processed Data Class with vehicle trajectory data
    :param retain: Float indicating the exact percent of best individuals (lowest scores) retained in each
    population as parents
    :param random_select: Float indicating the desired percent of random individuals (from the remaining individuals)
    selected as parents for diversity
    :param random_mutate: Float indicating the desired percent of individual parameters selected for mutation
    :param log_file: (optional) CSV File variable with filename.csv and save path
    = open(os.path.join(filename.csv,path),'w') & If included will provide descriptive information about
    each individual's parameter configuration.
    :return: survived_pop: List of individuals to move on to the next generation
    """
    # Mutation:
    # ----------------------------------------------------------------------------------------
    # For each individual, use a random number generator to determine whether each parameter should be mutated
    # therefore, if the random float between [0,1] is less than the "mutate" parameter, mutation will occur.
    no_dyn_param = len(population[0])  # Number of dynamic or calibratable parameters [t_rxn, V_des, a_des, d_des, d_lead, g_min]
    for i in range(len(population)):
        for j in range(random.randint(1, no_dyn_param)):
            if random_mutate > random.random():
                param_to_mutate = j
                population[i] = mutate_by_parameter(population[i], param_to_mutate)

    # Parent Selection:
    # ----------------------------------------------------------------------------------------
    # Create a list with all individuals and fitness scores
    pop_list_score_indiv = list()
    for i in range(len(population)):
        pop_list_score_indiv.append([get_individual_score(population[i], vehicle_data), population[i]])

    # Sort Individuals by Individual Score
    pop_list_score_indiv = sorted(pop_list_score_indiv, key=itemgetter(0))

    # Retain exact percentage of elite individuals
    retain_length = int(len(pop_list_score_indiv) * retain)
    parents = list()
    for i in range(retain_length):
        parents.append(pop_list_score_indiv[i][1])

    # Aggregate individuals not retained as parents
    non_parents = list()
    for i in range(len(pop_list_score_indiv) - retain_length):
        non_parents.append(pop_list_score_indiv[i + retain_length][1])

    # Randomly add other individuals to promote genetic diversity
    for i in range(len(non_parents)):
        if random_select > random.random():
            parents.append(non_parents[i])

    # Crossover (Generate Children):
    # ----------------------------------------------------------------------------------------
    # Crossover parents to create a new population of children - more children are created than are needed
    # to complete the population (parents + children); this way survivors can be selected using elitism
    # in the following step.
    no_children = len(population)
    children = list()
    while len(children) < no_children:
        # Select index of population for male and female parent
        male = random.randint(0, len(parents) - 1)
        female = random.randint(0, len(parents) - 1)
        if male != female:
            male_parent = parents[male]
            female_parent = parents[female]
            child = list()
            for k in range(len(population[0])):
                random_temp = random.random()
                if random_temp < (1 / 3):
                    child.append(male_parent[k])
                elif random_temp < (2 / 3):
                    child.append(female_parent[k])
                else:
                    average = round((male_parent[k] + female_parent[k]) / 2, 1)
                    child.append(average)
            children.append(child)
    # Add Children to Population
    parents.extend(children)

    # Survivor Selection:
    # ----------------------------------------------------------------------------------------
    # Select Survivors to continue to the next population using elitism
    pop_list_score_indiv_new = list()
    for i in range(len(parents)):
        pop_list_score_indiv_new.append([get_individual_score(parents[i], vehicle_data), parents[i]])

    # Sort Scored List by Score Value
    score_sorted_survived = sorted(pop_list_score_indiv_new, key=itemgetter(0))

    # Create Individuals Survived
    survived_pop = list()
    for i in range(len(population)):
        survived_pop.append(score_sorted_survived[i][1])

    return survived_pop


def mutate_by_parameter(individual,param_to_mutate):
    """
    Generates a random (mutated) value for one of the "calibratable/dynamic" parameters within the
    individual.
    :param individual: Array of model parameters: [zero,G_s,G_c,G_min,G_max,G_cfmax,v_a,v_s]
    :param param_to_mutate: Integer [0,1,2,3,4] that indicates which calibratable/dynamic parameter within the
    individual should be mutated
    :return: Updated Individual: Array of model parameters: [t_rxn,V_des,a_des,d_des,d_lead,g_min]
    """
    t_rxn,V_des,a_des,d_des,d_lead,g_min = individual

    if param_to_mutate == 0:  # t_rxn
        t_rxn = round((random.random()*(3-0.4)+0.4),1)
    elif param_to_mutate == 1:  # V_d
        V_des = round((random.random()*(40-10)+10),1)
    elif param_to_mutate == 2:  # a_d
        a_des = round((random.random()*(2.6-0.8)+0.8),1)
    elif param_to_mutate == 3:  # d_d
        d_des = round((random.random()*-(5.2-1.6)-1.6),1)
    elif param_to_mutate == 4:  # d_l
        d_lead = round((random.random()*-(4.5-3)-3),1)
    elif param_to_mutate == 5:  # S_gap
        g_min = round((random.random()*(4-2)+2),1)

    individual = [t_rxn,V_des,a_des,d_des,d_lead,g_min]

    return individual


def get_unique_individuals(population,log_file):
    """
    Identify the number of unique individuals within a population
    :param population: List of individuals
    :param log_file: CSV File variable with filename.csv and save path = open(os.path.join(filename.csv,path),'w')
    :return: List of the unique individuals within a population
    - Also printing the unique individuals to the log_file
    """
    # Identify the number of unique individuals remaining:
    unique_individuals = list()
    for i in range(len(population)):
        if population[i] not in unique_individuals:
            unique_individuals.append(population[i])
    print 'Unique Individuals in Final Population'
    log_file.write('\n'); log_file.write('\n')
    log_file.write('Unique Individuals in Final Population'); log_file.write('\n')
    for i in range(len(unique_individuals)):
        print unique_individuals[i]
        log_file.write('{}'.format(unique_individuals[i])); log_file.write('\n')
    print " "  # Formatting

    return unique_individuals


def get_best_individual(population,vehicle_data,log_file):
    """
    Extract individual from a population with the lowest individual score
    :param population: List of individuals
    :param vehicle_data: Instance of Processed Data Class with vehicle trajectory data
    :param log_file: CSV File variable with filename.csv and save path = open(os.path.join(filename.csv,path),'w')
    :return: Individual [t_rxn, V_f, a_f, b_f, b_l, g_min] with the lowest score
    """
    # Identify Individual with lowest score in the best population:
    min_score = np.inf
    min_index = np.nan
    for i in range(len(population)):
        score_temp = get_individual_score(population[i],vehicle_data,log_file)
        if score_temp < min_score:
            min_score = deepcopy(score_temp)
            min_index = deepcopy(i)
    low_indiv = population[min_index]
    return low_indiv


def plot_GA(plot_name,pop_score_history,fig_file=None):
    """
    Produces plots related to an entire population of individuals.
    :param plot_name: Select 'pop_score_history' or ...
    :param population: List of individuals
    :param pop_score_history: List of population scores associated with each generation
    :param fig_file: fig_file: File variable with filename and save path = os.path.join(filename,path)
    :return: No Return - if no fig_file provided, opens figure on screen
    """

    if plot_name == 'pop_score_history':
        fig = plt.figure(figsize=(16, 12))  # Size of figure
        fig.suptitle('GA Optimization: FHWA Driver Model', fontsize=18, fontweight='bold')
        plt.plot(pop_score_history)
        plt.xlabel('Generation')
        plt.ylabel('Grade')
        if fig_file is not None:
            fig.savefig(fig_file)
            fig.clf()
        else:
            plt.show()
    else:
        print "Plot Name Not Recognized"


def main():
    """
    Main function for running as a standalone script
    :return: N/A
    """
    #############################################################################################################
    #############################################################################################################
    # User Inputs
    #############################################################################################################
    #############################################################################################################

    # Initialize the test run parameters:
    # ----------------------------------------------------------------------------------------
    no_runs = 3  # Integer - indicating the number of iterations of each combination of the following variables
    # Initialize the genetic algorithm parameters
    pop_size = [10]  # List of integers - indicating the population size or number of individuals
    degrade = [2]  # List of integers - indicating the number of allowable consecutive generations with degrading population scores
    restart = [1]  # List of integers - indicating the number of allowable restarts at the best population
    p_retain = [0.2]  # List of floats - indicating the percent of the population that should be retained due to elitism
    random_select = [0.2]  # List of floats - indicating the approximate percent of the population that should be retained due to diversity
    random_mutate = [0.001]  # List of floats - indicating the approximate percent of the parameters within the individuals of a population that should be mutated

    # Input files names and paths:
    # ----------------------------------------------------------------------------------------
    # input_filename = 'wzdata_fw1_PC-hw-c_2017-06-12.csv'
    # input_source = 'fhwa_irv'
    input_filename = 'Event_ID_152218624.csv'
    input_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\02-NDS_Timeseries_Files'
    input_path = r'C:\Users\Britton\Dropbox\00-Education\Research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\02-NDS_Timeseries_Files'
    input_source = 'wy_nds_stac'
    stac_filename = '5228273_7_data_partition_edit.csv'
    stac_path = input_path

    # Identify save paths:
    # ----------------------------------------------------------------------------------------
    # Identify log path:
    log_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\07-Results_CF\Gipps Calibration'
    log_path = r'C:\Users\Britton\Dropbox\00-Education\Research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\07-Results_CF\Gipps Calibration'
    # Identify summary file path and file name
    summary_file_path = log_path
    summary_filename = 'summary.csv'

    #############################################################################################################
    #############################################################################################################
    # End User Inputs
    #############################################################################################################
    #############################################################################################################

    # Initialize summary file:
    # ----------------------------------------------------------------------------------------
    summary_file = open(os.path.join(summary_file_path, summary_filename), 'a')
    # Indicate fitness function logic in the summary file
    summary_file.write(
        'Test Number,Population Size,%Retain,%Random,%Mutate,Degrage,Restart,Computation Time,Number of Generations,'
        'Population Ave Score,Best Individual Score,Number of Unique Individuals, Best: T_rxn,'
        'Best: V_des, Best: a_des, Best: d_des, Best: d_lead, Best: g_min')
    summary_file.write('\n')
    summary_file.close()

    # Generate vehicle data
    # ----------------------------------------------------------------------------------------
    timestamp = int(time.time())  # Unique identifier for logging test
    vehicle_data = import_cf_data(input_source=input_source, input_filename=input_filename, input_path=input_path,
            log_path=log_path, timestamp=timestamp, stac_filename=stac_filename, stac_path=stac_path)

    # Perform the desired test runs:
    no_tests = len(pop_size) * len(degrade) * len(random_mutate) * len(p_retain) * len(restart) * no_runs
    test_number = 1
    for i in range(len(pop_size)):
        for j in range(len(degrade)):
            for k in range(len(random_mutate)):
                for m in range(len(random_select)):
                    for n in range(len(p_retain)):
                        for o in range(len(restart)):
                            for p in range(no_runs):
                                print "Test Number: {}/{}".format(test_number, no_tests)
                                # Initiate variables:
                                # ---------------------------------------------------------------------
                                # GA variables:
                                pop_size_this = pop_size[i]
                                degrade_this = degrade[j]
                                restart_this = restart[o]
                                p_retain_this = p_retain[n]
                                random_select_this = random_select[m]
                                random_mutate_this = random_mutate[k]
                                # File names:
                                log_filename = '{}__{}_{}_{}_{}_{}_{}_gippsCFM_GA_log_file.csv'.format(test_number,
                                    pop_size_this, degrade_this, restart_this, int(p_retain_this * 100),
                                    int(random_select_this * 100),int(random_mutate_this * 100))
                                log_file = open(os.path.join(log_path, log_filename), 'w')
                                fig_filename_pop_score = '{}__{}_{}_{}_{}_{}_{}_gippsCFM_GA_pop_score'.format(
                                    test_number, pop_size_this, degrade_this,
                                    restart_this, int(p_retain_this * 100), int(random_select_this * 100),
                                    int(random_mutate_this * 100))
                                fig_file_pop_score = os.path.join(log_path, fig_filename_pop_score)

                                # Provide description of test progress to python console
                                # ---------------------------------------------------------------------
                                print "    Pop Size: {}".format(pop_size_this)
                                print "    Allowed Degradation: {}".format(degrade_this)
                                print "    Allowed Restarts: {}".format(restart_this)
                                print "    Percent Retain: {}".format(p_retain_this)
                                print "    Random Selection: {}".format(random_select_this)
                                print "    Random Mutation: {}".format(random_mutate_this)
                                print " "

                                # Run Genetic Algorithm:
                                # ---------------------------------------------------------------------
                                best_population, run_time, no_generations = run_GA(pop_size=pop_size_this,
                                                                                     vehicle_data=vehicle_data,
                                                                                     degrade=degrade_this,
                                                                                     restart=restart_this,
                                                                                     p_retain=p_retain_this,
                                                                                     random_select=random_select_this,
                                                                                     random_mutate=random_mutate_this,
                                                                                     log_file=log_file,
                                                                                     fig_file_pop_score=fig_file_pop_score)

                                # Descriptive information:
                                # ---------------------------------------------------------------------
                                best_population_score = get_population_score(population=best_population,
                                                                             vehicle_data=vehicle_data)
                                unique_individuals = get_unique_individuals(population=best_population,
                                                                            log_file=log_file)
                                no_unique_individuals = len(unique_individuals)
                                best_individual = get_best_individual(population=unique_individuals,
                                                                      vehicle_data=vehicle_data, log_file=log_file)
                                best_individual_score = get_individual_score(individual=best_individual,
                                                                             vehicle_data=vehicle_data)
                                # Record information to the summary file
                                summary_file = open(os.path.join(log_path, summary_filename), 'a')
                                summary_file.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},'
                                                   .format(test_number, pop_size_this, p_retain_this,
                                                                  random_select_this,
                                                                  random_mutate_this, degrade_this, restart_this,
                                                                  run_time, no_generations,
                                                                  best_population_score, best_individual_score,
                                                                  no_unique_individuals,
                                                                  best_individual[0], best_individual[1],
                                                                  best_individual[2], best_individual[3],
                                                                  best_individual[4], best_individual[5]))
                                summary_file.write('\n')
                                summary_file.close()
                                log_file.close()

                                test_number += 1

##############################################################
# Run main() if script is being run directly
if __name__ == "__main__":
    main()