import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import gurobipy_pandas as gppd
from gurobi_ml import add_predictor_constr
import gurobipy as gp
from optimization_engine import optimization_engine
import plotly.graph_objects as go

################################# set page configuration #################################
st.set_page_config(layout="wide")


################################# Read env variables #################################
import os

# package used in jupyter notebook to read the variables in file .env
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

#Read env variables and save it as python variable
WLSACCESSID = os.environ.get("WLSACCESSID", "")
WLSSECRET = os.environ.get("WLSSECRET", "")
LICENSEID = int(os.environ.get("LICENSEID", ""))
PROJECT_GCP = os.environ.get("PROJECT_GCP", "")


################################# LOAD LICENCE GUROBI - using env variables #################################
params = {
"WLSACCESSID": WLSACCESSID,
"WLSSECRET": WLSSECRET,
"LICENSEID": LICENSEID
}
env = gp.Env(params=params)


######################## ORDER CODES THAT SHOW INFORMATION IN THE UI ########################
if __name__ == "__main__":


    ######################## ------------------------------------- FORM TO INPUT VALUES OF OPTIMIZER - SIDEBAR ------------------------------------- ########################
    with st.form(key ='Form1'):
        with st.sidebar:
            st.header('----- INPUT PARAMS TO RUN OPTIMIZATION -----')
            
            ############## PARAMETERS OF OPTIMIZATION PROBLEM ##############
            st.divider()
            col1_sidebar, col2_sidebar, col3_sidebar, col4_sidebar = st.columns(4)

            ### COLUMN 1 - lower bound
            col1_sidebar.write('**- lower bound -**')
            #col1_sidebar.write('**Input VNC**')
            input_X1_lower_bound = col1_sidebar.number_input("X1", min_value = 0.0, max_value = 10000.0, value = 0.0) # input_X1_lower_bound = 0
            input_Y1_lower_bound = col1_sidebar.number_input("Y1", min_value = 0.0, max_value = 10000.0, value = 0.0) # input_Y1_lower_bound = 0
            input_Z1_lower_bound = col1_sidebar.number_input("Z1", min_value = 0.0, max_value = 10000.0, value = 0.0) # input_Z1_lower_bound = 0
            input_X2_lower_bound = col1_sidebar.number_input("X2", min_value = 0.0, max_value = 10000.0, value = 0.0) # input_X2_lower_bound = 0
            input_Y2_lower_bound = col1_sidebar.number_input("Y2", min_value = 0.0, max_value = 10000.0, value = 0.0) # input_Y2_lower_bound = 0
            input_Y3_lower_bound = col1_sidebar.number_input("Y3", min_value = 0.0, max_value = 10000.0, value = 0.0) # input_Y3_lower_bound = 0
            input_X3_lower_bound = col1_sidebar.number_input("X3", min_value = 0.0, max_value = 10000.0, value = 0.0) # input_X3_lower_bound = 0
            input_TL1_lower_bound = col1_sidebar.number_input("TL1", min_value = 0.0, max_value = 10000.0, value = 0.0) # input_TL1_lower_bound = 100
            input_TL2_lower_bound = col1_sidebar.number_input("TL2", min_value = 0.0, max_value = 10000.0, value = 0.0) # input_TL2_lower_bound = 100
            input_TL3_lower_bound = col1_sidebar.number_input("TL3", min_value = 0.0, max_value = 10000.0, value = 0.0) # input_TL3_lower_bound = 100



            ### COLUMN 2 - upper bound
            col2_sidebar.write('**- upper bound -**')
            #col2_sidebar.write('**Input VNC**')
            input_X1_upper_bound = col2_sidebar.number_input("X1", min_value = 0.0, max_value = 10000.0, value = 1000.0) # input_X1_upper_bound = 1000
            input_Y1_upper_bound = col2_sidebar.number_input("Y1", min_value = 0.0, max_value = 10000.0, value = 400.0) # input_Y1_upper_bound = 400
            input_Z1_upper_bound = col2_sidebar.number_input("Z1", min_value = 0.0, max_value = 10000.0, value = 1000.0) # input_Z1_upper_bound = 1000
            input_X2_upper_bound = col2_sidebar.number_input("X2", min_value = 0.0, max_value = 10000.0, value = 1000.0) # input_X2_upper_bound = 1000
            input_Y2_upper_bound = col2_sidebar.number_input("Y2", min_value = 0.0, max_value = 10000.0, value = 500.0) # input_Y2_upper_bound = 500
            input_Y3_upper_bound = col2_sidebar.number_input("Y3", min_value = 0.0, max_value = 10000.0, value = 450.0) # input_Y3_upper_bound = 450
            input_X3_upper_bound = col2_sidebar.number_input("X3", min_value = 0.0, max_value = 10000.0, value = 1500.0) # input_X3_upper_bound = 1500
            input_TL1_upper_bound = col2_sidebar.number_input("TL1", min_value = 0.0, max_value = 100000.0, value = 20000.0) # input_TL1_upper_bound = 20000
            input_TL2_upper_bound = col2_sidebar.number_input("TL2", min_value = 0.0, max_value = 100000.0, value = 20000.0) # input_TL2_upper_bound = 20000
            input_TL3_upper_bound = col2_sidebar.number_input("TL3", min_value = 0.0, max_value = 100000.0, value = 20000.0) # input_TL3_upper_bound = 20000



            ### COLUMN 3 - rate change
            col3_sidebar.write('**- rate change -**')
            #col3_sidebar.write('**Input VNC**')
            input_X1_rate_change = col3_sidebar.number_input("X1", min_value = 0.0, max_value = 10000.0, value = 100.0) # input_X1_rate_change = 100
            input_Y1_rate_change = col3_sidebar.number_input("Y1", min_value = 0.0, max_value = 10000.0, value = 100.0) # input_Y1_rate_change = 100
            input_Z1_rate_change = col3_sidebar.number_input("Z1", min_value = 0.0, max_value = 10000.0, value = 100.0) # input_Z1_rate_change = 100
            input_X2_rate_change = col3_sidebar.number_input("X2 ", min_value = 0.0, max_value = 10000.0, value = 100.0) # input_X2_rate_change = 100
            input_Y2_rate_change = col3_sidebar.number_input("Y2", min_value = 0.0, max_value = 10000.0, value = 100.0) # input_Y2_rate_change = 100
            input_Y3_rate_change = col3_sidebar.number_input("Y3", min_value = 0.0, max_value = 10000.0, value = 100.0) # input_Y3_rate_change = 100
            input_X3_rate_change = col3_sidebar.number_input("X3", min_value = 0.0, max_value = 10000.0, value = 100.0) # input_X3_rate_change = 100
            input_TL1_rate_change = col3_sidebar.number_input("TL1", min_value = 0.0, max_value = 10000.0, value = 100.0) # input_TL1_rate_change = 100
            input_TL2_rate_change = col3_sidebar.number_input("TL2", min_value = 0.0, max_value = 10000.0, value = 100.0) # input_TL2_rate_change = 100
            input_TL3_rate_change = col3_sidebar.number_input("TL3", min_value = 0.0, max_value = 10000.0, value = 100.0) # input_TL3_rate_change = 100


            ##### COLUMN 3 - initial values
            col4_sidebar.write('**- initial values -**')
            #col4_sidebar.write('**Delta VC**')
            input_X1_initial = col4_sidebar.number_input("X1", min_value = 0.0, max_value = 10000.0, value = 50.0) # input_X1_initial = 50
            input_O1_initial = col4_sidebar.number_input("O1", min_value = 0.0, max_value = 10000.0, value = 50.0) # input_O1_initial = 50
            input_O2_initial = col4_sidebar.number_input("O2", min_value = 0.0, max_value = 10000.0, value = 50.0) # input_O2_initial = 50
            input_O3_initial = col4_sidebar.number_input("O3", min_value = 0.0, max_value = 10000.0, value = 50.0) # input_O3_initial = 50
            input_O4_initial = col4_sidebar.number_input("O4", min_value = 0.0, max_value = 10000.0, value = 200.0) # input_O4_initial = 200
            input_X2_initial = col4_sidebar.number_input("X2", min_value = 0.0, max_value = 10000.0, value = 5.0) # input_X2_initial = 5
            input_O5_initial = col4_sidebar.number_input("O5", min_value = 0.0, max_value = 10000.0, value = 5.0) # input_O5_initial = 5
            input_O6_initial = col4_sidebar.number_input("O6", min_value = 0.0, max_value = 10000.0, value = 5.0) # input_O6_initial = 5
            input_X3_initial = col4_sidebar.number_input("X3", min_value = 0.0, max_value = 10000.0, value = 5.0) # input_X3_initial = 5
            input_O7_initial = col4_sidebar.number_input("O7", min_value = 0.0, max_value = 10000.0, value = 4.0) # input_O7_initial = 4
            input_TL1_initial = col4_sidebar.number_input("TL1", min_value = 0.0, max_value = 10000.0, value = 500.0) # input_TL1_initial = 500
            input_TL2_initial = col4_sidebar.number_input("TL2", min_value = 0.0, max_value = 10000.0, value = 500.0) # input_TL2_initial = 500
            input_TL3_initial = col4_sidebar.number_input("TL3", min_value = 0.0, max_value = 10000.0, value = 500.0) # input_TL3_initial = 500


            ############## SUBMIT BUTTON ##############
            submitted_opt = st.form_submit_button(label = 'Run Optimization')




    ######################## ------------------------- RUN OPTIMIZATION WHEN USER SEND THE NEW VALUES OF OPTIMIZATION ------------------------- ########################
    if submitted_opt:

        ############################################# generate initial values - user input

        # read original initial values - default excel file
        path_folder_config_optimization = f'config/optimization_engine/config_optimization/'
        file_initvalues = 'InitialValues.xlsx'
        path_initvalues = path_folder_config_optimization + file_initvalues
        config_initvalues = pd.read_excel(path_initvalues)

        # update values with the input values
        config_initvalues.loc[config_initvalues['feature_name'] == 'X1', 'init_values'] = input_X1_initial
        config_initvalues.loc[config_initvalues['feature_name'] == 'O1', 'init_values'] = input_O1_initial
        config_initvalues.loc[config_initvalues['feature_name'] == 'O2', 'init_values'] = input_O2_initial
        config_initvalues.loc[config_initvalues['feature_name'] == 'O3', 'init_values'] = input_O3_initial
        config_initvalues.loc[config_initvalues['feature_name'] == 'O4', 'init_values'] = input_O4_initial
        config_initvalues.loc[config_initvalues['feature_name'] == 'X2', 'init_values'] = input_X2_initial
        config_initvalues.loc[config_initvalues['feature_name'] == 'O5', 'init_values'] = input_O5_initial
        config_initvalues.loc[config_initvalues['feature_name'] == 'O6', 'init_values'] = input_O6_initial
        config_initvalues.loc[config_initvalues['feature_name'] == 'X3', 'init_values'] = input_X3_initial
        config_initvalues.loc[config_initvalues['feature_name'] == 'O7', 'init_values'] = input_O7_initial
        config_initvalues.loc[config_initvalues['feature_name'] == 'TL1', 'init_values'] = input_TL1_initial
        config_initvalues.loc[config_initvalues['feature_name'] == 'TL2', 'init_values'] = input_TL2_initial
        config_initvalues.loc[config_initvalues['feature_name'] == 'TL3', 'init_values'] = input_TL3_initial



        ############################################# generate Lower bound, upper bound and rate change - user input
        # read original initial values - default excel file
        path_folder_config_optimization = f'config/optimization_engine/config_optimization/'
        file_allvariables = 'AllVariables.xlsx'
        path_allvariables = path_folder_config_optimization + file_allvariables
        config_allvariables = pd.read_excel(path_allvariables)

        # update values with the input values
        # X1
        config_allvariables.loc[config_initvalues['feature_name'] == 'X1', 'lower'] = input_X1_lower_bound
        config_allvariables.loc[config_initvalues['feature_name'] == 'X1', 'upper'] = input_X1_upper_bound
        config_allvariables.loc[config_initvalues['feature_name'] == 'X1', 'rate_change'] = input_X1_rate_change

        # Y1
        config_allvariables.loc[config_initvalues['feature_name'] == 'Y1', 'lower'] = input_Y1_lower_bound
        config_allvariables.loc[config_initvalues['feature_name'] == 'Y1', 'upper'] = input_Y1_upper_bound
        config_allvariables.loc[config_initvalues['feature_name'] == 'Y1', 'rate_change'] = input_Y1_rate_change

        # Z1
        config_allvariables.loc[config_initvalues['feature_name'] == 'Z1', 'lower'] = input_Z1_lower_bound
        config_allvariables.loc[config_initvalues['feature_name'] == 'Z1', 'upper'] = input_Z1_upper_bound
        config_allvariables.loc[config_initvalues['feature_name'] == 'Z1', 'rate_change'] = input_Z1_rate_change

        # X2
        config_allvariables.loc[config_initvalues['feature_name'] == 'X2', 'lower'] = input_X2_lower_bound
        config_allvariables.loc[config_initvalues['feature_name'] == 'X2', 'upper'] = input_X2_upper_bound
        config_allvariables.loc[config_initvalues['feature_name'] == 'X2', 'rate_change'] = input_X2_rate_change

        # Y2
        config_allvariables.loc[config_initvalues['feature_name'] == 'Y2', 'lower'] = input_Y2_lower_bound
        config_allvariables.loc[config_initvalues['feature_name'] == 'Y2', 'upper'] = input_Y2_upper_bound
        config_allvariables.loc[config_initvalues['feature_name'] == 'Y2', 'rate_change'] = input_Y2_rate_change

        # Y3
        config_allvariables.loc[config_initvalues['feature_name'] == 'Y3', 'lower'] = input_Y3_lower_bound
        config_allvariables.loc[config_initvalues['feature_name'] == 'Y3', 'upper'] = input_Y3_upper_bound
        config_allvariables.loc[config_initvalues['feature_name'] == 'Y3', 'rate_change'] = input_Y3_rate_change

        # X3
        config_allvariables.loc[config_initvalues['feature_name'] == 'X3', 'lower'] = input_X3_lower_bound
        config_allvariables.loc[config_initvalues['feature_name'] == 'X3', 'upper'] = input_X3_upper_bound
        config_allvariables.loc[config_initvalues['feature_name'] == 'X3', 'rate_change'] = input_X3_rate_change

        # TL1
        config_allvariables.loc[config_initvalues['feature_name'] == 'TL1', 'lower'] = input_TL1_lower_bound
        config_allvariables.loc[config_initvalues['feature_name'] == 'TL1', 'upper'] = input_TL1_upper_bound
        config_allvariables.loc[config_initvalues['feature_name'] == 'TL1', 'rate_change'] = input_TL1_rate_change

        # TL2
        config_allvariables.loc[config_initvalues['feature_name'] == 'TL2', 'lower'] = input_TL2_lower_bound
        config_allvariables.loc[config_initvalues['feature_name'] == 'TL2', 'upper'] = input_TL2_upper_bound
        config_allvariables.loc[config_initvalues['feature_name'] == 'TL2', 'rate_change'] = input_TL2_rate_change

        # TL3
        config_allvariables.loc[config_initvalues['feature_name'] == 'TL3', 'lower'] = input_TL3_lower_bound
        config_allvariables.loc[config_initvalues['feature_name'] == 'TL3', 'upper'] = input_TL3_upper_bound
        config_allvariables.loc[config_initvalues['feature_name'] == 'TL3', 'rate_change'] = input_TL3_rate_change



        ############################################# LOAD CONFIGURATION FILES FOR OPTIMIZER
        #################### 1.1 IndexTime file
        ## paths and files names
        path_folder_config_optimization = f'config/optimization_engine/config_optimization/'
        file_indextime = 'IndexTime.xlsx'
        path_indextime = path_folder_config_optimization + file_indextime
        
        # read file
        indextime = pd.read_excel(path_indextime)
        
        # set index
        index_set_time = pd.Index(indextime['IndexTime'].values)


        #################### 1.2 Decision variables
        # CREATED USING USER INPUTS


        #################### 1.3 Initial Values
        # CREATED USING USER INPUTS


        #################### 1.4 Define models to load
        # paths and file names
        path_folder_config_optimization = f'config/optimization_engine/config_optimization/'
        file_modelsml = 'ModelsML.xlsx'
        path_modelsml = path_folder_config_optimization + file_modelsml

        # read file
        config_modelsml = pd.read_excel(path_modelsml)


        #################### 1.5 Map tanks
        # paths and file names
        path_folder_config_optimization = f'config/optimization_engine/config_optimization/'
        file_maptanks = 'MapTanks.xlsx'
        path_maptanks = path_folder_config_optimization + file_maptanks

        # read file
        config_maptanks = pd.read_excel(path_maptanks)


        #################### 1.6 Map process Machine learning models features and target
        # paths and file names
        path_folder_config_optimization = f'config/optimization_engine/config_optimization/'
        file_mapprocess_mlmodels = 'MapProcessMLmodels.xlsx'
        path_mapprocess_mlmodels = path_folder_config_optimization + file_mapprocess_mlmodels

        # read filemapprocess_mlmodels
        config_mapprocess_mlmodels = pd.read_excel(path_mapprocess_mlmodels)




        ############################################# RUN OPTIMIZATION
        #################### i) Define parameters used to relax constraints (if it is necessary)
        # define list of decision vars that will have change in its rate change
        list_tags_to_relax_constraints = ['TL1', 'TL2', 'TL3']

        # define factor (percent = 0.1) to change the rate change of the decision vars. 
        # In this example all the decision var masked will change its value in the same factor
        param_factor_relax_constraints = 0.1
        factor_relax_constraints = 1 + param_factor_relax_constraints


        #################### ii) run optimization
        ######## do while. do define status solver // while status solver != 2 relaxing contrainsts (rate of change decision variables) until get a solution
        # initialize status
        status_solver = 0

        # initialize count iterations of relaxing constraints
        index_count_while = 0
        max_interations_relaxing_constraints = 40

        # initiliaze values of constraints to relax
        to_solver_config_allvariables = config_allvariables.copy()



        while (status_solver != 2) & (index_count_while <=max_interations_relaxing_constraints):
            # print('\n\n\n index while: ', index_count_while)

            ###### run optimization
            model_opt, status_solver, decision_var = optimization_engine(index_set_time,
                                            config_allvariables,
                                            config_initvalues,
                                            config_modelsml,
                                            config_maptanks,
                                            config_mapprocess_mlmodels,
                                            params)

            ##### get status solver - if solver get a solution - break the while
            if status_solver == 2:
                break
            
            # ###### relaxing constraints. if the solver return a value this values was delete, else the relaxing constraints are used in the while to get a solution
            # multiply rate change by factor of selected features
            mask = config_allvariables['feature_name'].isin(list_tags_to_relax_constraints)
            config_allvariables.loc[mask, 'rate_change'] = config_allvariables[mask]['rate_change'] * factor_relax_constraints

            # increse while
            index_count_while += 1


        # check status solver
        print('status solver: ' , status_solver)
        if status_solver != 2:
            print('Infeasible solution')


        # revisar qué valores de rate change quedan de las variables
        mask = config_allvariables['feature_name'].isin(list_tags_to_relax_constraints)
        rate_change_solution = config_allvariables[mask]





    ######################## ------------------------------------- INFO IN MAIN PAGE ------------------------------------- ########################


    # two tabs - first one show results - second show detail of model
    tab1, tab2 = st.tabs(["Results Optimization", "Details Optimization Modeling"])

    #### COLUMN1

    #tab1.markdown("### ----- RESULTS OPTIMIZATION -----")
    if submitted_opt:
        if status_solver == 2: # optimal solution was founded


            ############################################# get optimal values and save in a dataframe
            # create a dataframe with set as index
            solution = pd.DataFrame(index = index_set_time)

            # save optimal values - features of models (only the features)
            solution["var_X1"] = decision_var['X1'].gppd.X
            solution["var_O1"] = decision_var['O1'].gppd.X
            solution["var_O2"] = decision_var['O2'].gppd.X
            solution["var_O3"] = decision_var['O3'].gppd.X
            solution["var_Y1"] = decision_var['Y1'].gppd.X
            solution["var_O4"] = decision_var['O4'].gppd.X
            solution["var_Z1"] = decision_var['Z1'].gppd.X
            solution["var_X2"] = decision_var['X2'].gppd.X
            solution["var_O5"] = decision_var['O5'].gppd.X
            solution["var_O6"] = decision_var['O6'].gppd.X
            solution["var_Y2"] = decision_var['Y2'].gppd.X
            solution["var_Y3"] = decision_var['Y3'].gppd.X
            solution["var_X3"] = decision_var['X3'].gppd.X
            solution["var_O7"] = decision_var['O7'].gppd.X
            solution["var_TL1"] = decision_var['TL1'].gppd.X
            solution["var_TL2"] = decision_var['TL2'].gppd.X
            solution["var_TL3"] = decision_var['TL3'].gppd.X

            # get value objetive function
            opt_objetive_function = model_opt.ObjVal


            ############################################# show solutions
            # costs
            tab1.write('\n\n\n')
            tab1.markdown("##### ----- Objetive Optimization -----")
            tab1.write(f"\n The optimal Objetive of Optimization is: {opt_objetive_function}")
            
            # solution
            tab1.write('\n\n\n')
            tab1.markdown("##### ----- Solution Optimization -----")
            tab1.dataframe(solution)

            # download solution
            tab1.write('\n\n\n')
            tab1.markdown("##### ----- Download Optimization -----")
            final_solution = solution.to_csv()
            tab1.download_button("Download Optimal Solution", final_solution, file_name='final_solution.csv', key='csv_key')
            #os.remove('final_solution.csv')

            # show actual configuration of decision var relaxed. in this example only rate change is relaxed
            tab1.write('\n\n\n')
            tab1.markdown("##### ----- Show parameters decision var relaxed (rate change) -----")
            tab1.dataframe(rate_change_solution)


        else:
            tab1.write("Model is infeasible or unbounded - Change the input parameters")
    
    
    else: # else if the user doesn't click the button submit in the form
        pass

    #### COLUMN2
    tab2.markdown("### ----- INFO OPTIMIZATION PROBLEM -----")
    tab2.write(" to do ")















