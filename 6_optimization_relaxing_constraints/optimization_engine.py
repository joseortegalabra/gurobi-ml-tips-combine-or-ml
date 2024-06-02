import pickle
import pandas as pd
import numpy as np
import json

#gurobi
import gurobipy_pandas as gppd
from gurobi_ml import add_predictor_constr
import gurobipy as gp


def optimization_engine(index_set_time,
                        config_allvariables,
                        config_initvalues,
                        config_modelsml,
                        config_maptanks,
                        config_mapprocess_mlmodels,
                        params
                        ):
    """
    Given a certain parameters, ml models, etc. Give a optimal solution using gurobi.
    
    The optimization engine recibe the dataframes with the configuration files. This is done in this way because out of this script there are other code
    to relax the constraints (changing the values of the configuration files) and this script (optimization engine) recibe the exactly values to solve the 
    optimization problem
    
    Args:
        - index_set_time: cantidad de tiempos de la simulación
        - config_allvariables: archivo de configuración de todas las variables de decisión de optimizador (primarias, secundarias, observadas, tanques, etc), 
                limite inferior, limite superior, rate change. Muy importante indica el nombre de la variable el cual debe ser consistente en todos los archivos
        - config_initvalues: valores iniciales de las variables. Para que funcione bien el valor iniciar de las variables que son targets de modelos de ml no está fijado
        al valor inicial sino que el modelo de ML define el valor
        - config_modelsml: indica los modelos de ML a utilizar. para cada modelo indica el folder donde están los pkl de los modelos y el nombre del pkl a utilizar
            (en caso de querer utilizar diferentes modelos para ver diferentes soluciones o solucionar infactibilidades)
        - config_maptanks: mapea las variables que son de entrada, salida y nivel de cada tanque
        - config_mapprocess_mlmodels: mapea las variables que son features y targets de cada proceso (cada modelo de ML)

        - params (dictionary): with keys of licence of gurobi
    """


    #################################### 1. Load artifacts models ML - given the configuration file ####################################
    # for: for each model upload this pkl and save it into a dictionary
    # define a dictionary where the differents ml models are uploaded. the key of the names models is the column "name_process_model" (known the process
    # and the models)
    models_ml = {}
    for index_modelml in range(len(config_modelsml)):

        # get the name (ID) of ml models
        config_names_modelsml = config_modelsml.loc[index_modelml, 'name_process_model']
        print('\nname model - model id: ', config_names_modelsml)

        # define parameters to laod model. path (folder) and artifact (name file)
        path_folder = config_modelsml[config_modelsml['name_process_model'] == config_names_modelsml]['path_folder'].values[0]
        artifact_name = config_modelsml[config_modelsml['name_process_model'] == config_names_modelsml]['artifact_name'].values[0]
        extension_artifact = artifact_name.split('.')[-1]
        
        # define full path to model
        path_folder_model = f'artifacts/models/{path_folder}/'
        path_model_loaded = path_folder_model + artifact_name
        print('path model loade: ', path_folder)
        print('artifact model loaded: ', artifact_name)
        print('full path model: ', path_model_loaded)
        
        # load model pkl
        if extension_artifact == 'pkl':
            model_loaded = pd.read_pickle(path_model_loaded)
            print('loaded pkl')

        # load model excel - custom model
        if extension_artifact == 'xlsx':
            model_loaded = pd.read_excel(path_model_loaded)
            print('loaded excel')

        if extension_artifact == 'h5':
            print('keras - pass - todo')
            print('loaded h5')

        # save the model loaded into a dict
        models_ml[config_names_modelsml] = model_loaded



    #################################### 2. Create gurobi model ####################################
    env = gp.Env(params=params)
    model_opt = gp.Model('Example Optimization Model', env = env)




    #################################### 3. Create decision variables ####################################
    ##### define a for across the configuration table to create the decision vars and save it into a python dictionary
    decision_var = {}
    for index_var in range(len(config_allvariables)):

        # get config values
        config_names_decision_var = config_allvariables.loc[index_var, 'feature_name']
        config_description_decision_var = config_allvariables.loc[index_var, 'feature_name'] # use the feature name as a description in gurobi description
        print('defining decision variables: ', config_names_decision_var)

        # create decision var and save in the dictionary
        decision_var[config_names_decision_var] = gppd.add_vars(model_opt, 
                                                                index_set_time, 
                                                                name = config_description_decision_var,
                                                                #lb = -gp.GRB.INFINITY,
                                                                ub = gp.GRB.INFINITY
                                                            )




    #################################### 4. Set initial values decision variables t=0 ####################################
    # initial values decision variables - filter configuration file with only the decision var that have defined its initial values 
    # (it should be all except target variables)
    # define table with initial values (drop the nan values - decisions var not defined)
    config_initvalues_init = config_initvalues[config_initvalues['init_values'].isnull() == False]
    config_initvalues_init = config_initvalues_init.reset_index().drop(columns = 'index')

    for index_var in range(len(config_initvalues_init)):

        # get config values
        config_names_decision_var = config_initvalues_init.loc[index_var, 'feature_name']

        # get initial value
        initial_value_decision_var = config_initvalues_init[config_initvalues_init['feature_name'] == config_names_decision_var]['init_values'].values[0]

        # set initial value t=0 for all decision variables that needs this value
        print(f'set initial values decision variables: {config_names_decision_var} ||| Initial value: {initial_value_decision_var}')
        model_opt.addConstr(decision_var[config_names_decision_var]['t0'] == initial_value_decision_var,  
                            name = f'Initial Value {config_names_decision_var}')




    #################################### 5. Lower bound and Upper bound decision variables ####################################
    #################################### 5.1 Lower bound ####################################
    # lower bounds parameters - filter configuration file with only the decision var that have defined its lower bounds
    config_allvariables_lower_bounds = config_allvariables[config_allvariables['lower'].isnull() == False]
    config_allvariables_lower_bounds = config_allvariables_lower_bounds.reset_index().drop(columns = 'index') # reset index to count by index

    # generate constaint - lower bound
    for index_var in range(len(config_allvariables_lower_bounds)):

        # get config values
        config_names_decision_var = config_allvariables_lower_bounds.loc[index_var, 'feature_name']

        # get lower bound value
        lower_bound_decision_var = config_allvariables_lower_bounds[config_allvariables_lower_bounds['feature_name'] == config_names_decision_var]['lower'].values[0]

        # create contraint lowe bound
        print(f'set lower bound decision variables: {config_names_decision_var} ||| Lower bound: {lower_bound_decision_var}')
        gppd.add_constrs(model_opt, 
                        decision_var[config_names_decision_var],  # decision var
                        gp.GRB.GREATER_EQUAL,
                        lower_bound_decision_var,  # lower_bound_value
                        name = f'Lower bound {config_names_decision_var}')



    #################################### 5.2 upper bound ####################################
    # upper bounds parameters - filter configuration file with only the decision var that have defined its upper bounds
    config_allvariables_upper_bounds = config_allvariables[config_allvariables['upper'].isnull() == False]
    config_allvariables_upper_bounds = config_allvariables_upper_bounds.reset_index().drop(columns = 'index')

    # generate constaint - upper bound
    for index_var in range(len(config_allvariables_upper_bounds)):

        # get config values
        config_names_decision_var = config_allvariables_upper_bounds.loc[index_var, 'feature_name']

        # get upper bound value
        upper_bound_decision_var = config_allvariables_upper_bounds[config_allvariables_upper_bounds['feature_name'] == config_names_decision_var]['upper'].values[0]

        # create contraint upper bound
        print(f'set upper bound decision variables: {config_names_decision_var} ||| Upper bound: {upper_bound_decision_var}')
        gppd.add_constrs(model_opt, 
                        decision_var[config_names_decision_var],  # decision var
                        gp.GRB.LESS_EQUAL, 
                        upper_bound_decision_var,  # upper bound
                        name = f'Upper bound {config_names_decision_var}')



    #################################### 6. Rate change of decision variables across the time ####################################
    #################################### 6.1 define table with values of rate_change ####################################
    # rate change parameters - filter configuration file with only the decision var that have defined its rates changes
    config_allvariables_rate_change = config_allvariables[config_allvariables['rate_change'].isnull() == False]
    config_allvariables_rate_change = config_allvariables_rate_change.reset_index().drop(columns = 'index')



    #################################### 6.2 The rate change constraints is defined using absolute values. So it is necesary create an auxiliar decision variable ####################################
    ### create an auxiliar decion var "diff" for each decision var that has defined its rate change
    aux_decision_var = {}
    for index_var in range(len(config_allvariables_rate_change)):

        # get config values
        config_names_decision_var = config_allvariables_rate_change.loc[index_var, 'feature_name']
        print('create auxiliar variable diff "t" - "t-1": ', config_names_decision_var)

        # create decision var and save in the dictionary
        aux_decision_var[config_names_decision_var] = gppd.add_vars(model_opt, 
                                                                    index_set_time, 
                                                                    name = f'diff "t" - "t_1" of decision var: {config_names_decision_var}',
                                                                    lb = -gp.GRB.INFINITY,
                                                                    ub = gp.GRB.INFINITY
                                                                )

        # set initial value (diff t = 0 and t = -1) is set to cero because t = -1 is not defined
        model_opt.addConstr(aux_decision_var[config_names_decision_var]['t0'] == 0,  name = f'Initial Value diff {config_names_decision_var}')



    #################################### 6.3 Define rate change variable constraint for each decision variable and for each time ####################################
    # for each variable
    for index_var in range(len(config_allvariables_rate_change)):

        # get config values
        config_names_decision_var = config_allvariables_rate_change.loc[index_var, 'feature_name']
        print('rate change decision var: ', config_names_decision_var)

        # get rate change for this decision var
        rate_change_decision_var = config_allvariables_rate_change[config_allvariables_rate_change['feature_name'] == config_names_decision_var]['rate_change'].values[0]
        
        # for each time in this decision variable
        for index_time in range(1, len(index_set_time)):
            
            ### define time t and t-1
            time_t = index_set_time[index_time]
            time_t_1 = index_set_time[index_time-1]
        
            ### define constraints
            # positive segment
            model_opt.addConstr(aux_decision_var[config_names_decision_var][time_t] >= (decision_var[config_names_decision_var][time_t] - decision_var[config_names_decision_var][time_t_1]), 
                                name = f'diff {config_names_decision_var} positive segment {time_t} - {time_t_1}')

            # negative segment
            model_opt.addConstr(aux_decision_var[config_names_decision_var][time_t] >= -(decision_var[config_names_decision_var][time_t] - decision_var[config_names_decision_var][time_t_1]), 
                                name = f'diff {config_names_decision_var} negative segment {time_t} - {time_t_1}')

            # rate change
            model_opt.addConstr(aux_decision_var[config_names_decision_var][time_t] <= rate_change_decision_var, 
                                name = f'diff_var_X1 delta {time_t} - {time_t_1}')




    #################################### 7. Volumen change - across time - relation between decision variables ####################################
    # generate a list of tanks
    list_tanks = config_maptanks['tank'].unique().tolist()

    # filter by each tank and build the constraint creating left side of constraint according the number of inputs and outputs flows
    # FOR EACH TANK
    for name_tank in list_tanks:
        print('tank name: ', name_tank)

        # get a config map tanks dataframe with only the information of the tank that is consulting
        aux_config_maptanks = config_maptanks[config_maptanks['tank'] == name_tank]

        # FOR EACH TIME CONSTRAINT
        for index_time in range(1, len(index_set_time)):
            
            ### define time t and t-1
            time_t = index_set_time[index_time]
            time_t_1 = index_set_time[index_time-1]
        
            # BUILD THE LEFT CONSTRAINT WITH MULTIPLE INPUT AND OUTPUT FLOWS. ALSO IDENTIFY THE RIGHT SIDE CONSTRAINT
            left_side_constraint = 0
            #for tag_related_tank in aux_config_maptanks['tag'].to_list():
            for tag_related_tank in aux_config_maptanks['feature_name'].to_list():  # quizas este funciona

                # filter the configuration file only for the tag that is consulting
                aux_one_row_config_maptanks = aux_config_maptanks[aux_config_maptanks['feature_name'] == tag_related_tank]
        
                # identify if the tag is level variable adn the build the left side of constraint. also save the value to right side constraint
                if (aux_one_row_config_maptanks["input_output"] == 'L').values[0]:
                    left_side_constraint += decision_var[tag_related_tank][time_t_1]
                    right_side_constraint = decision_var[tag_related_tank][time_t]
        
                # identify if the tag is level variable adn the build the left side of constraint
                if (aux_one_row_config_maptanks["input_output"] == 'IN').values[0]:
                    left_side_constraint += decision_var[tag_related_tank][time_t]
                    
                # identify if the tag is level variable adn the build the left side of constraint
                if (aux_one_row_config_maptanks["input_output"] == 'OUT').values[0]:
                    left_side_constraint -= decision_var[tag_related_tank][time_t]

            ### define constraints
            model_opt.addConstr(left_side_constraint == right_side_constraint, 
                        name = f'new level of tank {name_tank}  in times: {time_t} - {time_t_1}')




    #################################### 8. Set values of observed variables ####################################
    ### get list of observed variables
    list_observed_variables = config_allvariables[config_allvariables['clasification'] == 'O']['feature_name'].tolist()

    ### get table with observed variables and its values
    config_initvalues_observed_variables = config_initvalues[config_initvalues['feature_name'].isin(list_observed_variables)]
    config_initvalues_observed_variables = config_initvalues_observed_variables.reset_index().drop(columns = 'index') # reset index to count by index

    for index_var in range(len(config_initvalues_observed_variables)):
        
        # get config values
        config_names_decision_var = config_initvalues_observed_variables.loc[index_var, 'feature_name']

        # get fixed values observed values
        fixed_values_observed_var = config_initvalues_observed_variables[config_initvalues_observed_variables['feature_name'] == config_names_decision_var]['init_values'].values[0]

        # add constraint observed variables fixed
        print(f'set fixed value observed values: {config_names_decision_var} ||| Fixed value observed: {fixed_values_observed_var}')
        gppd.add_constrs(model_opt,
                        decision_var[config_names_decision_var],  # decision var
                        gp.GRB.EQUAL,
                        fixed_values_observed_var,  # value observed variable
                        name = f'set value of observed variable: {config_names_decision_var}')




    #################################### 9. Load a Machine Learning Model as constraints that represent the relations in the process ####################################
    for index_modelml in range(len(config_modelsml)):

        ############ get the name (ID) of ml models ############
        config_names_modelsml = config_modelsml.loc[index_modelml, 'name_process_model']
        print('\nname model - model id: ', config_names_modelsml)


        ############ load artifact model ############
        model = models_ml[config_names_modelsml]
        type_model = type(model)

        
        ############ load parameters to build the machine learning models constraints ############
        # Get the list of features (with the SAME name used in the training of optimizer). and the SAME ORDER
        config_mapprocess_mlmodels_to_instance = config_mapprocess_mlmodels[config_mapprocess_mlmodels['name_process_model'] == config_names_modelsml]
        config_mapprocess_mlmodels_to_instance_features = config_mapprocess_mlmodels_to_instance[config_mapprocess_mlmodels_to_instance['clasificacion'] != 'T']
        list_features_instance = config_mapprocess_mlmodels_to_instance_features['feature_name'].tolist()
        
        # Get list of target
        config_mapprocess_mlmodels_to_instance_target = config_mapprocess_mlmodels_to_instance[config_mapprocess_mlmodels_to_instance['clasificacion'] == 'T']
        list_target_instance = config_mapprocess_mlmodels_to_instance_target['feature_name'].tolist()
        
        # Construct a list of targets of decision var acording the list of the target. # all the models predict only one target
        config_names_target_decision_var = list_target_instance[0]
        target_decision_var = decision_var[config_names_target_decision_var] 


        ############ build machine learning models constraints ############
        # identifity if the model loaded is a custom/rules or machine learning model
        if type(model) == pd.DataFrame:
            print('constraint custom model')
        
            # rename to remember the model is a dataframe with factors in custom model
            dataframe_model = model.copy()
            
            # for each feature in the model, get the decision var and multiply with its factor and build the left side constraint
            left_side_constraint = 0
            for index_feature_model in range(len(dataframe_model)):
            
                # get name of decision var
                config_names_decision_var = dataframe_model.loc[index_feature_model, 'feature_name']
                print('decision var: ', config_names_decision_var)
                
                # get factor of the decision var
                factor_decision_var = dataframe_model.loc[index_feature_model, 'factor_model']
                
                # multiply decision var with the factor and save in the left side constraint
                left_side_constraint += decision_var[config_names_decision_var] * factor_decision_var
        
        
            # define function as constraint
            gppd.add_constrs(model_opt, 
                            left_side_constraint,
                            gp.GRB.EQUAL, 
                            target_decision_var, # decision var target
                            name = f'function as constraint output predict {list_target_instance[0] }'
                            )
        
        else:
            print('constraint machine learning model')
        
            # Construct a list features of decision var acording the order in the list. 
            list_features_decision_var_instance = []
            for config_names_decision_var in list_features_instance:
                #print('feature - decision var append: ', config_names_decision_var)
                list_features_decision_var_instance.append(decision_var[config_names_decision_var])
            
            # Generate instance of machine learning model
            instance = pd.DataFrame(list_features_decision_var_instance).T
            instance.columns = list_features_instance
            
            # # Add machine learning model as constraint
            add_predictor_constr(gp_model = model_opt, 
                                predictor = model, 
                                input_vars = instance, # instance pandas gurobi
                                output_vars = target_decision_var, # decision var target
                                name = f'model_predict {list_target_instance[0]}'
                                )




    #################################### 10. Define objective optimization ####################################
    model_opt.setObjective(decision_var['Y1'].sum() + decision_var['Y2'].sum() + decision_var['Y3'].sum(),
                        gp.GRB.MAXIMIZE)
    



    #################################### 11. Optimize ####################################
    # solve optimization
    model_opt.optimize()
    status_solver = model_opt.Status

    # a solution was getting
    if status_solver == 2:
        return model_opt, status_solver, decision_var

    # there was a problem - infeasible solution
    else:
        print("There was a problem - infeasible solution")
        return 0, status_solver, 0