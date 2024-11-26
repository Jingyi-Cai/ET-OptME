import sys
sys.path.append('/home/sun/ETGEMS-10.20/')
import ETGEMs_function_protain as etgf
from ETGEMs_function_protain import *
import pandas as pd
import cobra
import gurobipy
import json
import multiprocessing
import os
from multiprocessing import Pool
from sympy import subsets
import pandas as pd
import matplotlib.pyplot as plt
import re



# 计算细胞生长
# 计算细胞生长
def calculate_biomass(Concretemodel_Need_Data,inputdic,model,path_strain):
    """Computational Modeling of Cell product in Four Modes

    Args:
    
    * model (cobra.Model): A Model object
    * Concretemodel_Need_Data (dir):The function of Model
    * inputdic(str):model、substrate、product、oxygence、mode
    * mode:enomic-scale Metabolic Network Model in four mode,"S" represent:  Genomic-scale Metabolic Network Model (GEMM),"ST" represent: thermodynamic constraints into GEM,"SE" represent:enzymatic constraints into genome-sacle metabolic model,"SET" represent:thermodynamic and enzymatic constraints into genome-sacle metabolic model.
    * path_strain(str): The strain name of the Model

    return: Modeling of biomass and Minmize enzyme cost and Thermodynamic maximum driving force

    """

    model.objective = inputdic['substrate']
    objvalue2=model.optimize().objective_value
    constr_coeff={}
    constr_coeff['fix_reactions']={}
    constr_coeff['substrate_constrain'] =(inputdic['substrate'],objvalue2)
    obj_name=inputdic['biomass']
    obj_target='maximize'
    B_value1 = 'None'
    totalE = 'None'
    #计量学模型
    if inputdic['mode'] == 'S':
        EcoECM_FBA_protainmodel=FBA_template2(set_obj_value=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel,'gurobi')
        v0_biomass=EcoECM_FBA_protainmodel.obj()
        print(f"maximum growth rate value is: {EcoECM_FBA_protainmodel.obj()}(mmol/gDW/h)")
        constr_coeff['fix_reactions'][inputdic['substrate']]=[value(EcoECM_FBA_protainmodel.reaction[inputdic['substrate']]),np.inf]
        constr_coeff['fix_reactions'][inputdic['biomass']]=[value(EcoECM_FBA_protainmodel.reaction[inputdic['biomass']]),np.inf]
        EcoECM_PFBA_protainmodel_wild=FBA_template2(set_obj_V_value=True,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_PFBA_protainmodel_wild,'gurobi')
        EcoECM_PFBA_protainmodel_wild.obj()
        bio=showflux(EcoECM_PFBA_protainmodel_wild)
    #酶模型
    if inputdic['mode'] == 'SE':
        if path_strain == 'iCW':
            Concretemodel_Need_Data['E_total']=0.56* np.mean([0.45311986236929197,0.4622348377433211,0.4600801040374112])
        if path_strain == 'iML1515':
            Concretemodel_Need_Data['E_total']=0.228
        EcoECM_FBA_protainmodel=FBA_template2(set_obj_value=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel,'gurobi')
        v0_biomass=EcoECM_FBA_protainmodel.obj()
        print(f"maximum growth rate value is: {EcoECM_FBA_protainmodel.obj()}(mmol/gDW/h)")
        # pfba
        constr_coeff['fix_reactions'][inputdic['substrate']]=[value(EcoECM_FBA_protainmodel.reaction[inputdic['substrate']]),np.inf]
        constr_coeff['fix_reactions'][inputdic['biomass']]=[value(EcoECM_FBA_protainmodel.reaction[inputdic['biomass']]),np.inf]
        constr_coeff['fix_E_total']=True
        EcoECM_PFBA_protainmodel_wild=FBA_template2(set_obj_V_value=True,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_PFBA_protainmodel_wild,'gurobi')
        EcoECM_PFBA_protainmodel_wild.obj()
        bio=showflux(EcoECM_PFBA_protainmodel_wild)
    # mini enzyme
    # mini enzyme
        constr_coeff={}
        constr_coeff['fix_reactions']={}
        constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
        constr_coeff['biomass_constrain']=(inputdic['biomass'],v0_biomass*0.95) 
        obj_name=inputdic['biomass']
        obj_target='minimize'
        EcoECM_FBA_protainmodel=FBA_template2(set_obj_sum_e=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel,'gurobi')
        totalE=EcoECM_FBA_protainmodel.obj()
        print(f"minimizing the total enzyme concentration  is: {EcoECM_FBA_protainmodel.obj()}((mmol/gDW)")
        print(EcoECM_FBA_protainmodel.obj())
    # 热力学模型
    if inputdic['mode'] == 'ST':
        Concretemodel_Need_Data['B_value']=0
        Concretemodel_Need_Data['K_value']=1249
        EcoECM_FBA_protainmodel=FBA_template2(set_obj_value=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel,'gurobi')
        v0_biomass=EcoECM_FBA_protainmodel.obj()
        print(f"maximum growth rate value is: {EcoECM_FBA_protainmodel.obj()}(mmol/gDW/h)")
        constr_coeff['fix_reactions'][inputdic['biomass']]=[value(EcoECM_FBA_protainmodel.reaction[inputdic['biomass']]),np.inf]
        EcoECM_FBA_protainmodel=FBA_template2(set_obj_B_value=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel,'gurobi')
        B_value1=EcoECM_FBA_protainmodel.obj()
        print(f"The maximum thermodynamic driving force value is : {EcoECM_FBA_protainmodel.obj()}")
        constr_coeff['fix_reactions'][inputdic['substrate']]=[value(EcoECM_FBA_protainmodel.reaction[inputdic['substrate']]),np.inf]
        constr_coeff['fix_reactions'][inputdic['biomass']]=[value(EcoECM_FBA_protainmodel.reaction[inputdic['biomass']]),np.inf]
        EcoECM_PFBA_protainmodel_wild=FBA_template2(set_obj_V_value=True,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_PFBA_protainmodel_wild,'gurobi')
        EcoECM_PFBA_protainmodel_wild.obj()
        bio=showflux(EcoECM_PFBA_protainmodel_wild)        
    # 酶热模型
    if inputdic['mode'] == 'SET':
        if path_strain == 'iCW':
            Concretemodel_Need_Data['E_total']=0.56* np.mean([0.45311986236929197,0.4622348377433211,0.4600801040374112])
        if path_strain == 'iML1515':
            Concretemodel_Need_Data['E_total']=0.228
        Concretemodel_Need_Data['B_value']=0
        Concretemodel_Need_Data['K_value']=1249
        EcoECM_FBA_protainmodel=FBA_template2(set_obj_value=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel,'gurobi')
        v0_biomass=EcoECM_FBA_protainmodel.obj()
        print(f"maximum growth rate value is: {EcoECM_FBA_protainmodel.obj()}(mmol/gDW/h)")
        constr_coeff['fix_reactions'][inputdic['biomass']]=[value(EcoECM_FBA_protainmodel.reaction[inputdic['biomass']]),np.inf]
        if path_strain == 'iML1515':
            constr_coeff['fix_reactions'][inputdic['biomass']]=[value(EcoECM_FBA_protainmodel.reaction[inputdic['biomass']])*0.9,np.inf]
        EcoECM_FBA_protainmodel_B1=FBA_template2(set_obj_B_value=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel_B1,'gurobi')
        B_value1=EcoECM_FBA_protainmodel_B1.obj()
        print(f"The maximum thermodynamic driving force value is : {EcoECM_FBA_protainmodel.obj()}")
        constr_coeff['fix_E_total']=True
        EcoECM_PFBA_protainmodel_wild=FBA_template2(set_obj_V_value=True,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_PFBA_protainmodel_wild,'gurobi')
        EcoECM_PFBA_protainmodel_wild.obj()
        bio=showflux(EcoECM_PFBA_protainmodel_wild)
    # mini enzyme
    # mini enzyme
        constr_coeff={}
        constr_coeff['fix_reactions']={}
        constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
        constr_coeff['biomass_constrain']=(inputdic['biomass'],v0_biomass*0.95) 
        obj_name=inputdic['biomass']
        obj_target='minimize'
        EcoECM_FBA_protainmodel=FBA_template2(set_obj_sum_e=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel,'gurobi')
        totalE=EcoECM_FBA_protainmodel.obj()
        print(f"minimizing the total enzyme concentration  is: {EcoECM_FBA_protainmodel.obj()}((mmol/gDW)")
    return B_value1,v0_biomass,bio,totalE,objvalue2
    return B_value1,v0_biomass,bio,totalE,objvalue2

def calculate_wildrange_reaction(Concretemodel_Need_Data,obj_name,inputdic,objvalue2,v0_biomass,totalE,path_strain,B_value1): 
    """
    Calculate the range of amounts of each enzyme in the model.
    Args:
    * Concretemodel_Need_Data (dict): The function of Model.
    * obj_name (Model.reaction): Objective reaction in the model.
    * inputdic (dict): Dictionary containing the following keys:
            * 'model': Model name.
            * 'substrate': Substrate information.
            * 'product': Product information.
            * 'oxygence': Oxygen conditions.
            * 'mode': Model mode. Four possible values:
                *  'S': Genomic-scale Metabolic Network Model (GEMM).
                *  'ST': Thermodynamic constraints into GEM.
                *  'SE': Enzymatic constraints into genome-scale metabolic model.
               *  'SET': Thermodynamic and enzymatic constraints into genome-scale metabolic model.
        * objvalue2 (float): Substrate uptake rate in the model.
        * v0_biomass (float): Biomass growth rate.
        * totalE (float): Minimum enzyme cost.
        * path_strain (str): The strain name of the Model.
       *  B_value1 (float): Thermodynamic maximum driving force.

    Returns:
        dict: A dictionary with the objective name as the key and a dictionary containing the range of enzyme amounts as the value.
    """
    if inputdic['mode'] == 'S':
        results = {}
        constr_coeff={}
        constr_coeff['fix_reactions']={}  
        # 固定底物摄入
        constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
        # 固定细胞最大生长
        constr_coeff['biomass_constrain']=(inputdic['biomass'],v0_biomass*0.95)  
        #计算酶量最小值
        obj_target = 'minimize'  
        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value=True, obj_name=obj_name, obj_target=obj_target, mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel, 'gurobi')  # gurobi求解器

        min_value = EcoECM_FBA_protainmodel.obj()
        print(f"Objective: {obj_name}, Maximize: {min_value}")
        #计算酶量最大值
        obj_target = 'maximize'
        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value=True, obj_name=obj_name, obj_target=obj_target,mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 
        max_value = EcoECM_FBA_protainmodel.obj()  
        print(f"Objective: {obj_name}, Maximize: {max_value}")
        results[obj_name] = {'range': [min_value, max_value]}
        #酶模型下的酶量分布
    if inputdic['mode'] == 'SE':
        results = {}
        constr_coeff={}
        constr_coeff['fix_reactions']={} 
        #固定最小总酶量
        Concretemodel_Need_Data['E_total']=totalE*1.001
        if path_strain == 'iML1515':
            Concretemodel_Need_Data['E_total']=totalE*1.5
        constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
        constr_coeff['biomass_constrain']=(inputdic['biomass'],v0_biomass*0.95) 
        obj_target = 'minimize' 

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value_e0=True, obj_name=obj_name, obj_target=obj_target, mode='SE', constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 
            min_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            min_value = 0
        min_value = EcoECM_FBA_protainmodel.obj()
        print(f"Objective: {obj_name}, Maximize: {min_value}")
        obj_target = 'maximize'   

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value_e0=True, obj_name=obj_name, obj_target=obj_target, mode='SE', constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi')  
            max_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            max_value = 100
        # max_value = EcoECM_FBA_protainmodel.obj() 
        print(f"Objective: {obj_name}, Maximize: {max_value}")
        results = {obj_name: {'range': [min_value, max_value]}} 
    #热力学模型下的酶量分布范围
    if inputdic['mode'] == 'ST':  
        results = {}
        constr_coeff={}
        #固定最大驱动力
        Concretemodel_Need_Data['B_value']= B_value1
        if path_strain == 'iML1515':
            Concretemodel_Need_Data['B_value']=B_value1*0.99        
        Concretemodel_Need_Data['K_value']=1249    
        constr_coeff['fix_reactions']={}  
        constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
        constr_coeff['biomass_constrain']=(inputdic['biomass'],v0_biomass*0.95)  

        obj_target = 'minimize' 

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value=True, obj_name=obj_name, obj_target=obj_target, mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel, 'gurobi')  
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 
            min_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            min_value = 0
        print(f"Objective: {obj_name}, Maximize: {min_value}")
        obj_target = 'maximize' 

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value=True, obj_name=obj_name, obj_target=obj_target,mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi')  
            max_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            max_value = 100
        print(f"Objective: {obj_name}, Maximize: {max_value}")
        results[obj_name] = {'range': [min_value, max_value]}
        #酶热模型求酶量分布范围
    if inputdic['mode'] == 'SET':
        results = {}
        constr_coeff = {}
        constr_coeff['fix_reactions'] = {} 
        #固定最小总酶量
        Concretemodel_Need_Data['E_total'] = totalE * 1.01
        # 固定热力学最大驱动力
        Concretemodel_Need_Data['B_value'] = B_value1 * 0.98
        constr_coeff['substrate_constrain'] = (inputdic['substrate'], 4.67)
        if path_strain == 'iML1515':
            Concretemodel_Need_Data['E_total']=totalE*1.01
            Concretemodel_Need_Data['B_value']=B_value1*0.99 
            constr_coeff['substrate_constrain'] = (inputdic['substrate'], 10)
        constr_coeff['biomass_constrain'] = (inputdic['biomass'], v0_biomass * 0.95) 

        obj_target = 'minimize'  

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value_e0=True, obj_name=obj_name, obj_target=obj_target, mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 
            min_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            min_value = 0

        print(f"Objective: {obj_name}, Minimize: {min_value}")

        obj_target = 'maximize'  

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value_e0=True, obj_name=obj_name, obj_target=obj_target,mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi')  
            max_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            max_value = 100

        print(f"Objective: {obj_name}, Maximize: {max_value}")

        results[obj_name] = {'range': [min_value, max_value]}

    return results


# 计算生长状态下的酶量分布范围
def calculate_wildrange(Concretemodel_Need_Data,obj_name,inputdic,objvalue2,v0_biomass,totalE,path_strain,B_value1): 
    """
    Calculate the range of amounts of each enzyme in the model.
    Args:
    * Concretemodel_Need_Data (dict): The function of Model.
    * obj_name (Model.reaction): Objective reaction in the model.
    * inputdic (dict): Dictionary containing the following keys:
            * 'model': Model name.
            * 'substrate': Substrate information.
            * 'product': Product information.
            * 'oxygence': Oxygen conditions.
            * 'mode': Model mode. Four possible values:
                *  'S': Genomic-scale Metabolic Network Model (GEMM).
                *  'ST': Thermodynamic constraints into GEM.
                *  'SE': Enzymatic constraints into genome-scale metabolic model.
               *  'SET': Thermodynamic and enzymatic constraints into genome-scale metabolic model.
        * objvalue2 (float): Substrate uptake rate in the model.
        * v0_biomass (float): Biomass growth rate.
        * totalE (float): Minimum enzyme cost.
        * path_strain (str): The strain name of the Model.
       *  B_value1 (float): Thermodynamic maximum driving force.

    Returns:
        dict: A dictionary with the objective name as the key and a dictionary containing the range of enzyme amounts as the value.
    """
    if inputdic['mode'] == 'S':
        results = {}
        constr_coeff={}
        constr_coeff['fix_reactions']={}  
        # 固定底物摄入
        constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
        # 固定细胞最大生长
        constr_coeff['biomass_constrain']=(inputdic['biomass'],v0_biomass*0.95)  
        #计算酶量最小值
        obj_target = 'minimize'  
        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value=True, obj_name=obj_name, obj_target=obj_target, mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel, 'gurobi')  # gurobi求解器

        min_value = EcoECM_FBA_protainmodel.obj()
        print(f"Objective: {obj_name}, Maximize: {min_value}")
        #计算酶量最大值
        obj_target = 'maximize'
        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value=True, obj_name=obj_name, obj_target=obj_target,mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 
        max_value = EcoECM_FBA_protainmodel.obj()  
        print(f"Objective: {obj_name}, Maximize: {max_value}")
        results[obj_name] = {'range': [min_value, max_value]}
        #酶模型下的酶量分布
    if inputdic['mode'] == 'SE':
        results = {}
        constr_coeff={}
        constr_coeff['fix_reactions']={} 
        #固定最小总酶量
        Concretemodel_Need_Data['E_total']=totalE*1.001
        if path_strain == 'iML1515':
            Concretemodel_Need_Data['E_total']=totalE*1.5
        constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
        constr_coeff['biomass_constrain']=(inputdic['biomass'],v0_biomass*0.95) 
        obj_target = 'minimize' 

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value_e=True, obj_name=obj_name, obj_target=obj_target, mode='SE', constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 
            min_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            min_value = 0
        min_value = EcoECM_FBA_protainmodel.obj()
        print(f"Objective: {obj_name}, Maximize: {min_value}")
        obj_target = 'maximize'   

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value_e=True, obj_name=obj_name, obj_target=obj_target, mode='SE', constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi')  
            max_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            max_value = 100
        # max_value = EcoECM_FBA_protainmodel.obj() 
        print(f"Objective: {obj_name}, Maximize: {max_value}")
        results = {obj_name: {'range': [min_value, max_value]}} 
    #热力学模型下的酶量分布范围
    if inputdic['mode'] == 'ST':  
        results = {}
        constr_coeff={}
        #固定最大驱动力
        Concretemodel_Need_Data['B_value']= B_value1
        if path_strain == 'iML1515':
            Concretemodel_Need_Data['B_value']=B_value1*0.99        
        Concretemodel_Need_Data['K_value']=1249    
        constr_coeff['fix_reactions']={}  
        constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
        constr_coeff['biomass_constrain']=(inputdic['biomass'],v0_biomass*0.95)  

        obj_target = 'minimize' 

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value=True, obj_name=obj_name, obj_target=obj_target, mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel, 'gurobi')  
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 
            min_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            min_value = 0
        print(f"Objective: {obj_name}, Maximize: {min_value}")
        obj_target = 'maximize' 

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value=True, obj_name=obj_name, obj_target=obj_target,mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi')  
            max_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            max_value = 100
        print(f"Objective: {obj_name}, Maximize: {max_value}")
        results[obj_name] = {'range': [min_value, max_value]}
        #酶热模型求酶量分布范围
    if inputdic['mode'] == 'SET':
        results = {}
        constr_coeff = {}
        constr_coeff['fix_reactions'] = {} 
        #固定最小总酶量
        Concretemodel_Need_Data['E_total'] = totalE * 1.01
        # 固定热力学最大驱动力
        Concretemodel_Need_Data['B_value'] = B_value1 * 0.98
        constr_coeff['substrate_constrain'] = (inputdic['substrate'], 4.67)
        if path_strain == 'iML1515':
            Concretemodel_Need_Data['E_total']=totalE*1.01
            Concretemodel_Need_Data['B_value']=B_value1*0.99 
            constr_coeff['substrate_constrain'] = (inputdic['substrate'], 10)
        constr_coeff['biomass_constrain'] = (inputdic['biomass'], v0_biomass * 0.95) 

        obj_target = 'minimize'  

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value_e=True, obj_name=obj_name, obj_target=obj_target, mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 
            min_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            min_value = 0

        print(f"Objective: {obj_name}, Minimize: {min_value}")

        obj_target = 'maximize'  

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value_e=True, obj_name=obj_name, obj_target=obj_target,mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi')  
            max_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            max_value = 100

        print(f"Objective: {obj_name}, Maximize: {max_value}")

        results[obj_name] = {'range': [min_value, max_value]}

    return results

#计算生产状态下的酶量分布范围
#计算生产状态下的酶量分布范围
def calculate_product(Concretemodel_Need_Data,inputdic,objvalue2,path_strain,v0_biomass):
    """Calculate  the range of amounts of each enzyme in the model.

    Args:
        Concretemodel_Need_Data (dict): Data needed for the concrete model.
        inputdic (dict): Dictionary containing input parameters such as substrate, biomass, product, and mode.
        objvalue2 (float): Objective value for substrate constraint.
        path_strain (str): The strain path identifier, e.g., 'iCW' or 'iML1515'.
        v0_biomass (float): Initial biomass value.calculate_biomasscalculate_biomass

    Returns:
        tuple: Contains B_value2, v1_product_max, pro, totalE2, EcoECM_FBA_protainmodel_B2.
            - B_value2 (float): Maximum driving force.
            - v1_product_max (float): Maximum product rate.
            - pro (dataframe): Flux data from the pFBA model.
            - totalE2 (float): Minimum enzyme amount.
            - EcoECM_FBA_protainmodel_B2: Final pFBA model.

    """

    constr_coeff={}
    constr_coeff['fix_reactions'] = {}
    # constr_coeff['fix_reactions']['EX_glc__D_e_reverse']=10
    # 固定底物摄入
    constr_coeff['substrate_constrain'] = (inputdic['substrate'],objvalue2)
    #固定细胞生长
    constr_coeff['biomass_constrain'] = (inputdic['biomass'],v0_biomass*0.1)
    #设置目标产品
    obj_name = inputdic['product']
    obj_target = 'maximize'
    B_value2 = 'None'
    totalE2 = 'None'
    EcoECM_FBA_protainmodel_B2 = 'None'
    #计量学模型的产品速率
    if inputdic['mode'] == 'S':
        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel,'gurobi')
        v1_product_max = EcoECM_FBA_protainmodel.obj()
        print(f"maximum product rate value is: {EcoECM_FBA_protainmodel.obj()}(mmol/gDW/h)")
        constr_coeff['fix_reactions'][inputdic['substrate']] = [value(EcoECM_FBA_protainmodel.reaction[inputdic['substrate']]),np.inf]
        constr_coeff['fix_reactions'][inputdic['product']] = [value(EcoECM_FBA_protainmodel.reaction[inputdic['product']]),np.inf]
        EcoECM_PFBA_protainmodel_pro = FBA_template2(set_obj_V_value=True,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_PFBA_protainmodel_pro,'gurobi')
        print(EcoECM_PFBA_protainmodel_pro.obj())
        pro = showflux(EcoECM_PFBA_protainmodel_pro)
    # 酶模型的产品速率
    if inputdic['mode'] == 'SE':
        if path_strain == 'iCW':
            Concretemodel_Need_Data['E_total']=0.56* np.mean([0.45311986236929197,0.4622348377433211,0.4600801040374112])
        if path_strain == 'iML1515':
            Concretemodel_Need_Data['E_total']=0.228
        EcoECM_FBA_protainmodel=FBA_template2(set_obj_value=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel,'gurobi')
        v1_product_max=EcoECM_FBA_protainmodel.obj()
        print(f"maximum growth rate value is: {EcoECM_FBA_protainmodel.obj()}(mmol/gDW/h)")
        # pfba
        constr_coeff['substrate_constrain'] = (inputdic['substrate'],value(EcoECM_FBA_protainmodel.reaction[inputdic['substrate']]))
        constr_coeff['biomass_constrain'] = (inputdic['biomass'],v0_biomass*0.1)
        constr_coeff['fix_E_total']=True
        EcoECM_PFBA_protainmodel_pro=FBA_template2(set_obj_V_value=True,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_PFBA_protainmodel_pro,'gurobi')
        EcoECM_PFBA_protainmodel_pro.obj()
        pro = showflux(EcoECM_PFBA_protainmodel_pro)
        # mini enzyme
        constr_coeff={}
        constr_coeff['fix_reactions']={}
        constr_coeff['substrate_constrain']=(inputdic['substrate'],value(EcoECM_FBA_protainmodel.reaction[inputdic['substrate']]))
        constr_coeff['biomass_constrain']=(inputdic['biomass'],v0_biomass*0.1) 
        constr_coeff['product_constrain']=(inputdic['product'],v1_product_max*0.95)
        obj_name=inputdic['product']
        obj_target='minimize'
        EcoECM_FBA_protainmodel=FBA_template2(set_obj_sum_e=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel,'gurobi')
        totalE2=EcoECM_FBA_protainmodel.obj()
        print(f"minimizing the total enzyme concentration  is: {EcoECM_FBA_protainmodel.obj()}((mmol/gDW)")
    #热力学模型的产品速率
    if inputdic['mode'] == 'ST':  
        Concretemodel_Need_Data['B_value']=0
        Concretemodel_Need_Data['K_value']=1249

        EcoECM_FBA_protainmodel=FBA_template2(set_obj_value=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel,'gurobi')
        v1_product_max=EcoECM_FBA_protainmodel.obj()
        print(f"maximum growth rate value is: {EcoECM_FBA_protainmodel.obj()}(mmol/gDW/h)")
        constr_coeff['fix_reactions'][inputdic['product']]=[value(EcoECM_FBA_protainmodel.reaction[inputdic['product']]*0.99),np.inf]
        EcoECM_FBA_protainmodel_B2=FBA_template2(set_obj_B_value=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel_B2,'gurobi')
        B_value2=EcoECM_FBA_protainmodel_B2.obj()
        print(f"The maximum thermodynamic driving force value is : {EcoECM_FBA_protainmodel.obj()}")
        constr_coeff['fix_reactions'][inputdic['substrate']]=[value(EcoECM_FBA_protainmodel.reaction[inputdic['substrate']]),np.inf]
        constr_coeff['fix_reactions'][inputdic['product']]=[value(EcoECM_FBA_protainmodel.reaction[inputdic['product']]),np.inf]
        EcoECM_PFBA_protainmodel_pro=FBA_template2(set_obj_V_value=True,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_PFBA_protainmodel_pro,'gurobi')
        print(EcoECM_PFBA_protainmodel_pro.obj())
        pro=showflux(EcoECM_PFBA_protainmodel_pro)
    # 酶热模型的产品速率
    if inputdic['mode'] == 'SET':
        if path_strain == 'iCW':
            Concretemodel_Need_Data['E_total']=0.56* np.mean([0.45311986236929197,0.4622348377433211,0.4600801040374112])
        if path_strain == 'iML1515':
            Concretemodel_Need_Data['E_total']=0.228
        Concretemodel_Need_Data['B_value']=0
        Concretemodel_Need_Data['K_value']=1249
        EcoECM_FBA_protainmodel=FBA_template2(set_obj_value=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel,'gurobi')
        v1_product_max=EcoECM_FBA_protainmodel.obj()
        print(f"maximum growth rate value is: {EcoECM_FBA_protainmodel.obj()}(mmol/gDW/h)")
        #固定产品，求最大驱动力
        constr_coeff['fix_reactions'][inputdic['product']]=[value(EcoECM_FBA_protainmodel.reaction[inputdic['product']])*0.9,np.inf]
        EcoECM_FBA_protainmodel_B2=FBA_template2(set_obj_B_value=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel_B2,'gurobi')
        B_value2=EcoECM_FBA_protainmodel_B2.obj()
        print(f"The maximum thermodynamic driving force value is : {EcoECM_FBA_protainmodel.obj()}")
        pro=showflux(EcoECM_FBA_protainmodel)
    # mini enzyme
    # mini enzyme
    # 求最小酶量
        constr_coeff={}
        constr_coeff['fix_reactions']={}
        constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
        constr_coeff['biomass_constrain']=(inputdic['biomass'],v0_biomass*0.1) 
        constr_coeff['product_constrain']=(inputdic['product'],v1_product_max*0.95)
        obj_name=inputdic['product']
        obj_target='minimize'
        EcoECM_FBA_protainmodel=FBA_template2(set_obj_sum_e=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel,'gurobi')
        totalE2=EcoECM_FBA_protainmodel.obj()
        print(f"minimizing the total enzyme concentration  is: {EcoECM_FBA_protainmodel.obj()}((mmol/gDW)")
    return B_value2,v1_product_max,pro,totalE2,EcoECM_FBA_protainmodel_B2



# 计算最大产品下的酶量分布情况
def calculate_over_reaction(Concretemodel_Need_Data,obj_name,inputdic,objvalue2,v0_biomass,v1_product_max,totalE2,path_strain,B_value2):
    """
    Calculate the range of enzyme levels for different models based on the specified mode.
    
    Args:
    - Concretemodel_Need_Data (dict): Dictionary containing concrete model data needed for calculations.
    - obj_name (str): Name of the objective reaction/product.
    - inputdic (dict): Dictionary containing various input parameters such as mode, substrate, biomass, and product.
    - objvalue2 (float): Value for the substrate constraint.
    - v0_biomass (float): Initial biomass value.
    - v1_product_max (float): Maximum value of the product.
    - totalE2 (float): Total enzyme value.
    - path_strain (str): Strain path, either 'iCW' or 'iML1515'.
    - B_value2 (float): B value for the thermodynamic model.

    Returns:
    - results (dict): Dictionary containing the range of values (min and max) for the objective reaction/product.
    """
    #  计量学模型的酶量分布
    if inputdic['mode'] == 'S':
        results = {}
        constr_coeff={}
        constr_coeff['fix_reactions']={} 
        constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
        constr_coeff['biomass_constrain']=(inputdic['biomass'],v0_biomass*0.1)  
        constr_coeff['product_constrain']=(inputdic['product'],v1_product_max*0.95)
        obj_target = 'minimize' 

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value=True, obj_name=obj_name, obj_target=obj_target, mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 
            min_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            min_value = 0
        # min_value = EcoECM_FBA_protainmodel.obj()  
        print(f"Objective: {obj_name}, Maximize: {min_value}")
        obj_target = 'maximize'  

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value=True, obj_name=obj_name, obj_target=obj_target,mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi')  
            max_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            max_value = 100
        print(f"Objective: {obj_name}, Maximize: {max_value}")
        results[obj_name] = {'range': [min_value, max_value]}
        # 酶约束模型的酶量分布
    if inputdic['mode'] == 'SE':
        results = {}
        constr_coeff={}
        constr_coeff['fix_reactions']={}  
        Concretemodel_Need_Data['E_total']=totalE2*1.001
        if path_strain == 'iML1515':
            Concretemodel_Need_Data['E_total']=totalE2*1.5
        constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
        constr_coeff['biomass_constrain']=(inputdic['biomass'],v0_biomass*0.1)
        constr_coeff['product_constrain']=(inputdic['product'],v1_product_max*0.95) 
        obj_target = 'minimize'  

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value_e0=True, obj_name=obj_name, obj_target=obj_target, mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 

        min_value = EcoECM_FBA_protainmodel.obj() 
        print(f"Objective: {obj_name}, Maximize: {min_value}")
        obj_target = 'maximize'
        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value_e0=True, obj_name=obj_name, obj_target=obj_target, mode='SE', constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel, 'gurobi')
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi')  
            max_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            max_value = 100
        # max_value = EcoECM_FBA_protainmodel.obj()  
        print(f"Objective: {obj_name}, Maximize: {max_value}")
        results = {obj_name: {'range': [min_value, max_value]}} 
        # 热力学模型的酶量分布
    if inputdic['mode'] == 'ST':  
        results = {}
        constr_coeff={}
        constr_coeff['fix_reactions']={}  
        Concretemodel_Need_Data['B_value']=B_value2
        if path_strain == 'iML1515':
            Concretemodel_Need_Data['B_value']=B_value2*0.99    
        constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
        constr_coeff['biomass_constrain']=(inputdic['biomass'],v0_biomass*0.1)
        constr_coeff['product_constrain']=(inputdic['product'],v1_product_max*0.95)  

        obj_target = 'minimize' 

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value=True, obj_name=obj_name, obj_target=obj_target, mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel, 'gurobi')  
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 
            min_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            min_value = 0
        # min_value = EcoECM_FBA_protainmodel.obj()  
        print(f"Objective: {obj_name}, Maximize: {min_value}")
        obj_target = 'maximize' 

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value=True, obj_name=obj_name, obj_target=obj_target,mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel, 'gurobi')  
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi')  
            max_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            max_value = 100
        print(f"Objective: {obj_name}, Maximize: {max_value}")
        results[obj_name] = {'range': [min_value, max_value]}
        #  酶热模型的酶量分布
    if inputdic['mode'] == 'SET':
        results = {}
        constr_coeff={}
        constr_coeff['fix_reactions']={} 
        Concretemodel_Need_Data['E_total']=totalE2*1.01
        Concretemodel_Need_Data['B_value']=B_value2*0.9
        if path_strain == 'iML1515':
            Concretemodel_Need_Data['E_total']=totalE2*1.01
            Concretemodel_Need_Data['B_value']=B_value2*0.9        
        constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
        constr_coeff['biomass_constrain']=(inputdic['biomass'],v0_biomass*0.1) 
        constr_coeff['product_constrain']=(inputdic['product'],v1_product_max*0.95) 
        obj_target = 'minimize'   

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value_e0=True, obj_name=obj_name, obj_target=obj_target, mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 
            min_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            min_value = 0

        print(f"Objective: {obj_name}, Minimize: {min_value}")

        obj_target = 'maximize'  

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value_e0=True, obj_name=obj_name, obj_target=obj_target,mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi')  
            max_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            max_value = 100

        print(f"Objective: {obj_name}, Maximize: {max_value}")

        results[obj_name] = {'range': [min_value, max_value]}
    return results


# 计算最大产品下的酶量分布情况
def calculate_over(Concretemodel_Need_Data,obj_name,inputdic,objvalue2,v0_biomass,v1_product_max,totalE2,path_strain,B_value2):
    """
    Calculate the range of enzyme levels for different models based on the specified mode.
    
    Args:
    - Concretemodel_Need_Data (dict): Dictionary containing concrete model data needed for calculations.
    - obj_name (str): Name of the objective reaction/product.
    - inputdic (dict): Dictionary containing various input parameters such as mode, substrate, biomass, and product.
    - objvalue2 (float): Value for the substrate constraint.
    - v0_biomass (float): Initial biomass value.
    - v1_product_max (float): Maximum value of the product.
    - totalE2 (float): Total enzyme value.
    - path_strain (str): Strain path, either 'iCW' or 'iML1515'.
    - B_value2 (float): B value for the thermodynamic model.

    Returns:
    - results (dict): Dictionary containing the range of values (min and max) for the objective reaction/product.
    """
    #  计量学模型的酶量分布
    if inputdic['mode'] == 'S':
        results = {}
        constr_coeff={}
        constr_coeff['fix_reactions']={} 
        constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
        constr_coeff['biomass_constrain']=(inputdic['biomass'],v0_biomass*0.1)  
        constr_coeff['product_constrain']=(inputdic['product'],v1_product_max*0.95)
        obj_target = 'minimize' 

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value=True, obj_name=obj_name, obj_target=obj_target, mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 
            min_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            min_value = 0
        # min_value = EcoECM_FBA_protainmodel.obj()  
        print(f"Objective: {obj_name}, Maximize: {min_value}")
        obj_target = 'maximize'  

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value=True, obj_name=obj_name, obj_target=obj_target,mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi')  
            max_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            max_value = 100
        print(f"Objective: {obj_name}, Maximize: {max_value}")
        results[obj_name] = {'range': [min_value, max_value]}
        # 酶约束模型的酶量分布
    if inputdic['mode'] == 'SE':
        results = {}
        constr_coeff={}
        constr_coeff['fix_reactions']={}  
        Concretemodel_Need_Data['E_total']=totalE2*1.001
        if path_strain == 'iML1515':
            Concretemodel_Need_Data['E_total']=totalE2*1.5
        constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
        constr_coeff['biomass_constrain']=(inputdic['biomass'],v0_biomass*0.1)
        constr_coeff['product_constrain']=(inputdic['product'],v1_product_max*0.95) 
        obj_target = 'minimize'  

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value_e=True, obj_name=obj_name, obj_target=obj_target, mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 

        min_value = EcoECM_FBA_protainmodel.obj() 
        print(f"Objective: {obj_name}, Maximize: {min_value}")
        obj_target = 'maximize'
        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value_e=True, obj_name=obj_name, obj_target=obj_target, mode='SE', constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi')  
            max_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            max_value = 100
        # max_value = EcoECM_FBA_protainmodel.obj()  
        print(f"Objective: {obj_name}, Maximize: {max_value}")
        results = {obj_name: {'range': [min_value, max_value]}} 
        # 热力学模型的酶量分布
    if inputdic['mode'] == 'ST':  
        results = {}
        constr_coeff={}
        constr_coeff['fix_reactions']={}  
        Concretemodel_Need_Data['B_value']=B_value2
        if path_strain == 'iML1515':
            Concretemodel_Need_Data['B_value']=B_value2*0.99    
        constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
        constr_coeff['biomass_constrain']=(inputdic['biomass'],v0_biomass*0.1)
        constr_coeff['product_constrain']=(inputdic['product'],v1_product_max*0.95)  

        obj_target = 'minimize' 

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value=True, obj_name=obj_name, obj_target=obj_target, mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel, 'gurobi')  
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 
            min_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            min_value = 0
        # min_value = EcoECM_FBA_protainmodel.obj()  
        print(f"Objective: {obj_name}, Maximize: {min_value}")
        obj_target = 'maximize' 

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value=True, obj_name=obj_name, obj_target=obj_target,mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel, 'gurobi')  
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi')  
            max_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            max_value = 100
        print(f"Objective: {obj_name}, Maximize: {max_value}")
        results[obj_name] = {'range': [min_value, max_value]}
        #  酶热模型的酶量分布
    if inputdic['mode'] == 'SET':
        results = {}
        constr_coeff={}
        constr_coeff['fix_reactions']={} 
        Concretemodel_Need_Data['E_total']=totalE2*1.01
        Concretemodel_Need_Data['B_value']=B_value2*0.9
        if path_strain == 'iML1515':
            Concretemodel_Need_Data['E_total']=totalE2*1.01
            Concretemodel_Need_Data['B_value']=B_value2*0.9        
        constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
        constr_coeff['biomass_constrain']=(inputdic['biomass'],v0_biomass*0.1) 
        constr_coeff['product_constrain']=(inputdic['product'],v1_product_max*0.95) 
        obj_target = 'minimize'   

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value_e=True, obj_name=obj_name, obj_target=obj_target, mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi') 
            min_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            min_value = 0

        print(f"Objective: {obj_name}, Minimize: {min_value}")

        obj_target = 'maximize'  

        EcoECM_FBA_protainmodel = FBA_template2(set_obj_value_e=True, obj_name=obj_name, obj_target=obj_target,mode=inputdic['mode'], constr_coeff=constr_coeff, Concretemodel_Need_Data=Concretemodel_Need_Data)
        try:
            Model_Solve(EcoECM_FBA_protainmodel, 'gurobi')  
            max_value = EcoECM_FBA_protainmodel.obj()
        except Exception as e:
            print(f"Error occurred while solving: {e}")
            max_value = 100

        print(f"Objective: {obj_name}, Maximize: {max_value}")

        results[obj_name] = {'range': [min_value, max_value]}
    return results

#  分别读取野生型和过表达型下的酶分布
def read_file(path_results):
    """
    Reads JSON files containing enzyme data from the specified directory.

    Args:
    - path_results (str): The path to the directory containing the 'wild-enzyme.json' and 'over-enzyme.json' files.

    Returns:
    - enzyme_results_data (dict): The data from the 'wild-enzyme.json' file.
    - enzyme_overresults2_data (dict): The data from the 'over-enzyme.json' file.
    """
    wild_file_path = os.path.join(path_results, 'wild-enzyme.json')
    over_file_path = os.path.join(path_results, 'over-enzyme.json') 
    with open(wild_file_path, 'r') as enzyme_results_file:
        enzyme_results_data = json.load(enzyme_results_file)
    with open(over_file_path, 'r') as enzyme_overresults2_file:
        enzyme_overresults2_data = json.load(enzyme_overresults2_file)
    return enzyme_results_data,enzyme_overresults2_data
def replace_none_with_zero(data):
    """
    Recursively replace all None values in a nested dictionary with 0.

    Args:
    - data (dict or list): The input data which can be a dictionary or list.

    Returns:
    - data: The modified data with None values replaced by 0.
    """
    if isinstance(data, dict):
        return {key: replace_none_with_zero(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [replace_none_with_zero(item) for item in data]
    else:
        return 0 if data is None else data

def read_file_rea(path_results):
    """
    Reads JSON files containing enzyme data from the specified directory.

    Args:
    - path_results (str): The path to the directory containing the 'wild-enzyme.json' and 'over-enzyme.json' files.

    Returns:
    - enzyme_results_data (dict): The data from the 'wild-enzyme.json' file with None values replaced by 0.
    - enzyme_overresults2_data (dict): The data from the 'over-enzyme.json' file with None values replaced by 0.
    """
    wild_file_path = os.path.join(path_results, 'wild-enzyme-reaction_e0.json')
    over_file_path = os.path.join(path_results, 'over-enzyme_reaction_e0.json')

    with open(wild_file_path, 'r') as enzyme_results_file:
        enzyme_results_data = json.load(enzyme_results_file)
        enzyme_results_data = replace_none_with_zero(enzyme_results_data)

    with open(over_file_path, 'r') as enzyme_overresults2_file:
        enzyme_overresults2_data = json.load(enzyme_overresults2_file)
        enzyme_overresults2_data = replace_none_with_zero(enzyme_overresults2_data)

    return enzyme_results_data, enzyme_overresults2_data
# 筛选需要改造的靶点
def compare_results(enzyme_results_data, enzyme_overresults_data,inputdic): 
    """
    Compares enzyme activity ranges between wild-type and overexpression data 
    to categorize them into knock-out (ko_flux), upregulated (up_flux), 
    and downregulated (down_flux) groups based on the mode specified.

    Args:
    - enzyme_results_data (dict): The wild-type enzyme data with activity ranges.
    - enzyme_overresults_data (dict): The overexpression enzyme data with activity ranges.
    - inputdic (dict): A dictionary containing the mode (e.g., 'S', 'ST', 'SE', 'SET').

    Returns:
    - ko_data (list): A list of tuples containing enzyme names and their wild-type activity ranges 
      for enzymes that are categorized as knock-out.
    - up_data (list): A list of tuples containing enzyme names and their wild-type activity ranges 
      for enzymes that are categorized as upregulated.
    - down_data (list): A list of tuples containing enzyme names and their wild-type activity ranges 
      for enzymes that are categorized as downregulated.
    """
    if inputdic['mode'] == 'S' or inputdic['mode'] == 'ST':
        # ko_flux
        mean_values = []

    
        for values in enzyme_results_data.values():
            if values.get('range') and len(values['range']) >= 2:
                range_values = values['range']
                mean_value = sum(range_values) / len(range_values)
                mean_values.append(mean_value)

    
        mean = np.mean(mean_values)
        std_value = np.std(mean_values)
        R = mean - 3 * std_value  

        ko_data = []

        for enzyme, values in enzyme_results_data.items():
            if values.get('range') and len(values['range']) >= 2:
                range_values = values['range']
                first_value = range_values[0]


                if enzyme in enzyme_overresults_data:
                    overresults_values = enzyme_overresults_data[enzyme]
                    if overresults_values.get('range') and len(overresults_values['range']) >= 2:
                        over_second_value = overresults_values['range'][1]


                        if first_value >= R and over_second_value <= R and range_values != [0, 0]:
                            ko_data.append((enzyme, range_values))

        # up flux
        up_data = []

        for enzyme, values in enzyme_results_data.items():
            if values.get('range') and len(values['range']) >= 2:
                range_values = values['range']
                second_value = range_values[1]
                
                if enzyme in enzyme_overresults_data:
                    overresults_values = enzyme_overresults_data[enzyme]
                    
                    if overresults_values.get('range') and len(overresults_values['range']) >= 2:
                        over_range_values = overresults_values['range']
                        over_first_value = over_range_values[0]
                        over_second_value = over_range_values[1]
                        
                        num_zeros = sum(1 for value in range_values if value == 0)
                        if num_zeros != 3 and second_value <= over_first_value and (range_values != [0, 0] or over_range_values != [0, 0]):
                            avg_values = sum(range_values) / len(range_values)
                            avg_over_values = sum(over_range_values) / len(over_range_values)
                            
                            if avg_values > 1e-1 or avg_over_values > 1e-1:
                                up_data.append((enzyme, range_values))


        # down_flux
        # down_flux
        down_data = []

        for enzyme, values in enzyme_results_data.items():
            if values.get('range') and len(values['range']) >= 2:
                range_values = values['range']
                first_value = range_values[0]

                if enzyme in enzyme_overresults_data:
                    overresults_values = enzyme_overresults_data[enzyme]
                    if overresults_values.get('range') and len(overresults_values['range']) >= 2:
                        over_range_values = overresults_values['range']
                        over_first_value = overresults_values['range'][0]
                        over_second_value = overresults_values['range'][1]

                        num_zeros = sum(1 for value in range_values if value == 0)

                        if num_zeros != 3 and first_value >= over_second_value and first_value >= over_first_value and (range_values != [0, 0] or over_range_values != [0, 0]):
                            avg_values = sum(range_values) / len(range_values)
                            avg_over_values = sum(over_range_values) / len(over_range_values) 

                            if avg_values > 1e-1 or avg_over_values > 1e-1:                       
                                down_data.append((enzyme, range_values))

    if inputdic['mode'] == 'SE' or inputdic['mode'] == 'SET':
        # ko_flux
        mean_values = []

    
        for values in enzyme_results_data.values():
            if values.get('range') and len(values['range']) >= 2:
                range_values = values['range']
                mean_value = sum(range_values) / len(range_values)
                mean_values.append(mean_value)

    
        mean = np.mean(mean_values)
        std_value = np.std(mean_values)
        R = mean - 3 * std_value  

        ko_data = []

        for enzyme, values in enzyme_results_data.items():
            if values.get('range') and len(values['range']) >= 2:
                range_values = values['range']
                first_value = range_values[0]


                if enzyme in enzyme_overresults_data:
                    overresults_values = enzyme_overresults_data[enzyme]
                    if overresults_values.get('range') and len(overresults_values['range']) >= 2:
                        over_second_value = overresults_values['range'][1]


                        if first_value >= R and over_second_value <= R and range_values != [0, 0]:
                            ko_data.append((enzyme, range_values))

        # up flux
        up_data = []

        for enzyme, values in enzyme_results_data.items():
            if values.get('range') and len(values['range']) >= 2:
                range_values = values['range']
                second_value = range_values[1]
                if enzyme in enzyme_overresults_data:
                    overresults_values = enzyme_overresults_data[enzyme]
                    if overresults_values.get('range') and len(overresults_values['range']) >= 2:
                        over_first_value = overresults_values['range'][0]
                        over_second_value = overresults_values['range'][1]


                        if second_value <= over_first_value and range_values != [0, 0]:
                            up_data.append((enzyme, range_values))

        # down_flux
        down_data = []

        for enzyme, values in enzyme_results_data.items():
            if values.get('range') and len(values['range']) >= 2:
                range_values = values['range']
                first_value = range_values[0]

                if enzyme in enzyme_overresults_data:
                    overresults_values = enzyme_overresults_data[enzyme]
                    if overresults_values.get('range') and len(overresults_values['range']) >= 2:
                        over_first_value = overresults_values['range'][0]
                        over_second_value = overresults_values['range'][1]


                        if first_value >= over_second_value and first_value >= over_first_value and range_values != [0, 0]:
                            down_data.append((enzyme, range_values))

    return ko_data,up_data,down_data



# get equation
def gene_reaction_map1(reaction_list,model):
    """
    Maps reaction IDs to their corresponding biochemical equations from the given model.

    Args:
    - reaction_list (list): A list of reaction IDs to be mapped.
    - model (Model): A metabolic model containing the reactions.

    Returns:
    - equation_dict (dict): A dictionary where the keys are reaction IDs and the values are their 
      corresponding biochemical equations.
    """
    equation_dict = {}

    for reaction_id in reaction_list:
        equation = model.reactions.get_by_id(reaction_id).reaction
        equation_dict[reaction_id] = equation
    return equation_dict
# wild enzyme cost 
def ref_e_con(inputdic,Concretemodel_Need_Data,totalE,path_strain,objvalue2):
    """
    Computes enzyme distribution and reaction values for a given metabolic model under specific constraints.

    Args:
    - inputdic (dict): A dictionary containing input parameters including substrate, biomass, and mode.
    - Concretemodel_Need_Data (dict): A dictionary containing data needed for the concrete model such as molecular weights and kcat values.
    - totalE (float): The total enzyme constraint value.
    - path_strain (str): The path strain identifier (e.g., 'iML1515').
    - objvalue2 (float): The substrate constraint value.

    Returns:
    - E_refdict (dict): A dictionary where the keys are genes and the values are the calculated enzyme amounts.
    - mw_dict (dict): A dictionary where the keys are genes and the values are their molecular weights.
    - reaction_dict_bio (dict): A dictionary where the keys are reaction IDs and the values are their reaction values if above a threshold.
    - kcat_dict (dict): A dictionary where the keys are reaction IDs and the values are their kcat values if they exist in the model.
    """
    # wild fix min_enz, biomass and B, maximize product, get value(model.reactions[i]) for i in reaction_list
    constr_coeff={}
    constr_coeff['fix_reactions']={}
    Concretemodel_Need_Data['E_total']=totalE
    if path_strain == 'iML1515':    
        Concretemodel_Need_Data['E_total']=totalE*1.118
    constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
    obj_name=inputdic['biomass']
    obj_target='maximize'

    EcoECM_FBA_protainmodel_MIN=FBA_template2(set_obj_value=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
    Model_Solve(EcoECM_FBA_protainmodel_MIN,'gurobi')
    print(EcoECM_FBA_protainmodel_MIN.obj())
    # wild e1
    e_ref={x: value(EcoECM_FBA_protainmodel_MIN.e1[x]) for x in EcoECM_FBA_protainmodel_MIN.e1}
    mw_dict=Concretemodel_Need_Data['mw_dict']
    reaction_dict = {x: value(EcoECM_FBA_protainmodel_MIN.reaction[x]) for x in EcoECM_FBA_protainmodel_MIN.reaction}
    reaction_dict_bio = {key: value for key, value in reaction_dict.items() if value > 1e-1}
    E_refdict = {}
    kcat_dict=Concretemodel_Need_Data['kcat_dict']
    kcat_dict={key:value for key,value in kcat_dict.items()if key in reaction_dict}
    for gene in e_ref:
        if gene in mw_dict:
            result = e_ref[gene] * mw_dict[gene]
            E_refdict[gene] = result

    E_refdict = {k: v for k, v in E_refdict.items() if v > 0}
    return E_refdict,mw_dict,reaction_dict_bio,kcat_dict


# get gene_reactions_map 
def gene_reaction_map(enzyme_list, filtered_reaction_dict,kcat_dict,Concretemodel_Need_Data,model):
    """
    Maps genes to their associated reactions and kcat values, generating detailed reaction equations and kcat mappings.

    Args:
    - enzyme_list (list): List of enzymes (genes) to be mapped.
    - filtered_reaction_dict (dict): Dictionary of filtered reactions with their descriptions.
    - kcat_dict (dict): Dictionary of kcat values for reactions.
    - Concretemodel_Need_Data (dict): Dictionary containing the enzyme reaction data and other necessary data.
    - model (object): Metabolic model object containing reactions.

    Returns:
    - new_gene_reaction_mapping (dict): Mapping of specified enzymes to their reaction equations.
    - gene_reaction_mapping (dict): Mapping of all genes to their reaction equations.
    - gene_kcat_mapping (dict): Mapping of genes to their reactions and corresponding kcat values.
    """

    enzyme_reaction = Concretemodel_Need_Data['enzyme_rxns_dict']
    gene_reaction_mapping = {}
    gene_kcat_mapping = {}
    new_gene_reaction_mapping = {}

    for gene, reactions in enzyme_reaction.items():
        reaction_equations = []
        for reaction_name in reactions:
            if reaction_name in filtered_reaction_dict:
                reaction = model.reactions.get_by_id(reaction_name)
                reaction_equation = f'{reaction_name} ({filtered_reaction_dict[reaction_name]}): {reaction.reaction}'
                reaction_equations.append(reaction_equation)
        if reaction_equations:
            gene_reaction_mapping[gene] = ", ".join(reaction_equations)

    for gene in enzyme_list:
        if gene in gene_reaction_mapping:
            new_gene_reaction_mapping[gene] = gene_reaction_mapping[gene]

    for gene, reactions in enzyme_reaction.items():
        for reaction_name in reactions:
            if reaction_name in kcat_dict:
                kcat_value = kcat_dict[reaction_name]
                if gene in gene_kcat_mapping:
                    gene_kcat_mapping[gene].append((reaction_name, kcat_value))
                else:
                    gene_kcat_mapping[gene] = [(reaction_name, kcat_value)]
    

    return new_gene_reaction_mapping, gene_reaction_mapping,gene_kcat_mapping




def reaction_flux(inputdic,Concretemodel_Need_Data,totalE2,path_strain,totalE,objvalue2,v0_biomass):
    """
    Determines the reaction fluxes in a metabolic model under given constraints and objectives.

    Args:
    - inputdic (dict): Dictionary containing input parameters such as substrate, biomass, product, and mode.
    - Concretemodel_Need_Data (dict): Dictionary containing necessary data for the concrete model.
    - totalE2 (float): Total enzyme concentration for the current condition.
    - path_strain (str): Identifier for the strain used in the pathway.
    - totalE (float): Total enzyme concentration for the reference condition.
    - objvalue2 (float): Objective value for the substrate constraint.
    - v0_biomass (float): Biomass constraint value.

    Returns:
    - reaction_dict (dict): Dictionary of reaction IDs and their corresponding flux values.
    - kcat_dict (dict): Dictionary of kcat values for reactions present in reaction_dict.
    - EcoECM_FBA_protainmodel_pro_max (object): The solved FBA model object.
    """
    constr_coeff={}
    constr_coeff['fix_reactions']={}
    Concretemodel_Need_Data['E_total']=totalE2
    if path_strain == 'iML1515':    
        Concretemodel_Need_Data['E_total']=totalE*1.01
    constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
    constr_coeff['biomass_constrain']=(inputdic['biomass'],v0_biomass*0.1)
    obj_name=inputdic['product']
    obj_target='maximize'

    EcoECM_FBA_protainmodel_pro_max=FBA_template2(set_obj_value=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
    Model_Solve(EcoECM_FBA_protainmodel_pro_max,'gurobi')
    print(EcoECM_FBA_protainmodel_pro_max.obj())

    # get reaction flux
    reaction_dict = {x: value(EcoECM_FBA_protainmodel_pro_max.reaction[x]) for x in EcoECM_FBA_protainmodel_pro_max.reaction}
    reaction_dict = {key: value for key, value in reaction_dict.items() if value > 1e-1}
    kcat_dict=Concretemodel_Need_Data['kcat_dict']
    kcat_dict={key:value for key,value in kcat_dict.items()if key in reaction_dict}
    return reaction_dict,kcat_dict,EcoECM_FBA_protainmodel_pro_max



# analysis enzyme usage_over
# e1*MW
def enzyme_usage_over(EcoECM_FBA_protainmodel_pro_max,Concretemodel_Need_Data):
    """
    Calculates the enzyme usage and normalizes it based on total enzyme concentration.

    Args:
    - EcoECM_FBA_protainmodel_pro_max (object): The solved FBA model object containing enzyme variables.
    - Concretemodel_Need_Data (dict): Dictionary containing necessary data for the concrete model, including molecular weights.

    Returns:
    - normalized_E_dict (dict): Dictionary of genes and their corresponding normalized enzyme usage as a percentage of total enzyme usage.
    - E_dict (dict): Dictionary of genes and their corresponding enzyme usage values calculated using molecular weights.
    - e_dict (dict): Dictionary of genes and their corresponding raw enzyme amounts from the FBA model.
    """
    e_dict = {x: value(EcoECM_FBA_protainmodel_pro_max.e1[x]) for x in EcoECM_FBA_protainmodel_pro_max.e1}
    mw_dict=Concretemodel_Need_Data['mw_dict']
    E_dict = {}
    for gene in e_dict:
        if gene in mw_dict:
            result = e_dict[gene] * mw_dict[gene]
            E_dict[gene] = result
    # sum e
    total_sum = sum(E_dict.values())
    print("sum:", total_sum)
    # next (single enzyme usage_over)
    normalized_E_dict = {gene: (value / total_sum) * 100 for gene, value in E_dict.items()}
    return normalized_E_dict,E_dict,e_dict




def fold_change(E_dict,E_refdict):
    """
    Calculates the fold change of enzyme usage between two datasets.

    Args:
    - E_dict (dict): Dictionary of genes and their corresponding enzyme usage values.
    - E_refdict (dict): Dictionary of genes and their corresponding reference enzyme usage values.

    Returns:
    - Fold_change (dict): Dictionary of genes and their corresponding fold change values, calculated as the ratio of enzyme usage in E_dict to E_refdict.
    """
    Fold_change={}
    for gene in E_dict:
        if gene in E_refdict:
            results = E_dict[gene]/E_refdict[gene]
            Fold_change[gene]=results
    return  Fold_change    


def basic(bio,pro,B_value1,v0_biomass,B_value2,v1_product_max,totalE,totalE2,inputdic):
    """
    Constructs dataframes for metabolite values and key information based on the specified mode.

    Args:
    - bio (dict): A dictionary containing metabolites and their corresponding values for the wildtype.
    - pro (dict): A dictionary containing metabolites and their corresponding values for the overexpressed condition.
    - B_value1 (float): MDF value for the wildtype.
    - v0_biomass (float): Growth value for the wildtype.
    - B_value2 (float): MDF value for the overexpressed condition.
    - v1_product_max (float): Maximum product value for the overexpressed condition.
    - totalE (float): Total enzyme usage for the wildtype.
    - totalE2 (float): Total enzyme usage for the overexpressed condition.
    - inputdic (dict): A dictionary specifying the mode of operation (e.g., 'S', 'ST', 'SE', 'SET').

    Returns:
    - bio_df (DataFrame): A DataFrame containing metabolite values for the wildtype.
    - pro_df (DataFrame): A DataFrame containing metabolite values for the overexpressed condition.
    - KeyInfo (DataFrame): A DataFrame containing key information based on the specified mode.
    """

    if inputdic['mode'] =='S':
        data = {
        'Metabolite': list(bio.keys()),
        'Value': list(bio.values())
        }
        bio_df = pd.DataFrame(data)
        data = {
        'Metabolite': list(pro.keys()),
        'Value': list(pro.values())
        }
        pro_df = pd.DataFrame(data)
        KeyInfo = pd.DataFrame() 

    if inputdic['mode'] =='ST':
        data = {
        'Metabolite': list(bio.keys()),
        'Value': list(bio.values())
        }
        bio_df = pd.DataFrame(data)
        data = {
        'Metabolite': list(pro.keys()),
        'Value': list(pro.values())
        }
        pro_df = pd.DataFrame(data)

        keyInfo=dict()
        keyInfo['wildtype']=dict()
        keyInfo['over']=dict()
        keyInfo['wildtype']['mdf']=B_value1
        keyInfo['wildtype']['growth']=v0_biomass
        keyInfo['over']['mdf']=B_value2
        keyInfo['over']['product']=v1_product_max
        KeyInfo = pd.DataFrame(keyInfo)
    
    if inputdic['mode'] =='SE':
        data = {
        'Metabolite': list(bio.keys()),
        'Value': list(bio.values())
        }
        bio_df = pd.DataFrame(data)
        data = {
        'Metabolite': list(pro.keys()),
        'Value': list(pro.values())
        }
        pro_df = pd.DataFrame(data)
        keyInfo=dict()
        keyInfo['wildtype']=dict()
        keyInfo['over']=dict()
        keyInfo['wildtype']['growth']=v0_biomass
        keyInfo['wildtype']['min_enz']=totalE
        keyInfo['over']['product']=v1_product_max
        keyInfo['over']['min_enz']=totalE2
        KeyInfo = pd.DataFrame(keyInfo)
    if inputdic['mode'] =='SET':
        data = {
        'Metabolite': list(bio.keys()),
        'Value': list(bio.values())
        }
        bio_df = pd.DataFrame(data)
        data = {
        'Metabolite': list(pro.keys()),
        'Value': list(pro.values())
        }
        pro_df = pd.DataFrame(data)
        keyInfo=dict()
        keyInfo['wildtype']=dict()
        keyInfo['over']=dict()
        keyInfo['wildtype']['mdf']=B_value1
        keyInfo['wildtype']['growth']=v0_biomass
        keyInfo['wildtype']['min_enz']=totalE
        keyInfo['over']['mdf']=B_value2
        keyInfo['over']['product']=v1_product_max
        keyInfo['over']['min_enz']=totalE2
        KeyInfo = pd.DataFrame(keyInfo)

    return bio_df,pro_df,KeyInfo




def bottleneck_reactions(EcoECM_FBA_protainmodel_B2,model):
    """
    Identifies bottleneck reactions based on flux and diffusion values from the given FBA model.

    Args:
    - EcoECM_FBA_protainmodel_B2 (object): An FBA model containing reaction flux and diffusion coefficients.
    - model (object): A model object used to retrieve reaction equations.

    Returns:
    - bottleneck_reactions_list (list): A list of reaction IDs that are identified as bottleneck reactions.
    - Df (DataFrame): A DataFrame containing details of the bottleneck reactions, including reaction ID, equation, diffusion value, and flux value.
    """
    Df_dict = {x: value(EcoECM_FBA_protainmodel_B2.Df[x]) for x in EcoECM_FBA_protainmodel_B2.Df}
    # get reaction and flux
    reaction={x:value(EcoECM_FBA_protainmodel_B2.reaction[x]) for x in EcoECM_FBA_protainmodel_B2.reaction}
    # get reaction>10-3
    filtered_reaction = {k: v for k, v in reaction.items() if v > 1e-3}
    # get df=B
    B_dict={x:value(EcoECM_FBA_protainmodel_B2.B[x]) for x in EcoECM_FBA_protainmodel_B2.B}
    B = next(iter(B_dict.values()))
    matching_keys = [key for key, value in Df_dict.items() if value == B]
    # find bottleneck_rxn
    matching_keys_set = set(matching_keys)
    filtered_reaction_set = set(filtered_reaction.keys())
    same_reactions = matching_keys_set.intersection(filtered_reaction_set)
    bottleneck_reactions_list = list(same_reactions)
    reaction_data = []
    for reaction_id in bottleneck_reactions_list:
        equation = model.reactions.get_by_id(reaction_id).reaction
        df_value = Df_dict.get(reaction_id, None)
        flux_value =reaction.get(reaction_id,None)
        reaction_data.append((reaction_id, equation, df_value,flux_value))

    Df = pd.DataFrame(reaction_data, columns=["Reaction_ID", "Equation", "df_Value","flux_value"])
    Df = Df.sort_values(by='flux_value', ascending=False)
    Df['df_Value'] = Df['df_Value'].apply(lambda x: format(x, '.3e') if isinstance(x, (int, float)) else x)
    Df['flux_value']=Df['flux_value'].apply(lambda x: format(x, '.3e') if isinstance(x, (int, float)) else x)
    return bottleneck_reactions_list,Df


def output(KeyInfo,path_results,bio_df,pro_df,meged_df,inputdic,Df):
    """
    Writes the results of the analysis to an Excel file, with sheets based on the specified mode.

    Args:
    - KeyInfo (DataFrame): A DataFrame containing key information about the analysis.
    - path_results (str): The path to the directory where the results will be saved.
    - bio_df (DataFrame): A DataFrame containing the biological data.
    - pro_df (DataFrame): A DataFrame containing the product data.
    - meged_df (DataFrame): A DataFrame containing merged data.
    - inputdic (dict): A dictionary containing input parameters, including the mode.
    - Df (DataFrame): A DataFrame containing details of bottleneck reactions (optional).

    Returns:
    - None: The function saves the output directly to an Excel file and does not return any values.
    """
    if inputdic['mode'] =='S':
        filename=os.path.join(path_results, 'results.xlsx')
        with pd.ExcelWriter(filename) as writer:
            bio_df.to_excel(writer,sheet_name='EX_ref',index=True)
            pro_df.to_excel(writer,sheet_name='EX_high',index=True)
            meged_df.to_excel(writer,sheet_name='MUST',index=True)

    if inputdic['mode'] =='ST':
        filename=os.path.join(path_results, 'results.xlsx')
        with pd.ExcelWriter(filename) as writer:
            KeyInfo.to_excel(writer,sheet_name='basic',index=True)
            bio_df.to_excel(writer,sheet_name='EX_ref',index=True)
            pro_df.to_excel(writer,sheet_name='EX_high',index=True)
            meged_df.to_excel(writer,sheet_name='MUST',index=True)
            Df.to_excel(writer,sheet_name='bottleneck-Rxn',index=True)

    if inputdic['mode'] =='SE':
        filename=os.path.join(path_results, 'results.xlsx')
        with pd.ExcelWriter(filename) as writer:
            KeyInfo.to_excel(writer,sheet_name='basic',index=True)
            bio_df.to_excel(writer,sheet_name='EX_ref',index=True)
            pro_df.to_excel(writer,sheet_name='EX_high',index=True)
            meged_df.to_excel(writer,sheet_name='MUST',index=True)

    if inputdic['mode'] =='SET':
        filename=os.path.join(path_results, 'results.xlsx')
        with pd.ExcelWriter(filename) as writer:
            KeyInfo.to_excel(writer,sheet_name='basic',index=True)
            bio_df.to_excel(writer,sheet_name='EX_ref',index=True)
            pro_df.to_excel(writer,sheet_name='EX_high',index=True)
            meged_df.to_excel(writer,sheet_name='MUST',index=True)
            Df.to_excel(writer,sheet_name='bottleneck-Rxn',index=True)   
def output_rea_e0(KeyInfo,path_results,bio_df,pro_df,meged_df,inputdic,Df):
    """
    Writes the results of the analysis to an Excel file, with sheets based on the specified mode.

    Args:
    - KeyInfo (DataFrame): A DataFrame containing key information about the analysis.
    - path_results (str): The path to the directory where the results will be saved.
    - bio_df (DataFrame): A DataFrame containing the biological data.
    - pro_df (DataFrame): A DataFrame containing the product data.
    - meged_df (DataFrame): A DataFrame containing merged data.
    - inputdic (dict): A dictionary containing input parameters, including the mode.
    - Df (DataFrame): A DataFrame containing details of bottleneck reactions (optional).

    Returns:
    - None: The function saves the output directly to an Excel file and does not return any values.
    """
    if inputdic['mode'] =='S':
        filename=os.path.join(path_results, 'results.xlsx')
        with pd.ExcelWriter(filename) as writer:
            bio_df.to_excel(writer,sheet_name='EX_ref',index=True)
            pro_df.to_excel(writer,sheet_name='EX_high',index=True)
            meged_df.to_excel(writer,sheet_name='MUST',index=True)

    if inputdic['mode'] =='ST':
        filename=os.path.join(path_results, 'results.xlsx')
        with pd.ExcelWriter(filename) as writer:
            KeyInfo.to_excel(writer,sheet_name='basic',index=True)
            bio_df.to_excel(writer,sheet_name='EX_ref',index=True)
            pro_df.to_excel(writer,sheet_name='EX_high',index=True)
            meged_df.to_excel(writer,sheet_name='MUST',index=True)
            Df.to_excel(writer,sheet_name='bottleneck-Rxn',index=True)

    if inputdic['mode'] =='SE':
        filename=os.path.join(path_results, 'results.xlsx')
        with pd.ExcelWriter(filename) as writer:
            KeyInfo.to_excel(writer,sheet_name='basic',index=True)
            bio_df.to_excel(writer,sheet_name='EX_ref',index=True)
            pro_df.to_excel(writer,sheet_name='EX_high',index=True)
            meged_df.to_excel(writer,sheet_name='MUST',index=True)

    if inputdic['mode'] =='SET':
        filename=os.path.join(path_results, 'results_reaction_e0.xlsx')
        with pd.ExcelWriter(filename) as writer:
            KeyInfo.to_excel(writer,sheet_name='basic',index=True)
            bio_df.to_excel(writer,sheet_name='EX_ref',index=True)
            pro_df.to_excel(writer,sheet_name='EX_high',index=True)
            meged_df.to_excel(writer,sheet_name='MUST',index=True)
            Df.to_excel(writer,sheet_name='bottleneck-Rxn',index=True)   


def must_df(inputdic,equation_dict,new_gene_reaction_mapping_bio,new_gene_reaction_mapping,E_refdict,E_dict,Fold_change,normalized_E_dict,e_dict,gene_kcat_mapping,enzyme_results_data,enzyme_overresults2_data,ko_data,up_data,down_data,Concretemodel_Need_Data):
    """
    Constructs a DataFrame summarizing enzyme usage and reaction fluxes based on the specified mode.

    Args:
    - inputdic (dict): A dictionary containing input parameters, including the mode.
    - equation_dict (dict): A dictionary mapping reaction IDs to their equations.
    - new_gene_reaction_mapping_bio (dict): Mapping of genes to biological reactions.
    - new_gene_reaction_mapping (dict): Mapping of genes to product reactions.
    - E_refdict (dict): Dictionary containing reference enzyme concentrations.
    - E_dict (dict): Dictionary containing enzyme concentrations after manipulation.
    - Fold_change (dict): Dictionary of fold changes for each gene.
    - normalized_E_dict (dict): Normalized enzyme usage data.
    - e_dict (dict): Dictionary of enzyme concentrations for genes.
    - gene_kcat_mapping (dict): Dictionary mapping genes to their kcat values.
    - enzyme_results_data (dict): Dictionary containing results data for wild-type enzymes.
    - enzyme_overresults2_data (dict): Dictionary containing results data for over-expressed enzymes.
    - ko_data (list): List of knockout gene data.
    - up_data (list): List of upregulated gene data.
    - down_data (list): List of downregulated gene data.
    - Concretemodel_Need_Data (dict): Dictionary containing concrete model data, including molecular weights.

    Returns:
    - DataFrame: A DataFrame summarizing enzyme usage, reaction fluxes, and manipulations.
    """
    if inputdic['mode'] == 'S':
        wild_data =[{'reaction':reaction,'flux_wild':[format(value,'.3e') for value in data['range']]}for reaction,data in enzyme_results_data.items()]
        df1=pd.DataFrame(wild_data)
        over_data =[{'reaction':reaction,'flux_over':[format(value,'.3e') for value in data['range']]}for reaction,data in enzyme_overresults2_data.items()]
        df2=pd.DataFrame(over_data)
        meged_df =pd.merge(df1,df2,on='reaction',how='inner')
        meged_df['equation'] = meged_df['reaction'].map(equation_dict)
        meged_df['manipulations'] = None

        for reaction, _ in ko_data:
            meged_df.loc[meged_df['reaction'] == reaction, 'manipulations'] = 'ko'
        for reaction, _ in up_data:
            meged_df.loc[meged_df['reaction'] == reaction, 'manipulations'] = 'Up'
        for reaction, _ in down_data:
            meged_df.loc[meged_df['reaction'] == reaction, 'manipulations'] = 'down'
    
    if inputdic['mode'] == 'ST':
        wild_data =[{'reaction':reaction,'flux_wild':[format(value,'.3e') for value in data['range']]}for reaction,data in enzyme_results_data.items()]
        df1=pd.DataFrame(wild_data)
        over_data =[{'reaction':reaction,'flux_over':[format(value,'.3e') for value in data['range']]}for reaction,data in enzyme_overresults2_data.items()]
        df2=pd.DataFrame(over_data)
        meged_df =pd.merge(df1,df2,on='reaction',how='inner')
        meged_df['equation'] = meged_df['reaction'].map(equation_dict)
        meged_df['manipulations'] = None

        for reaction, _ in ko_data:
            meged_df.loc[meged_df['reaction'] == reaction, 'manipulations'] = 'ko'
        for reaction, _ in up_data:
            meged_df.loc[meged_df['reaction'] == reaction, 'manipulations'] = 'Up'
        for reaction, _ in down_data:
            meged_df.loc[meged_df['reaction'] == reaction, 'manipulations'] = 'down'
    
    if inputdic['mode'] == 'SE':
        # make excel
        wild_data =[{'gene':gene,'enzyme_wild':[format(value,'.3e') for value in data['range']]}for gene,data in enzyme_results_data.items()]
        df1=pd.DataFrame(wild_data)
        over_data =[{'gene':gene,'enzyme_over':[format(value,'.3e') for value in data['range']]}for gene,data in enzyme_overresults2_data.items()]
        df2=pd.DataFrame(over_data)
        meged_df =pd.merge(df1,df2,on='gene',how='inner')
        meged_df['reaction_bio']=meged_df['gene'].map(new_gene_reaction_mapping_bio)
        meged_df['reaction_pro']=meged_df['gene'].map(new_gene_reaction_mapping)
        meged_df['ref_e_con(g/gDW)']=meged_df['gene'].map(E_refdict).apply(lambda x: format(x,'.3e'))
        meged_df['over_e_con(g/gDW)']=meged_df['gene'].map(E_dict).apply(lambda x: format(x,'.3e'))
        meged_df['Fold_change']=meged_df['gene'].map(Fold_change).apply(lambda x: format(x,'.3e'))
        meged_df['enzyme_usage_over']=meged_df['gene'].map(normalized_E_dict)
        meged_df['enzyme_usage_over'] = meged_df['enzyme_usage_over'].apply(float)
        meged_df = meged_df.sort_values(by='enzyme_usage_over', ascending=False)
        meged_df['enzyme_usage_over'] = meged_df['enzyme_usage_over'].apply(lambda x: format(x, '.3e'))
        meged_df.reset_index(drop=True, inplace=True)

        meged_df['manipulations'] = None

        for gene, _ in ko_data:
            meged_df.loc[meged_df['gene'] == gene, 'manipulations'] = 'ko'
        for gene, _ in up_data:
            meged_df.loc[meged_df['gene'] == gene, 'manipulations'] = 'Up'
        for gene, _ in down_data:
            meged_df.loc[meged_df['gene'] == gene, 'manipulations'] = 'down'
        meged_df['e1(mmol/gDW)']=meged_df['gene'].map(e_dict).apply(lambda x: format(x,'.3e'))
        meged_df['kcat(1/h)'] = meged_df['gene'].map(gene_kcat_mapping)
        mw_dict=Concretemodel_Need_Data['mw_dict']
        meged_df['mw(g/mg)'] = meged_df['gene'].map(mw_dict).apply(lambda x: format(x,'.3e'))

    if inputdic['mode'] =='SET':
        # make excel
        wild_data =[{'gene':gene,'enzyme_wild':[format(value,'.3e') for value in data['range']]}for gene,data in enzyme_results_data.items()]
        df1=pd.DataFrame(wild_data)
        over_data =[{'gene':gene,'enzyme_over':[format(value,'.3e') for value in data['range']]}for gene,data in enzyme_overresults2_data.items()]
        df2=pd.DataFrame(over_data)
        meged_df =pd.merge(df1,df2,on='gene',how='inner')
        meged_df['reaction_bio']=meged_df['gene'].map(new_gene_reaction_mapping_bio)
        meged_df['reaction_pro']=meged_df['gene'].map(new_gene_reaction_mapping)
        meged_df['ref_e_con(g/gDW)']=meged_df['gene'].map(E_refdict).apply(lambda x: format(x,'.3e'))
        meged_df['over_e_con(g/gDW)']=meged_df['gene'].map(E_dict).apply(lambda x: format(x,'.3e'))
        meged_df['Fold_change']=meged_df['gene'].map(Fold_change).apply(lambda x: format(x,'.3e'))
        meged_df['enzyme_usage_over']=meged_df['gene'].map(normalized_E_dict)
        meged_df['enzyme_usage_over'] = meged_df['enzyme_usage_over'].apply(float)
        meged_df = meged_df.sort_values(by='enzyme_usage_over', ascending=False)
        meged_df['enzyme_usage_over'] = meged_df['enzyme_usage_over'].apply(lambda x: format(x, '.3e'))
        meged_df.reset_index(drop=True, inplace=True)

        meged_df['manipulations'] = None

        for gene, _ in ko_data:
            meged_df.loc[meged_df['gene'] == gene, 'manipulations'] = 'ko'
        for gene, _ in up_data:
            meged_df.loc[meged_df['gene'] == gene, 'manipulations'] = 'Up'
        for gene, _ in down_data:
            meged_df.loc[meged_df['gene'] == gene, 'manipulations'] = 'down'
        meged_df['e1(mmol/gDW)']=meged_df['gene'].map(e_dict).apply(lambda x: format(x,'.3e'))
        meged_df['kcat(1/h)'] = meged_df['gene'].map(gene_kcat_mapping)
        mw_dict=Concretemodel_Need_Data['mw_dict']
        meged_df['mw(g/mg)'] = meged_df['gene'].map(mw_dict).apply(lambda x: format(x,'.3e'))
    return meged_df

def calculate_product_fseof(Concretemodel_Need_Data,inputdic,model):
    """
    Calculates the maximum product yield based on the specified input conditions and constraints.

    Args:
    - Concretemodel_Need_Data (dict): A dictionary containing model-specific data.
    - inputdic (dict): A dictionary containing input parameters, including substrate and product information.
    - model (Model): The metabolic model to be optimized.

    Returns:
    - float: The maximum product yield based on the specified constraints.
    - float: The objective value for the substrate optimization.
    """
    model.objective = inputdic['substrate']
    objvalue2=model.optimize().objective_value
    constr_coeff={}
    constr_coeff['fix_reactions']={}
    constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
    obj_name=inputdic['product']
    obj_target='maximize'
    if inputdic['mode'] == 'S':
        EcoECM_FBA_protainmodel=FBA_template2(set_obj_value=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel,'gurobi')
        print(EcoECM_FBA_protainmodel.obj())
        product=EcoECM_FBA_protainmodel.obj()
    if inputdic['mode'] == 'SE':
        Concretemodel_Need_Data['E_total']=0.56* np.mean([0.45311986236929197,0.4622348377433211,0.4600801040374112])
        EcoECM_FBA_protainmodel=FBA_template2(set_obj_value=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel,'gurobi')
        product=EcoECM_FBA_protainmodel.obj()
        print(EcoECM_FBA_protainmodel.obj())

    if inputdic['mode'] == 'ST':
        Concretemodel_Need_Data['B_value']=0
        Concretemodel_Need_Data['K_value']=1249
        EcoECM_FBA_protainmodel=FBA_template2(set_obj_value=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel,'gurobi')
        print(EcoECM_FBA_protainmodel.obj())
        product=EcoECM_FBA_protainmodel.obj()
    if inputdic['mode'] == 'SET':
        Concretemodel_Need_Data['E_total']=0.56* np.mean([0.45311986236929197,0.4622348377433211,0.4600801040374112])
        Concretemodel_Need_Data['B_value']=0
        Concretemodel_Need_Data['K_value']=1249
        EcoECM_FBA_protainmodel=FBA_template2(set_obj_value=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
        Model_Solve(EcoECM_FBA_protainmodel,'gurobi')
        print(EcoECM_FBA_protainmodel.obj())
        product=EcoECM_FBA_protainmodel.obj()

    return product,objvalue2



def gene_reaction_map2(enzyme_list, filtered_reaction_dict,kcat_dict,Concretemodel_Need_Data,model):
    enzyme_reaction = Concretemodel_Need_Data['enzyme_rxns_dict']
    gene_reaction_mapping = {}
    gene_kcat_mapping = {}
    new_gene_reaction_mapping = {}

    for gene, reactions in enzyme_reaction.items():
        reaction_equations = []
        for reaction_name in reactions:
            if reaction_name in filtered_reaction_dict:
                reaction = model.reactions.get_by_id(reaction_name)
                reaction_equation = f'{reaction_name} ({filtered_reaction_dict[reaction_name]}): {reaction.reaction}'
                reaction_equations.append(reaction_equation)
        if reaction_equations:
            gene_reaction_mapping[gene] = ", ".join(reaction_equations)

    for gene in enzyme_list:
        if gene in gene_reaction_mapping:
            new_gene_reaction_mapping[gene] = gene_reaction_mapping[gene]

    for gene, reactions in enzyme_reaction.items():
        for reaction_name in reactions:
            if reaction_name in kcat_dict:
                kcat_value = kcat_dict[reaction_name]
                if gene in gene_kcat_mapping:
                    gene_kcat_mapping[gene].append((reaction_name, kcat_value))
                else:
                    gene_kcat_mapping[gene] = [(reaction_name, kcat_value)]
    return new_gene_reaction_mapping, gene_reaction_mapping,gene_kcat_mapping



def biomass(product,inputdic,Concretemodel_Need_Data,objvalue2,model,model0):
    if inputdic['mode'] =='S':
        exlist = list(np.linspace(0,product,10))
        exlistn = []
        for i in exlist:
            i = format(i,'.2f')
            exlistn.append(float(i))
        FSEOFdf = pd.DataFrame()
        reactiondf = pd.DataFrame()
        obj_name=inputdic['biomass']
        for i in exlistn:
            cond = i
            obj_name=inputdic['biomass']
            obj_target='maximize'
            mode = 'S'
            constr_coeff={}
            constr_coeff['fix_reactions']={}
            constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
            constr_coeff['fix_reactions'][inputdic['product']] = [cond*0.99,np.inf]
            EcoECM_FBA_protainmodel_max=FBA_template2(set_obj_value=True,obj_name=obj_name,obj_target=obj_target,mode=mode,constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
            Model_Solve(EcoECM_FBA_protainmodel_max,'gurobi')
            biomass=EcoECM_FBA_protainmodel_max.obj()
            print(EcoECM_FBA_protainmodel_max.obj())
            constr_coeff['fix_reactions'][inputdic['substrate']]=[value(EcoECM_FBA_protainmodel_max.reaction[inputdic['substrate']]),np.inf]
            constr_coeff['fix_reactions'][inputdic['biomass']]=[value(EcoECM_FBA_protainmodel_max.reaction[inputdic['biomass']]),np.inf]
            EcoECM_PFBA_protainmodel_wild=FBA_template2(set_obj_V_value=True,mode=mode,constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
            Model_Solve(EcoECM_PFBA_protainmodel_wild,'gurobi')
            EcoECM_PFBA_protainmodel_wild.obj()
            bio=showflux(EcoECM_PFBA_protainmodel_wild)
                
            reaction_dict = conbine_flux(EcoECM_PFBA_protainmodel_wild,model0)
            model_reaction_solution = pd.DataFrame(list(reaction_dict.items()), columns=['reaction', 'flux'])

            reactiondf['reaction'] = model_reaction_solution['reaction']
            reactiondf['product = '+str(cond)] = model_reaction_solution['flux']     

    if inputdic['mode'] =='ST':
        exlist = list(np.linspace(0,product,10))
        exlistn = []
        for i in exlist:
            i = format(i,'.2f')
            exlistn.append(float(i))
        FSEOFdf = pd.DataFrame()
        reactiondf = pd.DataFrame()
        obj_name=inputdic['biomass']
        for i in exlistn:
            cond = i
            obj_name=inputdic['biomass']
            constr_coeff={}
            constr_coeff['fix_reactions']={}
            Concretemodel_Need_Data['B_value']=0
            Concretemodel_Need_Data['K_value']=1249
            constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
            constr_coeff['fix_reactions'][inputdic['product']] = [cond*0.9,np.inf]
            obj_target='maximize'
            mode = 'ST'
            EcoECM_FBA_protainmodel=FBA_template2(set_obj_value=True,obj_name=obj_name,obj_target=obj_target,mode=mode,constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
            Model_Solve(EcoECM_FBA_protainmodel,'gurobi')
            print(EcoECM_FBA_protainmodel.obj())
            constr_coeff['fix_reactions'][inputdic['biomass']]=[value(EcoECM_FBA_protainmodel.reaction[inputdic['biomass']]*0.99),np.inf]
            EcoECM_FBA_protainmodel_B2=FBA_template2(set_obj_B_value=True,obj_name=obj_name,obj_target=obj_target,mode=mode,constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
            Model_Solve(EcoECM_FBA_protainmodel_B2,'gurobi')
            B_value2=EcoECM_FBA_protainmodel_B2.obj()
            print(EcoECM_FBA_protainmodel.obj())
            constr_coeff={}
            constr_coeff['fix_reactions']={}
            Concretemodel_Need_Data['B_value']=B_value2*0.95
            constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
            constr_coeff['fix_reactions'][inputdic['product']] =  [cond*0.99,np.inf]
            EcoECM_FBA_protainmodel_max=FBA_template2(set_obj_value=True,obj_name=obj_name,obj_target=obj_target,mode=mode,constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
            Model_Solve(EcoECM_FBA_protainmodel_max,'gurobi')
            biomass=EcoECM_FBA_protainmodel_max.obj()
            print(EcoECM_FBA_protainmodel_max.obj())
            reaction_dict = conbine_flux(EcoECM_FBA_protainmodel_max,model0)
            model_reaction_solution = pd.DataFrame(list(reaction_dict.items()), columns=['reaction', 'flux'])

            reactiondf['reaction'] = model_reaction_solution['reaction']
            reactiondf['product = '+str(cond)] = model_reaction_solution['flux']    
    if inputdic['mode'] == 'SE':
        exlist = list(np.linspace(0,product,10))
        exlistn = []
        for i in exlist:
            i = format(i,'.2f')
            exlistn.append(float(i))
        FSEOFdf = pd.DataFrame()
        reactiondf = pd.DataFrame()
        obj_name = inputdic['biomass']
        for i in exlistn:
            cond = i
            obj_name = inputdic['biomass']
            obj_target='maximize'
            mode = 'SE'
            constr_coeff={}
            constr_coeff['fix_reactions']={}
            Concretemodel_Need_Data['E_total']=0.56* np.mean([0.45311986236929197,0.4622348377433211,0.4600801040374112])
            constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
            constr_coeff['fix_reactions'][inputdic['product']] =  [cond*0.95,np.inf]
            EcoECM_FBA_protainmodel=FBA_template2(set_obj_value=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
            Model_Solve(EcoECM_FBA_protainmodel,'gurobi')
            biomass=EcoECM_FBA_protainmodel.obj()
            print(EcoECM_FBA_protainmodel.obj())
            # mini enzyme
            constr_coeff={}
            constr_coeff['fix_reactions']={}
            constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
            constr_coeff['fix_reactions'][inputdic['biomass']]=[biomass*0.95,np.inf]
            constr_coeff['product_constrain']=(inputdic['product'],cond*0.9)
            obj_target = 'minimize'
            obj_name = inputdic['biomass']
            EcoECM_FBA_protainmodel=FBA_template2(set_obj_sum_e=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
            Model_Solve(EcoECM_FBA_protainmodel,'gurobi')
            totalE2=EcoECM_FBA_protainmodel.obj()
            print(EcoECM_FBA_protainmodel.obj())
            #带入totalE求product
            constr_coeff = {}
            constr_coeff['fix_reactions'] = {}
            # constr_coeff['fix_reactions']['EX_glc__D_e_reverse']=10
            Concretemodel_Need_Data['E_total'] = totalE2*1.2
            constr_coeff['substrate_constrain'] = (inputdic['substrate'],objvalue2)
            constr_coeff['fix_reactions'][inputdic['product']] =  [cond*0.95,np.inf]
            obj_target='maximize'
            EcoECM_FBA_protainmodel_max = FBA_template2(set_obj_value=True,obj_name=obj_name,obj_target=obj_target,mode=inputdic['mode'],constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
            Model_Solve(EcoECM_FBA_protainmodel_max,'gurobi')
            biomass_max = EcoECM_FBA_protainmodel_max.obj()
            print(EcoECM_FBA_protainmodel_max.obj())

            e1={x: value(EcoECM_FBA_protainmodel_max.e1[x]) for x in EcoECM_FBA_protainmodel_max.e1}
            model_e1_solution = pd.DataFrame(list(e1.items()), columns=['gene', 'e1'])
            reaction_dict = {x: value(EcoECM_FBA_protainmodel_max.reaction[x]) for x in EcoECM_FBA_protainmodel_max.reaction}
            kcat_dict=Concretemodel_Need_Data['kcat_dict']
            kcat_dict={key:value for key,value in kcat_dict.items()if key in reaction_dict}
            model_reaction_solution = pd.DataFrame(list(reaction_dict.items()), columns=['reaction', 'flux'])
            enzyme_list = list(Concretemodel_Need_Data['mw_dict'].keys()) 
            new_gene_reaction_mapping,gene_reaction_mapping,gene_kcat_mapping=gene_reaction_map2(enzyme_list, reaction_dict,kcat_dict,Concretemodel_Need_Data,model)
            model_e1_solution = pd.DataFrame(list(e1.items()), columns=['gene', 'e1'])
            FSEOFdf['gene'] = model_e1_solution['gene']
            FSEOFdf['gene = '+str(cond)] = FSEOFdf['gene'].map(new_gene_reaction_mapping)
            FSEOFdf['cond = '+str(cond)] = model_e1_solution['e1']
            reactiondf['reaction'] = model_reaction_solution['reaction']
            reactiondf['cond = '+str(cond)] = model_reaction_solution['flux']   
    if inputdic['mode'] == 'SET':
        exlist = list(np.linspace(0,product,10))
        exlistn = []
        for i in exlist:
            i = format(i,'.2f')
            exlistn.append(float(i))
        FSEOFdf = pd.DataFrame()
        reactiondf = pd.DataFrame()
        obj_name = inputdic['biomass']
        for i in exlistn:
            cond = i
            obj_name = inputdic['biomass']
            obj_target='maximize'
            mode = 'SET'
            constr_coeff={}
            constr_coeff['fix_reactions']={}
            Concretemodel_Need_Data['E_total']=0.56* np.mean([0.45311986236929197,0.4622348377433211,0.4600801040374112])
            Concretemodel_Need_Data['B_value']=0

            Concretemodel_Need_Data['K_value']=1249
            constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
            constr_coeff['fix_reactions'][inputdic['product']] =  [cond*0.95,np.inf]
            obj_target='maximize'

            EcoECM_FBA_protainmodel=FBA_template2(set_obj_value=True,obj_name=obj_name,obj_target=obj_target,mode=mode,constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
            Model_Solve(EcoECM_FBA_protainmodel,'gurobi')
            print(EcoECM_FBA_protainmodel.obj())
            constr_coeff['fix_reactions'][inputdic['biomass']]=[value(EcoECM_FBA_protainmodel.reaction[inputdic['biomass']]*0.9),np.inf]
            EcoECM_FBA_protainmodel_B2=FBA_template2(set_obj_B_value=True,obj_name=obj_name,obj_target=obj_target,mode=mode,constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
            Model_Solve(EcoECM_FBA_protainmodel_B2,'gurobi')
            B_value2=EcoECM_FBA_protainmodel_B2.obj()
            print(EcoECM_FBA_protainmodel.obj())
            constr_coeff={}
            constr_coeff['fix_reactions']={}
            Concretemodel_Need_Data['B_value']=B_value2
            # constr_coeff['biomass_constrain']=('EX_lys_L_e',cond*0.1)
            constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
            constr_coeff['fix_reactions'][inputdic['product']] =  [cond*0.99,np.inf]
            EcoECM_FBA_protainmodel=FBA_template2(set_obj_value=True,obj_name=obj_name,obj_target=obj_target,mode=mode,constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
            Model_Solve(EcoECM_FBA_protainmodel,'gurobi')
            biomass=EcoECM_FBA_protainmodel.obj()
            print(EcoECM_FBA_protainmodel.obj())
            constr_coeff={}
            constr_coeff['fix_reactions']={}
            constr_coeff['fix_reactions'][inputdic['biomass']]=[biomass*0.9,np.inf]
            constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
            constr_coeff['fix_reactions'][inputdic['product']] =  [cond*0.9,np.inf]
            obj_target='minimize'
            obj_name=inputdic['biomass']
            EcoECM_FBA_protainmodel=FBA_template2(set_obj_sum_e=True,obj_name=obj_name,obj_target=obj_target,mode=mode,constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
            Model_Solve(EcoECM_FBA_protainmodel,'gurobi')
            totalE2=EcoECM_FBA_protainmodel.obj()
            print(EcoECM_FBA_protainmodel.obj())

            constr_coeff={}
            constr_coeff['fix_reactions']={}
            # constr_coeff['fix_reactions']['EX_glc__D_e_reverse']=10
            Concretemodel_Need_Data['E_total']=totalE2*1.1
            Concretemodel_Need_Data['B_value']=B_value2*0.95
            Concretemodel_Need_Data['K_value']=1249
            constr_coeff['substrate_constrain']=(inputdic['substrate'],objvalue2)
            constr_coeff['fix_reactions'][inputdic['product']] = [cond*0.95,np.inf]
            # constr_coeff['biomass_constrain']=('EX_lys_L_e',cond*0.1)
            obj_target='maximize'
            EcoECM_FBA_protainmodel_max=FBA_template2(set_obj_value=True,obj_name=obj_name,obj_target=obj_target,mode=mode,constr_coeff=constr_coeff,Concretemodel_Need_Data=Concretemodel_Need_Data)
            Model_Solve(EcoECM_FBA_protainmodel_max,'gurobi')
            biomass_max=EcoECM_FBA_protainmodel_max.obj()
            print(EcoECM_FBA_protainmodel_max.obj())   
            e1={x: value(EcoECM_FBA_protainmodel_max.e1[x]) for x in EcoECM_FBA_protainmodel_max.e1}
            model_e1_solution = pd.DataFrame(list(e1.items()), columns=['gene', 'e1'])
            reaction_dict = {x: value(EcoECM_FBA_protainmodel_max.reaction[x]) for x in EcoECM_FBA_protainmodel_max.reaction}
            kcat_dict=Concretemodel_Need_Data['kcat_dict']
            kcat_dict={key:value for key,value in kcat_dict.items()if key in reaction_dict}
            model_reaction_solution = pd.DataFrame(list(reaction_dict.items()), columns=['reaction', 'flux'])
            enzyme_list = list(Concretemodel_Need_Data['mw_dict'].keys()) 
            new_gene_reaction_mapping,gene_reaction_mapping,gene_kcat_mapping=gene_reaction_map2(enzyme_list, reaction_dict,kcat_dict,Concretemodel_Need_Data,model)
            model_e1_solution = pd.DataFrame(list(e1.items()), columns=['gene', 'e1'])
            FSEOFdf['gene'] = model_e1_solution['gene']
            FSEOFdf['gene = '+str(cond)] = FSEOFdf['gene'].map(new_gene_reaction_mapping)
            FSEOFdf['cond = '+str(cond)] = model_e1_solution['e1']
            reactiondf['reaction'] = model_reaction_solution['reaction']
            reactiondf['cond = '+str(cond)] = model_reaction_solution['flux']               
    return FSEOFdf,reactiondf


def check_monotonicity(row):
    row_values = row[2:11] 
    increasing = all(row_values[i] < row_values[i + 1] for i in range(len(row_values) - 1))
    decreasing = all(row_values[i] > row_values[i + 1] for i in range(len(row_values) - 1))

    if increasing:
        return 'up'
    elif decreasing:
        return 'down'
    else:
        return 'unchanged'




def threshold(reactiondf):
    selected_columns = reactiondf.iloc[:, [2, 11]]
    mean_flux = selected_columns.mean(axis=1, numeric_only=True)
    reactiondf['mean_flux'] = mean_flux
    e_2_threshold = 1e-1
    reactiondf.loc[mean_flux <= e_2_threshold, 'manipulation'] = None
    up_reactions = reactiondf[reactiondf['manipulation'] == 'up']
    down_reactions = reactiondf[reactiondf['manipulation'] == 'down']

    product_flux = reactiondf.iloc[:, 1:11]
    # 检查并处理数据类型
    product_flux = product_flux.apply(pd.to_numeric, errors='coerce')
    # 计算每行的最大值和最小值的差值
    result_per_row = product_flux.apply(lambda row: round(row.max() - row.min(), 3), axis=1)
    # 将结果添加为新的一列
    reactiondf['result'] = result_per_row
    return up_reactions,down_reactions,result_per_row,reactiondf


def gene_reaction_map3(reaction_list,model_input):
    equation_dict = {}

    for reaction_id in reaction_list:
        equation = model_input.reactions.get_by_id(reaction_id).reaction
        equation_dict[reaction_id] = equation
    return equation_dict



def get_sort_key(x,FSEOFdf):
    columns = FSEOFdf.columns
    parts = x.split('=')
    if len(parts) == 2 and parts[0].strip() == 'gene':
        return float('inf'), float('inf')
    elif len(parts) == 2 and parts[0].strip() == 'cond':
        return float(parts[1]), float('inf')
    return float('inf'), float('inf')


def calculate_sum(row, column_name):
    pattern = r'\((.*?)\)'
    matches = re.findall(pattern, row)
    return sum(float(match.split(':')[0]) for match in matches)


def detail(FSEOFdf):
    columns = FSEOFdf.columns
    sorted_columns = sorted(columns, key=lambda x: get_sort_key(x, FSEOFdf))
    # 使用排序后的列名重新构建 DataFrame
    sorted_FSEOFdf = FSEOFdf[sorted_columns]
    FSEOFdf = FSEOFdf.copy()
    FSEOFdf.loc[:, 'manipulations'] = FSEOFdf.apply(lambda row: check_monotonicity(row), axis=1)

    # FSEOFdf['manipulations'] = FSEOFdf.apply(lambda row: check_monotonicity(row), axis=1)
    # 列名列表
    column_names = FSEOFdf.columns
    # 遍历列名，为每一列创建一个新列来存储括号内数值的总和
    for column_name in column_names:
        # 使用正则表达式提取数值部分
        match = re.search(r'(\d+\.\d+)', column_name)
        
        if match:
            new_column_name = f'sum={match.group(1)}'
            FSEOFdf.loc[:, new_column_name] = FSEOFdf[column_name].astype(str).apply(lambda x: calculate_sum(x, column_name)).copy()
    FSEOFdf = result(FSEOFdf)
    return FSEOFdf

def result(FSEOFdf):
    last_10_columns = FSEOFdf.iloc[:, -10:]
    mean_flux = last_10_columns.mean(axis=1)
    FSEOFdf = FSEOFdf.copy()
    FSEOFdf.loc[:, 'mean_fluxs'] = mean_flux.copy()
    # FSEOFdf['mean_fluxs'] = mean_flux
    e_2_threshold = 1e-1
    # FSEOFdf.loc[mean_flux <= e_2_threshold, 'manipulations'] = None
    return FSEOFdf



def output_fseof(path_results,inputdic,reactiondf,FSEOFdf):
    if inputdic['mode'] == 'S' or inputdic['mode'] =='ST':
        filename=os.path.join(path_results, 'results.xlsx')
        with pd.ExcelWriter(filename) as writer:
            reactiondf.to_excel(writer, sheet_name='test',index=True)
    if inputdic['mode'] == 'SE' or inputdic['mode'] == 'SET':
        filename=os.path.join(path_results, 'results.xlsx') 
        with pd.ExcelWriter(filename) as writer:
            FSEOFdf.to_excel(writer, sheet_name='test',index=True)
            reactiondf.to_excel(writer, sheet_name='reaction',index=True)


def gpr_map(reaction_list,model):
    gpr_dict = {}

    for reaction_id in reaction_list:
        equation = model.reactions.get_by_id(reaction_id).gene_reaction_rule
        gpr_dict[reaction_id] = equation
    return gpr_dict



# 获取反应对应基因的函数
def get_genes_for_reaction(model, reaction_id):
    try:
        # 尝试获取反应对象
        reaction = model.reactions.get_by_id(reaction_id)
        # 打印找到的反应对象
        # print(f"Found reaction: {reaction.id}")
        # 返回反应的基因反应规则
        return reaction.gene_reaction_rule
    except KeyError:
        # 打印找不到反应 ID 的信息
        # print(f"Reaction ID {reaction_id} not found. Searching for reactions containing '{reaction_id}_num'...")
        
        # 查找所有包含 reaction_id + '_num' 的反应 ID
        matched_reaction_ids = [rxn.id for rxn in model.reactions if f"{reaction_id}_num" in rxn.id]
        print(matched_reaction_ids)
        
        if matched_reaction_ids:
            # print(f"Found {len(matched_reaction_ids)} reactions containing '{reaction_id}_num': {matched_reaction_ids}")
            # 返回所有找到的反应对应的基因反应规则
            all_gene_rules = set()
            for reaction_id in matched_reaction_ids:
                # 提取每个反应的基因反应规则并更新集合
                all_gene_rules.add(model.reactions.get_by_id(reaction_id).gene_reaction_rule)
            return all_gene_rules
        else:
            # 如果没有找到任何反应，则返回空集合
            print(f"No reactions containing '{reaction_id}_num' were found.")
            return set()



def add_reactions_to_set(model,gene_id,reaction_set):
    try:
        # 获取基因对象
        gene = model.genes.get_by_id(gene_id.strip())
        # 获取与该基因相关的所有反应
        reaction_list = gene.reactions
        for reaction in reaction_list:
            reaction_set.add(reaction.id)
    except KeyError:
        # print(f"Gene ID {gene_id} not found in the model.")
        return ''

# 定义一个函数来获取反应的基因反应规则和方程式
def get_reaction_details(model,reaction_id):
    try:
        reaction = model.reactions.get_by_id(reaction_id)
        return reaction.gene_reaction_rule, reaction.reaction
    except KeyError:
        return '', ''
