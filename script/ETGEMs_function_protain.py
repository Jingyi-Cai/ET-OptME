"""ETGEMs_function.py

The code in this file reflects the Pyomo Concretemodel construction method of constrainted model. On the basis of this file, with a little modification, you can realize the constraints and object switching of various constrainted models mentioned in our manuscript.

"""

# IMPORTS
#External modules
import cobra
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
from equilibrator_api import ComponentContribution, Q_
from cobra.core import Reaction
from cobra.util.solver import set_objective
import re
from typing import Any, Dict, List
from numpy import mean
import copy

#蛋白量分布排序表格
def sort_protain(dictionarymodel,solved_Concretemodel,epool):
    # get daraframe of sorted protain
    protain_Data=get_dictionarymodel_data(dictionarymodel)
    efba_e1 = {}
    efba_emw={}
    efba_sort=[i   for i in range(1,len(protain_Data[1])+1)  ]   #[1,2,3,...]
    for i in protain_Data[1]:
        efba_e1[i] = value(solved_Concretemodel.e1[i])
        efba_emw[i]= value(solved_Concretemodel.e1[i])*protain_Data[1][i]
    efba_reaction={x:  protain_Data[2][x]  for x in protain_Data[1]}
    emw_percent={x: efba_emw[x]/epool for x in protain_Data[1]}
    pd_efba=pd.DataFrame()
    pd_efba=pd_efba.from_dict([protain_Data[1],efba_e1,efba_emw,emw_percent,efba_reaction]).T
    pd_efba.columns=['mw','e1','e*mw','e*mw_percent','reactions']
    pd_efba=pd_efba.sort_values(by='e*mw',ascending=False)
    pd_efba['sort']=efba_sort  

    sum_eMW="{:.3e}".format(sum(pd_efba['e*mw'].to_list()))
    sum_eMW_percent="{:.3e}".format(sum(pd_efba['e*mw_percent'].to_list()))

    pd_efba['mw'] = pd_efba['mw'].apply(lambda x: "{:.3e}".format(x))
    pd_efba['e1'] = pd_efba['e1'].apply(lambda x: "{:.3e}".format(x))
    pd_efba['e*mw'] = pd_efba['e*mw'].apply(lambda x: "{:.3e}".format(x))
    pd_efba['e*mw_percent'] = pd_efba['e*mw_percent'].apply(lambda x: "{:.3e}".format(x))

    new_row_data = {'e*mw': sum_eMW, 'e*mw_percent': sum_eMW_percent}
    pd_efba.loc['sum']=new_row_data
    return  pd_efba

#检查反应涉及的蛋白的信息
def check_reaction(reactionid,pd_efba,modelid,epool):   
    #get information of reaction.gene
    outputdict={}
    gpr=str(modelid.reactions.get_by_id(reactionid).gpr)
    if  gpr== '':
        return outputdict
    else: 
        if 'or' in gpr:
            gpr=gpr.split(' or ')
        else:
            gpr=[gpr]

        for gene in gpr:
            outputdict[gene]={}
            if 'and' in gene:
                geneflag1=gene.split(' and ')
                geneflag1.sort()
                gene0=' and '.join(geneflag1)
            if 'and' not  in gene:
                gene0=gene
            if gene0 in pd_efba.index:
                outputdict[gene]['pos']=pd_efba.loc[gene0,'sort']
                outputdict[gene]['E1*mw']=pd_efba.loc[gene0,'e_mw']
                outputdict[gene]['mw']=pd_efba.loc[gene0,'mw']
                outputdict[gene]['E1']=pd_efba.loc[gene0,'e1']
                outputdict[gene]['percent']=float(format(pd_efba.loc[gene0,'e_mw']/epool, '.2g'))
        return outputdict

# 得到求解后的模型的变量表格
def deal_protainKEFBA_result(Concretemodel_Data,solved_concentratemodel,dictionarymodel,modelid,epool,shiyan):
    
    pd_protain_kefba=sort_protain(dictionarymodel,solved_concentratemodel)
    reac={}
    flux = {}
    e={}
    e1={}
    rea_protain_e1={}
    p={}
    c={}
    #rea_met_c={}
    for i in dictionarymodel['reactions']:
        reac[i]=dictionarymodel['reactions'][i]['reaction']
        flux[i] = value(solved_concentratemodel.reaction[i])
    for i in Concretemodel_Data['e0']:
        e[i] = value(solved_concentratemodel.e[i])
    for i in Concretemodel_Data['totle_E']:
        e1[i] = value(solved_concentratemodel.e1[i])
    for i in dictionarymodel['reactions']:
        rea_protain_e1[i]=check_reaction(i,pd_protain_kefba,modelid,epool)
        # protain0=dictionary_model['reactions'][i]['gene_reaction_rule']
        # if protain0!='':
        #     protain=deal_iml1515gene(protain0) 
 
        #     rea_protain_e1[i]={protain:e1[protain]}
    for i in Concretemodel_Data['kmapp']:
        p[i] =  value(solved_concentratemodel.p[i])
    for i in Concretemodel_Data['allsubstr']:
        c[i] = value(solved_concentratemodel.c0[i])
    pd1=pd.DataFrame()
    pd1=pd1.from_dict([reac,flux,e,rea_protain_e1,p]).T
    pd1.columns=['reaction','kfba_v','kfba_e','kfba_protain','kfba_p']
    pd2=pd.DataFrame()
    pd2=pd2.from_dict([shiyan,c]).T
    pd2.columns=['实验浓度','计算浓度']
    return pd1,pd2

# 对带 and 的蛋白的基因进行排序
def deal_iml1515gene(gene):
    if 'and' in gene:
        geneflag1=gene.split(' and ')
        geneflag1.sort()
        gene0=' and '.join(geneflag1)
    else:
        gene0=gene
    return gene0

#KEFBA   不是以蛋白为中心
def EcoECM_KEFBA(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,E_total,C_total):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value
    kcat_dict=Concretemodel_Need_Data['kcat_dict']
    mw_dict2=Concretemodel_Need_Data['mw_dict2']
    enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
    e0=Concretemodel_Need_Data['e0']
    totle_E=Concretemodel_Need_Data['totle_E']
    km_dict=Concretemodel_Need_Data['kmapp']
    allsubstr=Concretemodel_Need_Data['allsubstr']
    allsubstr2=Concretemodel_Need_Data['allsubstr2']
    C_meta_lb=Concretemodel_Need_Data['C_meta_lb']
    C_meta_ub=Concretemodel_Need_Data['C_meta_ub']
    # P_lb=Concretemodel_Need_Data['P_lb']
    # P_ub=Concretemodel_Need_Data['P_ub']
    p0=list(km_dict.keys())
    EcoECM_protainmodel=Template_Concretemodel(C_meta_lb,C_meta_ub,set_constr6=True,mw_dict2=mw_dict2,C_total=C_total,set_avep=True,set_avec=True,set_constr7=True,km_dict=km_dict,set_constr5=True,add_dynamics_p_c=True,\
        p0=p0,allsubstr=allsubstr,allsubstr2=allsubstr2,e0=e0,totle_E=totle_E,kcat_dict=kcat_dict,enzyme_rxns_dict=enzyme_rxns_dict,add_e_e1=True,\
        reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        lb_list=lb_list,ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value,set_stoi_matrix=True,set_bound=False,E_total=E_total)
    EcoECM_protainmodel.write('./results/EcoECM_protainmodel.lp',io_options={'symbolic_solver_labels': True})   
    return EcoECM_protainmodel
#protain_KEFBA      以蛋白为中心
def EcoECM_protainKEFBA(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,E_total,C_total):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value
    kcat_dict=Concretemodel_Need_Data['kcat_dict']
    mw_dict=Concretemodel_Need_Data['mw_dict']
    enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
    e0=Concretemodel_Need_Data['e0']
    totle_E=Concretemodel_Need_Data['totle_E']
    km_dict=Concretemodel_Need_Data['kmapp']
    allsubstr=Concretemodel_Need_Data['allsubstr']
    allsubstr2=Concretemodel_Need_Data['allsubstr2']
    C_meta_lb=Concretemodel_Need_Data['C_meta_lb']
    C_meta_ub=Concretemodel_Need_Data['C_meta_ub']
    # P_lb=Concretemodel_Need_Data['P_lb']
    # P_ub=Concretemodel_Need_Data['P_ub']
    p0=list(km_dict.keys())
    EcoECM_protainmodel=Template_Concretemodel(C_meta_lb,C_meta_ub,C_total=C_total,set_avep=True,set_avec=True,set_constr7=True,km_dict=km_dict,set_constr5=True,add_dynamics_p_c=True,\
        p0=p0,allsubstr=allsubstr,allsubstr2=allsubstr2,e0=e0,totle_E=totle_E,kcat_dict=kcat_dict,mw_dict=mw_dict,enzyme_rxns_dict=enzyme_rxns_dict,add_e_e1=True,\
        set_constr3=True,set_constr4=True,reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        lb_list=lb_list,ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value,set_stoi_matrix=True,set_bound=False,E_total=E_total)
    EcoECM_protainmodel.write('./results/EcoECM_protainmodel.lp',io_options={'symbolic_solver_labels': True})   
    return EcoECM_protainmodel

#PFBA
def EcoECM_PFBA(Concretemodel_Need_Data,solved_concentratemodel,biomassid,strategy):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    lb_list=copy.deepcopy(Concretemodel_Need_Data['lb_list'])
    ub_list=copy.deepcopy(Concretemodel_Need_Data['ub_list'])
    if strategy==1:
        lb_list[biomassid]=value(solved_concentratemodel.reaction[biomassid])
        ub_list[biomassid]=value(solved_concentratemodel.reaction[biomassid])
    if strategy ==2:
        for i in solved_concentratemodel.reaction:
            if value(solved_concentratemodel.reaction[i])<=100:
                lb_list[i]=value(solved_concentratemodel.reaction[i])
                ub_list[i]=value(solved_concentratemodel.reaction[i])
    if strategy ==3:
        for  i in Concretemodel_Need_Data['kcat_dict']:
  
            lb_list[i]=value(solved_concentratemodel.reaction[i])
            ub_list[i]=value(solved_concentratemodel.reaction[i])
    EcoECM_protainmodel=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        lb_list=lb_list,ub_list=ub_list,set_stoi_matrix=True,set_obj_V_value=True)
    return EcoECM_protainmodel

#FBA
def EcoECM_FBA(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    EcoECM_protainmodel=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        lb_list=lb_list,ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value,set_stoi_matrix=True)
    return EcoECM_protainmodel


#EFBA
def EcoECM_protainEFBA(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,E_total):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value
    kcat_dict=Concretemodel_Need_Data['kcat_dict']
    mw_dict=Concretemodel_Need_Data['mw_dict']
    enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
    e0=Concretemodel_Need_Data['e0']
    totle_E=Concretemodel_Need_Data['totle_E']

    EcoECM_protainmodel=Template_Concretemodel(e0=e0,totle_E=totle_E,kcat_dict=kcat_dict,mw_dict=mw_dict,enzyme_rxns_dict=enzyme_rxns_dict,add_e_e1=True,\
        set_constr2=True,set_constr3=True,set_constr4=True,reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        lb_list=lb_list,ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value,set_stoi_matrix=True,set_bound=False,E_total=E_total)
    #EcoECM_protainmodel.write('./results/EcoECM_protainmodel.lp',io_options={'symbolic_solver_labels': True})   
    return EcoECM_protainmodel

#  from dictionary get kcat_dict,mw_dict,enzyme_rxns_dict,e0,totle_E
def get_dictionarymodel_data(dictionarymodel):
    kcat_dict={}
    mw_dict={}
    enzyme_rxns_dict={}
    e0=[]
    totle_E=[]
    for enz in dictionarymodel['enzyme']:
        flag_dict=dictionarymodel['enzyme'][enz]
        if flag_dict['MW'] !='null':
            mw_dict[enz]=flag_dict['MW']
        enzyme_rxns_dict[enz]=list(flag_dict['reactions'].keys())
        for rxn in flag_dict['reactions']:
            if flag_dict['reactions'][rxn]['kcat'] != {}:
                kcat_dict[rxn]=flag_dict['reactions'][rxn]['kcat']
    e0 = list(kcat_dict.keys())
    totle_E = list(enzyme_rxns_dict.keys()) 
    return kcat_dict,mw_dict,enzyme_rxns_dict,e0,totle_E
def get_dictionarymodel_data2(dictionarymodel,Concretemodel_Need_Data,kinetics_reactions):
    kcat_dict={}
    mw_dict={}
    enzyme_rxns_dict={}
    e0=[]
    totle_E=[]
    for enz in dictionarymodel['enzyme']:
        flag_dict=dictionarymodel['enzyme'][enz]
        if flag_dict['MW'] !='null':
            mw_dict[enz]=flag_dict['MW']
        enzyme_rxns_dict[enz]=list(flag_dict['reactions'].keys())
        for rxn in flag_dict['reactions']:
            if flag_dict['reactions'][rxn]['kcat'] != {}:
                kcat_dict[rxn]=flag_dict['reactions'][rxn]['kcat']
    e0 = list(kcat_dict.keys())
    totle_E = list(enzyme_rxns_dict.keys())   
    kmapp={}
    for enz in dictionarymodel['enzyme']:
        for rea in dictionarymodel['enzyme'][enz]['reactions']:
            if dictionarymodel['enzyme'][enz]['reactions'][rea]['km'] !={}:
                kmapp[rea]=dictionarymodel['enzyme'][enz]['reactions'][rea]['km'] 
    Concretemodel_Need_Data['kcat_dict']=kcat_dict
    Concretemodel_Need_Data['mw_dict']=mw_dict
    Concretemodel_Need_Data['enzyme_rxns_dict']=enzyme_rxns_dict
    Concretemodel_Need_Data['e0']=e0
    Concretemodel_Need_Data['totle_E']=totle_E
    kmapp2={}
    allsubstr=[]
    allsubstr2=[]
    P_lb={}
    P_ub={}
    for i in kinetics_reactions:
        if i in dictionarymodel['reactions'] and i in kmapp and i in kcat_dict:
            kmapp2[i]=kmapp[i]
        else:
            print(i)
    for i in kmapp2:
        P_lb[i]=0
        P_ub[i]=1
        if len(kmapp2[i])>2:
            print(i,kmapp2[i])
        if len(kmapp2[i])==2:
            ab =list(kmapp2[i].keys())
            allsubstr2.append(ab[0]+'_'+ab[1])
        for j in kmapp2[i]:
            allsubstr.append(j)
    allsubstr = list(set(allsubstr))
    allsubstr2 = list(set(allsubstr2))    
    Concretemodel_Need_Data['kmapp']=kmapp2
    Concretemodel_Need_Data['allsubstr']=allsubstr
    Concretemodel_Need_Data['allsubstr2']=allsubstr2
    Concretemodel_Need_Data['P_lb']=P_lb
    Concretemodel_Need_Data['P_ub']=P_ub      


def standardize_folder(folder: str) -> str:
    # Standardize for \ or / as path separator character.
    folder = folder.replace("\\", "/")

    if folder[-1] != "/":
        folder += "/"

    return folder 
# def get_data(model):
#     reaction_list=[]
#     metabolite_list=[]
#     lb_list={}
#     ub_list={}
#     coef_matrix={}
#     for rea in model.reactions:
#         reaction_list.append(rea.id)
#         lb_list[rea.id]=rea.lower_bound
#         ub_list[rea.id]=rea.upper_bound
#     for met in model.metabolites:
#         metabolite_list.append(met.id)
#     for i in model.reactions:
#         for j in i.metabolites:
#             coef_matrix[(j.id,i.id)]=i.get_coefficient(j.id)
#     return(reaction_list,metabolite_list, lb_list,ub_list,coef_matrix)
def get_data(model):
    reaction_list = []
    metabolite_list = []
    lb_list = {}
    ub_list = {}
    coef_matrix = {}

    # »ñÈ¡·´Ó¦ÁÐ±íºÍ´úÐ»ÎïÁÐ±í
    for rea in model.reactions:
        reaction_list.append(rea.id)
        lb_list[rea.id] = rea.lower_bound
        ub_list[rea.id] = rea.upper_bound

        # ±éÀú·´Ó¦Îï²¢¼ì²éÏµÊý
        for met in rea.metabolites:
            metabolite_list.append(met.id)

            # Ìí¼ÓÏµÊýÐÅÏ¢µ½ coef_matrix
            if (met.id, rea.id) not in coef_matrix:
                coef_matrix[(met.id, rea.id)] = rea.get_coefficient(met.id)

    # È¥³ýÖØ¸´µÄ´úÐ»Îï
    metabolite_list = list(set(metabolite_list))

    return reaction_list, metabolite_list, lb_list, ub_list, coef_matrix

def num_met(rxn,model,currency):
    items=model.reactions.get_by_id(rxn).metabolites
    item_dict={}    
    for x ,y in zip(list(items.keys()),list(items.values())):
        item_dict[x.id]=y
    i=0
    for j in item_dict.keys():
        if item_dict[j]<0 and j not in currency: 
            i=i+item_dict[j]   
    return abs(i)

#define double dict
def addtwodimdict(thedict, key_a, key_b, val): 
    if key_a in thedict:
        thedict[key_a].update({key_b: val})
    else:
        thedict.update({key_a:{key_b: val}})
#
def param4rxn(themet,thec,paramdic,organism,strain):

    

    thedict=dict()
    total_confidence=dict()
    total_confidence_value=[]
    for i in thec:
        theconf=0
        for j in themet:
            paramcoef=pd_param(i,j,organism,strain,paramdic)
            if len(paramcoef)>0:
                theconf+=paramcoef[1]
                addtwodimdict(thedict, i, j, paramcoef)
                total_confidence[i]=theconf
                total_confidence_value.append(theconf)
    if len(total_confidence_value):
        theindexes=[item for item in total_confidence.keys() if total_confidence[item] == max(total_confidence_value)] 
        return thedict[theindexes[0]]
    else:
        return {}
    
def getmetname(abbr,rev_metas):
    if abbr in rev_metas:
        met2=rev_metas[abbr]
    else:
        met2=abbr
    return met2
def get_met(model,rxn,rev_metas,currency):
    #currency=['h2o_c','h2o_e','h2o_p','co2_c','co2_e','co2_p','h_c','h_e','h_p','o2_c','o2_p','o2_e','hco3_c']
    items=model.reactions.get_by_id(rxn).metabolites
    item_dict={}    
    for x ,y in zip(list(items.keys()),list(items.values())):
        item_dict[x.id]=y
    list_met=[]
    for j in item_dict.keys():
        if item_dict[j]<0 and j not in currency: 
            jj=j[0:-2]
            list_met.append(jj) 

    list_met2=[getmetname(i, rev_metas) for i in list_met]
    if 'reverse' in rxn:
        rxn=rxn.replace("_reverse","")

    if  'ec-code' in model.reactions.get_by_id(rxn).annotation.keys():
        list_ec= model.reactions.get_by_id(rxn).annotation['ec-code']

        if type(list_ec)!=list:
            list_ec=[list_ec]

        return list_met2,list_ec
    else:
        return []
    

def mex_list(lis1,lis2):
    for xx in lis2:
        lis1.append(xx)
    return lis1

def pd_param(ec,substr,organism,strain,dict0):

    if   ec in dict0  :

        if "TRANSFER" in dict0[ec].keys():
            ec2=dict0[ec]["TRANSFER"]
            dict_param,dict_confidence=pd_param(ec2,substr,organism,strain,dict0)
        else:
            if substr in dict0[ec].keys():
                if organism in dict0[ec][substr].keys():
                    if strain in dict0[ec][substr][organism].keys():
                        #dict_param[substr]=dict0[ec][substr][strain][organism]

                        
                        dict_param=mean(dict0[ec][substr][organism][strain])
                        dict_confidence=4
                    else:
                        list0=[]                    
                        for j in dict0[ec][substr][organism].keys():                       
                            list0=mex_list(list0,dict0[ec][substr][organism][j])
                        dict_param=mean(list0)
                        dict_confidence=3
        
                else:
                    
                    list1=[]
                    for k in dict0[ec][substr].keys():
                        for k2 in dict0[ec][substr][k].keys():
                            list1=mex_list(list1,dict0[ec][substr][k][k2])
                    dict_param=mean(list1)
                    dict_confidence=2
            else:
                list2=[]
                for k in dict0[ec].keys():
                    for k2 in dict0[ec][k].keys():
                        for k3 in dict0[ec][k][k2].keys():
                            list2=mex_list(list2,dict0[ec][k][k2][k3])
                dict_param=mean(list2)            
                dict_confidence=1    
            return dict_param,dict_confidence
        return dict_param,dict_confidence    
    else:
        return 0,0

def check_zifu(zifu):
    list5=['0','1','2','3','4','5','6','7','8','9']
    if ' ' not in zifu:
        return zifu
    else:
        last=zifu.split(' ')[-1:][0]
        if len(last)>2 and last[:1]=='c' and last[1:2] in list5:
            m=zifu.split(' ')[:-1]
            return ' '.join(m)
        else:
            return zifu
def parse_bigg_metabolites_file(bigg_metabolites_file_path: str, json_output_folder: str) -> None:
    """Parses a BIGG metabolites text file and returns a dictionary for this file.

    As of 29/04/2019, a BIGG metabolites list of all BIGG-included metabolites
    is retrievable under http://bigg.ucsd.edu/data_access

    Arguments
    ----------
    * bigg_metabolites_file_path: str ~ The file path to the BIGG metabolites file.
      The usual file name (which has to be included too in this argument) is
      bigg_models_metabolites.txt
    * output_folder: str ~ The folder in which the JSON including the parsed BIGG
      metabolites file data is stored with the name 'bigg_id_name_mapping.json'

    Output
    ----------
    * A JSON file with the name 'bigg_id_name_mapping.json' in the given output folder,
      with the following structure:
    <pre>
     {
         "$BIGG_ID": "$CHEMICAL_OR_USUAL_NAME",
         (...),
         "$BIGG_ID": "$BIGG_ID",
         (...),
     }
    </pre>
    The BIGG ID <-> BIGG ID mapping is done for models which already use the BIGG IDs.
    """
    # Standardize output folder
    json_output_folder = standardize_folder(json_output_folder)

    # Open the BIGG metabolites file as string list, and remove all newlines
    with open(bigg_metabolites_file_path, "r") as f:
        lines = f.readlines()
    lines = [x.replace("\n", "") for x in lines if len(x) > 0]

    # Mapping variable which will store the BIGG ID<->
    bigg_id_name_mapping = {}
    # Go through each BIGG metabolites file line (which is a tab-separated file)
    # and retrieve the BIGG ID and the name (if there is a name for the given BIGG
    # ID)
    for line in lines:
        bigg_id = line.split("\t")[1]
        # Exception to check if there is no name :O
        try:
            name = line.split("\t")[2].lower()
        except Exception:
            continue
          
    #    bigg_id_name_mapping[name] = bigg_id
        name=check_zifu(name)
        bigg_id_name_mapping[bigg_id] = name

    # Write the JSON in the given folder :D
    json_write(json_output_folder+"bigg_id_name_mapping.json",
               bigg_id_name_mapping)
def json_write(path: str, dictionary: Dict[Any, Any]) -> None:
    """Writes a JSON file at the given path with the given dictionary as content.

    Arguments
    ----------
    * path: str ~  The path of the JSON file that shall be written
    * dictionary: Dict[Any, Any] ~ The dictionary which shalll be the content of
      the created JSON file
    """
    json_output = json.dumps(dictionary, indent=4)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json_output)
def parse_brenda_textfile_new(brenda_textfile_path,json_output_path):
    with open(brenda_textfile_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [x.replace("\n", "") for x in lines]

    # Go through each line and collect the organism lines and kcat lines for each EC number
    in_turnover_numbers = False
    in_organism_reference = False
    ec_number_kcat_lines_mapping: Dict[str, List[str]] = {}
    ec_number_organsism_lines_mapping: Dict[str, List[str]] = {}
    current_ec_number = ""
    organism_lines: List[str] = []
    kcat_lines: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("ID\t"):
            if current_ec_number != "":
                ec_number_organsism_lines_mapping[current_ec_number] = organism_lines
                ec_number_kcat_lines_mapping[current_ec_number] = kcat_lines
            current_ec_number = line.replace("ID\t", "").replace(" ()", "")
            organism_lines = []
            kcat_lines = []

        if len(line) == 0:
            in_turnover_numbers = False
            in_organism_reference = False
        elif line.startswith("PROTEIN"):
            in_organism_reference = True
            i += 1
            line = lines[i]
        elif line.startswith("KM_VALUE"):
            in_turnover_numbers = True
            i += 1
            line = lines[i]
            
        if in_organism_reference:
            if line.startswith("PR"):
                organism_lines.append("")
            if len(organism_lines[-1]) > 0:
                organism_lines[-1] += " "
            organism_lines[-1] += " " + line

        elif in_turnover_numbers:
            if line.startswith("KM"):
                kcat_lines.append("")
            if len(kcat_lines[-1]) > 0:
                kcat_lines[-1] += " "
            kcat_lines[-1] += line
            
        if len(line) == 0:
            in_turnover_numbers = False
            in_organism_reference = False
            
        i += 1

    # Create the BRENDA database dictionary using the collected kcat and organism lines
    # of each EC number :D
    ec_numbers = list(ec_number_kcat_lines_mapping.keys())
    brenda_kcat_database: Dict[str, Any] = {}
    for ec_number in ec_numbers:
        if "(transferred to " in ec_number:
            actual_ec_number = ec_number.split(" (transferred")[0]
            try:
                brenda_kcat_database[actual_ec_number] = {}
                brenda_kcat_database[actual_ec_number]["TRANSFER"] = \
                    ec_number.lower().replace("  ", " ").split(
                        "(transferred to ec")[1].replace(")", "").lstrip()
            except Exception:
                # Some transfers go to general subgroups instead of single EC numbers so that
                # no kcat database can be built from it D:
                print("WARNING: BRENDA text file line " + ec_number + " is not interpretable!")
            continue

        brenda_kcat_database[ec_number] = {}
        
        reference_number_organism_mapping = {}
        organism_lines = ec_number_organsism_lines_mapping[ec_number]
        for organism_line in organism_lines:
            reference_number = organism_line.split("#")[1]
            organism_line_split_first_part = organism_line.split("# ")[1]
            organism_line_split = organism_line_split_first_part.split(" ")
            organism_line_split = [
                x for x in organism_line_split if len(x) > 0]

            end = 1
            for part in organism_line_split:
                # Some organism names contain their SwissProt or UniProt ID,
                # since we don't nned them they are excluded
                if ("swissprot" in part.lower()) or \
                    (part.lower() == "and") or \
                    ("uniprot" in part.lower()) or \
                    ("genbank" in part.lower()) or \
                        ("trembl" in part.lower()):
                    end -= 2
                    break

                if ("<" in part) or ("(" in part):
                    end -= 1
                    break

                end += 1
            organism_name = " ".join(organism_line_split[:end])
            reference_number_organism_mapping[reference_number] = organism_name

        kcat_lines = ec_number_kcat_lines_mapping[ec_number]

        
        for kcat_line in kcat_lines:
            kcat_line = kcat_line
            # Exclude kcats of mutated/changed proteins since
            # they may not have a biological relevance

            reference_number = kcat_line.split("#")[1].split(",")[0]
            organism = reference_number_organism_mapping[reference_number]


            substrate = "".join(kcat_line.split("{")[1]).split("}")[0]

            substrate = substrate.lower()

            if substrate == 'more':
                continue

            # if substrate in bigg_id_name_mapping.keys():
            #     substrate = bigg_id_name_mapping[substrate]
            # else:
            #     substrate = "REST"

            if substrate not in brenda_kcat_database[ec_number].keys():
                brenda_kcat_database[ec_number][substrate] = {}
            if organism not in brenda_kcat_database[ec_number][substrate].keys():
                brenda_kcat_database[ec_number][substrate][organism]={}

            kcat_str = "".join(kcat_line.split("#")[2]).split("{")[0].lstrip().rstrip()
            kcatvalue = mean([float(x) for x in kcat_str.split("-") if len(x) > 0])

            if ("wild" in kcat_line.lower()):
                if 'wild-type' not in brenda_kcat_database[ec_number][substrate][organism].keys():
                    brenda_kcat_database[ec_number][substrate][organism]['wild-type']=[]
                brenda_kcat_database[ec_number][substrate][organism]['wild-type'].append(kcatvalue)
            elif ("mutant" in kcat_line.lower()) or ("mutated" in kcat_line.lower()):
                if 'mutant' not in brenda_kcat_database[ec_number][substrate][organism].keys():
                    brenda_kcat_database[ec_number][substrate][organism]['mutant']=[]
                brenda_kcat_database[ec_number][substrate][organism]['mutant'].append(kcatvalue)
            else:
                if 'else' not in brenda_kcat_database[ec_number][substrate][organism].keys():
                    brenda_kcat_database[ec_number][substrate][organism]['else']=[]
                brenda_kcat_database[ec_number][substrate][organism]['else'].append(kcatvalue) 

    # Write final BRENDA kcat database :D
    json_write(json_output_path,brenda_kcat_database)   

def json_load(path: str) :
    """Loads the given JSON file and returns it as dictionary.

    Arguments
    ----------
    * path: str ~ The path of the JSON file
    """
    with open(path) as f:
        dictionary = json.load(f)
    return dictionary



# def Get_Concretemodel_Need_Data(reaction_g0_file,metabolites_lnC_file,model_file,reaction_kcat_MW_file):
#     Concretemodel_Need_Data={}
#     reaction_g0=pd.read_csv(reaction_g0_file,index_col=0,sep='\t')
#     Concretemodel_Need_Data['reaction_g0']=reaction_g0
#     metabolites_lnC = pd.read_csv(metabolites_lnC_file, index_col=0, sep='\t')
#     Concretemodel_Need_Data['metabolites_lnC']=metabolites_lnC
#     if re.search('\.xml',model_file):
#         model = cobra.io.read_sbml_model(model_file)
#     elif re.search('\.json',model_file):
#         model = cobra.io.json.load_json_model(model_file)
#         dicmodel=json_load(model_file)
#     #convert_to_irreversible(model)
#     reaction_kcat_MW=pd.read_csv(reaction_kcat_MW_file,index_col=0)
#     Concretemodel_Need_Data['model']=model
#     Concretemodel_Need_Data['reaction_kcat_MW']=reaction_kcat_MW
#     [reaction_list,metabolite_list,lb_list,ub_list,coef_matrix]=Get_Model_Data(model)
#     Concretemodel_Need_Data['reaction_list']=reaction_list
#     Concretemodel_Need_Data['metabolite_list']=metabolite_list
#     Concretemodel_Need_Data['lb_list']=lb_list
#     Concretemodel_Need_Data['ub_list']=ub_list
#     Concretemodel_Need_Data['coef_matrix']=coef_matrix

#     def get_gene_reaction_dict(dicmodel):
#         reaction_gene_dict = {}
#         for reaction in dicmodel["reactions"]:
#             if reaction["gene_reaction_rule"] != "":
#                 reaction_gene_dict[reaction["id"]] = reaction["gene_reaction_rule"].replace(" ","_")
#         gene_reaction_dict = {}
#         for key, value in reaction_gene_dict.items():
#                 if value not in gene_reaction_dict:
#                     gene_reaction_dict[value] = [key]
#                 else:
#                     gene_reaction_dict[value].append(key)
#         return gene_reaction_dict

#     gene_reaction_dict=get_gene_reaction_dict(dicmodel)
#     totle_E=list(gene_reaction_dict.keys())
#     e0=list(reaction_kcat_MW.index)
#     Concretemodel_Need_Data['totle_E']=totle_E
#     Concretemodel_Need_Data['gene_reaction_dict']=gene_reaction_dict
#     Concretemodel_Need_Data['e0']=e0
    
#     theMW={}

#     for i in dicmodel['reactions']:
#         # print(type(i['kcat']))
#         if i['kcat']!='' and i['kcat_MW']!='':
#             theMW[i['gene_reaction_rule']]=(i['kcat']/i['kcat_MW'])

#     Concretemodel_Need_Data['theMW']=theMW
    
#     return Concretemodel_Need_Data,model,dicmodel  #,e0

def remove_unused_metabolites(model):
    metabolite_used_list = []
    for reaction in model.reactions:
        for metabolite in reaction.metabolites:
            metabolite_used_list.append(metabolite.id)
            metabolite_used_list = list(set(metabolite_used_list))
    metabolite_unused_num = 1
    while(metabolite_unused_num > 0):
        metabolite_unused_num = 0
        for metabolite in model.metabolites:
            if metabolite.id not in metabolite_used_list:
                metabolite_unused_num += 1
                model.remove_metabolites([metabolite])

def isoenzyme_split(model):
    """Split isoenzyme reaction to mutiple reaction

    Arguments
    ----------
    * model: cobra.Model.
    
    :return: new cobra.Model.
    """  
    for r in model.reactions:
        if re.search(" or ", r.gene_reaction_rule):
            rea = r.copy()
            gene = r.gene_reaction_rule.split(" or ")
            for index, value in enumerate(gene):
                if index == 0:
                    r.id = r.id + "_num1"
                    r.gene_reaction_rule = value
                else:
                    r_add = rea.copy()
                    r_add.id = rea.id + "_num" + str(index+1)
                    r_add.gene_reaction_rule = value
                    model.add_reactions([r_add])
    for r in model.reactions:
        r.gene_reaction_rule = r.gene_reaction_rule.strip("( )")
    return model

# dictionary_model:  json_load(model_file)
def get_gene_reaction_dict(dictionary_model):
    reaction_gene_dict = {}
    for reaction in dictionary_model["reactions"]:
        if reaction["gene_reaction_rule"] != "":
            reaction_gene_dict[reaction["id"]] = reaction["gene_reaction_rule"].replace(" ","_")
    gene_reaction_dict = {}
    for key, value in reaction_gene_dict.items():
            if value not in gene_reaction_dict:
                gene_reaction_dict[value] = [key]
            else:
                gene_reaction_dict[value].append(key)
    return gene_reaction_dict

def trans_model2standard_json_etgem(model_file):
    if model_file.split(".")[-1] == 'json':
        model = cobra.io.load_json_model(model_file)
    elif model_file.split(".")[-1] == 'xml':
        model = cobra.io.read_sbml_model(model_file)
    remove_unused_metabolites(model)
    convert_to_irreversible(model)
    model = isoenzyme_split(model)
    model_name = model_file.split('/')[-1].split('.')[0]
    json_path = "./%s_irreversible.json" % model_name
    print(json_path)
    cobra.io.save_json_model(model, json_path)
    dictionary_model = json_load(json_path)
    for eachreaction in dictionary_model['reactions']:
        eachreaction["g0"] = -99999
    for eachmetabolite in dictionary_model['metabolites']:
        if eachmetabolite["id"][-1] != 'c':
            eachmetabolite["concentration_ub"] = 0.0
            eachmetabolite["concentration_lb"] = 0.0
        elif eachmetabolite["id"][-1] == 'c':
            eachmetabolite["concentration_ub"] = -3.912023005
            eachmetabolite["concentration_lb"] = -14.50865774

    reaction_gene_dict = {}
    for reaction in dictionary_model["reactions"]:
        if reaction["gene_reaction_rule"] != "":
            flag1=reaction["gene_reaction_rule"].split(' and ')
            flag1.sort(key=None, reverse=False)
            flag2=' and '.join(flag1)
            reaction_gene_dict[reaction["id"]] = flag2
    gene_reaction_dict = {}
    for key, value in reaction_gene_dict.items():
            if value not in gene_reaction_dict:
                gene_reaction_dict[value] = [key]
            else:
                gene_reaction_dict[value].append(key)
    enzyme_dict = {}
    for key,value in gene_reaction_dict.items():
        new_enzyme = {}
        new_enzyme["gpr"] = key
        gene_list = key.split("_and_")
        enzyme_gene_dict = {}
        for gene in gene_list:
            enzyme_gene_dict[gene] = 1
        new_enzyme["genes"] = enzyme_gene_dict
        new_enzyme["MW"] = "null"
        reaction_dict = {}
        for reaction in value:
            
            single_reaction_dict = {}
            single_reaction_dict["kcat"] = {}
            single_reaction_dict["km"] = {}
            # single_reaction_dict["ki"] = "null"
            # single_reaction_dict["dGr0"] = get_g0_by_id(dictionary_model,reaction)
            # if "annotation" in get_reaction_by_id(dictionary_model,reaction).keys():
            #     if "ec-code" in get_reaction_by_id(dictionary_model,reaction)["annotation"].keys():
            #         single_reaction_dict["EC"] = get_reaction_by_id(dictionary_model,reaction)["annotation"]["ec-code"]
            # single_reaction_dict["SA"] = "null"
            # single_reaction_dict["cofactor"] = "null"
            # single_reaction_dict["inhibitor"] = "null"
            # single_reaction_dict["ACTIVATING_COMPOUND"] = "null"
            reaction_dict[reaction] = single_reaction_dict
        new_enzyme["reactions"] = reaction_dict
        enzyme_dict[key] = new_enzyme
    dictionary_model["enzyme"] = enzyme_dict
    dictionary_model['metabolites']
    
    dict_met={}
    for eachmetabolite in dictionary_model['metabolites']:
        dict_met[eachmetabolite["id"]]=eachmetabolite
    dictionary_model['metabolites']=dict_met 
    dict_rea={}
    for eachmetabolite in dictionary_model['reactions']:
        dict_rea[eachmetabolite["id"]]=eachmetabolite
    dictionary_model['reactions']=dict_rea 
    dict_gene={}
    for eachmetabolite in dictionary_model['genes']:
        dict_gene[eachmetabolite["id"]]=eachmetabolite
    dictionary_model['genes']=dict_gene     
    reactions_reaction={}
    for eachmetabolite in dictionary_model['reactions']:
        dictionary_model['reactions'][eachmetabolite]['reaction']=model.reactions.get_by_id(eachmetabolite).reaction
    
    return dictionary_model

def get_g0_by_id(dictionary_model,reaction_id):
    for reaction in dictionary_model['reactions']:
        if reaction['id'] == reaction_id:
            result_g0 = reaction['g0']
            return result_g0
    raise KeyError(
        f"{reaction_id}"
    )
def get_reaction_by_id(dictionary_model,reaction_id):
    for reaction in dictionary_model['reactions']:
        if reaction['id'] == reaction_id:
            return reaction
    raise KeyError(
        f"{reaction_id}"
    )

def trans_model2enzyme_core_model(model):    
    reaction_gene_dict = {}
    for reaction in model["reactions"]:
        if reaction["gene_reaction_rule"] != "":
            reaction_gene_dict[reaction["id"]] = reaction["gene_reaction_rule"].replace(" ","_")
    gene_reaction_dict = {}
    for key, value in reaction_gene_dict.items():
            if value not in gene_reaction_dict:
                gene_reaction_dict[value] = [key]
            else:
                gene_reaction_dict[value].append(key)
    enzyme_dict = {}
    for key,value in gene_reaction_dict.items():
        new_enzyme = {}
        new_enzyme["gpr"] = key
        gene_list = key.split("_and_")
        enzyme_gene_dict = {}
        for gene in gene_list:
            enzyme_gene_dict[gene] = 1
        new_enzyme["genes"] = enzyme_gene_dict
        new_enzyme["MW"] = "null"
        for reaction in value:
            reaction_dict = {}
            single_reaction_dict = {}
            single_reaction_dict["kcat"] = {}
            single_reaction_dict["km"] = {}
            single_reaction_dict["ki"] = "null"
            single_reaction_dict["dGr0"] = get_g0_by_id(model,reaction)
            if "annotation" in get_reaction_by_id(model,reaction).keys():
                if "ec-code" in get_reaction_by_id(model,reaction)["annotation"].keys():
                    single_reaction_dict["EC"] = get_reaction_by_id(model,reaction)["annotation"]["ec-code"]
            single_reaction_dict["SA"] = "null"
            single_reaction_dict["cofactor"] = "null"
            single_reaction_dict["inhibitor"] = "null"
            single_reaction_dict["ACTIVATING_COMPOUND"] = "null"
            reaction_dict[reaction] = single_reaction_dict
        new_enzyme["reactions"] = reaction_dict
        enzyme_dict[key] = new_enzyme
    model["enzyme"] = enzyme_dict
    
#Extracting information from GEM (iML1515 model)
def Get_Model_Data_old(model):
    """Returns reaction_list,metabolite_list,lb_list,ub_list,coef_matrix from model.
    
    Notes: 
    ----------
    *model： is in SBML format (.xml).
    """
    reaction_list=[]
    metabolite_list=[]
    lb_list={}
    ub_list={}
    coef_matrix={}
    for rea in model.reactions:
        reaction_list.append(rea.id)
        lb_list[rea.id]=rea.lower_bound
        ub_list[rea.id]=rea.upper_bound
        for met in model.metabolites:
            metabolite_list.append(met.id)
            try:
                rea.get_coefficient(met.id)  
            except:
                pass
            else:
                coef_matrix[met.id,rea.id]=rea.get_coefficient(met.id)
    # print(ub_list)
    reaction_list=list(set(reaction_list))
    metabolite_list=list(set(metabolite_list))
    return(reaction_list,metabolite_list,lb_list,ub_list,coef_matrix)
# def Get_Model_Data(model):
#     """Returns reaction_list,metabolite_list,lb_list,ub_list,coef_matrix from model.
    
#     Notes: 
#     ----------
#     *model： is in SBML format (.xml).
#     """
#     reaction_list=[]
#     metabolite_list=[]
#     lb_list={}
#     ub_list={}
#     coef_matrix={}
#     for rea in model.reactions:
#         reaction_list.append(rea.id)
        
#         if rea.lower_bound+rea.upper_bound<0:
#             print(str(rea.id) + ' has a negative flux range')
#             print('ub is : ' + str(rea.upper_bound))
#             print('lb is : ' + str(rea.lower_bound))
#             lb_list[rea.id]=rea.lower_bound+1000
#             ub_list[rea.id]=rea.upper_bound+1000
#             for met in model.metabolites:
#                 metabolite_list.append(met.id)
#                 try:
#                     rea.get_coefficient(met.id)
#                     # print(rea.get_coefficient(met.id))
#                 except:
#                     pass
#                 else:
#                     coef_matrix[met.id,rea.id]=-rea.get_coefficient(met.id)
#                     # print(-rea.get_coefficient(met.id))
#         else:
#             lb_list[rea.id]=rea.lower_bound
#             ub_list[rea.id]=rea.upper_bound                
#             for met in model.metabolites:
#                 metabolite_list.append(met.id)
#                 try:
#                     rea.get_coefficient(met.id)  
#                 except:
#                     pass
#                 else:
#                     coef_matrix[met.id,rea.id]=rea.get_coefficient(met.id)
#     reaction_list=list(set(reaction_list))
#     metabolite_list=list(set(metabolite_list))
#     return(reaction_list,metabolite_list,lb_list,ub_list,coef_matrix)

# dictionary_model:  json_load(model_file)
def get_gene_reaction_dict(dictionary_model):
    reaction_gene_dict = {}
    for reaction in dictionary_model["reactions"]:
        if reaction["gene_reaction_rule"] != "":
            reaction_gene_dict[reaction["id"]] = reaction["gene_reaction_rule"].replace(" ","_")
    gene_reaction_dict = {}
    for key, value in reaction_gene_dict.items():
            if value not in gene_reaction_dict:
                gene_reaction_dict[value] = [key]
            else:
                gene_reaction_dict[value].append(key)
    return gene_reaction_dict

def Get_Model_Data(model):
    """Returns reaction_list,metabolite_list,lb_list,ub_list,coef_matrix from model.
    
    Notes: 
    ----------
    *model? is in SBML format (.xml).
    """
    reaction_list=[]
    metabolite_list=[]
    lb_list={}
    ub_list={}
    coef_matrix={}
    for rea in model.reactions:
        reaction_list.append(rea.id)
        
        if rea.lower_bound+rea.upper_bound<0:
            print(str(rea.id) + ' has a negative flux range')
            print('ub is : ' + str(rea.upper_bound))
            print('lb is : ' + str(rea.lower_bound))
            lb_list[rea.id]=rea.lower_bound+1000
            ub_list[rea.id]=rea.upper_bound+1000
            for met in rea.metabolites:
                coef_matrix[met.id,rea.id]=-rea.get_coefficient(met.id)

        else:
            lb_list[rea.id]=rea.lower_bound
            ub_list[rea.id]=rea.upper_bound                
            for met in rea.metabolites:
                coef_matrix[met.id,rea.id]=-rea.get_coefficient(met.id)
    for met in model.metabolites:
        metabolite_list.append(met.id)
    reaction_list=list(set(reaction_list))
    return(reaction_list,metabolite_list,lb_list,ub_list,coef_matrix)

def convert_to_irreversible(model):
    """Split reversible reactions into two irreversible reactions

    These two reactions will proceed in opposite directions. This
    guarentees that all reactions in the model will only allow
    positive flux values, which is useful for some modeling problems.

    Arguments
    ----------
    * model: cobra.Model ~ A Model object which will be modified in place.

    """
    #warn("deprecated, not applicable for optlang solvers", DeprecationWarning)
    reactions_to_add = []
    coefficients = {}
    for reaction in model.reactions:
        # If a reaction is reverse only, the forward reaction (which
        # will be constrained to 0) will be left in the model.
        if reaction.lower_bound < 0 and reaction.upper_bound > 0:
            reverse_reaction = Reaction(reaction.id + "_reverse")
            reverse_reaction.name=reaction.name
            reverse_reaction.lower_bound = max(0, -reaction.upper_bound)
            reverse_reaction.upper_bound = -reaction.lower_bound
            coefficients[
                reverse_reaction] = reaction.objective_coefficient * -1
            reaction.lower_bound = max(0, reaction.lower_bound)
            reaction.upper_bound = max(0, reaction.upper_bound)
            # Make the directions aware of each other
            reverse_reaction.notes = reaction.notes
            reverse_reaction.annotation = reaction.annotation
            reaction.notes["reflection"] = reverse_reaction.id
            reverse_reaction.notes["reflection"] = reaction.id
            reaction_dict = {k: v * -1
                             for k, v in reaction._metabolites.items()}
            reverse_reaction.add_metabolites(reaction_dict)
            reverse_reaction._model = reaction._model
            reverse_reaction._genes = reaction._genes
            for gene in reaction._genes:
                gene._reaction.add(reverse_reaction)
            reverse_reaction.subsystem = reaction.subsystem
            reverse_reaction.gene_reaction_rule = reaction.gene_reaction_rule
            reactions_to_add.append(reverse_reaction)
            
    model.add_reactions(reactions_to_add)
    set_objective(model, coefficients, additive=True)
    
#Encapsulating parameters used in Concretemodel
def Get_Concretemodel_Need_Data(model_file):
    Concretemodel_Need_Data={}
    if re.search('\.xml',model_file):
        model = cobra.io.read_sbml_model(model_file)
    elif re.search('\.json',model_file):
        model = cobra.io.json.load_json_model(model_file)
    Concretemodel_Need_Data['model']=model
    [reaction_list,metabolite_list,lb_list,ub_list,coef_matrix]=get_data(model)
    Concretemodel_Need_Data['reaction_list']=reaction_list
    Concretemodel_Need_Data['metabolite_list']=metabolite_list
    Concretemodel_Need_Data['lb_list']=lb_list
    Concretemodel_Need_Data['ub_list']=ub_list
    Concretemodel_Need_Data['coef_matrix']=coef_matrix
    return (Concretemodel_Need_Data)


#set_obj_value,set_metabolite,set_Df: only 'True' and 'False'
    #  kcat_dict:   reaction:value    
    #  mw_dict  :   enzyme:value
    #  enzyme_rxns_dict:     enzyme:rxns_list  
def Template_Concretemodel(C_meta_lb=None,C_meta_ub=None,set_constr6=False,mw_dict2=None,C_total=None,set_avep=False,set_avec=False,set_constr7=False,\
    km_dict=None,set_constr9=False,set_constr10=False,biomass_name=None,v0_biomass=None,set_constr11=False,set_constr5=False,add_dynamics_p_c=False,p0=None,allsubstr=None,allsubstr2=None,\
    e0=None,totle_E=None,kcat_dict=None,mw_dict=None,enzyme_rxns_dict=None,add_e_e1=False,set_constr2=False,set_constr3=False,\
    set_constr4=False,reaction_list=None,metabolite_list=None,coef_matrix=None,metabolites_lnC=None,reaction_g0=None,reaction_kcat_MW=None,lb_list=None,\
    ub_list=None,obj_name=None,K_value=None,obj_target=None,set_obj_value=False,set_substrate_ini=False,set_biomass_ini_e=False,substrate_name=None,substrate_value=None,\
    set_biomass_ini=False,biomass_value=None,biomass_id=None,set_metabolite=False,set_enzyme_value=False,set_Df=False,set_obj_B_value=False,set_stoi_matrix=False,\
    set_bound=False,set_enzyme_constraint=False,set_enzyme_constraint_e=False,set_integer=False,set_metabolite_ratio=False,set_thermodynamics=False,B_value=None,\
    set_obj_E_value=False,set_obj_V_value=False,set_obj_TM_value=False,set_obj_Met_value=False,set_obj_single_E_value=False,E_total=None,\
    Bottleneck_reaction_list=None,set_Bottleneck_reaction=False,set_obj_totalE=False):
    


    """According to the parameter conditions provided by the user, the specific pyomo model is returned.

    Notes
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC:Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...

    * reaction_kcat_MW: The enzymatic data of kcat (turnover number, /h) divided by MW (molecular weight, kDa).
    The format is as follows (in .csv file):
    kcat,MW,kcat_MW
    AADDGT,49389.2889,40.6396,1215.299582180927
    GALCTND,75632.85923999999,42.5229,1778.63831582512
    ...
    
    * lb_list: Lower bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * ub_list: Upper bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * obj_name: Name of object, such as set_obj_value, set_obj_single_E_value, set_obj_TM_value and set_obj_Met_value.    
    * K_value: the maximum value minus the minimum value of reaction thermodynamic driving force (1249, maxDFi-minDFi).
    * obj_target: Type of object function (maximize or minimize).
    * set_obj_value: Set the flux as the object function (True or False).
    * set_substrate_ini: Adding substrate amount constraints (True or False).
    * substrate_name: Substrate input reaction ID in the model (such as EX_glc__D_e_reverse).
    * substrate_value: Set the upper bound for substrate input reaction flux (10 mmol/h/gDW).
    * set_biomass_ini: Set the lower bound for biomass synthesis reaction flux (True or False).
    * biomass_value: The lower bound of biomass synthesis reaction flux.
    * biomass_id: Biomass synthesis reaction ID in the model (BIOMASS_Ec_iML1515_core_75p37M).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    * set_obj_B_value: The object function is the maximizing thermodynamic driving force of a pathway (True or False)
    * set_stoi_matrix: Adding flux balance constraints (True or False).
    * set_bound: Set the upper and lower bounds of reaction flux (True or False).
    * set_enzyme_constraint: Adding enzymamic constraints (True or False).
    * set_integer: Adding binary variables constraints (True or False).
    * set_metabolite_ratio: Adding concentration ratio constraints for metabolites (True or False).
    * set_thermodynamics: Adding thermodynamic constraints (True or False).
    * B_value: The value of maximizing the minimum thermodynamic driving force (MDF).
    * set_obj_E_value: The object function is the minimum enzyme cost of a pathway (True or False).
    * set_obj_V_value: The object function is the pFBA of a pathway (True or False)
    * set_obj_TM_value: The object function is the thermodynamic driving force of a reaction (True or False).
    * set_obj_Met_value: The object function is the concentration of a metabolite (True or False).
    * set_obj_single_E_value: The object function is the enzyme cost of a reaction (True or False).     
    * E_total: Total amount constraint of enzymes (0.114).
    * Bottleneck_reaction_list: A list extracted from the result file automatically.
    * set_Bottleneck_reaction: Adding integer variable constraints for specific reaction (True or False).
    """

    # e0 = []
    # for i in kcat_dict:
    #     if i in reaction_list:
    #         e0.append(i)
    # totle_E = list(enzyme_rxns_dict.keys())

    Concretemodel = ConcreteModel()
    # Concretemodel.metabolite = pyo.Var(metabolite_list,  within=Reals)
    # Concretemodel.Df = pyo.Var(reaction_list,  within=Reals)
    # Concretemodel.B = pyo.Var()
    Concretemodel.reaction = pyo.Var(reaction_list,   bounds=lambda m, i: (lb_list[i], ub_list[i]))
    # Concretemodel.z = pyo.Var(reaction_list,  within=pyo.Binary)
    if set_Df:
        reaction_listDF=[j for j in reaction_g0.index if j in reaction_list]
        ConMet_list= list(set(metabolite_list).intersection(set(metabolites_lnC.index)))
        Concretemodel.metabolite = pyo.Var(ConMet_list,  within=Reals) # metabolite concentration variable
        Concretemodel.Df = pyo.Var(reaction_listDF,  within=Reals) # thermodynamic driving force variable--reactions
        Concretemodel.z = pyo.Var(reaction_listDF,  within=pyo.Binary)    # binary variable
        Concretemodel.B = pyo.Var()    
    else:
        Concretemodel.metabolite = pyo.Var(metabolite_list,  within=Reals)
    #Set upper and lower bounds of metabolite concentration
    if set_metabolite:
        def set_metabolite_c(m,i):
            return  inequality(metabolites_lnC.loc[i,'lnClb'], m.metabolite[i], metabolites_lnC.loc[i,'lnCub'])
        # Concretemodel.set_metabolite= Constraint(metabolite_list,rule=set_metabolite) 
        Concretemodel.set_metabolite= Constraint(ConMet_list,rule=set_metabolite_c)  

    #thermodynamic driving force expression for reactions
    if set_Df:
        def set_Df_c(m,j):
            # return  m.Df[j]==-reaction_g0.loc[j,'g0']-2.579*sum(coef_matrix[i,j]*m.metabolite[i]  for i in metabolite_list if (i,j) in coef_matrix.keys())
            return  m.Df[j]==-reaction_g0.loc[j,'g0']-2.579*sum(coef_matrix[i,j]*m.metabolite[i]  for i in ConMet_list if (i,j) in coef_matrix.keys())
        # rg0=list(set(list(reaction_g0.index)).intersection(set(reaction_list))) 

        Concretemodel.set_Df = Constraint(reaction_listDF,rule=set_Df_c)
    
    #Set the maximum flux as the object function
    if set_obj_value:   
        def set_obj_value_c(m):
            return m.reaction[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_value_c, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_value_c, sense=minimize)
               
            
                
    #Set the value of maximizing the minimum thermodynamic driving force as the object function
    if set_obj_B_value:             
        def set_obj_B_value_c(m):
            return m.B  
        Concretemodel.obj = Objective(rule=set_obj_B_value_c, sense=maximize)  

    #Set the minimum enzyme cost of a pathway as the object function
    if set_obj_E_value:             
        def set_obj_E_value_c(m):
            return sum(m.reaction[j]/(reaction_kcat_MW.loc[j,'kcat/MW']) for j in reaction_kcat_MW.index)
        Concretemodel.obj = Objective(rule=set_obj_E_value_c, sense=minimize)  


    #Minimizing the flux sum of pathway (pFBA)
    if set_obj_V_value:             
        def set_obj_V_value_c(m):
            return sum(m.reaction[j] for j in reaction_list)
        Concretemodel.obj = Objective(rule=set_obj_V_value_c, sense=minimize)  

    
    if set_obj_totalE:  
        Concretemodel.totalE = pyo.Var()
        def set_obj_totalE_c(m):
            return m.totalE
        Concretemodel.obj = Objective(rule=set_obj_totalE_c, sense=minimize)  
    # else:
    #     Concretemodel.totalE = Constraint(expr=Concretemodel.totalE==E_total)


    #To calculate the variability of enzyme usage of single reaction.
    if set_obj_single_E_value:             
        def set_obj_single_E_value_c(m):
            return m.reaction[obj_name]/(reaction_kcat_MW.loc[obj_name,'kcat/MW'])
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_single_E_value_c, sense=maximize) 
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_single_E_value_c, sense=minimize)

    #To calculate the variability of thermodynamic driving force (Dfi) of single reaction.
    if set_obj_TM_value:   
        def set_obj_TM_value_c(m):
            return m.Df[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_TM_value_c, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_TM_value_c, sense=minimize)

    #To calculate the concentration variability of metabolites.
    if set_obj_Met_value:   
        def set_obj_Met_value_c(m):
            return m.metabolite[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_Met_value_c, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_Met_value_c, sense=minimize)

    #Adding flux balance constraints （FBA）
    if set_stoi_matrix:
        def set_stoi_matrix_c(m,i):
            return sum(coef_matrix[i,j]*m.reaction[j]  for j in reaction_list if (i,j) in coef_matrix.keys() )==0
        Concretemodel.set_stoi_matrix = Constraint( metabolite_list,rule=set_stoi_matrix_c)

    #Adding the upper and lower bound constraints of reaction flux
    if set_bound:
        def set_bound_c(m,j):
            return inequality(lb_list[j],m.reaction[j],ub_list[j])
        Concretemodel.set_bound = Constraint(reaction_list,rule=set_bound_c) 

    #Set the upper bound for substrate input reaction flux
    if set_substrate_ini:
        def set_substrate_ini_c(m): 
            return m.reaction[substrate_name] <= substrate_value
        Concretemodel.set_substrate_ini = Constraint(rule=set_substrate_ini_c)   
        
    #Set the lower bound for biomass synthesis reaction flux
    if set_biomass_ini:
        def set_biomass_ini_c(m): 
            return m.reaction[biomass_id] >=biomass_value
        Concretemodel.set_biomass_ini = Constraint(rule=set_biomass_ini_c)  
    
        #Set upper and lower bounds of metabolite concentration
     

  #蛋白为中心酶约束
    #enzyme cons  设置单酶和蛋白变量
    if add_e_e1:
        Concretemodel.e = pyo.Var(e0, within=NonNegativeReals)
        Concretemodel.e1 = pyo.Var(totle_E, within=NonNegativeReals)      

    if set_biomass_ini_e:
        def set_biomass_ini_e_c(m): 
            return m.e1[biomass_id]*mw_dict[biomass_id] >=biomass_value
        Concretemodel.set_biomass_ini_e = Constraint(rule=set_biomass_ini_e_c)  
           
    if set_constr2:
        Concretemodel.set_constr2 = ConstraintList()
        for i in e0:
            Concretemodel.set_constr2.add(Concretemodel.e[i] * kcat_dict[i] >= Concretemodel.reaction[i])
    #   蛋白浓度=单酶加和
    if set_constr3:    
        Concretemodel.set_constr3 = ConstraintList()
        for i in totle_E:
            Concretemodel.set_constr3.add(Concretemodel.e1[i] == sum(Concretemodel.e[j] for j in enzyme_rxns_dict[i] if j in e0))
 
    #   蛋白量加和  <=  epool
    if set_constr4:   
        Concretemodel.constr4 = Constraint(expr=sum(Concretemodel.e1[i] * mw_dict[i] for i in enzyme_rxns_dict.keys() if i in mw_dict) <= E_total)
 
 #   酶量加和  <=  epool  不是以蛋白为中心
    if  set_constr6:
        Concretemodel.constr6 =Constraint(expr=sum(  Concretemodel.e[i] * mw_dict2[i]    for i in e0)   <=E_total)

    #  动力学约束
    if add_dynamics_p_c:
        Concretemodel.p = pyo.Var(p0, bounds=(0, 1))
        Concretemodel.avep = pyo.Var(bounds=(0, 1))
        Concretemodel.c0 = pyo.Var(allsubstr, bounds=lambda m, x: (C_meta_lb[x], C_meta_ub[x]))
        Concretemodel.c1 = pyo.Var(allsubstr2)
    if set_constr5:
        Concretemodel.con_enz = ConstraintList()
        for i in [i for i in reaction_list if i in kcat_dict]:
            if i in km_dict:
                Concretemodel.con_enz.add(Concretemodel.e[i] * kcat_dict[i]*Concretemodel.p[i]  == Concretemodel.reaction[i])
            else:
                Concretemodel.con_enz.add(Concretemodel.e[i] * kcat_dict[i]*Concretemodel.avep  == Concretemodel.reaction[i])
    if set_avep:
            # average p
        Concretemodel.con_avep = Constraint(expr=sum(Concretemodel.p[i] for i in p0)==len(p0)*Concretemodel.avep)
    if set_avec:
        # average c
        Concretemodel.con_avec = Constraint(expr=sum(Concretemodel.c0[i]  for i in allsubstr) <= C_total )
    if set_constr7:
        Concretemodel.con_sub = ConstraintList()
        for i in km_dict:
            substrate = list(km_dict[i])
            if len(substrate) == 2:
                ccomb = substrate[0] + '_' + substrate[1]
                Concretemodel.con_sub.add(Concretemodel.c1[ccomb] == Concretemodel.c0[substrate[0]] * Concretemodel.c0[substrate[1]])
                Concretemodel.con_sub.add(Concretemodel.p[i] * km_dict[i][substrate[0]][0] * Concretemodel.c0[substrate[1]] + Concretemodel.p[i] * km_dict[i][substrate[1]][0] * Concretemodel.c0[substrate[0]] + Concretemodel.p[i] * Concretemodel.c1[ccomb] == Concretemodel.c1[ccomb])
            if len(substrate) == 1:
                Concretemodel.con_sub.add(Concretemodel.p[i] * km_dict[i][substrate[0]][0] + Concretemodel.p[i] * Concretemodel.c0[substrate[0]] == Concretemodel.c0[substrate[0]])    

     #    确定酶范围
    # v_biomass    >=v0_biomass*0.95
    # if set_constr9:
    #     Concretemodel.constr9 = Constraint(expr=Concretemodel.reaction[biomass_name] >= v0_biomass * 0.95)
    # 对v_biomass进行约束
    if set_constr10:
        Concretemodel.constr10 = Constraint(expr=Concretemodel.reaction[biomass_name] >= v0_biomass * 0.1)    
    # # 对v_product进行约束
    # if set_constr11:
    #     Concretemodel.constr11 = Constraint(expr=Concretemodel.reaction[product_name] >= v1_product_max * 0.95)
            
        
    #Adding enzymamic constraints
    if set_enzyme_constraint:
        def set_enzyme_constraint_c(m):           
            return sum( m.reaction[j]/(reaction_kcat_MW.loc[j,'kcat/MW']) for j in reaction_kcat_MW.index)<= E_total
        Concretemodel.set_enzyme_constraint = Constraint(rule=set_enzyme_constraint_c)
    if set_enzyme_constraint_e:
        def set_enzyme_constraint_e_c(m):
            return sum( m.reaction[j]/(reaction_kcat_MW.loc[j,'kcat/MW']) for j in reaction_kcat_MW.index)<= m.totalE
        Concretemodel.set_enzyme_constraint_e = Constraint(rule=set_enzyme_constraint_e_c)    
    #Adding thermodynamic MDF(B) object function
    if set_obj_B_value:
        def set_obj_B_value_c(m,j):
            return m.B<=(m.Df[j]+(1-m.z[j])*K_value)
        Concretemodel.set_obj_B_value = Constraint(reaction_listDF, rule=set_obj_B_value_c)

    #Adding thermodynamic constraints
    if set_thermodynamics:
        def set_thermodynamics_c(m,j):
            return (m.Df[j]+(1-m.z[j])*K_value)>= B_value
        Concretemodel.set_thermodynamics = Constraint(reaction_listDF, rule=set_thermodynamics_c)
        
    #Adding binary variables constraints
    if set_integer:
        def set_integer_c(m,j):
            return m.reaction[j]<=m.z[j]*ub_list[j] 
        Concretemodel.set_integer = Constraint(reaction_listDF,rule=set_integer_c)    

    #Adding concentration ratio constraints for metabolites
    if set_metabolite_ratio:
        def set_atp_adp(m):
            return m.metabolite['atp_c']-m.metabolite['adp_c']==np.log(10)
        def set_adp_amp(m):
            return m.metabolite['adp_c']-m.metabolite['amp_c']==np.log(1)
        def set_nad_nadh(m):
            return m.metabolite['nad_c']-m.metabolite['nadh_c']==np.log(10)
        def set_nadph_nadp(m):
            return m.metabolite['nadph_c']-m.metabolite['nadp_c']==np.log(10)
        def set_hco3_co2(m):
            return m.metabolite['hco3_c']-m.metabolite['co2_c']==np.log(2)

        Concretemodel.set_atp_adp = Constraint(rule=set_atp_adp) 
        Concretemodel.set_adp_amp = Constraint(rule=set_adp_amp) 
        Concretemodel.set_nad_nadh = Constraint(rule=set_nad_nadh) 
        Concretemodel.set_nadph_nadp = Constraint(rule=set_nadph_nadp) 
        Concretemodel.set_hco3_co2 = Constraint(rule=set_hco3_co2)

    #Adding Bottleneck reaction constraints
    if set_Bottleneck_reaction:
        def set_Bottleneck_reaction_c(m,j):
            return m.z[j]==1 
        Concretemodel.set_Bottleneck_reaction = Constraint(Bottleneck_reaction_list,rule=set_Bottleneck_reaction_c) 

    return Concretemodel


def FBA_template(set_obj_sum_e=False,set_constr8=False,name_mapping=None,set_constr3=False,set_constr4=False,set_constr9=False,set_constr10=False,set_constr11=False,set_constr12=False,set_constr13=False,C_total=None,set_avep=False,set_avec=False,set_constr7=False,\
                 km_dict=None,set_constr5=False,add_dynamics_p_c=False,p0=None,allsubstr=None,allsubstr2=None,v0_biomass=None,biomass_name=None,v1_product_max=None,product_name=None,\
    reaction_list=None,metabolite_list=None,coef_matrix=None,metabolites_lnC=None,reaction_g0=None,reaction_kcat_MW=None,lb_list=None,enzyme_rxns_dict=None,\
    ub_list=None,e0=None,totle_E=None,obj_name=None,K_value=None,obj_target=None,set_obj_value=False,set_substrate_ini=False,substrate_name=None,substrate_value=None,\
    set_biomass_ini=False,biomass_value=None,biomass_id=None,set_metabolite=False,set_Df=False,set_kinetic=False,set_obj_B_value=False,set_stoi_matrix=False,\
    set_bound=False,set_enzyme_constraint=False,set_enzyme_value=False,set_integer=False,set_metabolite_ratio=False,set_thermodynamics=False,B_value=None,kcat_dict=None,mw_dict=None,\
    set_obj_value_e=False,set_obj_E_value=False,set_obj_V_value=False,set_obj_TM_value=False,set_obj_Met_value=False,set_obj_single_E_value=False,E_total=None,\
    Bottleneck_reaction_list=None,set_Bottleneck_reaction=False):
    
    """According to the parameter conditions provided by the user, the specific pyomo model is returned.

    Notes
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC:Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...

    * reaction_kcat_MW: The enzymatic data of kcat (turnover number, /h) divided by MW (molecular weight, kDa).
    The format is as follows (in .csv file):
    kcat,MW,kcat_MW
    AADDGT	231887.7737	40.6396	5705.956104
    AAMYL	23490.41652	56.63940007	414.736323
    ...
    
    * lb_list: Lower bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * ub_list: Upper bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * obj_name: Name of object, such as set_obj_value, set_obj_single_E_value, set_obj_TM_value and set_obj_Met_value.    
    * K_value: the maximum value minus the minimum value of reaction thermodynamic driving force (1249, "max(DFi,max)-min(DFi,min)").
    * obj_target: Type of object function (maximize or minimize).
    * set_obj_value: Set the flux as the object function (True or False).
    * set_substrate_ini: Adding substrate amount constraints (True or False).
    * substrate_name: Substrate input reaction ID in the model (such as EX_glc__D_e_reverse).
    * substrate_value: Set the upper bound for substrate input reaction flux (10 mmol/h/gDW).
    * set_biomass_ini: Set the lower bound for biomass synthesis reaction flux (True or False).
    * biomass_value: The lower bound of biomass synthesis reaction flux.
    * biomass_id: Biomass synthesis reaction ID in the model (BIOMASS_Ec_iML1515_core_75p37M).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    * set_obj_B_value: The object function is the maximizing thermodynamic driving force of a pathway (True or False)
    * set_stoi_matrix: Adding flux balance constraints (True or False).
    * set_bound: Set the upper and lower bounds of reaction flux (True or False).
    * set_enzyme_constraint: Adding enzymamic constraints (True or False).
    * set_integer: Adding binary variables constraints (True or False).
    * set_metabolite_ratio: Adding concentration ratio constraints for metabolites (True or False).
    * set_thermodynamics: Adding thermodynamic constraints (True or False).
    * B_value: The value of maximizing the minimum thermodynamic driving force (MDF).
    * set_obj_E_value: The object function is the minimum enzyme cost of a pathway (True or False).
    * set_obj_V_value: The object function is the pFBA of a pathway (True or False)
    * set_obj_TM_value: The object function is the thermodynamic driving force of a reaction (True or False).
    * set_obj_Met_value: The object function is the concentration of a metabolite (True or False).
    * set_obj_single_E_value: The object function is the enzyme cost of a reaction (True or False).     
    * E_total: Total amount constraint of effective enzymes pool (0.13).
    * Bottleneck_reaction_list: A list extracted from the result file automatically.
    * set_Bottleneck_reaction: Adding integer variable constraints for specific reaction (True or False).
    """
    Concretemodel = ConcreteModel()
    
    if reaction_kcat_MW is not None:
    # check if kcat_MW is a column in reaction_kcat_MW
        if 'kcat_MW' in reaction_kcat_MW.columns:
            kcatmw='kcat_MW'
        elif 'kcat/mw' in reaction_kcat_MW.columns:
            kcatmw='kcat/mw'

    Concretemodel.reaction = pyo.Var(reaction_list,  within=NonNegativeReals) # reaction flux variable    
    if set_Df:
        reaction_listDF=[j for j in reaction_g0.index if j in reaction_list]
        ConMet_list= list(set(metabolite_list).intersection(set(metabolites_lnC.index)))
        Concretemodel.metabolite = pyo.Var(ConMet_list,  within=Reals) # metabolite concentration variable
        Concretemodel.Df = pyo.Var(reaction_listDF,  within=Reals) # thermodynamic driving force variable--reactions
        Concretemodel.z = pyo.Var(reaction_listDF,  within=pyo.Binary)    # binary variable
        Concretemodel.B = pyo.Var()     # thermodynamic driving force variable--mdfz
    # else:
    #
    # Concretemodel.metabolite = pyo.Var(metabolite_list,  within=Reals)
    #     Concretemodel.Df = pyo.Var(reaction_list,  within=Reals) # thermodynamic driving force variable--reactions

    #Set upper and lower bounds of metabolite concentration
    if set_enzyme_value:
        Concretemodel.e = pyo.Var(e0, within=NonNegativeReals)
        Concretemodel.e1 = pyo.Var(totle_E, within=NonNegativeReals)              

    
    if set_metabolite:
        def set_metabolite(m,i):
            return  inequality(metabolites_lnC.loc[i,'lnClb'], m.metabolite[i], metabolites_lnC.loc[i,'lnCub'])
        Concretemodel.set_metabolite= Constraint(ConMet_list,rule=set_metabolite)        

    #thermodynamic driving force expression for reactions
    if set_Df:
        def set_Df(m,j):
            return  m.Df[j]==-reaction_g0.loc[j,'g0']-2.579*sum(coef_matrix[i,j]*m.metabolite[i]  for i in ConMet_list if (i,j) in coef_matrix.keys())
        Concretemodel.set_Df = Constraint(reaction_listDF,rule=set_Df)
    
    #Set the maximum flux as the object function
    if set_obj_value:   
        def set_obj_value(m):
            return m.reaction[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_value, sense=minimize)
            
    #Set the value of maximizing the minimum thermodynamic driving force as the object function
    if set_obj_B_value:             
        def set_obj_B_value(m):
            return m.B  
        Concretemodel.obj = Objective(rule=set_obj_B_value, sense=maximize)  
    if set_obj_sum_e:             
        def set_obj_sum_e(m):
            return sum(Concretemodel.e1[i] * mw_dict[i] for i in totle_E if i in mw_dict)

        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_sum_e, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_sum_e, sense=minimize)
# 每个酶的分布
    if set_obj_value_e:   
        def set_obj_value_e(m):
            return m.e1[obj_name]*mw_dict[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_value_e, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_value_e, sense=minimize)
    #Set the minimum enzyme cost of a pathway as the object function
    if set_obj_E_value:             
        def set_obj_E_value(m):
            return sum(m.reaction[j]/(reaction_kcat_MW.loc[j,kcatmw]) for j in reaction_kcat_MW.index if j in reaction_list)
        Concretemodel.obj = Objective(rule=set_obj_E_value, sense=minimize)  

    #Minimizing the flux sum of pathway (pFBA)
    if set_obj_V_value:             
        def set_obj_V_value(m):
            return sum(m.reaction[j] for j in reaction_list)
        Concretemodel.obj = Objective(rule=set_obj_V_value, sense=minimize)  

    #To calculate the variability of enzyme usage of single reaction.
    if set_obj_single_E_value:             
        def set_obj_single_E_value(m):
            return m.reaction[obj_name]/(reaction_kcat_MW.loc[obj_name,kcatmw])
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_single_E_value, sense=maximize) 
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_single_E_value, sense=minimize)

    #To calculate the variability of thermodynamic driving force (Dfi) of single reaction.
    if set_obj_TM_value:   
        def set_obj_TM_value(m):
            return m.Df[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_TM_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_TM_value, sense=minimize)

    #To calculate the concentration variability of metabolites.
    if set_obj_Met_value:   
        def set_obj_Met_value(m):
            return m.metabolite[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_Met_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_Met_value, sense=minimize)

    #Adding flux balance constraints （FBA）
    if set_stoi_matrix:
        def set_stoi_matrix(m,i):
            return sum(coef_matrix[i,j]*m.reaction[j]  for j in reaction_list if (i,j) in coef_matrix.keys() )==0
        Concretemodel.set_stoi_matrix = Constraint( metabolite_list,rule=set_stoi_matrix)

    #Adding the upper and lower bound constraints of reaction flux
    if set_bound:
        def set_bound(m,j):
            return inequality(lb_list[j],m.reaction[j],ub_list[j])
        Concretemodel.set_bound = Constraint(reaction_list,rule=set_bound) 

    #Set the upper bound for substrate input reaction flux
    if set_substrate_ini:
        def set_substrate_ini(m): 
            return m.reaction[substrate_name] <= substrate_value
        Concretemodel.set_substrate_ini = Constraint(rule=set_substrate_ini)   
        
    #Set the lower bound for biomass synthesis reaction flux
    if set_biomass_ini:
        def set_biomass_ini(m): 
            return m.reaction[biomass_id] >=biomass_value
        Concretemodel.set_biomass_ini = Constraint(rule=set_biomass_ini)
        
        
      #Adding enzymamic constraints
    if set_enzyme_constraint:
        # flux- enzyme relationship
        Concretemodel.set_constr2 = ConstraintList()
        for i in e0:
            Concretemodel.set_constr2.add(Concretemodel.e[i] *kcat_dict[i] >= Concretemodel.reaction[i])
        # protein  concentration
        Concretemodel.set_constr3 = ConstraintList()  
        for i in totle_E:   
            Concretemodel.set_constr3.add(Concretemodel.e1[i] == sum(Concretemodel.e[j] for j in enzyme_rxns_dict[i] if j in e0))
        # total protein concentration
        if not set_obj_sum_e:
            Concretemodel.constr4 = Constraint(expr=sum(Concretemodel.e1[i] * mw_dict[i] for i in totle_E if i in mw_dict) <= E_total)
 
 
    #Adding dynamic constraints
    if add_dynamics_p_c:
            Concretemodel.p = pyo.Var(p0, bounds=(0, 1))
            Concretemodel.avep = pyo.Var(bounds=(0, 1))
            Concretemodel.c0 = pyo.Var(allsubstr)
            # Concretemodel.c0 = pyo.Var(allsubstr, bounds=lambda m, x: (C_meta_lb[x], C_meta_ub[x]))
            Concretemodel.c1 = pyo.Var(allsubstr2)
    if set_constr5:
        Concretemodel.con_enz = ConstraintList()
        for i in e0:
            if i in km_dict:
                Concretemodel.con_enz.add(Concretemodel.e[i] * kcat_dict[i]*Concretemodel.p[i]  == Concretemodel.reaction[i])
            else:
                Concretemodel.con_enz.add(Concretemodel.e[i] * kcat_dict[i]*Concretemodel.avep  == Concretemodel.reaction[i])


    if set_avep:
            # average p
        Concretemodel.con_avep = Constraint(expr=sum(Concretemodel.p[i] for i in p0)==len(p0)*Concretemodel.avep)
    if set_avec:
        # average c
        Concretemodel.con_avec = Constraint(expr=sum(Concretemodel.c0[i]  for i in allsubstr) <= C_total )
    if set_constr7:
        Concretemodel.con_sub = ConstraintList()
        for i in km_dict:
            substrate = list(km_dict[i])
            if len(substrate) == 2:
                ccomb = substrate[0] + '_' + substrate[1]
                Concretemodel.con_sub.add(Concretemodel.c1[ccomb] == Concretemodel.c0[substrate[0]] * Concretemodel.c0[substrate[1]])
                Concretemodel.con_sub.add(Concretemodel.p[i] * km_dict[i][substrate[0]][0] * Concretemodel.c0[substrate[1]] + Concretemodel.p[i] * km_dict[i][substrate[1]][0] * Concretemodel.c0[substrate[0]] + Concretemodel.p[i] * Concretemodel.c1[ccomb] == Concretemodel.c1[ccomb])
            if len(substrate) == 1:
                Concretemodel.con_sub.add(Concretemodel.p[i] * km_dict[i][substrate[0]][0] + Concretemodel.p[i] * Concretemodel.c0[substrate[0]] == Concretemodel.c0[substrate[0]])    
    # name_mapping
    if set_constr8:
        Concretemodel.set_constr8 = ConstraintList()
        for i in allsubstr:
            Concretemodel.set_constr8.add( Concretemodel.metabolite[name_mapping[i]+'_c'],Concretemodel.c0[i]  )
    
 #    确定酶范围
    # v_biomass    >=v0_biomass*0.95
    if set_constr9:
        Concretemodel.constr9 = Constraint(expr=Concretemodel.reaction[biomass_name] >= v0_biomass * 0.95)
    # 对v_biomass进行约束
    if set_constr10:
        Concretemodel.constr10 = Constraint(expr=Concretemodel.reaction[biomass_name] >= v0_biomass * 0.1)    
    # 对v_product进行约束
    if set_constr11:
        Concretemodel.constr11 = Constraint(expr=Concretemodel.reaction[product_name] >= v1_product_max * 0.95)

    #Adding thermodynamic MDF(B) object function
    if set_obj_B_value:
        def set_obj_B_value(m,j):
            return m.B<=(m.Df[j]+(1-m.z[j])*K_value)
        Concretemodel.set_obj_B_value = Constraint(reaction_listDF, rule=set_obj_B_value)

    #Adding thermodynamic constraints
    if set_thermodynamics:
        def set_thermodynamics(m,j):
            return (m.Df[j]+(1-m.z[j])*K_value)>= B_value
        Concretemodel.set_thermodynamics = Constraint(reaction_listDF, rule=set_thermodynamics)
        
    #Adding binary variables constraints
    if set_integer:
        def set_integer(m,j):
            return m.reaction[j]<=m.z[j]*ub_list[j] 
        Concretemodel.set_integer = Constraint(reaction_listDF,rule=set_integer)    

    #Adding concentration ratio constraints for metabolites
    if set_metabolite_ratio:
        def set_atp_adp(m):
            return m.metabolite['atp_c']-m.metabolite['adp_c']==np.log(10)
        def set_adp_amp(m):
            return m.metabolite['adp_c']-m.metabolite['amp_c']==np.log(1)
        def set_nad_nadh(m):
            return m.metabolite['nad_c']-m.metabolite['nadh_c']==np.log(10)
        def set_nadph_nadp(m):
            return m.metabolite['nadph_c']-m.metabolite['nadp_c']==np.log(10)
        def set_hco3_co2(m):
            return m.metabolite['hco3_c']-m.metabolite['co2_c']==np.log(2)

        Concretemodel.set_atp_adp = Constraint(rule=set_atp_adp) 
        Concretemodel.set_adp_amp = Constraint(rule=set_adp_amp) 
        Concretemodel.set_nad_nadh = Constraint(rule=set_nad_nadh) 
        Concretemodel.set_nadph_nadp = Constraint(rule=set_nadph_nadp) 
        Concretemodel.set_hco3_co2 = Constraint(rule=set_hco3_co2)

    #Adding Bottleneck reaction constraints
    if set_Bottleneck_reaction:
        def set_Bottleneck_reaction(m,j):
            return m.z[j]==1 
        Concretemodel.set_Bottleneck_reaction = Constraint(Bottleneck_reaction_list,rule=set_Bottleneck_reaction) 

    return Concretemodel

	
		
		
		
def FBA_template2(coef_matrix=None,metabolites_lnC=None,reaction_g0=None,lb_list=None,ub_list=None,reaction_kcat_MW=None,B_value=None,kcat_dict=None,mw_dict=None,K_value=None,\
            product_value=None,biomass_value=None,substrate_value=None,E_total=None,\
            product_name=None,reaction_list=None,metabolite_list=None,enzyme_rxns_dict=None,e0=None,totle_E=None,obj_name=None,obj_target=None,\
            biomass_id=None,substrate_name=None,Bottleneck_reaction_list=None,biomass_name=None,\
            set_obj_B_value=False,set_obj_sum_e=False,set_obj_value_e=False,set_obj_E_value=False,set_obj_single_E_value=False,set_obj_X_value=False,\
            set_obj_value=False,set_obj_V_value=False,set_obj_TM_value=False,set_obj_Met_value=False,set_obj_Two_value=False,\
            set_metabolite=False,set_Df=False,set_thermodynamics0=False,set_thermodynamics=False,set_integer=False,set_fix_sum_e=False,\
            set_enzyme_constraint=False,set_substrate_ini=False, set_biomass_ini=False,set_product_ini=False,set_reactions_ini=False,\
            set_metabolite_ratio=False,set_Bottleneck_reaction=False,set_Df_value=False,set_enzyme_value=False,set_max_sum_e=False,\
            mode=None,constr_coeff=None,Concretemodel_Need_Data=None):
    
    """According to the parameter conditions provided by the user, the specific pyomo model is returned.

    Notes
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC:Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'¡ã) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...

    * reaction_kcat_MW: The enzymatic data of kcat (turnover number, /h) divided by MW (molecular weight, kDa).
    The format is as follows (in .csv file):
    kcat,MW,kcat_MW
    AADDGT	231887.7737	40.6396	5705.956104
    AAMYL	23490.41652	56.63940007	414.736323
    ...
    
    * lb_list: Lower bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * ub_list: Upper bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * obj_name: Name of object, such as set_obj_value, set_obj_single_E_value, set_obj_TM_value and set_obj_Met_value.    
    * K_value: the maximum value minus the minimum value of reaction thermodynamic driving force (1249, "max(DFi,max)-min(DFi,min)").
    * obj_target: Type of object function (maximize or minimize).
    * set_obj_value: Set the flux as the object function (True or False).
    * set_substrate_ini: Adding substrate amount constraints (True or False).
    * substrate_name: Substrate input reaction ID in the model (such as EX_glc__D_e_reverse).
    * substrate_value: Set the upper bound for substrate input reaction flux (10 mmol/h/gDW).
    * set_biomass_ini: Set the lower bound for biomass synthesis reaction flux (True or False).
    * biomass_value: The lower bound of biomass synthesis reaction flux.
    * biomass_id: Biomass synthesis reaction ID in the model (BIOMASS_Ec_iML1515_core_75p37M).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    * set_obj_B_value: The object function is the maximizing thermodynamic driving force of a pathway (True or False)
    * set_stoi_matrix: Adding flux balance constraints (True or False).
    * set_bound: Set the upper and lower bounds of reaction flux (True or False).
    * set_enzyme_constraint: Adding enzymamic constraints (True or False).
    * set_integer: Adding binary variables constraints (True or False).
    * set_metabolite_ratio: Adding concentration ratio constraints for metabolites (True or False).
    * set_thermodynamics: Adding thermodynamic constraints (True or False).
    * B_value: The value of maximizing the minimum thermodynamic driving force (MDF).
    * set_obj_E_value: The object function is the minimum enzyme cost of a pathway (True or False).
    * set_obj_V_value: The object function is the pFBA of a pathway (True or False)
    * set_obj_TM_value: The object function is the thermodynamic driving force of a reaction (True or False).
    * set_obj_Met_value: The object function is the concentration of a metabolite (True or False).
    * set_obj_single_E_value: The object function is the enzyme cost of a reaction (True or False).     
    * E_total: Total amount constraint of effective enzymes pool (0.13).
    * Bottleneck_reaction_list: A list extracted from the result file automatically.
    * set_Bottleneck_reaction: Adding integer variable constraints for specific reaction (True or False).
    """
    Concretemodel = ConcreteModel() 
    
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    if   mode=='S':
        if set_obj_X_value:
            set_obj_Two_value=True
    if   mode=='ST':
        if set_obj_B_value:
            set_thermodynamics0=True
        if not set_obj_B_value:
            set_thermodynamics=True
        set_Df_value=True
        set_metabolite=True
        set_Df=True
        set_integer=True
        set_metabolite_ratio=True
        
        metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
        reaction_g0=Concretemodel_Need_Data['reaction_g0']
        B_value=Concretemodel_Need_Data['B_value']
        K_value=Concretemodel_Need_Data['K_value']
        
        
    if  mode=='SE': 
        set_enzyme_constraint=True
        set_enzyme_value=True
        set_max_sum_e=True
        # if set_obj_sum_e or set_fix_sum_e:
        #     set_max_sum_e=False
        # if not set_obj_sum_e:
        #     set_max_sum_e=True
            
        kcat_dict=Concretemodel_Need_Data['kcat_dict']
        mw_dict=Concretemodel_Need_Data['mw_dict']
        enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
        e0=Concretemodel_Need_Data['e0']
        totle_E=Concretemodel_Need_Data['totle_E']
        E_total=Concretemodel_Need_Data['E_total']
           
    if  mode=='SET':   
        if set_obj_X_value:
            set_obj_Two_value=True 
        if set_obj_B_value:
            set_thermodynamics0=True
        if not set_obj_B_value:
            set_thermodynamics=True
        if set_obj_sum_e or set_fix_sum_e:
            set_max_sum_e=False
        if not set_obj_sum_e:
            set_max_sum_e=True
        
        set_stoi_matrix=True
        set_bound=True   
        set_Df_value=True
        set_enzyme_value=True
        set_metabolite=True
        set_Df=True
        set_metabolite_ratio=True
        set_integer=True 
        set_enzyme_constraint=True    
        kcat_dict=Concretemodel_Need_Data['kcat_dict']
        mw_dict=Concretemodel_Need_Data['mw_dict']
        enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
        e0=Concretemodel_Need_Data['e0']
        totle_E=Concretemodel_Need_Data['totle_E']
        E_total=Concretemodel_Need_Data['E_total']   
        metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
        reaction_g0=Concretemodel_Need_Data['reaction_g0']
        B_value=Concretemodel_Need_Data['B_value']
        K_value=Concretemodel_Need_Data['K_value']        
                    
    if constr_coeff !=None:
        if 'substrate_constrain' in constr_coeff:
            set_substrate_ini=True
            substrate_value=constr_coeff['substrate_constrain'][1]
            substrate_name=constr_coeff['substrate_constrain'][0]
        if 'biomass_constrain' in constr_coeff:
            set_biomass_ini=True
            biomass_value=constr_coeff['biomass_constrain'][1]
            biomass_id=constr_coeff['biomass_constrain'][0]
        if 'product_constrain' in constr_coeff:
            set_product_ini=True
            product_value=constr_coeff['product_constrain'][1]
            product_name=constr_coeff['product_constrain'][0]
        if 'fix_reactions' in constr_coeff and constr_coeff['fix_reactions'] !={}:
            set_reactions_ini=True
        if 'fix_E_total' in constr_coeff:
            set_fix_sum_e=True
            set_max_sum_e=False
        # if 'fix_E_total' in constr_coeff:
        #     set_fix_sum_e=True
        #     fix_sum_e=constr_coeff['fix_E_total']
        #     set_max_sum_e=False          
    if mode =='S0':
        Concretemodel.reaction = pyo.Var(reaction_list)
    elif mode =='S' or mode =='SE' or mode =='SET' or mode =='ST':
           # Concretemodel.metabolite = pyo.Var(metabolite_list,  within=Reals)  
        Concretemodel.reaction = pyo.Var(reaction_list,  within=NonNegativeReals,bounds=lambda m, i: (lb_list[i], ub_list[i]))
    # Concretemodel.reaction = pyo.Var(reaction_list,  within=NonNegativeReals,bounds=lambda m, i: (lb_list[i], ub_list[i])) # reaction flux variable  

    # Concretemodel.X = pyo.Var() 
    # SV=0
    # Concretemodel.set_stoi_matrix = ConstraintList()
    # for i in metabolite_list:
    #     Concretemodel.set_stoi_matrix.add(sum(coef_matrix[i,j]*Concretemodel.reaction[j]  for j in reaction_list if (i,j) in coef_matrix.keys() )==0)
    def set_stoi_matrix(m,i):
        return sum(coef_matrix[i,j]*m.reaction[j]  for j in reaction_list if (i,j) in coef_matrix.keys() )==0
    Concretemodel.set_stoi_matrix = Constraint( metabolite_list,rule=set_stoi_matrix)
    # LB¡¢UB

    def set_bound(m,j):
        return inequality(lb_list[j],m.reaction[j],ub_list[j])
    Concretemodel.set_bound = Constraint(reaction_list,rule=set_bound)   
    if set_obj_Two_value:
        # def set_obj_Two_value(m,obj_name,biomass_name):
        #     return m.X==(m.reaction[obj_name]*m.reaction[biomass_name])
        Concretemodel.set_obj_Two_value = Constraint(expr=Concretemodel.reaction[obj_name]*Concretemodel.reaction[biomass_name]==Concretemodel.X)

    if reaction_kcat_MW is not None:
    # check if kcat_MW is a column in reaction_kcat_MW
        if 'kcat_MW' in reaction_kcat_MW.columns:
            kcatmw='kcat_MW'
        elif 'kcat/mw' in reaction_kcat_MW.columns:
            kcatmw='kcat/mw'
            
    #           ****ÉèÖÃ±äÁ¿****    
    # ÈÈÁ¦Ñ§±äÁ¿ÉèÖÃ    
    if set_Df_value:
        reaction_listDF=[j for j in reaction_g0.index if j in reaction_list]
        ConMet_list= list(set(metabolite_list).intersection(set(metabolites_lnC.index)))
        Concretemodel.metabolite = pyo.Var(ConMet_list,  within=Reals) # metabolite concentration variable
        Concretemodel.Df = pyo.Var(reaction_listDF,  within=Reals) # thermodynamic driving force variable--reactions
        Concretemodel.z = pyo.Var(reaction_listDF,  within=pyo.Binary)    # binary variable
        Concretemodel.B = pyo.Var()     # thermodynamic driving force variable--mdfz
 
    # µ°°×ÎªÖÐÐÄÃ¸±äÁ¿ÉèÖÃ
    if set_enzyme_value:
        Concretemodel.e = pyo.Var(e0, within=NonNegativeReals)
        Concretemodel.e1 = pyo.Var(totle_E, within=NonNegativeReals)              

    #           ****ÉèÖÃÄ¿±êº¯Êý****
    
    # ÒÔB_valueÎªÄ¿±êº¯Êý
    if set_obj_X_value:
        def set_obj_X_value(m):
            return m.X
        Concretemodel.obj = Objective(rule=set_obj_X_value, sense=maximize) 
    if set_obj_B_value:             
        def set_obj_B_value(m):
            return m.B  
        Concretemodel.obj = Objective(rule=set_obj_B_value, sense=maximize)  
    # ÒÔ×Üµ°°×Á¿¼ÓºÍÎªÄ¿±êº¯Êý
    if set_obj_sum_e:             
        def set_obj_sum_e(m):
            return sum(Concretemodel.e1[i] * mw_dict[i] for i in totle_E if i in mw_dict)
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_sum_e, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_sum_e, sense=minimize)
    # ÒÔ×îÐ¡»¯Ã¸Á¿¼ÓºÍÎªÄ¿±êº¯Êý£¨²»ÊÇÒÔµ°°×ÎªÖÐÐÄ£©   
    if set_obj_E_value:             
        def set_obj_E_value(m):
            return sum(m.reaction[j]/(reaction_kcat_MW.loc[j,kcatmw]) for j in reaction_kcat_MW.index if j in reaction_list)
        Concretemodel.obj = Objective(rule=set_obj_E_value, sense=minimize)      
    # ÒÔµ¥Ò»µ°°×ÎªÄ¿±êº¯Êý
    if set_obj_value_e:   
        def set_obj_value_e(m):
            return m.e1[obj_name]*mw_dict[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_value_e, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_value_e, sense=minimize)    
    # ÒÔµ¥Ò»µ°°×ÎªÄ¿±êº¯Êý£¨²»ÊÇÒÔµ°°×ÎªÖÐÐÄ£©
    if set_obj_single_E_value:             
        def set_obj_single_E_value(m):
            return m.reaction[obj_name]/(reaction_kcat_MW.loc[obj_name,kcatmw])
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_single_E_value, sense=maximize) 
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_single_E_value, sense=minimize)
    # ÒÔÄ³¸ö·´Ó¦ÎªÄ¿±êº¯Êý
    if set_obj_value:   
        def set_obj_value(m):
            return m.reaction[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_value, sense=minimize)
    # 2个变量
    # if set_obj_Two_value:   
    #     def set_obj_Two_value(m):
    #         return m.reaction[obj_name]*m.reaction[biomass_name]
    #     if obj_target=='maximize':
    #         Concretemodel.obj = Objective(rule=set_obj_Two_value, sense=maximize)
    #     elif obj_target=='minimize':
    #         Concretemodel.obj = Objective(rule=set_obj_Two_value, sense=minimize)    
    #×îÐ¡»¯Í¨Á¿ºÍÎªÄ¿±êº¯Êý (pFBA)
    if set_obj_V_value:             
        def set_obj_V_value(m):
            return sum(m.reaction[j] for j in reaction_list)
        Concretemodel.obj = Objective(rule=set_obj_V_value, sense=minimize)  
    #To calculate the variability of thermodynamic driving force (Dfi) of single reaction.
    if set_obj_TM_value:   
        def set_obj_TM_value(m):
            return m.Df[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_TM_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_TM_value, sense=minimize)
    #ÒÔ´úÐ»ÎïÅ¨¶ÈÎªÄ¿±êº¯Êý£¨ÈÈÁ¦Ñ§£©
    if set_obj_Met_value:   
        def set_obj_Met_value(m):
            return m.metabolite[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_Met_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_Met_value, sense=minimize)
 
 
 
 
    #           ****Ìí¼ÓÔ¼Êø****
    
    #  ÈÈÁ¦Ñ§Ô¼Êøset_metabolite¡¢set_Df¡¢set_obj_B_value£¨B_valueÎª±äÁ¿£©¡¢set_thermodynamics£¨B_valueÎª¶¨Öµ£©¡¢set_integer
    if set_metabolite:
        def set_metabolite(m,i):
            return  inequality(metabolites_lnC.loc[i,'lnClb'], m.metabolite[i], metabolites_lnC.loc[i,'lnCub'])
        Concretemodel.set_metabolite= Constraint(ConMet_list,rule=set_metabolite)        
    #thermodynamic driving force expression for reactions
    if set_Df:
        def set_Df(m,j):
            return  m.Df[j]==-reaction_g0.loc[j,'g0']-2.579*sum(coef_matrix[i,j]*m.metabolite[i]  for i in ConMet_list if (i,j) in coef_matrix.keys())
        Concretemodel.set_Df = Constraint(reaction_listDF,rule=set_Df)
    #Adding thermodynamic MDF(B) object function
    if set_thermodynamics0:
        def set_thermodynamics0(m,j):
            return m.B<=(m.Df[j]+(1-m.z[j])*K_value)
        Concretemodel.set_obj_B_value = Constraint(reaction_listDF, rule=set_thermodynamics0)
    #Adding thermodynamic constraints
    if set_thermodynamics:
        def set_thermodynamics(m,j):
            return (m.Df[j]+(1-m.z[j])*K_value)>= B_value
        Concretemodel.set_thermodynamics = Constraint(reaction_listDF, rule=set_thermodynamics)
    #Adding binary variables constraints
    if set_integer:
        def set_integer(m,j):
            return m.reaction[j]<=m.z[j]*ub_list[j] 
        Concretemodel.set_integer = Constraint(reaction_listDF,rule=set_integer)     
        
        
    # Ã¸Ô¼Êø
    if set_enzyme_constraint:
        # flux- enzyme relationship
        Concretemodel.set_constr2 = ConstraintList()
        for i in e0:
            Concretemodel.set_constr2.add(Concretemodel.e[i] *kcat_dict[i] >= Concretemodel.reaction[i])
        # protein  concentration
        Concretemodel.set_constr3 = ConstraintList()  
        for i in totle_E:   
            Concretemodel.set_constr3.add(Concretemodel.e1[i] == sum(Concretemodel.e[j] for j in enzyme_rxns_dict[i] if j in e0))
        # total protein concentration
    if set_max_sum_e:
        Concretemodel.set_max_sum_e = Constraint(expr=sum(Concretemodel.e1[i] * mw_dict[i] for i in totle_E if i in mw_dict) <= E_total)
    if set_fix_sum_e:
        Concretemodel.set_max_sum_e = Constraint(expr=sum(Concretemodel.e1[i] * mw_dict[i] for i in totle_E if i in mw_dict) == E_total) 
        
    # ×î´óµ×ÎïÉãÈëÔ¼Êø
    if set_substrate_ini:
        def set_substrate_ini(m): 
            return m.reaction[substrate_name] <= substrate_value
        Concretemodel.set_substrate_ini = Constraint(rule=set_substrate_ini)       
    # ¹Ì¶¨biomass·¶Î§
    if set_biomass_ini:
        def set_biomass_ini(m): 
            return m.reaction[biomass_id] >=biomass_value
        Concretemodel.set_biomass_ini = Constraint(rule=set_biomass_ini)
    # ¹Ì¶¨product·¶Î§
    if set_product_ini:
        Concretemodel.set_product_ini = Constraint(expr=Concretemodel.reaction[product_name] >= product_value)
    # ÅúÁ¿¹Ì¶¨·´Ó¦Í¨Á¿
    if set_reactions_ini:
        Concretemodel.set_reactions_ini = ConstraintList()
        for i in constr_coeff['fix_reactions']:
            Concretemodel.set_reactions_ini.add(Concretemodel.reaction[i] >= constr_coeff['fix_reactions'][i][0])
            Concretemodel.set_reactions_ini.add(Concretemodel.reaction[i] <= constr_coeff['fix_reactions'][i][1])

    #Adding concentration ratio constraints for metabolites
    if set_metabolite_ratio:
        def set_atp_adp(m):
            return m.metabolite['atp_c']-m.metabolite['adp_c']==np.log(10)
        def set_adp_amp(m):
            return m.metabolite['adp_c']-m.metabolite['amp_c']==np.log(1)
        def set_nad_nadh(m):
            return m.metabolite['nad_c']-m.metabolite['nadh_c']==np.log(10)
        def set_nadph_nadp(m):
            return m.metabolite['nadph_c']-m.metabolite['nadp_c']==np.log(10)
        def set_hco3_co2(m):
            return m.metabolite['hco3_c']-m.metabolite['co2_c']==np.log(2)

        Concretemodel.set_atp_adp = Constraint(rule=set_atp_adp) 
        Concretemodel.set_adp_amp = Constraint(rule=set_adp_amp) 
        Concretemodel.set_nad_nadh = Constraint(rule=set_nad_nadh) 
        Concretemodel.set_nadph_nadp = Constraint(rule=set_nadph_nadp) 
        Concretemodel.set_hco3_co2 = Constraint(rule=set_hco3_co2)

    #Adding Bottleneck reaction constraints
    if set_Bottleneck_reaction:
        def set_Bottleneck_reaction(m,j):
            return m.z[j]==1 
        Concretemodel.set_Bottleneck_reaction = Constraint(Bottleneck_reaction_list,rule=set_Bottleneck_reaction) 

    return Concretemodel



def FBA_template3(coef_matrix=None,metabolites_lnC=None,reaction_g0=None,lb_list=None,ub_list=None,reaction_kcat_MW=None,B_value=None,kcat_dict=None,mw_dict=None,K_value=None,\
            product_value=None,biomass_value=None,substrate_value=None,E_total=None,\
            product_name=None,reaction_list=None,metabolite_list=None,enzyme_rxns_dict=None,e0=None,totle_E=None,obj_name=None,obj_target=None,\
            biomass_id=None,substrate_name=None,Bottleneck_reaction_list=None,\
            set_obj_B_value=False,set_obj_sum_e=False,set_obj_value_e=False,set_obj_E_value=False,set_obj_single_E_value=False,\
            set_obj_value=False,set_obj_V_value=False,set_obj_TM_value=False,set_obj_Met_value=False,\
            set_metabolite=False,set_Df=False,set_thermodynamics0=False,set_thermodynamics=False,set_integer=False,set_fix_sum_e=False,\
            set_enzyme_constraint=False,set_substrate_ini=False, set_biomass_ini=False,set_product_ini=False,set_reactions_ini=False,\
            set_metabolite_ratio=False,set_Bottleneck_reaction=False,set_Df_value=False,set_enzyme_value=False,set_max_sum_e=False,\
            mode=None,constr_coeff=None,Concretemodel_Need_Data=None):
    
  
    Concretemodel = ConcreteModel()
    
    
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    if   mode=='ST':
        if set_obj_B_value:
            set_thermodynamics0=True
        if not set_obj_B_value:
            set_thermodynamics=True
        set_Df_value=True
        set_metabolite=True
        set_Df=True
        set_integer=True
        set_metabolite_ratio=True
        
        metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
        reaction_g0=Concretemodel_Need_Data['reaction_g0']
        B_value=Concretemodel_Need_Data['B_value']
        K_value=Concretemodel_Need_Data['K_value']
        
        
    if  mode=='SE': 
        set_enzyme_constraint=True
        set_enzyme_value=True
        set_max_sum_e=True
        # if set_obj_sum_e or set_fix_sum_e:
        #     set_max_sum_e=False
        # if not set_obj_sum_e:
        #     set_max_sum_e=True
            
        kcat_dict=Concretemodel_Need_Data['kcat_dict']
        mw_dict=Concretemodel_Need_Data['mw_dict']
        # enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
        e0=Concretemodel_Need_Data['e0']
        # totle_E=Concretemodel_Need_Data['totle_E']
        E_total=Concretemodel_Need_Data['E_total']
           
    if  mode=='SET':    
        if set_obj_B_value:
            set_thermodynamics0=True
        if not set_obj_B_value:
            set_thermodynamics=True
        if set_obj_sum_e or set_fix_sum_e:
            set_max_sum_e=False
        if not set_obj_sum_e:
            set_max_sum_e=True
        
        set_stoi_matrix=True
        set_bound=True   
        set_Df_value=True
        set_enzyme_value=True
        set_metabolite=True
        set_Df=True
        set_metabolite_ratio=True
        set_integer=True 
        set_enzyme_constraint=True    
        kcat_dict=Concretemodel_Need_Data['kcat_dict']
        mw_dict=Concretemodel_Need_Data['mw_dict']
        # enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
        e0=Concretemodel_Need_Data['e0']
        # totle_E=Concretemodel_Need_Data['totle_E']
        E_total=Concretemodel_Need_Data['E_total']   
        metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
        reaction_g0=Concretemodel_Need_Data['reaction_g0']
        B_value=Concretemodel_Need_Data['B_value']
        K_value=Concretemodel_Need_Data['K_value']        
                    
    if constr_coeff !=None:
        if 'substrate_constrain' in constr_coeff:
            set_substrate_ini=True
            substrate_value=constr_coeff['substrate_constrain'][1]
            substrate_name=constr_coeff['substrate_constrain'][0]
        if 'biomass_constrain' in constr_coeff:
            set_biomass_ini=True
            biomass_value=constr_coeff['biomass_constrain'][1]
            biomass_id=constr_coeff['biomass_constrain'][0]
        if 'product_constrain' in constr_coeff:
            set_product_ini=True
            product_value=constr_coeff['product_constrain'][1]
            product_name=constr_coeff['product_constrain'][0]
        if 'fix_reactions' in constr_coeff and constr_coeff['fix_reactions'] !={}:
            set_reactions_ini=True
        if 'fix_E_total' in constr_coeff:
            set_fix_sum_e=True
            fix_sum_e=constr_coeff['fix_E_total']
            set_max_sum_e=False

    Concretemodel.reaction = pyo.Var(reaction_list,  within=NonNegativeReals,bounds=lambda m, i: (lb_list[i], ub_list[i])) # reaction flux variable  
    # SV=0
    # Concretemodel.set_stoi_matrix = ConstraintList()
    # for i in metabolite_list:
    #     Concretemodel.set_stoi_matrix.add(sum(coef_matrix[i,j]*Concretemodel.reaction[j]  for j in reaction_list if (i,j) in coef_matrix.keys() )==0)
    def set_stoi_matrix(m,i):
        return sum(coef_matrix[i,j]*m.reaction[j]  for j in reaction_list if (i,j) in coef_matrix.keys() )==0
    Concretemodel.set_stoi_matrix = Constraint( metabolite_list,rule=set_stoi_matrix)
    # LB¡¢UB

    def set_bound(m,j):
        return inequality(lb_list[j],m.reaction[j],ub_list[j])
    Concretemodel.set_bound = Constraint(reaction_list,rule=set_bound)   
    
    
    if reaction_kcat_MW is not None:
    # check if kcat_MW is a column in reaction_kcat_MW
        if 'kcat_MW' in reaction_kcat_MW.columns:
            kcatmw='kcat_MW'
        elif 'kcat/mw' in reaction_kcat_MW.columns:
            kcatmw='kcat/mw'
            
    #           ****ÉèÖÃ±äÁ¿****    
    # ÈÈÁ¦Ñ§±äÁ¿ÉèÖÃ    
    if set_Df_value:
        reaction_listDF=[j for j in reaction_g0.index if j in reaction_list]
        ConMet_list= list(set(metabolite_list).intersection(set(metabolites_lnC.index)))
        Concretemodel.metabolite = pyo.Var(ConMet_list,  within=Reals) # metabolite concentration variable
        Concretemodel.Df = pyo.Var(reaction_listDF,  within=Reals) # thermodynamic driving force variable--reactions
        Concretemodel.z = pyo.Var(reaction_listDF,  within=pyo.Binary)    # binary variable
        Concretemodel.B = pyo.Var()     # thermodynamic driving force variable--mdfz
 
    # µ°°×ÎªÖÐÐÄÃ¸±äÁ¿ÉèÖÃ
    if set_enzyme_value:
        Concretemodel.e = pyo.Var(e0, within=NonNegativeReals)
            

    #           ****ÉèÖÃÄ¿±êº¯Êý****
    
    # ÒÔB_valueÎªÄ¿±êº¯Êý
    if set_obj_B_value:             
        def set_obj_B_value(m):
            return m.B  
        Concretemodel.obj = Objective(rule=set_obj_B_value, sense=maximize)  
    # ÒÔ×ÜÃ¸Á¿¼ÓºÍÎªÄ¿±êº¯Êý
    if set_obj_sum_e:             
        def set_obj_sum_e(m):
            return sum(Concretemodel.e[i] * mw_dict[i] for i in e0)
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_sum_e, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_sum_e, sense=minimize)
    
    # ÒÔµ¥Ò»Ã¸ÎªÄ¿±êº¯Êý
    if set_obj_value_e:   
        def set_obj_value_e(m):
            return m.e[obj_name]*mw_dict[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_value_e, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_value_e, sense=minimize)    

    # ÒÔÄ³¸ö·´Ó¦ÎªÄ¿±êº¯Êý
    if set_obj_value:   
        def set_obj_value(m):
            return m.reaction[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_value, sense=minimize)           
    #×îÐ¡»¯Í¨Á¿ºÍÎªÄ¿±êº¯Êý (pFBA)
    if set_obj_V_value:             
        def set_obj_V_value(m):
            return sum(m.reaction[j] for j in reaction_list)
        Concretemodel.obj = Objective(rule=set_obj_V_value, sense=minimize)  
    #To calculate the variability of thermodynamic driving force (Dfi) of single reaction.
    if set_obj_TM_value:   
        def set_obj_TM_value(m):
            return m.Df[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_TM_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_TM_value, sense=minimize)
    #ÒÔ´úÐ»ÎïÅ¨¶ÈÎªÄ¿±êº¯Êý£¨ÈÈÁ¦Ñ§£©
    if set_obj_Met_value:   
        def set_obj_Met_value(m):
            return m.metabolite[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_Met_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_Met_value, sense=minimize)
 
 
 
 
    #           ****Ìí¼ÓÔ¼Êø****
    
    #  ÈÈÁ¦Ñ§Ô¼Êøset_metabolite¡¢set_Df¡¢set_obj_B_value£¨B_valueÎª±äÁ¿£©¡¢set_thermodynamics£¨B_valueÎª¶¨Öµ£©¡¢set_integer
    if set_metabolite:
        def set_metabolite(m,i):
            return  inequality(metabolites_lnC.loc[i,'lnClb'], m.metabolite[i], metabolites_lnC.loc[i,'lnCub'])
        Concretemodel.set_metabolite= Constraint(ConMet_list,rule=set_metabolite)        
    #thermodynamic driving force expression for reactions
    if set_Df:
        def set_Df(m,j):
            return  m.Df[j]==-reaction_g0.loc[j,'g0']-2.579*sum(coef_matrix[i,j]*m.metabolite[i]  for i in ConMet_list if (i,j) in coef_matrix.keys())
        Concretemodel.set_Df = Constraint(reaction_listDF,rule=set_Df)
    #Adding thermodynamic MDF(B) object function
    if set_thermodynamics0:
        def set_thermodynamics0(m,j):
            return m.B<=(m.Df[j]+(1-m.z[j])*K_value)
        Concretemodel.set_obj_B_value = Constraint(reaction_listDF, rule=set_thermodynamics0)
    #Adding thermodynamic constraints
    if set_thermodynamics:
        def set_thermodynamics(m,j):
            return (m.Df[j]+(1-m.z[j])*K_value)>= B_value
        Concretemodel.set_thermodynamics = Constraint(reaction_listDF, rule=set_thermodynamics)
    #Adding binary variables constraints
    if set_integer:
        def set_integer(m,j):
            return m.reaction[j]<=m.z[j]*ub_list[j] 
        Concretemodel.set_integer = Constraint(reaction_listDF,rule=set_integer)     
        
        
    # Ã¸Ô¼Êø
    if set_enzyme_constraint:
        # flux- enzyme relationship
        Concretemodel.set_constr2 = ConstraintList()
        for i in e0:
            Concretemodel.set_constr2.add(Concretemodel.e[i] *kcat_dict[i] >= Concretemodel.reaction[i])
        # total protein concentration
    if set_max_sum_e:
        Concretemodel.set_max_sum_e = Constraint(expr=sum(Concretemodel.e[i] * mw_dict[i] for i in  e0) <= E_total)
    if set_fix_sum_e:
        Concretemodel.set_fix_sum_e = Constraint(expr=sum(Concretemodel.e[i] * mw_dict[i] for i in  e0) == fix_sum_e) 

    # ×î´óµ×ÎïÉãÈëÔ¼Êø
    if set_substrate_ini:
        def set_substrate_ini(m): 
            return m.reaction[substrate_name] <= substrate_value
        Concretemodel.set_substrate_ini = Constraint(rule=set_substrate_ini)       
    # ¹Ì¶¨biomass·¶Î§
    if set_biomass_ini:
        def set_biomass_ini(m): 
            return m.reaction[biomass_id] >=biomass_value
        Concretemodel.set_biomass_ini = Constraint(rule=set_biomass_ini)
    # ¹Ì¶¨product·¶Î§
    if set_product_ini:
        Concretemodel.set_product_ini = Constraint(expr=Concretemodel.reaction[product_name] >= product_value)
    # ÅúÁ¿¹Ì¶¨·´Ó¦Í¨Á¿
    if set_reactions_ini:
        Concretemodel.set_reactions_ini = ConstraintList()
        for i in constr_coeff['fix_reactions']:
            Concretemodel.set_reactions_ini.add(Concretemodel.reaction[i] >= constr_coeff['fix_reactions'][i][0])
            Concretemodel.set_reactions_ini.add(Concretemodel.reaction[i] <= constr_coeff['fix_reactions'][i][1])

    #Adding concentration ratio constraints for metabolites
    if set_metabolite_ratio:
        def set_atp_adp(m):
            return m.metabolite['atp_c']-m.metabolite['adp_c']==np.log(10)
        def set_adp_amp(m):
            return m.metabolite['adp_c']-m.metabolite['amp_c']==np.log(1)
        def set_nad_nadh(m):
            return m.metabolite['nad_c']-m.metabolite['nadh_c']==np.log(10)
        def set_nadph_nadp(m):
            return m.metabolite['nadph_c']-m.metabolite['nadp_c']==np.log(10)
        def set_hco3_co2(m):
            return m.metabolite['hco3_c']-m.metabolite['co2_c']==np.log(2)

        Concretemodel.set_atp_adp = Constraint(rule=set_atp_adp) 
        Concretemodel.set_adp_amp = Constraint(rule=set_adp_amp) 
        Concretemodel.set_nad_nadh = Constraint(rule=set_nad_nadh) 
        Concretemodel.set_nadph_nadp = Constraint(rule=set_nadph_nadp) 
        Concretemodel.set_hco3_co2 = Constraint(rule=set_hco3_co2)

    #Adding Bottleneck reaction constraints
    if set_Bottleneck_reaction:
        def set_Bottleneck_reaction(m,j):
            return m.z[j]==1 
        Concretemodel.set_Bottleneck_reaction = Constraint(Bottleneck_reaction_list,rule=set_Bottleneck_reaction) 

    return Concretemodel





	

def Get_Max_Min_Df(Concretemodel_Need_Data,obj_name,obj_target,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']

    Concretemodel = Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,obj_name=obj_name,obj_target=obj_target,\
        set_obj_TM_value=True,set_metabolite=True,set_Df=True)
    """Calculation both of maximum and minimum thermodynamic driving force for reactions,without metabolite ratio constraints.
    
    Notes:
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.

    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC: Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...
    
    * obj_name: Object reaction ID.
    * obj_target: Type of object function (maximize or minimize).    
    * set_obj_TM_value: set_obj_TM_value: The object function is the thermodynamic driving force of a reaction (True or False).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    """ 
    max_min_Df_list=pd.DataFrame()  
    opt=Model_Solve(Concretemodel,solver)

    if obj_target=='maximize':
        max_min_Df_list.loc[obj_name,'max_value']=opt.obj() 
    elif obj_target=='minimize':
        max_min_Df_list.loc[obj_name,'min_value']=opt.obj() 

    return max_min_Df_list

def FBA_template4(coef_matrix=None,metabolites_lnC=None,reaction_g0=None,lb_list=None,ub_list=None,reaction_kcat_MW=None,B_value=None,kcat_dict=None,mw_dict=None,K_value=None,\
            product_value=None,biomass_value=None,substrate_value=None,E_total=None,\
            product_name=None,reaction_list=None,metabolite_list=None,enzyme_rxns_dict=None,e0=None,totle_E=None,obj_name=None,obj_target=None,\
            biomass_id=None,substrate_name=None,Bottleneck_reaction_list=None,set_obj_y_value=False,\
            set_obj_B_value=False,set_obj_sum_e=False,set_obj_value_e=False,set_obj_E_value=False,set_obj_single_E_value=False,\
            set_obj_value=False,set_obj_V_value=False,set_obj_TM_value=False,set_obj_Met_value=False,\
            set_metabolite=False,set_Df=False,set_thermodynamics0=False,set_thermodynamics=False,set_integer=False,set_fix_sum_e=False,\
            set_enzyme_constraint=False,set_substrate_ini=False, set_biomass_ini=False,set_product_ini=False,set_reactions_ini=False,\
            set_metabolite_ratio=False,set_Bottleneck_reaction=False,set_Df_value=False,set_enzyme_value=False,set_max_sum_e=False,\
            mode=None,constr_coeff=None,Concretemodel_Need_Data=None,num_e1=None,M=None):
    
    """According to the parameter conditions provided by the user, the specific pyomo model is returned.

    Notes
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC:Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'Â°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...

    * reaction_kcat_MW: The enzymatic data of kcat (turnover number, /h) divided by MW (molecular weight, kDa).
    The format is as follows (in .csv file):
    kcat,MW,kcat_MW
    AADDGT	231887.7737	40.6396	5705.956104
    AAMYL	23490.41652	56.63940007	414.736323
    ...
    
    * lb_list: Lower bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * ub_list: Upper bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * obj_name: Name of object, such as set_obj_value, set_obj_single_E_value, set_obj_TM_value and set_obj_Met_value.    
    * K_value: the maximum value minus the minimum value of reaction thermodynamic driving force (1249, "max(DFi,max)-min(DFi,min)").
    * obj_target: Type of object function (maximize or minimize).
    * set_obj_value: Set the flux as the object function (True or False).
    * set_substrate_ini: Adding substrate amount constraints (True or False).
    * substrate_name: Substrate input reaction ID in the model (such as EX_glc__D_e_reverse).
    * substrate_value: Set the upper bound for substrate input reaction flux (10 mmol/h/gDW).
    * set_biomass_ini: Set the lower bound for biomass synthesis reaction flux (True or False).
    * biomass_value: The lower bound of biomass synthesis reaction flux.
    * biomass_id: Biomass synthesis reaction ID in the model (BIOMASS_Ec_iML1515_core_75p37M).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    * set_obj_B_value: The object function is the maximizing thermodynamic driving force of a pathway (True or False)
    * set_stoi_matrix: Adding flux balance constraints (True or False).
    * set_bound: Set the upper and lower bounds of reaction flux (True or False).
    * set_enzyme_constraint: Adding enzymamic constraints (True or False).
    * set_integer: Adding binary variables constraints (True or False).
    * set_metabolite_ratio: Adding concentration ratio constraints for metabolites (True or False).
    * set_thermodynamics: Adding thermodynamic constraints (True or False).
    * B_value: The value of maximizing the minimum thermodynamic driving force (MDF).
    * set_obj_E_value: The object function is the minimum enzyme cost of a pathway (True or False).
    * set_obj_V_value: The object function is the pFBA of a pathway (True or False)
    * set_obj_TM_value: The object function is the thermodynamic driving force of a reaction (True or False).
    * set_obj_Met_value: The object function is the concentration of a metabolite (True or False).
    * set_obj_single_E_value: The object function is the enzyme cost of a reaction (True or False).     
    * E_total: Total amount constraint of effective enzymes pool (0.13).
    * Bottleneck_reaction_list: A list extracted from the result file automatically.
    * set_Bottleneck_reaction: Adding integer variable constraints for specific reaction (True or False).
    """
    Concretemodel = ConcreteModel()
    
    
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    if   mode=='ST':
        if set_obj_B_value:
            set_thermodynamics0=True
        if not set_obj_B_value:
            set_thermodynamics=True
        set_Df_value=True
        set_metabolite=True
        set_Df=True
        set_integer=True
        set_metabolite_ratio=True
        
        metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
        reaction_g0=Concretemodel_Need_Data['reaction_g0']
        B_value=Concretemodel_Need_Data['B_value']
        K_value=Concretemodel_Need_Data['K_value']
        
        
    if  mode=='SE': 
        set_enzyme_constraint=True
        set_enzyme_value=True
        if set_obj_sum_e or set_fix_sum_e:
            set_max_sum_e=False
        if not set_obj_sum_e:
            set_max_sum_e=True
            
        kcat_dict=Concretemodel_Need_Data['kcat_dict']
        mw_dict=Concretemodel_Need_Data['mw_dict']
        enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
        e0=Concretemodel_Need_Data['e0']
        totle_E=Concretemodel_Need_Data['totle_E']
        E_total=Concretemodel_Need_Data['E_total']
        y1_protain=Concretemodel_Need_Data['y1_protain']
        # listy=[]
        # if len(y1_protain)>0:
        #     for i in y1_protain:
        #         for j in enzyme_rxns_dict[i]:
        #             listy.append(j)
        listy = []
        if len(y1_protain) > 0:
            for i in y1_protain:
                for j in Concretemodel_Need_Data['enzyme_rxns_dict'][i]:
                    if j in Concretemodel_Need_Data['kcat_dict']:
                        listy.append(j)

    if  mode=='SET':    
        if set_obj_B_value:
            set_thermodynamics0=True
        if not set_obj_B_value:
            set_thermodynamics=True
        if set_obj_sum_e or set_fix_sum_e:
            set_max_sum_e=False
        if not set_obj_sum_e:
            set_max_sum_e=True
            
        set_Df_value=True
        set_enzyme_value=True
        set_metabolite=True
        set_Df=True
        set_metabolite_ratio=True
        set_integer=True 
        set_enzyme_constraint=True    
        kcat_dict=Concretemodel_Need_Data['kcat_dict']
        mw_dict=Concretemodel_Need_Data['mw_dict']
        enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
        e0=Concretemodel_Need_Data['e0']
        totle_E=Concretemodel_Need_Data['totle_E']
        E_total=Concretemodel_Need_Data['E_total']   
        metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
        reaction_g0=Concretemodel_Need_Data['reaction_g0']
        B_value=Concretemodel_Need_Data['B_value']
        K_value=Concretemodel_Need_Data['K_value']        
        y1_protain=Concretemodel_Need_Data['y1_protain']
        listy = []
        if len(y1_protain) > 0:
            for i in y1_protain:
                for j in Concretemodel_Need_Data['enzyme_rxns_dict'][i]:
                    if j in Concretemodel_Need_Data['kcat_dict']:
                        listy.append(j) 
    if constr_coeff !=None:
        if 'substrate_constrain' in constr_coeff:
            set_substrate_ini=True
            substrate_value=constr_coeff['substrate_constrain'][1]
            substrate_name=constr_coeff['substrate_constrain'][0]
        if 'biomass_constrain' in constr_coeff:
            set_biomass_ini=True
            biomass_value=constr_coeff['biomass_constrain'][1]
            biomass_id=constr_coeff['biomass_constrain'][0]
        if 'product_constrain' in constr_coeff:
            set_product_ini=True
            product_value=constr_coeff['product_constrain'][1]
            product_name=constr_coeff['product_constrain'][0]
        if 'fix_reactions' in constr_coeff and constr_coeff['fix_reactions'] !={}:
            set_reactions_ini=True
        if 'fix_E_total' in constr_coeff:
            set_fix_sum_e=True
            set_max_sum_e=False
    Concretemodel.reaction = pyo.Var(reaction_list)
    #Concretemodel.reaction = pyo.Var(reaction_list,  within=NonNegativeReals) # reaction flux variable  
    # SV=0
    def set_stoi_matrix(m,i):
        return sum(coef_matrix[i,j]*m.reaction[j]  for j in reaction_list if (i,j) in coef_matrix.keys() )==0
    Concretemodel.set_stoi_matrix_c = Constraint( metabolite_list,rule=set_stoi_matrix)
    # LBãUB
    def set_bound(m,j):
        return inequality(lb_list[j],m.reaction[j],ub_list[j])
    Concretemodel.set_bound_c = Constraint(reaction_list,rule=set_bound)   
    
    
    if reaction_kcat_MW is not None:
    # check if kcat_MW is a column in reaction_kcat_MW
        if 'kcat_MW' in reaction_kcat_MW.columns:
            kcatmw='kcat_MW'
        elif 'kcat/mw' in reaction_kcat_MW.columns:
            kcatmw='kcat/mw'
            
    #           ****è®¾ç½®åé****    
    # ç­åå­¦åéè®¾ç½®    
    if set_Df_value:
        reaction_listDF=[j for j in reaction_g0.index if j in reaction_list]
        ConMet_list= list(set(metabolite_list).intersection(set(metabolites_lnC.index)))
        Concretemodel.metabolite = pyo.Var(ConMet_list,  within=Reals) # metabolite concentration variable
        Concretemodel.Df = pyo.Var(reaction_listDF,  within=Reals) # thermodynamic driving force variable--reactions
        Concretemodel.z = pyo.Var(reaction_listDF,  within=pyo.Binary)    # binary variable
        #Concretemodel.B = pyo.Var()     # thermodynamic driving force variable--mdfz
 
    # èç½ä¸ºä¸­å¿é¶åéè®¾ç½®
    if set_enzyme_value:
        Concretemodel.e = pyo.Var(e0, within=NonNegativeReals)
        totle_E=[x for x in totle_E if x in mw_dict ]
        Concretemodel.e1 = pyo.Var(totle_E, within=NonNegativeReals)    
        if len(listy)>0:
            Concretemodel.y = pyo.Var(listy,within=NonNegativeReals)
        if len(y1_protain)>0:
            Concretemodel.y1 = pyo.Var(y1_protain, within=Binary)         

    #           ****è®¾ç½®ç®æ å½æ°****
    
    # ä»¥B_valueä¸ºç®æ å½æ°
    if set_obj_B_value:             
        def set_obj_B_value(m):
            return m.B  
        Concretemodel.obj = Objective(rule=set_obj_B_value, sense=maximize)  
    # ä»¥æ»èç½éå åä¸ºç®æ å½æ°
    if set_obj_sum_e:             
        def set_obj_sum_e(m):
            return sum(Concretemodel.e1[i] * mw_dict[i] for i in totle_E if i in mw_dict)
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_sum_e, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_sum_e, sense=minimize)
    # ä»¥æå°åé¶éå åä¸ºç®æ å½æ°ï¼ä¸æ¯ä»¥èç½ä¸ºä¸­å¿ï¼   
    if set_obj_E_value:             
        def set_obj_E_value(m):
            return sum(m.reaction[j]/(reaction_kcat_MW.loc[j,kcatmw]) for j in reaction_kcat_MW.index if j in reaction_list)
        Concretemodel.obj = Objective(rule=set_obj_E_value, sense=minimize)      
    # ä»¥åä¸èç½ä¸ºç®æ å½æ°
    if set_obj_value_e:   
        def set_obj_value_e(m):
            return m.e1[obj_name]*mw_dict[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_value_e, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_value_e, sense=minimize)    
    # ä»¥åä¸èç½ä¸ºç®æ å½æ°ï¼ä¸æ¯ä»¥èç½ä¸ºä¸­å¿ï¼
    if set_obj_single_E_value:             
        def set_obj_single_E_value(m):
            return m.reaction[obj_name]/(reaction_kcat_MW.loc[obj_name,kcatmw])
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_single_E_value, sense=maximize) 
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_single_E_value, sense=minimize)
    # ä»¥æä¸ªååºä¸ºç®æ å½æ°
    if set_obj_value:   
        def set_obj_value(m):
            return m.reaction[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_value, sense=minimize)           
    #æå°åééåä¸ºç®æ å½æ° (pFBA)
    if set_obj_V_value:             
        def set_obj_V_value(m):
            return sum(m.reaction[j] for j in reaction_list)
        Concretemodel.obj = Objective(rule=set_obj_V_value, sense=minimize)  
    if set_obj_y_value:             
        def set_obj_y_value(m):
            return sum(m.y[j] for j in listy)
        Concretemodel.obj = Objective(rule=set_obj_y_value, sense=minimize)  
    #To calculate the variability of thermodynamic driving force (Dfi) of single reaction.
    if set_obj_TM_value:   
        def set_obj_TM_value(m):
            return m.Df[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_TM_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_TM_value, sense=minimize)
    #ä»¥ä»£è°¢ç©æµåº¦ä¸ºç®æ å½æ°ï¼ç­åå­¦ï¼
    if set_obj_Met_value:   
        def set_obj_Met_value(m):
            return m.metabolite[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_Met_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_Met_value, sense=minimize)
 
 
 
 
    #           ****æ·»å çº¦æ****
    
    #  ç­åå­¦çº¦æset_metaboliteãset_Dfãset_obj_B_valueï¼B_valueä¸ºåéï¼ãset_thermodynamicsï¼B_valueä¸ºå®å¼ï¼ãset_integer
    if set_metabolite:
        def set_metabolite(m,i):
            return  inequality(metabolites_lnC.loc[i,'lnClb'], m.metabolite[i], metabolites_lnC.loc[i,'lnCub'])
        Concretemodel.set_metabolite= Constraint(ConMet_list,rule=set_metabolite)        
    #thermodynamic driving force expression for reactions
    if set_Df:
        def set_Df(m,j):
            return  m.Df[j]==-reaction_g0.loc[j,'g0']-2.579*sum(coef_matrix[i,j]*m.metabolite[i]  for i in ConMet_list if (i,j) in coef_matrix.keys())
        Concretemodel.set_Df = Constraint(reaction_listDF,rule=set_Df)
    #Adding thermodynamic MDF(B) object function
    if set_thermodynamics0:
        def set_thermodynamics0(m,j):
            return m.B<=(m.Df[j]+(1-m.z[j])*K_value)
        Concretemodel.set_obj_B_value = Constraint(reaction_listDF, rule=set_thermodynamics0)
    #Adding thermodynamic constraints
    if set_thermodynamics:
        def set_thermodynamics(m,j):
            return (m.Df[j]+(1-m.z[j])*K_value)>= B_value
        Concretemodel.set_thermodynamics = Constraint(reaction_listDF, rule=set_thermodynamics)
    #Adding binary variables constraints
    if set_integer:
        def set_integer(m,j):
            return m.reaction[j]<=m.z[j]*ub_list[j] 
        Concretemodel.set_integer = Constraint(reaction_listDF,rule=set_integer)     
        
        
    # é¶çº¦æ
    if set_enzyme_constraint:
        # flux- enzyme relationship
        Concretemodel.set_constr15 = ConstraintList()  
        if len(y1_protain)>0:
            for i in y1_protain:   
                for j in enzyme_rxns_dict[i]:
                    if j in e0 and j in kcat_dict:
                        Concretemodel.set_constr15.add(  Concretemodel.y[j] <= Concretemodel.y1[i]* M[j][1] ) 
                        Concretemodel.set_constr15.add(  Concretemodel.y[j] >= Concretemodel.y1[i]* M[j][0]) 
                        
        Concretemodel.set_constr2 = ConstraintList()
        for i in e0:
            if len(listy)>0:
                if i in listy:
                    Concretemodel.set_constr2.add(Concretemodel.e[i] * (kcat_dict[i] + Concretemodel.y[i]) >= Concretemodel.reaction[i])
                else:
                    Concretemodel.set_constr2.add(Concretemodel.e[i] *kcat_dict[i]  >= Concretemodel.reaction[i])
            else:
                Concretemodel.set_constr2.add(Concretemodel.e[i] *kcat_dict[i]  >= Concretemodel.reaction[i])
        # protein  concentration
        Concretemodel.set_constr3 = ConstraintList()  
        for i in totle_E:   
            Concretemodel.set_constr3.add(Concretemodel.e1[i] == sum(Concretemodel.e[j] for j in enzyme_rxns_dict[i] if j in e0))
        # if set_obj_y_value != False:
        #     Concretemodel.set_constr16 = ConstraintList()
        #     for i in y1_dict: 
        #         Concretemodel.set_constr16.add(  Concretemodel.y1[i] == y1_dict[i] ) 
        # total protein concentration
    if set_max_sum_e:
        if len(y1_protain) >0:
            Concretemodel.set_y = Constraint(expr=sum(Concretemodel.y1[i] for i in y1_protain if i in mw_dict) <=num_e1 )
        Concretemodel.set_max_sum_e = Constraint(expr= sum(Concretemodel.e1[i] * mw_dict[i] for i in totle_E if i  in mw_dict)<= E_total)
    if set_fix_sum_e:
        Concretemodel.set_max_sum_e = Constraint(expr=sum(Concretemodel.e1[i] * mw_dict[i] for i in totle_E if i in mw_dict) == E_total) 
        
    # æå¤§åºç©æå¥çº¦æ
    if set_substrate_ini:
        def set_substrate_ini(m): 
            return m.reaction[substrate_name] <= substrate_value
        Concretemodel.set_substrate_ini = Constraint(rule=set_substrate_ini)       
    # åºå®biomassèå´
    if set_biomass_ini:
        def set_biomass_ini(m): 
            return m.reaction[biomass_id] >=biomass_value
        Concretemodel.set_biomass_ini = Constraint(rule=set_biomass_ini)
    # åºå®productèå´
    if set_product_ini:
        Concretemodel.set_product_ini = Constraint(expr=Concretemodel.reaction[product_name] >= product_value)
    # æ¹éåºå®ååºéé
    if set_reactions_ini:
        Concretemodel.set_reactions_ini = ConstraintList()
        for i in constr_coeff['fix_reactions']:
            Concretemodel.set_reactions_ini.add(Concretemodel.reaction[i] >= constr_coeff['fix_reactions'][i][0])
            Concretemodel.set_reactions_ini.add(Concretemodel.reaction[i] <= constr_coeff['fix_reactions'][i][1])

    #Adding concentration ratio constraints for metabolites
    if set_metabolite_ratio:
        def set_atp_adp(m):
            return m.metabolite['atp_c']-m.metabolite['adp_c']==np.log(10)
        def set_adp_amp(m):
            return m.metabolite['adp_c']-m.metabolite['amp_c']==np.log(1)
        def set_nad_nadh(m):
            return m.metabolite['nad_c']-m.metabolite['nadh_c']==np.log(10)
        def set_nadph_nadp(m):
            return m.metabolite['nadph_c']-m.metabolite['nadp_c']==np.log(10)
        def set_hco3_co2(m):
            return m.metabolite['hco3_c']-m.metabolite['co2_c']==np.log(2)

        Concretemodel.set_atp_adp = Constraint(rule=set_atp_adp) 
        Concretemodel.set_adp_amp = Constraint(rule=set_adp_amp) 
        Concretemodel.set_nad_nadh = Constraint(rule=set_nad_nadh) 
        Concretemodel.set_nadph_nadp = Constraint(rule=set_nadph_nadp) 
        Concretemodel.set_hco3_co2 = Constraint(rule=set_hco3_co2)

    #Adding Bottleneck reaction constraints
    if set_Bottleneck_reaction:
        def set_Bottleneck_reaction(m,j):
            return m.z[j]==1 
        Concretemodel.set_Bottleneck_reaction = Constraint(Bottleneck_reaction_list,rule=set_Bottleneck_reaction) 
    Concretemodel.write('/home/sun/ETGEMS-10.20/ETM/lp/EcoECM_protainmodel.lp',io_options={'symbolic_solver_labels': True})  
    return Concretemodel

def FBA_template5(coef_matrix=None,metabolites_lnC=None,reaction_g0=None,lb_list=None,ub_list=None,reaction_kcat_MW=None,B_value=None,kcat_dict=None,mw_dict=None,K_value=None,\
            product_value=None,biomass_value=None,substrate_value=None,E_total=None,\
            product_name=None,reaction_list=None,metabolite_list=None,enzyme_rxns_dict=None,e0=None,totle_E=None,obj_name=None,obj_target=None,\
            biomass_id=None,substrate_name=None,Bottleneck_reaction_list=None,\
            set_obj_B_value=False,set_obj_sum_e=False,set_obj_value_e=False,set_obj_E_value=False,set_obj_single_E_value=False,\
            set_obj_value=False,set_obj_V_value=False,set_obj_TM_value=False,set_obj_Met_value=False,\
            set_metabolite=False,set_Df=False,set_thermodynamics0=False,set_thermodynamics=False,set_integer=False,set_fix_sum_e=False,\
            set_enzyme_constraint=False,set_substrate_ini=False, set_biomass_ini=False,set_product_ini=False,set_reactions_ini=False,\
            set_metabolite_ratio=False,set_Bottleneck_reaction=False,set_Df_value=False,set_enzyme_value=False,set_max_sum_e=False,\
            mode=None,constr_coeff=None,Concretemodel_Need_Data=None,num_e1=None,set_abs_value=False):
    
    """According to the parameter conditions provided by the user, the specific pyomo model is returned.

    Notes
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC:Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...

    * reaction_kcat_MW: The enzymatic data of kcat (turnover number, /h) divided by MW (molecular weight, kDa).
    The format is as follows (in .csv file):
    kcat,MW,kcat_MW
    AADDGT	231887.7737	40.6396	5705.956104
    AAMYL	23490.41652	56.63940007	414.736323
    ...
    
    * lb_list: Lower bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * ub_list: Upper bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * obj_name: Name of object, such as set_obj_value, set_obj_single_E_value, set_obj_TM_value and set_obj_Met_value.    
    * K_value: the maximum value minus the minimum value of reaction thermodynamic driving force (1249, "max(DFi,max)-min(DFi,min)").
    * obj_target: Type of object function (maximize or minimize).
    * set_obj_value: Set the flux as the object function (True or False).
    * set_substrate_ini: Adding substrate amount constraints (True or False).
    * substrate_name: Substrate input reaction ID in the model (such as EX_glc__D_e_reverse).
    * substrate_value: Set the upper bound for substrate input reaction flux (10 mmol/h/gDW).
    * set_biomass_ini: Set the lower bound for biomass synthesis reaction flux (True or False).
    * biomass_value: The lower bound of biomass synthesis reaction flux.
    * biomass_id: Biomass synthesis reaction ID in the model (BIOMASS_Ec_iML1515_core_75p37M).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    * set_obj_B_value: The object function is the maximizing thermodynamic driving force of a pathway (True or False)
    * set_stoi_matrix: Adding flux balance constraints (True or False).
    * set_bound: Set the upper and lower bounds of reaction flux (True or False).
    * set_enzyme_constraint: Adding enzymamic constraints (True or False).
    * set_integer: Adding binary variables constraints (True or False).
    * set_metabolite_ratio: Adding concentration ratio constraints for metabolites (True or False).
    * set_thermodynamics: Adding thermodynamic constraints (True or False).
    * B_value: The value of maximizing the minimum thermodynamic driving force (MDF).
    * set_obj_E_value: The object function is the minimum enzyme cost of a pathway (True or False).
    * set_obj_V_value: The object function is the pFBA of a pathway (True or False)
    * set_obj_TM_value: The object function is the thermodynamic driving force of a reaction (True or False).
    * set_obj_Met_value: The object function is the concentration of a metabolite (True or False).
    * set_obj_single_E_value: The object function is the enzyme cost of a reaction (True or False).     
    * E_total: Total amount constraint of effective enzymes pool (0.13).
    * Bottleneck_reaction_list: A list extracted from the result file automatically.
    * set_Bottleneck_reaction: Adding integer variable constraints for specific reaction (True or False).
    """
    Concretemodel = ConcreteModel()
    
    
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    if   mode=='ST':
        if set_obj_B_value:
            set_thermodynamics0=True
        if not set_obj_B_value:
            set_thermodynamics=True
        set_Df_value=True
        set_metabolite=True
        set_Df=True
        set_integer=True
        set_metabolite_ratio=True
        
        metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
        reaction_g0=Concretemodel_Need_Data['reaction_g0']
        B_value=Concretemodel_Need_Data['B_value']
        K_value=Concretemodel_Need_Data['K_value']
        
        
    if  mode=='SE': 
        set_enzyme_constraint=True
        set_enzyme_value=True
        if set_obj_sum_e or set_fix_sum_e:
            set_max_sum_e=False
        if not set_obj_sum_e:
            set_max_sum_e=True
            
        kcat_dict=Concretemodel_Need_Data['kcat_dict']
        mw_dict=Concretemodel_Need_Data['mw_dict']
        enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
        e0=Concretemodel_Need_Data['e0']
        totle_E=Concretemodel_Need_Data['totle_E']
        E_total=Concretemodel_Need_Data['E_total']
        y1_protain=Concretemodel_Need_Data['y1_protain']
        listy=[]
        for i in y1_protain:
            for j in enzyme_rxns_dict[i]:
                listy.append(j)
        C13_dict=Concretemodel_Need_Data['C13_flux_dict']
        C13_keys=list(C13_dict.keys())
        C13_reaction=Concretemodel_Need_Data['C13_reaction_dict']
    if  mode=='SET':    
        if set_obj_B_value:
            set_thermodynamics0=True
        if not set_obj_B_value:
            set_thermodynamics=True
        if set_obj_sum_e or set_fix_sum_e:
            set_max_sum_e=False
        if not set_obj_sum_e:
            set_max_sum_e=True
            
        set_Df_value=True
        set_enzyme_value=True
        set_metabolite=True
        set_Df=True
        set_metabolite_ratio=True
        set_integer=True 
        set_enzyme_constraint=True    
        kcat_dict=Concretemodel_Need_Data['kcat_dict']
        mw_dict=Concretemodel_Need_Data['mw_dict']
        enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
        e0=Concretemodel_Need_Data['e0']
        totle_E=Concretemodel_Need_Data['totle_E']
        E_total=Concretemodel_Need_Data['E_total']   
        metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
        reaction_g0=Concretemodel_Need_Data['reaction_g0']
        B_value=Concretemodel_Need_Data['B_value']
        K_value=Concretemodel_Need_Data['K_value']   
        y1_protain=Concretemodel_Need_Data['y1_protain']
        listy=[]
        for i in y1_protain:
            for j in enzyme_rxns_dict[i]:
                listy.append(j)     
        C13_dict=Concretemodel_Need_Data['C13_flux_dict']
        C13_keys=list(C13_dict.keys())
        C13_reaction=Concretemodel_Need_Data['C13_reaction_dict']
                    
    if constr_coeff !=None:
        if 'substrate_constrain' in constr_coeff:
            set_substrate_ini=True
            substrate_value=constr_coeff['substrate_constrain'][1]
            substrate_name=constr_coeff['substrate_constrain'][0]
        if 'biomass_constrain' in constr_coeff:
            set_biomass_ini=True
            biomass_value=constr_coeff['biomass_constrain'][1]
            biomass_id=constr_coeff['biomass_constrain'][0]
        if 'product_constrain' in constr_coeff:
            set_product_ini=True
            product_value=constr_coeff['product_constrain'][1]
            product_name=constr_coeff['product_constrain'][0]
        if 'fix_reactions' in constr_coeff and constr_coeff['fix_reactions'] !={}:
            set_reactions_ini=True
        if 'fix_E_total' in constr_coeff:
            set_fix_sum_e=True
            set_max_sum_e=False
    Concretemodel.reaction = pyo.Var(reaction_list)
    #Concretemodel.reaction = pyo.Var(reaction_list,  within=NonNegativeReals) # reaction flux variable  
    # SV=0
    def set_stoi_matrix(m,i):
        return sum(coef_matrix[i,j]*m.reaction[j]  for j in reaction_list if (i,j) in coef_matrix.keys() )==0
    Concretemodel.set_stoi_matrix_c = Constraint( metabolite_list,rule=set_stoi_matrix)
    # LB、UB
    def set_bound(m,j):
        return inequality(lb_list[j],m.reaction[j],ub_list[j])
    Concretemodel.set_bound_c = Constraint(reaction_list,rule=set_bound)   
    
    
    if reaction_kcat_MW is not None:
    # check if kcat_MW is a column in reaction_kcat_MW
        if 'kcat_MW' in reaction_kcat_MW.columns:
            kcatmw='kcat_MW'
        elif 'kcat/mw' in reaction_kcat_MW.columns:
            kcatmw='kcat/mw'
            
    #           ****设置变量****    
    # 热力学变量设置    
    if set_Df_value:
        reaction_listDF=[j for j in reaction_g0.index if j in reaction_list]
        ConMet_list= list(set(metabolite_list).intersection(set(metabolites_lnC.index)))
        Concretemodel.metabolite = pyo.Var(ConMet_list,  within=Reals) # metabolite concentration variable
        Concretemodel.Df = pyo.Var(reaction_listDF,  within=Reals) # thermodynamic driving force variable--reactions
        Concretemodel.z = pyo.Var(reaction_listDF,  within=pyo.Binary)    # binary variable
        #Concretemodel.B = pyo.Var()     # thermodynamic driving force variable--mdfz
 
    # 蛋白为中心酶变量设置
    if set_enzyme_value:
        Concretemodel.e = pyo.Var(e0, within=NonNegativeReals)
        totle_E=[x for x in totle_E if x in mw_dict ]
        Concretemodel.e1 = pyo.Var(totle_E, within=NonNegativeReals)    
        Concretemodel.y = pyo.Var(listy, within=Binary)
        Concretemodel.y1 = pyo.Var(y1_protain, within=Binary)    
        Concretemodel.abs=pyo.Var(C13_keys, within=NonNegativeReals)    

    #           ****设置目标函数****
    
    # 以B_value为目标函数
    if set_obj_B_value:             
        def set_obj_B_value(m):
            return m.B  
        Concretemodel.obj = Objective(rule=set_obj_B_value, sense=maximize)  
    # 以总蛋白量加和为目标函数
    if set_obj_sum_e:             
        def set_obj_sum_e(m):
            return sum(Concretemodel.e1[i] * mw_dict[i] for i in totle_E if i in mw_dict)
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_sum_e, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_sum_e, sense=minimize)
    # 以最小化酶量加和为目标函数（不是以蛋白为中心）   
    if set_obj_E_value:             
        def set_obj_E_value(m):
            return sum(m.reaction[j]/(reaction_kcat_MW.loc[j,kcatmw]) for j in reaction_kcat_MW.index if j in reaction_list)
        Concretemodel.obj = Objective(rule=set_obj_E_value, sense=minimize)      
    # 以单一蛋白为目标函数
    if set_obj_value_e:   
        def set_obj_value_e(m):
            return m.e1[obj_name]*mw_dict[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_value_e, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_value_e, sense=minimize)    
    # 以单一蛋白为目标函数（不是以蛋白为中心）
    if set_obj_single_E_value:             
        def set_obj_single_E_value(m):
            return m.reaction[obj_name]/(reaction_kcat_MW.loc[obj_name,kcatmw])
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_single_E_value, sense=maximize) 
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_single_E_value, sense=minimize)
    # 以某个反应为目标函数
    if set_obj_value:   
        def set_obj_value(m):
            return m.reaction[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_value, sense=minimize)           
    #最小化通量和为目标函数 (pFBA)
    if set_obj_V_value:             
        def set_obj_V_value(m):
            return sum(m.reaction[j] for j in reaction_list)
        Concretemodel.obj = Objective(rule=set_obj_V_value, sense=minimize)  

    #最小化abs和为目标函数 (pFBA)
    if set_abs_value:             
        def set_abs_value(m):
            return sum(m.abs[j] for j in C13_keys)
        Concretemodel.obj = Objective(rule=set_abs_value, sense=minimize)  

    #To calculate the variability of thermodynamic driving force (Dfi) of single reaction.
    if set_obj_TM_value:   
        def set_obj_TM_value(m):
            return m.Df[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_TM_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_TM_value, sense=minimize)
    #以代谢物浓度为目标函数（热力学）
    if set_obj_Met_value:   
        def set_obj_Met_value(m):
            return m.metabolite[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_Met_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_Met_value, sense=minimize)
 
 
 
 
    #           ****添加约束****
    
    #  热力学约束set_metabolite、set_Df、set_obj_B_value（B_value为变量）、set_thermodynamics（B_value为定值）、set_integer
    if set_metabolite:
        def set_metabolite(m,i):
            return  inequality(metabolites_lnC.loc[i,'lnClb'], m.metabolite[i], metabolites_lnC.loc[i,'lnCub'])
        Concretemodel.set_metabolite= Constraint(ConMet_list,rule=set_metabolite)        
    #thermodynamic driving force expression for reactions
    if set_Df:
        def set_Df(m,j):
            return  m.Df[j]==-reaction_g0.loc[j,'g0']-2.579*sum(coef_matrix[i,j]*m.metabolite[i]  for i in ConMet_list if (i,j) in coef_matrix.keys())
        Concretemodel.set_Df = Constraint(reaction_listDF,rule=set_Df)
    #Adding thermodynamic MDF(B) object function
    if set_thermodynamics0:
        def set_thermodynamics0(m,j):
            return m.B<=(m.Df[j]+(1-m.z[j])*K_value)
        Concretemodel.set_obj_B_value = Constraint(reaction_listDF, rule=set_thermodynamics0)
    #Adding thermodynamic constraints
    if set_thermodynamics:
        def set_thermodynamics(m,j):
            return (m.Df[j]+(1-m.z[j])*K_value)>= B_value
        Concretemodel.set_thermodynamics = Constraint(reaction_listDF, rule=set_thermodynamics)
    #Adding binary variables constraints
    if set_integer:
        def set_integer(m,j):
            return m.reaction[j]<=m.z[j]*ub_list[j] 
        Concretemodel.set_integer = Constraint(reaction_listDF,rule=set_integer)     
        
        
    # 酶约束
    if set_enzyme_constraint:
        # flux- enzyme relationship
        Concretemodel.set_constr15 = ConstraintList()  
        for i in y1_protain:   
            for j in enzyme_rxns_dict[i]:
                if j in e0:
                    Concretemodel.set_constr15.add(Concretemodel.y1[i] == Concretemodel.y[j] )
            
                    
        Concretemodel.set_constr2 = ConstraintList()
        for i in e0:
            if i in listy:
                Concretemodel.set_constr2.add(Concretemodel.e[i] *kcat_dict[i] + 1000 * Concretemodel.y[i] >= Concretemodel.reaction[i])
            else:
                Concretemodel.set_constr2.add(Concretemodel.e[i] *kcat_dict[i]  >= Concretemodel.reaction[i])
        # protein  concentration
        Concretemodel.set_constr3 = ConstraintList()  
        for i in totle_E:   
            Concretemodel.set_constr3.add(Concretemodel.e1[i] == sum(Concretemodel.e[j] for j in enzyme_rxns_dict[i] if j in e0))

        Concretemodel.set_c13 = ConstraintList()  
        for i in C13_dict:
            Concretemodel.set_c13.add( Concretemodel.abs[i] >=   sum( -Concretemodel.reaction[j] if '_reverse' in  j else  Concretemodel.reaction[j]  for j in C13_reaction[i])   - C13_dict[i]    )
            Concretemodel.set_c13.add( Concretemodel.abs[i] >=   C13_dict[i] -  sum(-Concretemodel.reaction[j] if '_reverse' in  j else  Concretemodel.reaction[j]  for j in C13_reaction[i]))

        Concretemodel.set_biomass = Constraint(expr=Concretemodel.reaction[biomass_id] >= biomass_value)

        # total protein concentration
    if set_max_sum_e:
        # Concretemodel.set_protainY =ConstraintList()
        # for i in ['b3731 and b3732 and b3733 and b3734 and b3735 and b3736 and b3737 and b3738','b3731 and b3732 and b3733 and b3734 and b3735 and b3736 and b3737 and b3738 and b3739']:
        #     Concretemodel.set_protainY.add(Concretemodel.y[i]==0)
        Concretemodel.set_y = Constraint(expr=sum(Concretemodel.y1[i] for i in y1_protain if i in mw_dict) <=num_e1 )
        Concretemodel.set_max_sum_e = Constraint(expr=sum(Concretemodel.e1[i] * mw_dict[i] * (1-Concretemodel.y1[i] ) for i in totle_E if i in y1_protain) + sum(Concretemodel.e1[i] * mw_dict[i] for i in totle_E if i  not in y1_protain)<= E_total)
    if set_fix_sum_e:
        Concretemodel.set_max_sum_e = Constraint(expr=sum(Concretemodel.e1[i] * mw_dict[i] for i in totle_E if i in mw_dict) == E_total) 
        
    # 最大底物摄入约束
    if set_substrate_ini:
        def set_substrate_ini(m): 
            return m.reaction[substrate_name] <= substrate_value
        Concretemodel.set_substrate_ini = Constraint(rule=set_substrate_ini)       
    # 固定biomass范围
    if set_biomass_ini:
        def set_biomass_ini(m): 
            return m.reaction[biomass_id] >=biomass_value
        Concretemodel.set_biomass_ini = Constraint(rule=set_biomass_ini)
    # 固定product范围
    if set_product_ini:
        Concretemodel.set_product_ini = Constraint(expr=Concretemodel.reaction[product_name] >= product_value)
    # 批量固定反应通量
    if set_reactions_ini:
        Concretemodel.set_reactions_ini = ConstraintList()
        for i in constr_coeff['fix_reactions']:
            Concretemodel.set_reactions_ini.add(Concretemodel.reaction[i] == constr_coeff['fix_reactions'][i])

    #Adding concentration ratio constraints for metabolites
    if set_metabolite_ratio:
        def set_atp_adp(m):
            return m.metabolite['atp_c']-m.metabolite['adp_c']==np.log(10)
        def set_adp_amp(m):
            return m.metabolite['adp_c']-m.metabolite['amp_c']==np.log(1)
        def set_nad_nadh(m):
            return m.metabolite['nad_c']-m.metabolite['nadh_c']==np.log(10)
        def set_nadph_nadp(m):
            return m.metabolite['nadph_c']-m.metabolite['nadp_c']==np.log(10)
        def set_hco3_co2(m):
            return m.metabolite['hco3_c']-m.metabolite['co2_c']==np.log(2)

        Concretemodel.set_atp_adp = Constraint(rule=set_atp_adp) 
        Concretemodel.set_adp_amp = Constraint(rule=set_adp_amp) 
        Concretemodel.set_nad_nadh = Constraint(rule=set_nad_nadh) 
        Concretemodel.set_nadph_nadp = Constraint(rule=set_nadph_nadp) 
        Concretemodel.set_hco3_co2 = Constraint(rule=set_hco3_co2)

    #Adding Bottleneck reaction constraints
    if set_Bottleneck_reaction:
        def set_Bottleneck_reaction(m,j):
            return m.z[j]==1 
        Concretemodel.set_Bottleneck_reaction = Constraint(Bottleneck_reaction_list,rule=set_Bottleneck_reaction) 

    return Concretemodel

def FBA_template6(coef_matrix=None,metabolites_lnC=None,reaction_g0=None,lb_list=None,ub_list=None,reaction_kcat_MW=None,B_value=None,kcat_dict=None,mw_dict=None,K_value=None,\
            product_value=None,biomass_value=None,substrate_value=None,E_total=None,\
            product_name=None,reaction_list=None,metabolite_list=None,enzyme_rxns_dict=None,e0=None,totle_E=None,obj_name=None,obj_target=None,\
            biomass_id=None,substrate_name=None,Bottleneck_reaction_list=None,\
            set_obj_B_value=False,set_obj_sum_e=False,set_obj_value_e=False,set_obj_E_value=False,set_obj_single_E_value=False,\
            set_obj_value=False,set_obj_V_value=False,set_obj_TM_value=False,set_obj_Met_value=False,\
            set_metabolite=False,set_Df=False,set_thermodynamics0=False,set_thermodynamics=False,set_integer=False,set_fix_sum_e=False,\
            set_enzyme_constraint=False,set_substrate_ini=False, set_biomass_ini=False,set_product_ini=False,set_reactions_ini=False,\
            set_metabolite_ratio=False,set_Bottleneck_reaction=False,set_Df_value=False,set_enzyme_value=False,set_max_sum_e=False,\
            mode=None,constr_coeff=None,Concretemodel_Need_Data=None,num_e1=None,set_abs_value=False,M=None):
    
    """According to the parameter conditions provided by the user, the specific pyomo model is returned.

    Notes
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC:Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...

    * reaction_kcat_MW: The enzymatic data of kcat (turnover number, /h) divided by MW (molecular weight, kDa).
    The format is as follows (in .csv file):
    kcat,MW,kcat_MW
    AADDGT	231887.7737	40.6396	5705.956104
    AAMYL	23490.41652	56.63940007	414.736323
    ...
    
    * lb_list: Lower bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * ub_list: Upper bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * obj_name: Name of object, such as set_obj_value, set_obj_single_E_value, set_obj_TM_value and set_obj_Met_value.    
    * K_value: the maximum value minus the minimum value of reaction thermodynamic driving force (1249, "max(DFi,max)-min(DFi,min)").
    * obj_target: Type of object function (maximize or minimize).
    * set_obj_value: Set the flux as the object function (True or False).
    * set_substrate_ini: Adding substrate amount constraints (True or False).
    * substrate_name: Substrate input reaction ID in the model (such as EX_glc__D_e_reverse).
    * substrate_value: Set the upper bound for substrate input reaction flux (10 mmol/h/gDW).
    * set_biomass_ini: Set the lower bound for biomass synthesis reaction flux (True or False).
    * biomass_value: The lower bound of biomass synthesis reaction flux.
    * biomass_id: Biomass synthesis reaction ID in the model (BIOMASS_Ec_iML1515_core_75p37M).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    * set_obj_B_value: The object function is the maximizing thermodynamic driving force of a pathway (True or False)
    * set_stoi_matrix: Adding flux balance constraints (True or False).
    * set_bound: Set the upper and lower bounds of reaction flux (True or False).
    * set_enzyme_constraint: Adding enzymamic constraints (True or False).
    * set_integer: Adding binary variables constraints (True or False).
    * set_metabolite_ratio: Adding concentration ratio constraints for metabolites (True or False).
    * set_thermodynamics: Adding thermodynamic constraints (True or False).
    * B_value: The value of maximizing the minimum thermodynamic driving force (MDF).
    * set_obj_E_value: The object function is the minimum enzyme cost of a pathway (True or False).
    * set_obj_V_value: The object function is the pFBA of a pathway (True or False)
    * set_obj_TM_value: The object function is the thermodynamic driving force of a reaction (True or False).
    * set_obj_Met_value: The object function is the concentration of a metabolite (True or False).
    * set_obj_single_E_value: The object function is the enzyme cost of a reaction (True or False).     
    * E_total: Total amount constraint of effective enzymes pool (0.13).
    * Bottleneck_reaction_list: A list extracted from the result file automatically.
    * set_Bottleneck_reaction: Adding integer variable constraints for specific reaction (True or False).
    """
    Concretemodel = ConcreteModel()
    
    
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    if   mode=='ST':
        if set_obj_B_value:
            set_thermodynamics0=True
        if not set_obj_B_value:
            set_thermodynamics=True
        set_Df_value=True
        set_metabolite=True
        set_Df=True
        set_integer=True
        set_metabolite_ratio=True
        
        metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
        reaction_g0=Concretemodel_Need_Data['reaction_g0']
        B_value=Concretemodel_Need_Data['B_value']
        K_value=Concretemodel_Need_Data['K_value']
        
        
    if  mode=='SE': 
        set_enzyme_constraint=True
        set_enzyme_value=True
        if set_obj_sum_e or set_fix_sum_e:
            set_max_sum_e=False
        if not set_obj_sum_e:
            set_max_sum_e=True
            
        kcat_dict=Concretemodel_Need_Data['kcat_dict']
        mw_dict=Concretemodel_Need_Data['mw_dict']
        enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
        e0=Concretemodel_Need_Data['e0']
        totle_E=Concretemodel_Need_Data['totle_E']
        E_total=Concretemodel_Need_Data['E_total']
        y1_protain=Concretemodel_Need_Data['y1_protain']
        listy = []
        if len(y1_protain) > 0:
            for i in y1_protain:
                for j in Concretemodel_Need_Data['enzyme_rxns_dict'][i]:
                    if j in Concretemodel_Need_Data['kcat_dict']:
                        listy.append(j)
        C13_dict=Concretemodel_Need_Data['C13_flux_dict']
        C13_keys=list(C13_dict.keys())
        C13_reaction=Concretemodel_Need_Data['C13_reaction_dict']
    if  mode=='SET':    
        if set_obj_B_value:
            set_thermodynamics0=True
        if not set_obj_B_value:
            set_thermodynamics=True
        if set_obj_sum_e or set_fix_sum_e:
            set_max_sum_e=False
        if not set_obj_sum_e:
            set_max_sum_e=True
            
        set_Df_value=True
        set_enzyme_value=True
        set_metabolite=True
        set_Df=True
        set_metabolite_ratio=True
        set_integer=True 
        set_enzyme_constraint=True    
        kcat_dict=Concretemodel_Need_Data['kcat_dict']
        mw_dict=Concretemodel_Need_Data['mw_dict']
        enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
        e0=Concretemodel_Need_Data['e0']
        totle_E=Concretemodel_Need_Data['totle_E']
        E_total=Concretemodel_Need_Data['E_total']   
        metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
        reaction_g0=Concretemodel_Need_Data['reaction_g0']
        B_value=Concretemodel_Need_Data['B_value']
        K_value=Concretemodel_Need_Data['K_value']   
        y1_protain=Concretemodel_Need_Data['y1_protain']
        listy = []
        if len(y1_protain) > 0:
            for i in y1_protain:
                for j in Concretemodel_Need_Data['enzyme_rxns_dict'][i]:
                    if j in Concretemodel_Need_Data['kcat_dict']:
                        listy.append(j)    
        C13_dict=Concretemodel_Need_Data['C13_flux_dict']
        C13_keys=list(C13_dict.keys())
        C13_reaction=Concretemodel_Need_Data['C13_reaction_dict']
                    
    if constr_coeff !=None:
        if 'substrate_constrain' in constr_coeff:
            set_substrate_ini=True
            substrate_value=constr_coeff['substrate_constrain'][1]
            substrate_name=constr_coeff['substrate_constrain'][0]
        if 'biomass_constrain' in constr_coeff:
            set_biomass_ini=True
            biomass_value=constr_coeff['biomass_constrain'][1]
            biomass_id=constr_coeff['biomass_constrain'][0]
        if 'product_constrain' in constr_coeff:
            set_product_ini=True
            product_value=constr_coeff['product_constrain'][1]
            product_name=constr_coeff['product_constrain'][0]
        if 'fix_reactions' in constr_coeff and constr_coeff['fix_reactions'] !={}:
            set_reactions_ini=True
        if 'fix_E_total' in constr_coeff:
            set_fix_sum_e=True
            set_max_sum_e=False
    Concretemodel.reaction = pyo.Var(reaction_list)
    #Concretemodel.reaction = pyo.Var(reaction_list,  within=NonNegativeReals) # reaction flux variable  
    # SV=0
    def set_stoi_matrix(m,i):
        return sum(coef_matrix[i,j]*m.reaction[j]  for j in reaction_list if (i,j) in coef_matrix.keys() )==0
    Concretemodel.set_stoi_matrix_c = Constraint( metabolite_list,rule=set_stoi_matrix)
    # LB、UB
    def set_bound(m,j):
        return inequality(lb_list[j],m.reaction[j],ub_list[j])
    Concretemodel.set_bound_c = Constraint(reaction_list,rule=set_bound)   
    
    
    if reaction_kcat_MW is not None:
    # check if kcat_MW is a column in reaction_kcat_MW
        if 'kcat_MW' in reaction_kcat_MW.columns:
            kcatmw='kcat_MW'
        elif 'kcat/mw' in reaction_kcat_MW.columns:
            kcatmw='kcat/mw'
            
    #           ****设置变量****    
    # 热力学变量设置    
    if set_Df_value:
        reaction_listDF=[j for j in reaction_g0.index if j in reaction_list]
        ConMet_list= list(set(metabolite_list).intersection(set(metabolites_lnC.index)))
        Concretemodel.metabolite = pyo.Var(ConMet_list,  within=Reals) # metabolite concentration variable
        Concretemodel.Df = pyo.Var(reaction_listDF,  within=Reals) # thermodynamic driving force variable--reactions
        Concretemodel.z = pyo.Var(reaction_listDF,  within=pyo.Binary)    # binary variable
        #Concretemodel.B = pyo.Var()     # thermodynamic driving force variable--mdfz
 
    # 蛋白为中心酶变量设置
    if set_enzyme_value:
        Concretemodel.e = pyo.Var(e0, within=NonNegativeReals)
        totle_E=[x for x in totle_E if x in mw_dict ]
        Concretemodel.e1 = pyo.Var(totle_E, within=NonNegativeReals)    
        if len(listy)>0:
            Concretemodel.y = pyo.Var(listy,within=NonNegativeReals)
        if len(y1_protain)>0:
            Concretemodel.y1 = pyo.Var(y1_protain, within=Binary)    
        Concretemodel.abs=pyo.Var(C13_keys, within=NonNegativeReals)    

    #           ****设置目标函数****
    
    # 以B_value为目标函数
    if set_obj_B_value:             
        def set_obj_B_value(m):
            return m.B  
        Concretemodel.obj = Objective(rule=set_obj_B_value, sense=maximize)  
    # 以总蛋白量加和为目标函数
    if set_obj_sum_e:             
        def set_obj_sum_e(m):
            return sum(Concretemodel.e1[i] * mw_dict[i] for i in totle_E if i in mw_dict)
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_sum_e, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_sum_e, sense=minimize)
    # 以最小化酶量加和为目标函数（不是以蛋白为中心）   
    if set_obj_E_value:             
        def set_obj_E_value(m):
            return sum(m.reaction[j]/(reaction_kcat_MW.loc[j,kcatmw]) for j in reaction_kcat_MW.index if j in reaction_list)
        Concretemodel.obj = Objective(rule=set_obj_E_value, sense=minimize)      
    # 以单一蛋白为目标函数
    if set_obj_value_e:   
        def set_obj_value_e(m):
            return m.e1[obj_name]*mw_dict[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_value_e, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_value_e, sense=minimize)    
    # 以单一蛋白为目标函数（不是以蛋白为中心）
    if set_obj_single_E_value:             
        def set_obj_single_E_value(m):
            return m.reaction[obj_name]/(reaction_kcat_MW.loc[obj_name,kcatmw])
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_single_E_value, sense=maximize) 
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_single_E_value, sense=minimize)
    # 以某个反应为目标函数
    if set_obj_value:   
        def set_obj_value(m):
            return m.reaction[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_value, sense=minimize)           
    #最小化通量和为目标函数 (pFBA)
    if set_obj_V_value:             
        def set_obj_V_value(m):
            return sum(m.reaction[j] for j in reaction_list)
        Concretemodel.obj = Objective(rule=set_obj_V_value, sense=minimize)  

    #最小化abs和为目标函数 (pFBA)
    if set_abs_value:             
        def set_abs_value(m):
            return sum(m.abs[j] for j in C13_keys)
        Concretemodel.obj = Objective(rule=set_abs_value, sense=minimize)  

    #To calculate the variability of thermodynamic driving force (Dfi) of single reaction.
    if set_obj_TM_value:   
        def set_obj_TM_value(m):
            return m.Df[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_TM_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_TM_value, sense=minimize)
    #以代谢物浓度为目标函数（热力学）
    if set_obj_Met_value:   
        def set_obj_Met_value(m):
            return m.metabolite[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_Met_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_Met_value, sense=minimize)
 
 
 
 
    #           ****添加约束****
    
    #  热力学约束set_metabolite、set_Df、set_obj_B_value（B_value为变量）、set_thermodynamics（B_value为定值）、set_integer
    if set_metabolite:
        def set_metabolite(m,i):
            return  inequality(metabolites_lnC.loc[i,'lnClb'], m.metabolite[i], metabolites_lnC.loc[i,'lnCub'])
        Concretemodel.set_metabolite= Constraint(ConMet_list,rule=set_metabolite)        
    #thermodynamic driving force expression for reactions
    if set_Df:
        def set_Df(m,j):
            return  m.Df[j]==-reaction_g0.loc[j,'g0']-2.579*sum(coef_matrix[i,j]*m.metabolite[i]  for i in ConMet_list if (i,j) in coef_matrix.keys())
        Concretemodel.set_Df = Constraint(reaction_listDF,rule=set_Df)
    #Adding thermodynamic MDF(B) object function
    if set_thermodynamics0:
        def set_thermodynamics0(m,j):
            return m.B<=(m.Df[j]+(1-m.z[j])*K_value)
        Concretemodel.set_obj_B_value = Constraint(reaction_listDF, rule=set_thermodynamics0)
    #Adding thermodynamic constraints
    if set_thermodynamics:
        def set_thermodynamics(m,j):
            return (m.Df[j]+(1-m.z[j])*K_value)>= B_value
        Concretemodel.set_thermodynamics = Constraint(reaction_listDF, rule=set_thermodynamics)
    #Adding binary variables constraints
    if set_integer:
        def set_integer(m,j):
            return m.reaction[j]<=m.z[j]*ub_list[j] 
        Concretemodel.set_integer = Constraint(reaction_listDF,rule=set_integer)     
        
        
    # 酶约束
    if set_enzyme_constraint:
        # flux- enzyme relationship
        Concretemodel.set_constr15 = ConstraintList()  
        if len(y1_protain)>0:
            for i in y1_protain:   
                for j in enzyme_rxns_dict[i]:
                    if j in e0 and j in kcat_dict:
                        Concretemodel.set_constr15.add(  Concretemodel.y[j] <= Concretemodel.y1[i]* M[j][1] ) 
                        Concretemodel.set_constr15.add(  Concretemodel.y[j] >= Concretemodel.y1[i]* M[j][0]) 
                    
        Concretemodel.set_constr2 = ConstraintList()
        for i in e0:
            if len(listy)>0:
                if i in listy:
                    Concretemodel.set_constr2.add(Concretemodel.e[i] * (kcat_dict[i] + Concretemodel.y[i]) >= Concretemodel.reaction[i])
                else:
                    Concretemodel.set_constr2.add(Concretemodel.e[i] *kcat_dict[i]  >= Concretemodel.reaction[i])
            else:
                Concretemodel.set_constr2.add(Concretemodel.e[i] *kcat_dict[i]  >= Concretemodel.reaction[i])

        # protein  concentration
        Concretemodel.set_constr3 = ConstraintList()  
        for i in totle_E:   
            Concretemodel.set_constr3.add(Concretemodel.e1[i] == sum(Concretemodel.e[j] for j in enzyme_rxns_dict[i] if j in e0))

        Concretemodel.set_c13 = ConstraintList()  
        for i in C13_dict:
            Concretemodel.set_c13.add( Concretemodel.abs[i] >=   sum( -Concretemodel.reaction[j] if '_reverse' in  j else  Concretemodel.reaction[j]  for j in C13_reaction[i])   - C13_dict[i]    )
            Concretemodel.set_c13.add( Concretemodel.abs[i] >=   C13_dict[i] -  sum(-Concretemodel.reaction[j] if '_reverse' in  j else  Concretemodel.reaction[j]  for j in C13_reaction[i]))

        Concretemodel.set_biomass = Constraint(expr=Concretemodel.reaction[biomass_id] >= biomass_value)

        # total protein concentration
    if set_max_sum_e:
        # Concretemodel.set_protainY =ConstraintList()
        # for i in ['b3731 and b3732 and b3733 and b3734 and b3735 and b3736 and b3737 and b3738','b3731 and b3732 and b3733 and b3734 and b3735 and b3736 and b3737 and b3738 and b3739']:
        #     Concretemodel.set_protainY.add(Concretemodel.y[i]==0)
        # Concretemodel.set_y = Constraint(expr=sum(Concretemodel.y1[i] for i in y1_protain if i in mw_dict) <=num_e1 )
        # Concretemodel.set_max_sum_e = Constraint(expr=sum(Concretemodel.e1[i] * mw_dict[i] * (1-Concretemodel.y1[i] ) for i in totle_E if i in y1_protain) + sum(Concretemodel.e1[i] * mw_dict[i] for i in totle_E if i  not in y1_protain)<= E_total)
        if len(y1_protain) >0:
            Concretemodel.set_y = Constraint(expr=sum(Concretemodel.y1[i] for i in y1_protain if i in mw_dict) <=num_e1 )
        Concretemodel.set_max_sum_e = Constraint(expr= sum(Concretemodel.e1[i] * mw_dict[i] for i in totle_E if i  in mw_dict)<= E_total)

    if set_fix_sum_e:
        Concretemodel.set_max_sum_e = Constraint(expr=sum(Concretemodel.e1[i] * mw_dict[i] for i in totle_E if i in mw_dict) == E_total) 
        
    # 最大底物摄入约束
    if set_substrate_ini:
        def set_substrate_ini(m): 
            return m.reaction[substrate_name] <= substrate_value
        Concretemodel.set_substrate_ini = Constraint(rule=set_substrate_ini)       
    # 固定biomass范围
    if set_biomass_ini:
        def set_biomass_ini(m): 
            return m.reaction[biomass_id] >=biomass_value
        Concretemodel.set_biomass_ini = Constraint(rule=set_biomass_ini)
    # 固定product范围
    if set_product_ini:
        Concretemodel.set_product_ini = Constraint(expr=Concretemodel.reaction[product_name] >= product_value)
    # 批量固定反应通量
    if set_reactions_ini:
        Concretemodel.set_reactions_ini = ConstraintList()
        for i in constr_coeff['fix_reactions']:
            Concretemodel.set_reactions_ini.add(Concretemodel.reaction[i] == constr_coeff['fix_reactions'][i])

    #Adding concentration ratio constraints for metabolites
    if set_metabolite_ratio:
        def set_atp_adp(m):
            return m.metabolite['atp_c']-m.metabolite['adp_c']==np.log(10)
        def set_adp_amp(m):
            return m.metabolite['adp_c']-m.metabolite['amp_c']==np.log(1)
        def set_nad_nadh(m):
            return m.metabolite['nad_c']-m.metabolite['nadh_c']==np.log(10)
        def set_nadph_nadp(m):
            return m.metabolite['nadph_c']-m.metabolite['nadp_c']==np.log(10)
        def set_hco3_co2(m):
            return m.metabolite['hco3_c']-m.metabolite['co2_c']==np.log(2)

        Concretemodel.set_atp_adp = Constraint(rule=set_atp_adp) 
        Concretemodel.set_adp_amp = Constraint(rule=set_adp_amp) 
        Concretemodel.set_nad_nadh = Constraint(rule=set_nad_nadh) 
        Concretemodel.set_nadph_nadp = Constraint(rule=set_nadph_nadp) 
        Concretemodel.set_hco3_co2 = Constraint(rule=set_hco3_co2)

    #Adding Bottleneck reaction constraints
    if set_Bottleneck_reaction:
        def set_Bottleneck_reaction(m,j):
            return m.z[j]==1 
        Concretemodel.set_Bottleneck_reaction = Constraint(Bottleneck_reaction_list,rule=set_Bottleneck_reaction) 

    return Concretemodel
def FBA_template7(coef_matrix=None,metabolites_lnC=None,reaction_g0=None,lb_list=None,ub_list=None,reaction_kcat_MW=None,B_value=None,kcat_dict=None,mw_dict=None,K_value=None,\
            product_value=None,biomass_value=None,substrate_value=None,E_total=None,\
            product_name=None,reaction_list=None,metabolite_list=None,enzyme_rxns_dict=None,e0=None,totle_E=None,obj_name=None,obj_target=None,\
            biomass_id=None,substrate_name=None,Bottleneck_reaction_list=None,minabs_error=None,set_abs_constraint=False,\
            set_obj_B_value=False,set_obj_sum_e=False,set_obj_value_e=False,set_obj_E_value=False,set_obj_single_E_value=False,\
            set_obj_value=False,set_obj_V_value=False,set_obj_TM_value=False,set_obj_Met_value=False,\
            set_metabolite=False,set_Df=False,set_thermodynamics0=False,set_thermodynamics=False,set_integer=False,set_fix_sum_e=False,\
            set_enzyme_constraint=False,set_substrate_ini=False, set_biomass_ini=False,set_product_ini=False,set_reactions_ini=False,\
            set_metabolite_ratio=False,set_Bottleneck_reaction=False,set_Df_value=False,set_enzyme_value=False,set_FC_value=False,set_max_sum_e=False,\
            mode=None,constr_coeff=None,Concretemodel_Need_Data=None,num_e1=None,set_abs_value=False,M=None):
    
    """According to the parameter conditions provided by the user, the specific pyomo model is returned.

    Notes
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC:Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...

    * reaction_kcat_MW: The enzymatic data of kcat (turnover number, /h) divided by MW (molecular weight, kDa).
    The format is as follows (in .csv file):
    kcat,MW,kcat_MW
    AADDGT	231887.7737	40.6396	5705.956104
    AAMYL	23490.41652	56.63940007	414.736323
    ...
    
    * lb_list: Lower bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * ub_list: Upper bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * obj_name: Name of object, such as set_obj_value, set_obj_single_E_value, set_obj_TM_value and set_obj_Met_value.    
    * K_value: the maximum value minus the minimum value of reaction thermodynamic driving force (1249, "max(DFi,max)-min(DFi,min)").
    * obj_target: Type of object function (maximize or minimize).
    * set_obj_value: Set the flux as the object function (True or False).
    * set_substrate_ini: Adding substrate amount constraints (True or False).
    * substrate_name: Substrate input reaction ID in the model (such as EX_glc__D_e_reverse).
    * substrate_value: Set the upper bound for substrate input reaction flux (10 mmol/h/gDW).
    * set_biomass_ini: Set the lower bound for biomass synthesis reaction flux (True or False).
    * biomass_value: The lower bound of biomass synthesis reaction flux.
    * biomass_id: Biomass synthesis reaction ID in the model (BIOMASS_Ec_iML1515_core_75p37M).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    * set_obj_B_value: The object function is the maximizing thermodynamic driving force of a pathway (True or False)
    * set_stoi_matrix: Adding flux balance constraints (True or False).
    * set_bound: Set the upper and lower bounds of reaction flux (True or False).
    * set_enzyme_constraint: Adding enzymamic constraints (True or False).
    * set_integer: Adding binary variables constraints (True or False).
    * set_metabolite_ratio: Adding concentration ratio constraints for metabolites (True or False).
    * set_thermodynamics: Adding thermodynamic constraints (True or False).
    * B_value: The value of maximizing the minimum thermodynamic driving force (MDF).
    * set_obj_E_value: The object function is the minimum enzyme cost of a pathway (True or False).
    * set_obj_V_value: The object function is the pFBA of a pathway (True or False)
    * set_obj_TM_value: The object function is the thermodynamic driving force of a reaction (True or False).
    * set_obj_Met_value: The object function is the concentration of a metabolite (True or False).
    * set_obj_single_E_value: The object function is the enzyme cost of a reaction (True or False).     
    * E_total: Total amount constraint of effective enzymes pool (0.13).
    * Bottleneck_reaction_list: A list extracted from the result file automatically.
    * set_Bottleneck_reaction: Adding integer variable constraints for specific reaction (True or False).
    """
    Concretemodel = ConcreteModel()
    
    
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    if   mode=='ST':
        if set_obj_B_value:
            set_thermodynamics0=True
        if not set_obj_B_value:
            set_thermodynamics=True
        set_Df_value=True
        set_metabolite=True
        set_Df=True
        set_integer=True
        set_metabolite_ratio=True
        
        metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
        reaction_g0=Concretemodel_Need_Data['reaction_g0']
        B_value=Concretemodel_Need_Data['B_value']
        K_value=Concretemodel_Need_Data['K_value']
        
        
    if  mode=='SE': 
        set_enzyme_constraint=True
        set_enzyme_value=True
        if set_obj_sum_e or set_fix_sum_e:
            set_max_sum_e=False
        if not set_obj_sum_e:
            set_max_sum_e=True
        # if set_FC_value:
        #     set_abs_constraint=True
            
        kcat_dict=Concretemodel_Need_Data['kcat_dict']
        mw_dict=Concretemodel_Need_Data['mw_dict']
        enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
        e0=Concretemodel_Need_Data['e0']
        totle_E=Concretemodel_Need_Data['totle_E']
        E_total=Concretemodel_Need_Data['E_total']
        y1_protain=Concretemodel_Need_Data['y1_protain']
        listy = []
        if len(y1_protain) > 0:
            for i in y1_protain:
                for j in Concretemodel_Need_Data['enzyme_rxns_dict'][i]:
                    if j in Concretemodel_Need_Data['kcat_dict']:
                        listy.append(j)
        C13_dict=Concretemodel_Need_Data['C13_flux_dict']
        C13_keys=list(C13_dict.keys())
        C13_reaction=Concretemodel_Need_Data['C13_reaction_dict']
    if  mode=='SET':    
        if set_obj_B_value:
            set_thermodynamics0=True
        if not set_obj_B_value:
            set_thermodynamics=True
        if set_obj_sum_e or set_fix_sum_e:
            set_max_sum_e=False
        if not set_obj_sum_e:
            set_max_sum_e=True
            
        set_Df_value=True
        set_enzyme_value=True
        set_metabolite=True
        set_Df=True
        set_metabolite_ratio=True
        set_integer=True 
        set_enzyme_constraint=True    
        kcat_dict=Concretemodel_Need_Data['kcat_dict']
        mw_dict=Concretemodel_Need_Data['mw_dict']
        enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
        e0=Concretemodel_Need_Data['e0']
        totle_E=Concretemodel_Need_Data['totle_E']
        E_total=Concretemodel_Need_Data['E_total']   
        metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
        reaction_g0=Concretemodel_Need_Data['reaction_g0']
        B_value=Concretemodel_Need_Data['B_value']
        K_value=Concretemodel_Need_Data['K_value']   
        y1_protain=Concretemodel_Need_Data['y1_protain']
        listy = []
        if len(y1_protain) > 0:
            for i in y1_protain:
                for j in Concretemodel_Need_Data['enzyme_rxns_dict'][i]:
                    if j in Concretemodel_Need_Data['kcat_dict']:
                        listy.append(j)    
        C13_dict=Concretemodel_Need_Data['C13_flux_dict']
        C13_keys=list(C13_dict.keys())
        C13_reaction=Concretemodel_Need_Data['C13_reaction_dict']
                    
    if constr_coeff !=None:
        if 'substrate_constrain' in constr_coeff:
            set_substrate_ini=True
            substrate_value=constr_coeff['substrate_constrain'][1]
            substrate_name=constr_coeff['substrate_constrain'][0]
        if 'biomass_constrain' in constr_coeff:
            set_biomass_ini=True
            biomass_value=constr_coeff['biomass_constrain'][1]
            biomass_id=constr_coeff['biomass_constrain'][0]
        if 'product_constrain' in constr_coeff:
            set_product_ini=True
            product_value=constr_coeff['product_constrain'][1]
            product_name=constr_coeff['product_constrain'][0]
        if 'fix_reactions' in constr_coeff and constr_coeff['fix_reactions'] !={}:
            set_reactions_ini=True
        if 'fix_E_total' in constr_coeff:
            set_fix_sum_e=True
            set_max_sum_e=False
    Concretemodel.reaction = pyo.Var(reaction_list)
    #Concretemodel.reaction = pyo.Var(reaction_list,  within=NonNegativeReals) # reaction flux variable  
    # SV=0
    def set_stoi_matrix(m,i):
        return sum(coef_matrix[i,j]*m.reaction[j]  for j in reaction_list if (i,j) in coef_matrix.keys() )==0
    Concretemodel.set_stoi_matrix_c = Constraint( metabolite_list,rule=set_stoi_matrix)
    # LB、UB
    def set_bound(m,j):
        return inequality(lb_list[j],m.reaction[j],ub_list[j])
    Concretemodel.set_bound_c = Constraint(reaction_list,rule=set_bound)   
    
    
    if reaction_kcat_MW is not None:
    # check if kcat_MW is a column in reaction_kcat_MW
        if 'kcat_MW' in reaction_kcat_MW.columns:
            kcatmw='kcat_MW'
        elif 'kcat/mw' in reaction_kcat_MW.columns:
            kcatmw='kcat/mw'
            
    #           ****设置变量****    
    # 热力学变量设置    
    if set_Df_value:
        reaction_listDF=[j for j in reaction_g0.index if j in reaction_list]
        ConMet_list= list(set(metabolite_list).intersection(set(metabolites_lnC.index)))
    #    Concretemodel.metabolite = pyo.Var(ConMet_list,  within=Reals) # metabolite concentration variable
    #    Concretemodel.Df = pyo.Var(reaction_listDF,  within=Reals) # thermodynamic driving force variable--reactions
    #    Concretemodel.z = pyo.Var(reaction_listDF,  within=pyo.Binary)    # binary variable
        #Concretemodel.B = pyo.Var()     # thermodynamic driving force variable--mdfz
 
    # 蛋白为中心酶变量设置

    # add variable with bounds in pyomo: Concretemodel.e = pyo.Var(e0, within=NonNegativeReals,bounds=(0, 100))
    # ConcreteModel.e=pyo.Var(e0,)
    if set_enzyme_value:
        Concretemodel.e = pyo.Var(e0, bounds = (0, 0.1))
        totle_E=[x for x in totle_E if x in mw_dict ]
        Concretemodel.e1 = pyo.Var(totle_E, bounds = (0, 0.1))    
        Concretemodel.FC_rxn = pyo.Var(listy,bounds = (0, 500))
        Concretemodel.FC = pyo.Var(y1_protain,bounds = (0,500))
        if len(y1_protain)>0:
            Concretemodel.y1 = pyo.Var(y1_protain, within=Binary)    
        Concretemodel.abs=pyo.Var(C13_keys,bounds = (0, 100))   


    # if set_enzyme_value:
    #     Concretemodel.e = pyo.Var(e0, within=NonNegativeReals)
    #     totle_E=[x for x in totle_E if x in mw_dict ]
    #     Concretemodel.e1 = pyo.Var(totle_E, within=NonNegativeReals)    
    #     Concretemodel.FC_rxn = Var(listy,within=NonNegativeReals)
    #     Concretemodel.FC = pyo.Var(y1_protain,within=NonNegativeReals)
    #     if len(y1_protain)>0:
    #         Concretemodel.y1 = pyo.Var(y1_protain, within=Binary)    
    #     Concretemodel.abs=pyo.Var(C13_keys, within=NonNegativeReals)    

    #           ****设置目标函数****
    
    # 以B_value为目标函数
    if set_obj_B_value:             
        def set_obj_B_value(m):
            return m.B  
        Concretemodel.obj = Objective(rule=set_obj_B_value, sense=maximize)  
    # 以总蛋白量加和为目标函数
    if set_obj_sum_e:             
        def set_obj_sum_e(m):
            return sum(Concretemodel.e1[i] * mw_dict[i] for i in totle_E if i in mw_dict)
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_sum_e, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_sum_e, sense=minimize)
    # 以最小化酶量加和为目标函数（不是以蛋白为中心）   
    if set_obj_E_value:             
        def set_obj_E_value(m):
            return sum(m.reaction[j]/(reaction_kcat_MW.loc[j,kcatmw]) for j in reaction_kcat_MW.index if j in reaction_list)
        Concretemodel.obj = Objective(rule=set_obj_E_value, sense=minimize)      
    # 以单一蛋白为目标函数
    if set_obj_value_e:   
        def set_obj_value_e(m):
            return m.e1[obj_name]*mw_dict[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_value_e, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_value_e, sense=minimize)    
    # 以单一蛋白为目标函数（不是以蛋白为中心）
    if set_obj_single_E_value:             
        def set_obj_single_E_value(m):
            return m.reaction[obj_name]/(reaction_kcat_MW.loc[obj_name,kcatmw])
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_single_E_value, sense=maximize) 
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_single_E_value, sense=minimize)
    # 以某个反应为目标函数
    if set_obj_value:   
        def set_obj_value(m):
            return m.reaction[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_value, sense=minimize)           
    #最小化通量和为目标函数 (pFBA)
    if set_obj_V_value:             
        def set_obj_V_value(m):
            return sum(m.reaction[j] for j in reaction_list)
        Concretemodel.obj = Objective(rule=set_obj_V_value, sense=minimize)  

    #最小化abs和为目标函数 (pFBA)
    if set_abs_value:             
        def set_abs_value(m):
            return sum(m.abs[j] for j in C13_keys)
        Concretemodel.obj = Objective(rule=set_abs_value, sense=minimize)  
    if set_FC_value:             
        def set_FC_value(m):
            return sum(m.FC_rxn[j] for j in listy)
        Concretemodel.obj = Objective(rule=set_FC_value, sense=minimize)  
    #To calculate the variability of thermodynamic driving force (Dfi) of single reaction.
    if set_obj_TM_value:   
        def set_obj_TM_value(m):
            return m.Df[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_TM_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_TM_value, sense=minimize)
    #以代谢物浓度为目标函数（热力学）
    if set_obj_Met_value:   
        def set_obj_Met_value(m):
            return m.metabolite[obj_name]
        if obj_target=='maximize':
            Concretemodel.obj = Objective(rule=set_obj_Met_value, sense=maximize)
        elif obj_target=='minimize':
            Concretemodel.obj = Objective(rule=set_obj_Met_value, sense=minimize)
 
 
 
 
    #           ****添加约束****
    
    #  热力学约束set_metabolite、set_Df、set_obj_B_value（B_value为变量）、set_thermodynamics（B_value为定值）、set_integer
    if set_metabolite:
        def set_metabolite(m,i):
            return  inequality(metabolites_lnC.loc[i,'lnClb'], m.metabolite[i], metabolites_lnC.loc[i,'lnCub'])
        Concretemodel.set_metabolite= Constraint(ConMet_list,rule=set_metabolite)        
    #thermodynamic driving force expression for reactions
    if set_Df:
        def set_Df(m,j):
            return  m.Df[j]==-reaction_g0.loc[j,'g0']-2.579*sum(coef_matrix[i,j]*m.metabolite[i]  for i in ConMet_list if (i,j) in coef_matrix.keys())
        Concretemodel.set_Df = Constraint(reaction_listDF,rule=set_Df)
    #Adding thermodynamic MDF(B) object function
    if set_thermodynamics0:
        def set_thermodynamics0(m,j):
            return m.B<=(m.Df[j]+(1-m.z[j])*K_value)
        Concretemodel.set_obj_B_value = Constraint(reaction_listDF, rule=set_thermodynamics0)
    #Adding thermodynamic constraints
    if set_thermodynamics:
        def set_thermodynamics(m,j):
            return (m.Df[j]+(1-m.z[j])*K_value)>= B_value
        Concretemodel.set_thermodynamics = Constraint(reaction_listDF, rule=set_thermodynamics)
    #Adding binary variables constraints
    if set_integer:
        def set_integer(m,j):
            return m.reaction[j]<=m.z[j]*ub_list[j] 
        Concretemodel.set_integer = Constraint(reaction_listDF,rule=set_integer)     

        
        
    # 酶约束
    if set_enzyme_constraint:
        # flux- enzyme relationship
        Concretemodel.set_constr15 = ConstraintList()  
        if len(y1_protain)>0:
            for i in y1_protain:   
                # Concretemodel.set_constr15.add(  Concretemodel.FC[i] <= (1-Concretemodel.y1[i])*M + 1000 ) 
                # Concretemodel.set_constr15.add(  Concretemodel.FC[i] >= -(1-Concretemodel.y1[i])*M + 50.01) 
                # Concretemodel.set_constr15.add(  Concretemodel.FC[i] <= -Concretemodel.y1[i]*M + 1) 
                # Concretemodel.set_constr15.add(  Concretemodel.FC[i] >= -Concretemodel.y1[i]*M + 1)
                Concretemodel.set_constr15.add(Concretemodel.FC[i] <= 499 * Concretemodel.y1[i] +1)
                Concretemodel.set_constr15.add(Concretemodel.FC[i] >= -0.99 * Concretemodel.y1[i] +1)                 
                for j in enzyme_rxns_dict[i]:
                    if j in e0 and j in kcat_dict:
                        Concretemodel.set_constr15.add(Concretemodel.FC_rxn[j] == Concretemodel.FC[i])       
        Concretemodel.set_constr2 = ConstraintList()
        # 每一个酶对应的反应的FC相同，缺一个对应的j
        
        for i in e0:
            if len(y1_protain)>0:
                if i in listy:
                    # Concretemodel.set_constr2.add(expr=Concretemodel.FC[i] == Concretemodel.FC[j])          
                    Concretemodel.set_constr2.add(Concretemodel.e[i] * (kcat_dict[i]*Concretemodel.FC_rxn[i]) >= Concretemodel.reaction[i])
                else:
                    Concretemodel.set_constr2.add(Concretemodel.e[i] *kcat_dict[i]  >= Concretemodel.reaction[i])
        else:
            Concretemodel.set_constr2.add(Concretemodel.e[i] *kcat_dict[i]  >= Concretemodel.reaction[i])

        # protein  concentration
        Concretemodel.set_constr3 = ConstraintList()  
        for i in totle_E:   
            Concretemodel.set_constr3.add(Concretemodel.e1[i] == sum(Concretemodel.e[j] for j in enzyme_rxns_dict[i] if j in e0))

        Concretemodel.set_c13 = ConstraintList()  
        for i in C13_dict:
            Concretemodel.set_c13.add( Concretemodel.abs[i] >=   sum( -Concretemodel.reaction[j] if '_reverse' in  j else  Concretemodel.reaction[j]  for j in C13_reaction[i])- C13_dict[i])
            Concretemodel.set_c13.add( Concretemodel.abs[i] >=   C13_dict[i] -  sum(-Concretemodel.reaction[j] if '_reverse' in  j else  Concretemodel.reaction[j]  for j in C13_reaction[i]))

        Concretemodel.set_biomass = Constraint(expr=Concretemodel.reaction[biomass_id] >= biomass_value)

        # total protein concentration
    if set_max_sum_e:
        # Concretemodel.set_protainY =ConstraintList()
        # for i in ['b3731 and b3732 and b3733 and b3734 and b3735 and b3736 and b3737 and b3738','b3731 and b3732 and b3733 and b3734 and b3735 and b3736 and b3737 and b3738 and b3739']:
        #     Concretemodel.set_protainY.add(Concretemodel.y[i]==0)
        # Concretemodel.set_y = Constraint(expr=sum(Concretemodel.y1[i] for i in y1_protain if i in mw_dict) <=num_e1 )
        # Concretemodel.set_max_sum_e = Constraint(expr=sum(Concretemodel.e1[i] * mw_dict[i] * (1-Concretemodel.y1[i] ) for i in totle_E if i in y1_protain) + sum(Concretemodel.e1[i] * mw_dict[i] for i in totle_E if i  not in y1_protain)<= E_total)
        if len(y1_protain) >0:
            Concretemodel.set_y = Constraint(expr=sum(Concretemodel.y1[i] for i in y1_protain if i in mw_dict) <=num_e1 )
        Concretemodel.set_max_sum_e = Constraint(expr= sum(Concretemodel.e1[i] * mw_dict[i] for i in totle_E if i  in mw_dict)<= E_total)

    if set_fix_sum_e:
        Concretemodel.set_max_sum_e = Constraint(expr=sum(Concretemodel.e1[i] * mw_dict[i] for i in totle_E if i in mw_dict) == E_total) 

    if set_FC_value:
        Concretemodel.set_abs_constraint = Constraint(expr=sum(Concretemodel.abs[j] for j in C13_keys) <= minabs_error)

    # if set_abs_constraint:             
    #     def set_abs_value(m):
    #         return sum(m.abs[j] for j in C13_keys) <= minabs_error
    #     Concretemodel.set_abs_constraint = Constraint(rule=set_abs_constraint)  

    # 最大底物摄入约束
    if set_substrate_ini:
        def set_substrate_ini(m): 
            return m.reaction[substrate_name] <= substrate_value
        Concretemodel.set_substrate_ini = Constraint(rule=set_substrate_ini)       
    # 固定biomass范围
    if set_biomass_ini:
        def set_biomass_ini(m): 
            return m.reaction[biomass_id] >=biomass_value
        Concretemodel.set_biomass_ini = Constraint(rule=set_biomass_ini)
    # 固定product范围
    if set_product_ini:
        Concretemodel.set_product_ini = Constraint(expr=Concretemodel.reaction[product_name] >= product_value)
    # 批量固定反应通量
    if set_reactions_ini:
        Concretemodel.set_reactions_ini = ConstraintList()
        for i in constr_coeff['fix_reactions']:
            Concretemodel.set_reactions_ini.add(Concretemodel.reaction[i] == constr_coeff['fix_reactions'][i])

    #Adding concentration ratio constraints for metabolites
    if set_metabolite_ratio:
        def set_atp_adp(m):
            return m.metabolite['atp_c']-m.metabolite['adp_c']==np.log(10)
        def set_adp_amp(m):
            return m.metabolite['adp_c']-m.metabolite['amp_c']==np.log(1)
        def set_nad_nadh(m):
            return m.metabolite['nad_c']-m.metabolite['nadh_c']==np.log(10)
        def set_nadph_nadp(m):
            return m.metabolite['nadph_c']-m.metabolite['nadp_c']==np.log(10)
        def set_hco3_co2(m):
            return m.metabolite['hco3_c']-m.metabolite['co2_c']==np.log(2)

        Concretemodel.set_atp_adp = Constraint(rule=set_atp_adp) 
        Concretemodel.set_adp_amp = Constraint(rule=set_adp_amp) 
        Concretemodel.set_nad_nadh = Constraint(rule=set_nad_nadh) 
        Concretemodel.set_nadph_nadp = Constraint(rule=set_nadph_nadp) 
        Concretemodel.set_hco3_co2 = Constraint(rule=set_hco3_co2)

    #Adding Bottleneck reaction constraints
    if set_Bottleneck_reaction:
        def set_Bottleneck_reaction(m,j):
            return m.z[j]==1 
        Concretemodel.set_Bottleneck_reaction = Constraint(Bottleneck_reaction_list,rule=set_Bottleneck_reaction) 

    return Concretemodel
def Get_Max_Min_Df_Ratio(Concretemodel_Need_Data,obj_name,obj_target,solver):

    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']

    Concretemodel = Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,obj_name=obj_name,obj_target=obj_target,\
        set_obj_TM_value=True,set_metabolite=True,set_Df=True,set_metabolite_ratio=True)
    """Calculation both of maximum and minimum thermodynamic driving force for reactions,with metabolite ratio constraints.
    
    Notes:
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.

    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC: Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...
    
    * obj_name: Object reaction ID.
    * obj_target: Type of object function (maximize or minimize).    
    * set_obj_TM_value: set_obj_TM_value: The object function is the thermodynamic driving force of a reaction (True or False).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    * set_metabolite_ratio: Adding concentration ratio constraints for metabolites (True or False).   
    """
    max_min_Df_list=pd.DataFrame()
    opt = Model_Solve(Concretemodel,solver)   
    if obj_target=='maximize':
        max_min_Df_list.loc[obj_name,'max_value']=opt.obj() 
    elif obj_target=='minimize':
        max_min_Df_list.loc[obj_name,'min_value']=opt.obj() 

    return max_min_Df_list

# #Solving the MDF (B value)
# def MDF_Calculation(Concretemodel_Need_Data,biomass_value,biomass_id,substrate_name,substrate_value,K_value,E_total,solver):
#     reaction_list=Concretemodel_Need_Data['reaction_list']
#     metabolite_list=Concretemodel_Need_Data['metabolite_list']
#     coef_matrix=Concretemodel_Need_Data['coef_matrix']
#     metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
#     reaction_g0=Concretemodel_Need_Data['reaction_g0']
#     reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
#     lb_list=Concretemodel_Need_Data['lb_list']
#     ub_list=Concretemodel_Need_Data['ub_list']
#     ub_list[substrate_name]=substrate_value

#     Concretemodel=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
#         metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
#         K_value=K_value,set_substrate_ini=True,substrate_name=substrate_name,substrate_value=substrate_value,set_biomass_ini=True,\
#         biomass_value=biomass_value,biomass_id=biomass_id,set_metabolite=True,set_Df=True,set_obj_B_value=True,set_stoi_matrix=True,\
#         set_bound=True,set_enzyme_constraint=True,E_total=E_total,set_integer=True,set_metabolite_ratio=True)
#     opt=Model_Solve(Concretemodel,solver)
#     #B_value=format(Concretemodel.obj(), '.3f')
#     B_value=opt.obj()-0.000001
#     return B_value
#Solving the MDF (B value)
def MDF_Calculation(Concretemodel_Need_Data,biomass_value,biomass_id,substrate_name,substrate_value,K_value,E_total,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=copy.deepcopy(Concretemodel_Need_Data['ub_list'])
    ub_list[substrate_name]=substrate_value
    kcat_dict=Concretemodel_Need_Data['kcat_dict']
    mw_dict=Concretemodel_Need_Data['mw_dict']
    enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
    e0=Concretemodel_Need_Data['e0']
    totle_E=Concretemodel_Need_Data['totle_E']


    Concretemodel=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,mw_dict=mw_dict,\
        enzyme_rxns_dict=enzyme_rxns_dict,totle_E=totle_E,set_enzyme_value=True,e0=e0,set_enzyme_constraint=True,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,kcat_dict=kcat_dict,lb_list=lb_list,ub_list=ub_list,\
        K_value=K_value,substrate_name=substrate_name,substrate_value=substrate_value,set_biomass_ini=True,\
        biomass_value=biomass_value,biomass_id=biomass_id,set_substrate_ini=True,set_metabolite=True,set_Df=True,set_obj_B_value=True,\
        set_stoi_matrix=True,set_bound=True,E_total=E_total,set_integer=True,set_metabolite_ratio=True)
    opt=Model_Solve(Concretemodel,solver)
    #B_value=format(Concretemodel.obj(), '.3f')
    B_value=opt.obj()-0.000001
    return B_value

def MDF_Calculation_v(Concretemodel_Need_Data,biomass_value,biomass_id,substrate_name,substrate_value,K_value,E_total,solver,biomass_name,v0_biomass):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=copy.deepcopy(Concretemodel_Need_Data['ub_list'])
    ub_list[substrate_name]=substrate_value
    kcat_dict=Concretemodel_Need_Data['kcat_dict']
    mw_dict=Concretemodel_Need_Data['mw_dict']
    enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
    e0=Concretemodel_Need_Data['e0']
    totle_E=Concretemodel_Need_Data['totle_E']
    
    Concretemodel=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,mw_dict=mw_dict,\
        enzyme_rxns_dict=enzyme_rxns_dict,totle_E=totle_E,set_enzyme_value=True,e0=e0,set_enzyme_constraint=True,\
            metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,kcat_dict=kcat_dict,lb_list=lb_list,ub_list=ub_list,\
        K_value=K_value,substrate_name=substrate_name,substrate_value=substrate_value,set_biomass_ini=True,\
        biomass_value=biomass_value,biomass_id=biomass_id,set_substrate_ini=True,set_metabolite=True,set_Df=True,set_obj_B_value=True,set_constr10=True,\
            biomass_name=biomass_name,v0_biomass=v0_biomass,set_stoi_matrix=True,set_bound=True,E_total=E_total,set_integer=True,set_metabolite_ratio=True)



    opt=Model_Solve(Concretemodel,solver)
    #B_value=format(Concretemodel.obj(), '.3f')
    B_value=opt.obj()-0.000001
    Concretemodel.write('./lpfiles/mdf.lp',io_options={'symbolic_solver_labels': True})
    return B_value,Concretemodel

# only thermodynamic
def MDF_Calculation_t(Concretemodel_Need_Data,biomass_value,biomass_id,substrate_name,substrate_value,K_value,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=copy.deepcopy(Concretemodel_Need_Data['ub_list'])
    ub_list[substrate_name]=substrate_value

    Concretemodel=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
            metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,lb_list=lb_list,ub_list=ub_list,\
        K_value=K_value,substrate_name=substrate_name,substrate_value=substrate_value,set_biomass_ini=True,\
        biomass_value=biomass_value,biomass_id=biomass_id,set_substrate_ini=True,set_metabolite=True,set_Df=True,set_obj_B_value=True,set_constr10=False,\
        set_stoi_matrix=True,set_bound=True,set_integer=True,set_metabolite_ratio=True)
    opt=Model_Solve(Concretemodel,solver)
    #B_value=format(Concretemodel.obj(), '.3f')
    B_value=opt.obj()-0.000001
    # Concretemodel.write('./lpfiles/mdf.lp',io_options={'symbolic_solver_labels': True})
    return B_value,Concretemodel
def MDF_Calculation_tv(Concretemodel_Need_Data,biomass_value,biomass_id,substrate_name,substrate_value,K_value,solver,biomass_name,v0_biomass):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=copy.deepcopy(Concretemodel_Need_Data['ub_list'])
    ub_list[substrate_name]=substrate_value

    Concretemodel=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
            metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,lb_list=lb_list,ub_list=ub_list,\
        K_value=K_value,substrate_name=substrate_name,substrate_value=substrate_value,set_biomass_ini=True,\
        biomass_value=biomass_value,biomass_id=biomass_id,set_substrate_ini=True,set_metabolite=True,set_Df=True,set_obj_B_value=True,set_constr10=True,\
            biomass_name=biomass_name,v0_biomass=v0_biomass,set_stoi_matrix=True,set_bound=True,set_integer=True,set_metabolite_ratio=True)
    opt=Model_Solve(Concretemodel,solver)
    #B_value=format(Concretemodel.obj(), '.3f')
    B_value=opt.obj()-0.000001
    # Concretemodel.write('./lpfiles/mdf.lp',io_options={'symbolic_solver_labels': True})
    return B_value,Concretemodel

    
#Constructing a GEM (iML1515 model) using Pyomo Concretemodel framework
def EcoGEM(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    EcoGEM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
            lb_list=lb_list,ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
            substrate_name=substrate_name,substrate_value=substrate_value,set_stoi_matrix=True,set_bound=True)
    return EcoGEM

#Constructing a enzymatic constraints model (EcoECM) using Pyomo Concretemodel framework
def EcoECM(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,E_total):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    EcoECM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_value=True,\
        set_substrate_ini=True,substrate_name=substrate_name,substrate_value=substrate_value,set_stoi_matrix=True,set_bound=True,\
        set_enzyme_constraint=True,E_total=E_total)
    return EcoECM

#Constructing a thermodynamic constraints model (EcoTCM) using Pyomo Concretemodel framework
def EcoTCM(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,K_value,B_value):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    EcoTCM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,lb_list=lb_list,ub_list=ub_list,\
        obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value,set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
        set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,K_value=K_value,B_value=B_value)
    return EcoTCM

#Constructing a enzymatic and thermodynamic constraints model (EcoETM) using Pyomo Concretemodel framework
def EcoETM(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,E_total,K_value,B_value):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    EcoETM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
        set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,K_value=K_value,B_value=B_value,\
        set_enzyme_constraint=True,E_total=E_total)
    return EcoETM

#Solving programming problems
#Solving programming problems
def Model_Solve(model,solver):
    opt = pyo.SolverFactory(solver)
    opt.solve(model)
    return model

#Maximum growth rate calculation
def Max_Growth_Rate_Calculation(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,K_value,E_total,B_value,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    Concretemodel=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        obj_name=obj_name,K_value=K_value,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value,set_metabolite=True,set_Df=True,set_stoi_matrix=True,\
        set_bound=True,set_enzyme_constraint=True,E_total=E_total,set_integer=True,set_metabolite_ratio=True,\
        set_thermodynamics=True,B_value=B_value)
    opt=Model_Solve(Concretemodel,solver)
    return opt.obj()

#Minimum enzyme cost calculation
def Min_Enzyme_Cost_Calculation(Concretemodel_Need_Data,biomass_value,biomass_id,substrate_name,substrate_value,K_value,E_total,B_value,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    Concretemodel=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        K_value=K_value,set_biomass_ini=True,biomass_value=biomass_value,biomass_id=biomass_id,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value,set_metabolite=True,set_Df=True,set_stoi_matrix=True,\
        set_bound=True,set_enzyme_constraint=True,E_total=E_total,set_integer=True,set_metabolite_ratio=True,\
        set_thermodynamics=True,B_value=B_value,set_obj_E_value=True)
    opt=Model_Solve(Concretemodel,solver)
    min_E=opt.obj()
    return min_E

#Minimum flux sum calculation（pFBA）
def Min_Flux_Sum_Calculation(Concretemodel_Need_Data,biomass_value,biomass_id,substrate_name,substrate_value,K_value,E_total,B_value,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    Concretemodel=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        K_value=K_value,set_biomass_ini=True,biomass_value=biomass_value,biomass_id=biomass_id,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value,set_metabolite=True,set_Df=True,set_stoi_matrix=True,\
        set_bound=True,set_enzyme_constraint=True,E_total=E_total,set_integer=True,set_metabolite_ratio=True,\
        set_thermodynamics=True,B_value=B_value,set_obj_V_value=True)
    opt=Model_Solve(Concretemodel,solver)

    min_V=opt.obj()
    return [min_V,Concretemodel]

#Determination of bottleneck reactions by analysing the variability of thermodynamic driving force
def Get_Max_Min_Df_Complete(Concretemodel_Need_Data,obj_name,obj_target,K_value,B_value,max_biomass_under_mdf,biomass_id,E_total,substrate_name,substrate_value,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    Concretemodel = Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,\
        ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_TM_value=True,set_metabolite=True,set_Df=True,set_metabolite_ratio=True,set_bound=True,\
        set_stoi_matrix=True,set_substrate_ini=True,K_value=K_value,substrate_name=substrate_name,substrate_value=substrate_value,set_thermodynamics=True,B_value=B_value,\
        set_biomass_ini=True,biomass_id=biomass_id,biomass_value=max_biomass_under_mdf,set_integer=True,set_enzyme_constraint=True,E_total=E_total)
    """Calculation both of maximum and minimum thermodynamic driving force for reactions in a special list.
    
    Notes:
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.

    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC: Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...
    
    * reaction_kcat_MW: The enzymatic data of kcat (turnover number, /h) divided by MW (molecular weight, kDa).
    The format is as follows (in .csv file):
    kcat,MW,kcat_MW
    AADDGT,49389.2889,40.6396,1215.299582180927
    GALCTND,75632.85923999999,42.5229,1778.63831582512
    ...
    
    * lb_list: Lower bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * ub_list: Upper bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * ub_list[substrate_name]: substrate_value (the upper bound for substrate input reaction flux)
    * obj_name: Object reaction ID.
    * obj_target: Type of object function (maximize or minimize). 
    * set_obj_TM_value: The object function is the thermodynamic driving force of a reaction (True or False).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    * set_metabolite_ratio: Adding concentration ratio constraints for metabolites (True or False).
    * set_bound: Set the upper and lower bounds of reaction flux (True or False).
    * set_stoi_matrix: Adding flux balance constraints (True or False).
    * set_biomass_ini: Set the lower bound for biomass synthesis reaction flux (True or False).
    * K_value: the maximum value minus the minimum value of reaction thermodynamic driving force (1249, maxDFi-minDFi).
    * substrate_name: Substrate input reaction ID in the model (such as EX_glc__D_e_reverse).
    * substrate_value: Set the upper bound for substrate input reaction flux (10 mM).
    * set_thermodynamics: Adding thermodynamic constraints (True or False).
    * B_value: The value of maximizing the minimum thermodynamic driving force (MDF).
    * set_biomass_ini: Set the lower bound for biomass synthesis reaction flux (True or False).
    * biomass_id: Biomass synthesis reaction ID in the model (BIOMASS_Ec_iML1515_core_75p37M).
    * biomass_value: The lower bound of biomass synthesis reaction flux.
    * set_integer: Adding binary variables constraints (True or False).
    * set_enzyme_constraint: Adding enzymamic constraints (True or False).
    * E_total: Total amount constraint of enzymes (0.114).
    """
    max_min_Df_list=pd.DataFrame()
    opt=Model_Solve(Concretemodel,solver)    
    if obj_target=='maximize':
        max_min_Df_list.loc[obj_name,'max_value']=opt.obj() 
    elif obj_target=='minimize':
        max_min_Df_list.loc[obj_name,'min_value']=opt.obj() 

    return max_min_Df_list

#Determination of limiting metabolites by analysing the concentration variability
def Get_Max_Min_Met_Concentration(Concretemodel_Need_Data,obj_name,obj_target,K_value,B_value,max_biomass_under_mdf,biomass_id,E_total,substrate_name,substrate_value,Bottleneck_reaction_list,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    Concretemodel = Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,\
        ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_Met_value=True,set_metabolite=True,set_Df=True,set_metabolite_ratio=True,set_bound=True,\
        set_stoi_matrix=True,set_substrate_ini=True,K_value=K_value,substrate_name=substrate_name,substrate_value=substrate_value,set_thermodynamics=True,B_value=B_value,\
        set_biomass_ini=True,biomass_id=biomass_id,biomass_value=max_biomass_under_mdf,set_integer=True,set_enzyme_constraint=True,E_total=E_total,\
        Bottleneck_reaction_list=Bottleneck_reaction_list,set_Bottleneck_reaction=True)
    """Calculation of the maximum and minimum concentrations for metabolites in a specific list.

    Notes：
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.

    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC: Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...
    
    * reaction_kcat_MW: The enzymatic data of kcat (turnover number, /h) divided by MW (molecular weight, kDa).
    The format is as follows (in .csv file):
    kcat,MW,kcat_MW
    AADDGT,49389.2889,40.6396,1215.299582180927
    GALCTND,75632.85923999999,42.5229,1778.63831582512
    ...
    
    * lb_list: Lower bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * ub_list: Upper bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * obj_name: Object reaction ID.
    * obj_target: Type of object function (maximize or minimize).   
    * set_obj_Met_value: The object function is the concentration of a metabolite (True or False).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    * set_metabolite_ratio: Adding concentration ratio constraints for metabolites (True or False).
    * set_bound: Set the upper and lower bounds of reaction flux (True or False).
    * set_stoi_matrix: Adding flux balance constraints (True or False).
    * set_substrate_ini: Adding substrate amount constraints (True or False).
    * K_value: the maximum value minus the minimum value of reaction thermodynamic driving force (1249, maxDFi-minDFi).
    * substrate_name: Substrate input reaction ID in the model (such as EX_glc__D_e_reverse).
    * substrate_value: Set the upper bound for substrate input reaction flux (10 mM).
    * set_thermodynamics: Adding thermodynamic constraints (True or False).
    * B_value: The value of maximizing the minimum thermodynamic driving force (MDF).
    * set_biomass_ini: Set the lower bound for biomass synthesis reaction flux (True or False).
    * biomass_id: Biomass synthesis reaction ID in the model (BIOMASS_Ec_iML1515_core_75p37M).
    * biomass_value: The lower bound of biomass synthesis reaction flux.
    * set_enzyme_constraint: Adding enzymamic constraints (True or False).
    * set_integer: Adding binary variables constraints (True or False)
    * E_total: Total amount constraint of enzymes (0.114).
    * Bottleneck_reaction_list: A list extracted from the result file automatically.
    * set_Bottleneck_reaction: Adding integer variable constraints for specific reaction (True or False).
    """  
    max_min_Df_list=pd.DataFrame()
    opt=Model_Solve(Concretemodel,solver)   
    if obj_target=='maximize':
        max_min_Df_list.loc[obj_name,'max_value']=opt.obj() 
    elif obj_target=='minimize':
        max_min_Df_list.loc[obj_name,'min_value']=opt.obj() 

    return max_min_Df_list

#calculate enzyme's overexpression and attenuation by objective's value(max/min)
def Get_Max_Min_R0(Concretemodel_Need_Data,obj_name,obj_target,K_value,B_value,max_biomass_under_mdf,biomass_id,E_total,substrate_name,substrate_value,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value
    totle_E=Concretemodel_Need_Data['totle_E']
    gene_reaction_dict=Concretemodel_Need_Data['gene_reaction_dict']
    e0=Concretemodel_Need_Data['e0']
    theMW=Concretemodel_Need_Data['theMW']

    Concretemodel = Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,theMW=theMW,lb_list=lb_list,\
        ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_single_E_value=True,set_metabolite=True,set_Df=True,set_metabolite_ratio=True,set_bound=True,\
        set_enzyme_constraint=False,set_stoi_matrix=True,set_substrate_ini=True,K_value=K_value,substrate_name=substrate_name,substrate_value=substrate_value,set_thermodynamics=True,B_value=B_value,\
        set_biomass_ini=True,biomass_id=biomass_id,biomass_value=max_biomass_under_mdf,set_integer=True,E_total=E_total,\
           enz_def_constr=True,mul_func_e1=True,enz_sum_constr=True,e0=e0,totle_E=totle_E,gene_reaction_dict=gene_reaction_dict)
    """Calculation of the maximum and minimum enzyme cost for reactions in a specific list.

    Notes:
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.

    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC: Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...
    
    * reaction_kcat_MW: The enzymatic data of kcat (turnover number, /h) divided by MW (molecular weight, kDa).
    The format is as follows (in .csv file):
    kcat,MW,kcat_MW
    AADDGT,49389.2889,40.6396,1215.299582180927
    GALCTND,75632.85923999999,42.5229,1778.63831582512
    ...
    
    * lb_list: Lower bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * ub_list: Upper bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * obj_name: Object reaction ID.
    * obj_target: Type of object function (maximize or minimize).  
    * set_obj_single_E_value: The object function is the enzyme cost of a reaction (True or False).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
lowerGet_Concretemodel_Need_Data    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    * set_metabolite_ratio: Adding concentration ratio constraints for metabolites (True or False).
    * set_bound: Set the upper and lower bounds of reaction flux (True or False).
    * set_stoi_matrix: Adding flux balance constraints (True or False).
    * set_substrate_ini: Adding substrate amount constraints (True or False).
    * K_value: the maximum value minus the minimum value of reaction thermodynamic driving force (1249, maxDFi-minDFi).
    * substrate_name: Substrate input reaction ID in the model (such as EX_glc__D_e_reverse).
    * substrate_value: Set the upper bound for substrate input reaction flux (10 mM).
    * set_thermodynamics: Adding thermodynamic constraints (True or False).
    * B_value: The value of maximizing the minimum thermodynamic driving force (MDF).
    * set_biomass_ini: Set the lower bound for biomass synthesis reaction flux (True or False).
    * biomass_id: Biomass synthesis reaction ID in the model (BIOMASS_Ec_iML1515_core_75p37M).
    * biomass_value: The lower bound of biomass synthesis reaction flux.
    * set_integer: Adding binary variables constraints (True or False).
    * set_enzyme_constraint: Adding enzymamic constraints (True or False).
    * E_total: Total amount constraint of enzymes (0.114).
    """  

    max_min_Df_list=pd.DataFrame()
    Concretemodel.write('result_max_min_e0.lp',io_options={'symbolic_solver_labels': True})
    opt=Model_Solve(Concretemodel,solver)
    if obj_target=='maximize':
        max_min_Df_list.loc[obj_name,'max_value']=opt.obj() 
    elif obj_target=='minimize':
        max_min_Df_list.loc[obj_name,'min_value']=opt.obj() 
    # print(max_min_Df_list)
    return max_min_Df_list

#calculate enzyme's overexpression and attenuation by objective's value(max/min)
def Get_Max_Min_E0(Concretemodel_Need_Data,obj_name,obj_target,K_value,B_value,max_biomass_under_mdf,biomass_id,E_total,substrate_name,substrate_value,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value
    totle_E=Concretemodel_Need_Data['totle_E']
    gene_reaction_dict=Concretemodel_Need_Data['gene_reaction_dict']
    e0=Concretemodel_Need_Data['e0']
    theMW=Concretemodel_Need_Data['theMW']

    Concretemodel = Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,theMW=theMW,lb_list=lb_list,\
        ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_single_E_value=True,set_metabolite=True,set_Df=True,set_metabolite_ratio=True,set_bound=True,\
        set_enzyme_constraint=False,set_stoi_matrix=True,set_substrate_ini=True,K_value=K_value,substrate_name=substrate_name,substrate_value=substrate_value,set_thermodynamics=True,B_value=B_value,\
        set_biomass_ini=True,biomass_id=biomass_id,biomass_value=max_biomass_under_mdf,set_integer=True,E_total=E_total,\
        set_obj_e1_value=True,enz_def_constr=True,mul_func_e1=True,enz_sum_constr=True,e0=e0,totle_E=totle_E,gene_reaction_dict=gene_reaction_dict)
    """Calculation of the maximum and minimum enzyme cost for reactions in a specific list.

    Notes:
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.

    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC: Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...
    
    * reaction_kcat_MW: The enzymatic data of kcat (turnover number, /h) divided by MW (molecular weight, kDa).
    The format is as follows (in .csv file):
    kcat,MW,kcat_MW
    AADDGT,49389.2889,40.6396,1215.299582180927
    GALCTND,75632.85923999999,42.5229,1778.63831582512
    ...
    
    * lb_list: Lower bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * ub_list: Upper bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * obj_name: Object reaction ID.
    * obj_target: Type of object function (maximize or minimize).  
    * set_obj_single_E_value: The object function is the enzyme cost of a reaction (True or False).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
lowerGet_Concretemodel_Need_Data    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    * set_metabolite_ratio: Adding concentration ratio constraints for metabolites (True or False).
    * set_bound: Set the upper and lower bounds of reaction flux (True or False).
    * set_stoi_matrix: Adding flux balance constraints (True or False).
    * set_substrate_ini: Adding substrate amount constraints (True or False).
    * K_value: the maximum value minus the minimum value of reaction thermodynamic driving force (1249, maxDFi-minDFi).
    * substrate_name: Substrate input reaction ID in the model (such as EX_glc__D_e_reverse).
    * substrate_value: Set the upper bound for substrate input reaction flux (10 mM).
    * set_thermodynamics: Adding thermodynamic constraints (True or False).
    * B_value: The value of maximizing the minimum thermodynamic driving force (MDF).
    * set_biomass_ini: Set the lower bound for biomass synthesis reaction flux (True or False).
    * biomass_id: Biomass synthesis reaction ID in the model (BIOMASS_Ec_iML1515_core_75p37M).
    * biomass_value: The lower bound of biomass synthesis reaction flux.
    * set_integer: Adding binary variables constraints (True or False).
    * set_enzyme_constraint: Adding enzymamic constraints (True or False).
    * E_total: Total amount constraint of enzymes (0.114).
    """  

    max_min_Df_list=pd.DataFrame()
    Concretemodel.write('result_max_min_e0.lp',io_options={'symbolic_solver_labels': True})
    opt=Model_Solve(Concretemodel,solver)
    if obj_target=='maximize':
        max_min_Df_list.loc[obj_name,'max_value']=opt.obj() 
    elif obj_target=='minimize':
        max_min_Df_list.loc[obj_name,'min_value']=opt.obj() 
    # print(max_min_Df_list)
    return max_min_Df_list

#Determination of key enzymes by analysing the enzyme cost variability
def Get_Max_Min_E(Concretemodel_Need_Data,obj_name,obj_target,K_value,B_value,max_biomass_under_mdf,biomass_id,E_total,substrate_name,substrate_value,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    Concretemodel = Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,\
        ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_single_E_value=True,set_metabolite=True,set_Df=True,set_metabolite_ratio=True,set_bound=True,\
        set_stoi_matrix=True,set_substrate_ini=True,K_value=K_value,substrate_name=substrate_name,substrate_value=substrate_value,set_thermodynamics=True,B_value=B_value,\
        set_biomass_ini=True,biomass_id=biomass_id,biomass_value=max_biomass_under_mdf,set_integer=True,set_enzyme_constraint=True,E_total=E_total)
    """Calculation of the maximum and minimum enzyme cost for reactions in a specific list.

    Notes:
    ----------
    * reaction_list: List of reaction IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.

    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolites_lnC: Upper and lower bound information of metabolite concentration (Natural logarithm).
    The format is as follows (in .txt file):
    id	lnClb	lnCub
    2pg_c	-14.50865774	-3.912023005
    13dpg_c	-14.50865774	-3.912023005
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...
    
    * reaction_kcat_MW: The enzymatic data of kcat (turnover number, /h) divided by MW (molecular weight, kDa).
    The format is as follows (in .csv file):
    kcat,MW,kcat_MW
    AADDGT,49389.2889,40.6396,1215.299582180927
    GALCTND,75632.85923999999,42.5229,1778.63831582512
    ...
    
    * lb_list: Lower bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * ub_list: Upper bound of reaction flux. It is obtained by analyzing SBML model with get_data(model) function.
    * obj_name: Object reaction ID.
    * obj_target: Type of object function (maximize or minimize).  
    * set_obj_single_E_value: The object function is the enzyme cost of a reaction (True or False).
    * set_metabolite: Set the upper and lower bounds of metabolite concentration (True or False).
    * set_Df: Adding thermodynamic driving force expression for reactions (True or False).
    * set_metabolite_ratio: Adding concentration ratio constraints for metabolites (True or False).
    * set_bound: Set the upper and lower bounds of reaction flux (True or False).
    * set_stoi_matrix: Adding flux balance constraints (True or False).
    * set_substrate_ini: Adding substrate amount constraints (True or False).
    * K_value: the maximum value minus the minimum value of reaction thermodynamic driving force (1249, maxDFi-minDFi).
    * substrate_name: Substrate input reaction ID in the model (such as EX_glc__D_e_reverse).
    * substrate_value: Set the upper bound for substrate input reaction flux (10 mM).
    * set_thermodynamics: Adding thermodynamic constraints (True or False).
    * B_value: The value of maximizing the minimum thermodynamic driving force (MDF).
    * set_biomass_ini: Set the lower bound for biomass synthesis reaction flux (True or False).
    * biomass_id: Biomass synthesis reaction ID in the model (BIOMASS_Ec_iML1515_core_75p37M).
    * biomass_value: The lower bound of biomass synthesis reaction flux.
    * set_integer: Adding binary variables constraints (True or False).
    * set_enzyme_constraint: Adding enzymamic constraints (True or False).
    * E_total: Total amount constraint of enzymes (0.114).
    """  

    max_min_Df_list=pd.DataFrame()
    opt=Model_Solve(Concretemodel,solver)
    if obj_target=='maximize':
        max_min_Df_list.loc[obj_name,'max_value']=opt.obj() 
    elif obj_target=='minimize':
        max_min_Df_list.loc[obj_name,'min_value']=opt.obj() 

    return max_min_Df_list

#Solving maximum growth by different models
def Max_OBJ_By_Four_Model(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,K_value,B_value,E_total,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value

    biomass_list=pd.DataFrame()
    ub_list[substrate_name]=substrate_value
    EcoGEM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        lb_list=lb_list,ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value,set_stoi_matrix=True,set_bound=True)
    opt=Model_Solve(EcoGEM,solver)
    biomass_list.loc[substrate_value,'iML1515']=opt.obj()

    EcoECM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_value=True,\
        set_substrate_ini=True,substrate_name=substrate_name,substrate_value=substrate_value,set_stoi_matrix=True,set_bound=True,\
        set_enzyme_constraint=True,E_total=E_total)
    opt=Model_Solve(EcoECM,solver)
    biomass_list.loc[substrate_value,'EcoECM']=opt.obj()

    EcoTCM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,lb_list=lb_list,ub_list=ub_list,\
        obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value,set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
        set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,K_value=K_value,B_value=B_value)
    opt=Model_Solve(EcoTCM,solver)
    biomass_list.loc[substrate_value,'EcoTCM(Dfi>=0)']=opt.obj()

    EcoETM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
        set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,K_value=K_value,B_value=B_value,\
        set_enzyme_constraint=True,E_total=E_total)
    opt=Model_Solve(EcoETM,solver)
    biomass_list.loc[substrate_value,'EcoETM']=opt.obj()

    return biomass_list,EcoGEM,EcoECM,EcoTCM,EcoETM

#Solving MDF value under preset growth rate
def Max_MDF_By_model(Concretemodel_Need_Data,substrate_name,substrate_value,biomass_value,biomass_id,K_value,E_total,obj_enz_constraint,obj_no_enz_constraint,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=Concretemodel_Need_Data['ub_list']
    ub_list[substrate_name]=substrate_value
    MDF_list=pd.DataFrame()

    if biomass_value<=obj_no_enz_constraint:
        EcoTCM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
            metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,lb_list=lb_list,ub_list=ub_list,\
            K_value=K_value,set_substrate_ini=True,substrate_name=substrate_name,substrate_value=substrate_value,set_biomass_ini=True,\
            biomass_value=biomass_value,biomass_id=biomass_id,set_metabolite=True,set_Df=True,set_obj_B_value=True,set_stoi_matrix=True,\
            set_bound=True,set_integer=True,set_metabolite_ratio=True)

        opt=Model_Solve(EcoTCM,solver)
        MDF_list.loc[biomass_value,'EcoTCM']=opt.obj()
    else:
        MDF_list.loc[biomass_value,'EcoTCM']=None

    if biomass_value<=obj_enz_constraint:
        EcoETM=Template_Concretemodel(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
            metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
            K_value=K_value,set_substrate_ini=True,substrate_name=substrate_name,substrate_value=substrate_value,set_biomass_ini=True,\
            biomass_value=biomass_value,biomass_id=biomass_id,set_metabolite=True,set_Df=True,set_obj_B_value=True,set_stoi_matrix=True,\
            set_bound=True,set_enzyme_constraint=True,E_total=E_total,set_integer=True,set_metabolite_ratio=True)

        opt=Model_Solve(EcoETM,solver)
        MDF_list.loc[biomass_value,'EcoETM']=opt.obj()
    else:
        MDF_list.loc[biomass_value,'EcoETM']=None
        
    return MDF_list

def Get_Results_Thermodynamics(model,Concretemodel,reaction_kcat_MW,reaction_g0,coef_matrix,metabolite_list):
    """The formatting of the calculated results, includes the metabolic flux, binary variable values, thermodynamic driving force of reactions, the enzyme amount and the metabolite concentrations. The value of "-9999" means that the missing of kinetic (kcat) or thermodynamickcat (drG'°) parameters.
    
    Notes:
    ----------
    * model: is in SBML format (.xml).
    * Concretemodel: Pyomo Concretemodel.
    
    * reaction_kcat_MW: The enzymatic data of kcat (turnover number, /h) divided by MW (molecular weight, kDa).
    The format is as follows (in .csv file):
    kcat,MW,kcat_MW
    AADDGT,49389.2889,40.6396,1215.299582180927
    GALCTND,75632.85923999999,42.5229,1778.63831582512
    ...
    
    * reaction_g0: Thermodynamic parameter of reactions (drG'°) .
    The format is as follows (in .txt file):
    reaction	g0
    13PPDH2	-21.3
    13PPDH2_reverse	21.3
    ...
    
    * coef_matrix: The model coefficient matrix.
    It is obtained by analyzing SBML model with get_data(model) function.
    
    * metabolite_list: List of metabolite IDs for the model.
    It is obtained by analyzing SBML model with get_data(model) function.
    """
    result_dataframe = pd.DataFrame()
    for eachreaction in Concretemodel.reaction:
        flux=Concretemodel.reaction[eachreaction].value
        z=Concretemodel.z[eachreaction].value
        result_dataframe.loc[eachreaction,'flux']=flux
        result_dataframe.loc[eachreaction,'z']=z  
        if eachreaction in reaction_g0.index:
            result_dataframe.loc[eachreaction,'f']=-reaction_g0.loc[eachreaction,'g0']-2.579*sum(coef_matrix[i,eachreaction]*Concretemodel.metabolite[i].value  for i in metabolite_list if (i,eachreaction) in coef_matrix.keys())
        else:
            result_dataframe.loc[eachreaction,'f']=-9999
        if eachreaction in reaction_kcat_MW.index:
            result_dataframe.loc[eachreaction,'enz']= flux/(reaction_kcat_MW.loc[eachreaction,'kcat_MW'])
        else:
            result_dataframe.loc[eachreaction,'enz']= -9999 
            
        tmp=model.reactions.get_by_id(eachreaction)
        met_list=''
        for met in tmp.metabolites:    
            met_list=met_list+';'+str(met.id)+' : '+str(np.exp(Concretemodel.metabolite[met.id].value))
        result_dataframe.loc[eachreaction,'met_concentration']= met_list  
        
    return(result_dataframe)

#Visualization of calculation results
def Draw_Biomass_By_Glucose_rate(Biomass_list,save_file):
    plt.figure(figsize=(15, 10), dpi=300)

    plt.plot(Biomass_list.index, Biomass_list[Biomass_list.columns[0]], color="black", linewidth=3.0, linestyle="--", label=Biomass_list.columns[0])
    plt.plot(Biomass_list.index, Biomass_list[Biomass_list.columns[1]], color="red", linewidth=3.0, linestyle="-", label=Biomass_list.columns[1])
    plt.plot(Biomass_list.index, Biomass_list[Biomass_list.columns[2]], color="cyan", linewidth=3.0, linestyle="-", label=Biomass_list.columns[2])
    plt.plot(Biomass_list.index, Biomass_list[Biomass_list.columns[3]], color="darkorange", linewidth=3.0, linestyle="-", label=Biomass_list.columns[3])

    font1 = {
    'weight' : 'normal',
    'size'   : 20,
    }

    plt.legend(loc="upper left",prop=font1)

    plt.xlim(0, 15)
    plt.ylim(0, 1.4)

    plt.tick_params(labelsize=23)
    plt.xticks([0, 1, 2,3, 4,5, 6, 7, 8, 9, 10, 11, 12, 13,14,15])
    plt.yticks([0.2, 0.4, 0.6,0.8, 1.0,1.2, 1.4])

    font2 = {
    'weight' : 'normal',
    'size'   : 25,
    }
    plt.xlabel("Glucose uptake rate (mmol/gDW/h)",font2)
    plt.ylabel("Growth rate ($\mathregular{h^-1}$)",font2)
    plt.savefig(save_file)
    plt.show()

def Draw_MDF_By_Growth_rate(MDF_list,save_file):
    plt.figure(figsize=(15, 10), dpi=300)
    MDF_list=MDF_list.sort_index(ascending=True) 
    plt.plot(MDF_list.index, MDF_list[MDF_list.columns[0]], color="cyan", linewidth=3.0, linestyle="-", label=MDF_list.columns[0])
    plt.plot(MDF_list.index, MDF_list[MDF_list.columns[1]], color="darkorange", linewidth=3.0, linestyle="-", label=MDF_list.columns[1])
    font1 = {
    'weight' : 'normal',
    'size'   : 23,
    }

    font2 = {
    'weight' : 'normal',
    'size'   : 30,
    }

    plt.ylabel("MDF of pathways (kJ/mol)",font2)
    plt.xlabel("Growth rate ($\mathregular{h^-1}$)",font2)

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_position(('data', 0.3))
    ax.spines['bottom'].set_position(('data', 0))
    plt.legend(loc="lower left",prop=font1)
    plt.xlim(0.3, 0.9)
    plt.ylim(-26, 3)

    plt.tick_params(labelsize=23)

    plt.xticks([0.3, 0.4, 0.5,0.6, 0.7,0.8, 0.9])
    plt.yticks([-26, -22, -18,-14, -10,-6, -2,2])

    #plt.scatter([0.633], [2.6670879363966336], s=80, color="red")
    #plt.scatter([0.6756], [-0.48186643213771774], s=80, color="red")
    #plt.scatter([0.7068], [-9.486379882991386], s=80, color="red")
    #plt.scatter([0.852], [2.6670879363966336], s=80, color="red")
    #plt.scatter([0.855], [1.4290141211096987], s=80, color="red")
    #plt.scatter([0.867], [0.06949515162540898], s=80, color="red")
    #plt.scatter([0.872], [-0.8364187795859692], s=80, color="red")
    #plt.scatter([0.876], [-9.486379882991372], s=80, color="red")

    plt.savefig(save_file)
    plt.show()

def get_recation_g0(model,p_h,p_mg,ionic_strength,temperature):
    #get ΔG'° use equilibrator_api
    cc = None
    while cc is None:
        try:
            cc = ComponentContribution()
        except JSONDecodeError:
            logger.warning('Waiting for zenodo.org... Retrying in 5s')
            sleep(5)

    #cc = ComponentContribution()
    cc.p_h = Q_(p_h)
    cc.p_mg = Q_(p_mg)
    cc.ionic_strength = Q_(ionic_strength)
    cc.temperature = Q_(temperature)
    
    reaction_g0={}
    for eachr in model.reactions:
        if 'EX_' not in eachr.id:
            reaction_left=[]
            reaction_right=[]
            for k, v in eachr.metabolites.items():
                if str(k).endswith('_c'):
                    k_new = "_c".join(str(k).split('_c')[:-1])
                elif str(k).endswith('_p'):
                    k_new = "_p".join(str(k).split('_p')[:-1])
                elif str(k).endswith('_e'):
                    k_new = "_e".join(str(k).split('_e')[:-1]) 
                    
                #kegg,chebi,metanetx;kegg:C00002 + CHEBI:15377 = metanetx.chemical:MNXM7 + bigg.metabolite:pi
                if v<0:                    
                    reaction_left.append(str(-v)+' bigg.metabolite:'+k_new)
                else:
                    reaction_right.append(str(v)+' bigg.metabolite:'+k_new)
            reaction_equ=(' + ').join(reaction_left)+' -> '+(' + ').join(reaction_right)
            #print(reaction_equ)
            #get ΔG'° use equilibrator_api
            try:
                equilibrator_api_reaction = cc.parse_reaction_formula(reaction_equ)
                #print(reaction_equ)
                #print("The reaction is " + ("" if equilibrator_api_reaction.is_balanced() else "not ") + "balanced")
                dG0_prime = cc.standard_dg_prime(equilibrator_api_reaction)
            except:
                #pass
                print('error reaction: '+eachr.id)
                print(reaction_equ)
            else:  
                reaction_g0[eachr.id]={}
                reaction_g0[eachr.id]['reaction']=eachr.id
                reaction_g0[eachr.id]['equ']=eachr.reaction
                #dG0 (kilojoule/mole)
                reaction_g0[eachr.id]['g0']=f"{dG0_prime}".split(') ')[0].split('(')[1].split(' +/- ')[0]
              
    return reaction_g0

def get_recation_g0_local(model,p_h,p_mg,ionic_strength,temperature):
    #get ΔG'° use equilibrator_api
    lc = LocalCompoundCache()
    cc = ComponentContribution(ccache = lc.ccache)
    cc.p_h = Q_(p_h)
    cc.p_mg = Q_(p_mg)
    cc.ionic_strength = Q_(ionic_strength)
    cc.temperature = Q_(temperature)
    
    reaction_g0={}
    for eachr in model.reactions:
        if 'EX_' not in eachr.id:
            reaction_left=[]
            reaction_right=[]
            for k, v in eachr.metabolites.items():
                if str(k).endswith('_c'):
                    k_new = "_c".join(str(k).split('_c')[:-1])
                elif str(k).endswith('_p'):
                    k_new = "_p".join(str(k).split('_p')[:-1])
                elif str(k).endswith('_e'):
                    k_new = "_e".join(str(k).split('_e')[:-1]) 
                    
                #kegg,chebi,metanetx;kegg:C00002 + CHEBI:15377 = metanetx.chemical:MNXM7 + bigg.metabolite:pi
                if v<0:                    
                    reaction_left.append(str(-v)+' bigg.metabolite:'+k_new)
                else:
                    reaction_right.append(str(v)+' bigg.metabolite:'+k_new)
            if (' + ').join(reaction_left) !=(' + ').join(reaction_right):
                reaction_equ=(' + ').join(reaction_left)+' -> '+(' + ').join(reaction_right)
                #print(reaction_equ)
                #get ΔG'° use equilibrator_api
                try:
                    equilibrator_api_reaction = cc.parse_reaction_formula(reaction_equ)
                    #print("The reaction is " + ("" if equilibrator_api_reaction.is_balanced() else "not ") + "balanced")
                    dG0_prime = cc.standard_dg_prime(equilibrator_api_reaction)
                except:
                    #pass
                    print('error reaction: '+eachr.id)
                else:  
                    reaction_g0[eachr.id]={}
                    reaction_g0[eachr.id]['reaction']=eachr.id
                    reaction_g0[eachr.id]['equ']=reaction_equ
                    reaction_g0[eachr.id]['dG0 (kilojoule/mole)']=f"{dG0_prime}".split(') ')[0].split('(')[1]
                    #dG0 (kilojoule/mole)
                    reaction_g0[eachr.id]['g0']=f"{dG0_prime}".split(') ')[0].split('(')[1].split(' +/- ')[0]
    return reaction_g0

#将拆分后模型的通量整合为拆分前模型的通量, solved_Concretemodel:求解后的模型，modelid：拆分前的模型
def conbine_flux(solved_Concretemodel,modelid):   
    #get flux of stoichiometry from E_protain_FBA

    flux_efba={ i: value(solved_Concretemodel.reaction[i])   for i in solved_Concretemodel.reaction}
    reaction_list=[]
    for rea in modelid.reactions:
        reaction_list.append(rea.id)
    fluxnew={}
    for i in reaction_list:
        fluxnew[i]=0
        for j in flux_efba.keys():
            if i == j:
                fluxnew[i]=fluxnew[i]+flux_efba[i]
            if i+'_num' ==j[:-1]:
                fluxnew[i]=fluxnew[i]+flux_efba[j]
            if i +'_reverse'  == j or  i+'_reverse_num' == j[:-1]:
                fluxnew[i]=fluxnew[i]-flux_efba[j]
    return fluxnew
def conbine_e(solved_Concretemodel,modelid):   
    #get flux of stoichiometry from E_protain_FBA

    e_efba={ i: value(solved_Concretemodel.e[i])   for i in solved_Concretemodel.e}
    reaction_list=[]
    for rea in modelid.reactions:
        reaction_list.append(rea.id)
    fluxnew={}
    for i in reaction_list:
        fluxnew[i]=0
        for j in e_efba.keys():
            if i == j:
                fluxnew[i]=fluxnew[i]+e_efba[i]
            if i+'_num' ==j[:-1]:
                fluxnew[i]=fluxnew[i]+e_efba[j]
            if i +'_reverse'  == j or  i+'_reverse_num' == j[:-1]:
                fluxnew[i]=fluxnew[i]+e_efba[j]
    return fluxnew
def conbine_G0(solved_Concretemodel,modelid):   
    #get flux of stoichiometry from E_protain_FBA

    g0_efba={ i: value(solved_Concretemodel.Df[i])   for i in solved_Concretemodel.Df}
    reaction_list=[]
    for rea in modelid.reactions:
        reaction_list.append(rea.id)
    fluxnew={}
    for i in reaction_list:
        if i in g0_efba:
            fluxnew[i]=g0_efba[i]
        if i+'_num1' in g0_efba:
            fluxnew[i]=g0_efba[i+'_num1']
    return fluxnew
def Get_Concretemodel_Need_Data_g0(Concretemodel_Need_Data,reaction_g0_file,metabolites_lnC_file,reaction_kcat_MW_file):
    reaction_g0=pd.read_csv(reaction_g0_file,index_col=0,sep='\t')
    Concretemodel_Need_Data['reaction_g0']=reaction_g0
    metabolites_lnC = pd.read_csv(metabolites_lnC_file, index_col=0,sep='\t')
    Concretemodel_Need_Data['metabolites_lnC']=metabolites_lnC
    reaction_kcat_MW=pd.read_csv(reaction_kcat_MW_file,index_col=0)
    Concretemodel_Need_Data['reaction_kcat_MW']=reaction_kcat_MW

def EGFBA(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,K_value,B_value,E_total,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=copy.deepcopy(Concretemodel_Need_Data['ub_list'])
    ub_list[substrate_name]=substrate_value
    kcat_dict=Concretemodel_Need_Data['kcat_dict']
    mw_dict=Concretemodel_Need_Data['mw_dict']
    enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
    e0=Concretemodel_Need_Data['e0']
    totle_E=Concretemodel_Need_Data['totle_E']

    EcoETM=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,enzyme_rxns_dict=enzyme_rxns_dict,\
        totle_E=totle_E,e0=e0,metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
        set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,K_value=K_value,B_value=B_value,\
        set_enzyme_constraint=True,set_enzyme_value=True,E_total=E_total,kcat_dict=kcat_dict,mw_dict=mw_dict)
    # #EcoETM=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
    #     metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
    #     obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
    #     substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
    #     set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,K_value=K_value,B_value=B_value,\
    #     set_enzyme_constraint=True,E_total=E_total)
    Model_Solve(EcoETM,solver)
    # EcoETM.write('./results/GEFBA.lp',io_options={'symbolic_solver_labels': True})

    return EcoETM

# e'k
def EGFBA_wild(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,K_value,B_value,E_total,solver,v0_biomass,biomass_name):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=copy.deepcopy(Concretemodel_Need_Data['ub_list'])
    ub_list[substrate_name]=substrate_value
    kcat_dict=Concretemodel_Need_Data['kcat_dict']
    mw_dict=Concretemodel_Need_Data['mw_dict']
    enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
    e0=Concretemodel_Need_Data['e0']
    totle_E=Concretemodel_Need_Data['totle_E']


    EcoETM=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,enzyme_rxns_dict=enzyme_rxns_dict,\
        totle_E=totle_E,e0=e0,metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        obj_name=obj_name,obj_target=obj_target,set_obj_value_e=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
        set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,set_constr9=True,K_value=K_value,B_value=B_value,\
        set_enzyme_constraint=True,set_enzyme_value=True,E_total=E_total,kcat_dict=kcat_dict,mw_dict=mw_dict,v0_biomass=v0_biomass,biomass_name=biomass_name)
    EcoETM.write('./lpfiles/GEFBA.lp',io_options={'symbolic_solver_labels': True})

    Model_Solve(EcoETM,solver)
    #EcoETM.write('./results/GEFBA.lp',io_options={'symbolic_solver_labels': True})

    return EcoETM


# 求野生型通量分布
def EGFBA_wild_flux(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,K_value,B_value,E_total,solver,v0_biomass,biomass_name):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=copy.deepcopy(Concretemodel_Need_Data['ub_list'])
    ub_list[substrate_name]=substrate_value
    kcat_dict=Concretemodel_Need_Data['kcat_dict']
    mw_dict=Concretemodel_Need_Data['mw_dict']
    enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
    e0=Concretemodel_Need_Data['e0']
    totle_E=Concretemodel_Need_Data['totle_E']


    EcoETM=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,enzyme_rxns_dict=enzyme_rxns_dict,\
        totle_E=totle_E,e0=e0,metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
        set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,set_constr9=True,K_value=K_value,B_value=B_value,\
        set_enzyme_constraint=True,set_enzyme_value=True,E_total=E_total,kcat_dict=kcat_dict,mw_dict=mw_dict,v0_biomass=v0_biomass,biomass_name=biomass_name)


    Model_Solve(EcoETM,solver)
    # EcoETM.write('./results/GEFBA.lp',io_options={'symbolic_solver_labels': True})

    return EcoETM

# 求v_product_max
def EGFBA_product(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,K_value,B_value,E_total,solver,v0_biomass,biomass_name):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=copy.deepcopy(Concretemodel_Need_Data['ub_list'])
    ub_list[substrate_name]=substrate_value
    kcat_dict=Concretemodel_Need_Data['kcat_dict']
    mw_dict=Concretemodel_Need_Data['mw_dict']
    enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
    e0=Concretemodel_Need_Data['e0']
    totle_E=Concretemodel_Need_Data['totle_E']

    EcoETM=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,enzyme_rxns_dict=enzyme_rxns_dict,\
        totle_E=totle_E,e0=e0,metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
        set_Df=True,set_integer=True,set_metabolite_ratio=False,set_thermodynamics=True,set_constr10=True,K_value=K_value,B_value=B_value,\
        set_enzyme_constraint=True,set_enzyme_value=True,E_total=E_total,kcat_dict=kcat_dict,mw_dict=mw_dict,v0_biomass=v0_biomass,biomass_name=biomass_name)
    # #EcoETM=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
    #     metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
    #     obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
    #     substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
    #     set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,K_value=K_value,B_value=B_value,\
    #     set_enzyme_constraint=True,E_total=E_total)
    Model_Solve(EcoETM,solver)
    # EcoETM.write('./results/GEFBA.lp',io_options={'symbolic_solver_labels': True})

    return EcoETM


# 求ek
def EGFBA_over(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,K_value,B_value,E_total,solver,v0_biomass,biomass_name,v1_product_max,product_name):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=copy.deepcopy(Concretemodel_Need_Data['ub_list'])
    ub_list[substrate_name]=substrate_value
    kcat_dict=Concretemodel_Need_Data['kcat_dict']
    mw_dict=Concretemodel_Need_Data['mw_dict']
    enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
    e0=Concretemodel_Need_Data['e0']
    totle_E=Concretemodel_Need_Data['totle_E']

    EcoETM=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,enzyme_rxns_dict=enzyme_rxns_dict,\
        totle_E=totle_E,e0=e0,metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        obj_name=obj_name,obj_target=obj_target,set_obj_value_e=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
        set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,set_constr10=True,set_constr11=True,K_value=K_value,B_value=B_value,\
        set_enzyme_constraint=True,set_enzyme_value=True,E_total=E_total,kcat_dict=kcat_dict,mw_dict=mw_dict,v0_biomass=v0_biomass,biomass_name=biomass_name,\
        v1_product_max=v1_product_max,product_name=product_name)
    # #EcoETM=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
    #     metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
    #     obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
    #     substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
    #     set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,K_value=K_value,B_value=B_value,\
    #     set_enzyme_constraint=True,E_total=E_total)
    Model_Solve(EcoETM,solver)
    # EcoETM.write('./results/GEFBA.lp',io_options={'symbolic_solver_labels': True}) 
    #
    return EcoETM



# 求通量分布
def EGFBA_over_flux(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,K_value,B_value,E_total,solver,v0_biomass,biomass_name,v1_product_max,product_name):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=copy.deepcopy(Concretemodel_Need_Data['ub_list'])
    ub_list[substrate_name]=substrate_value
    kcat_dict=Concretemodel_Need_Data['kcat_dict']
    mw_dict=Concretemodel_Need_Data['mw_dict']
    enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
    e0=Concretemodel_Need_Data['e0']
    totle_E=Concretemodel_Need_Data['totle_E']

    EcoETM=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,enzyme_rxns_dict=enzyme_rxns_dict,\
        totle_E=totle_E,e0=e0,metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
        set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,set_constr10=True,set_constr11=True,K_value=K_value,B_value=B_value,\
        set_enzyme_constraint=True,set_enzyme_value=True,E_total=E_total,kcat_dict=kcat_dict,mw_dict=mw_dict,v0_biomass=v0_biomass,biomass_name=biomass_name,v1_product_max=v1_product_max,product_name=product_name)
    # #EcoETM=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
    #     metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
    #     obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
    #     substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
    #     set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,K_value=K_value,B_value=B_value,\
    #     set_enzyme_constraint=True,E_total=E_total)
    Model_Solve(EcoETM,solver)
    # EcoETM.write('./results/GEFBA.lp',io_options={'symbolic_solver_labels': True}) 
    return EcoETM    
def EGFBA_over_flux2(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,K_value,B_value,E_total,solver,v0_biomass,biomass_name,v1_product_max,product_name):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=copy.deepcopy(Concretemodel_Need_Data['ub_list'])
    ub_list[substrate_name]=substrate_value
    kcat_dict=Concretemodel_Need_Data['kcat_dict']
    mw_dict=Concretemodel_Need_Data['mw_dict']
    enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
    e0=Concretemodel_Need_Data['e0']
    totle_E=Concretemodel_Need_Data['totle_E']

    EcoETM=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,enzyme_rxns_dict=enzyme_rxns_dict,\
        totle_E=totle_E,e0=e0,metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
        set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,set_constr10=True,set_constr13=True,K_value=K_value,B_value=B_value,\
        set_enzyme_constraint=True,set_enzyme_value=True,E_total=E_total,kcat_dict=kcat_dict,mw_dict=mw_dict,v0_biomass=v0_biomass,biomass_name=biomass_name,v1_product_max=v1_product_max,product_name=product_name)
    # #EcoETM=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
    #     metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
    #     obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
    #     substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
    #     set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,K_value=K_value,B_value=B_value,\
    #     set_enzyme_constraint=True,E_total=E_total)
    Model_Solve(EcoETM,solver)
    # EcoETM.write('./results/GEFBA.lp',io_options={'symbolic_solver_labels': True}) 
    return EcoETM    

def SUM_E_GEFBA(biomass_value,Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,K_value,B_value,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=copy.deepcopy(Concretemodel_Need_Data['lb_list'])
    ub_list=copy.deepcopy(Concretemodel_Need_Data['ub_list'])
    lb_list[obj_name]=biomass_value
    ub_list[obj_name]=biomass_value
    ub_list[substrate_name]=substrate_value
    kcat_dict=Concretemodel_Need_Data['kcat_dict']
    mw_dict=Concretemodel_Need_Data['mw_dict']
    enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
    e0=Concretemodel_Need_Data['e0']
    totle_E=Concretemodel_Need_Data['totle_E']

    EcoETM0=FBA_template(set_obj_sum_e=True,reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,enzyme_rxns_dict=enzyme_rxns_dict,\
        totle_E=totle_E,e0=e0,metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        obj_name=obj_name,obj_target=obj_target,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
        set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,K_value=K_value,B_value=B_value,\
        set_enzyme_constraint=True,set_enzyme_value=True,kcat_dict=kcat_dict,mw_dict=mw_dict)
    # #EcoETM=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
    #     metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
    #     obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
    #     substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
    #     set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,K_value=K_value,B_value=B_value,\
    #     set_enzyme_constraint=True,E_total=E_total)
    Model_Solve(EcoETM0,solver)
#     #EcoETM.write('./results/GEFBA.lp',io_options={'symbolic_solver_labels': True})

#     return EcoETM0
def SUM_E_GEFBA_m(Concretemodel_Need_Data,obj_target,substrate_name,substrate_value,K_value,B_value,solver,v0_biomass,biomass_name):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=copy.deepcopy(Concretemodel_Need_Data['ub_list'])
    ub_list[substrate_name]=substrate_value
    kcat_dict=Concretemodel_Need_Data['kcat_dict']
    mw_dict=Concretemodel_Need_Data['mw_dict']
    enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
    e0=Concretemodel_Need_Data['e0']
    totle_E=Concretemodel_Need_Data['totle_E']


    EcoETM=FBA_template(set_obj_sum_e=True,reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,enzyme_rxns_dict=enzyme_rxns_dict,\
        totle_E=totle_E,e0=e0,metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        obj_target=obj_target,set_obj_value_e=False,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
        set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,set_constr9=True,K_value=K_value,B_value=B_value,\
        set_enzyme_constraint=True,set_enzyme_value=True,kcat_dict=kcat_dict,mw_dict=mw_dict,v0_biomass=v0_biomass,biomass_name=biomass_name)
    # EcoETM.write('./lpfiles/GEFBA.lp',io_options={'symbolic_solver_labels': True})

    Model_Solve(EcoETM,solver)
    #EcoETM.write('./results/GEFBA.lp',io_options={'symbolic_solver_labels': True})

    return EcoETM

def SUM_E_GEFBA_V(Concretemodel_Need_Data,obj_target,substrate_name,substrate_value,K_value,B_value,solver,v0_biomass,biomass_name,v1_product_max,product_name):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    reaction_kcat_MW=Concretemodel_Need_Data['reaction_kcat_MW']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=copy.deepcopy(Concretemodel_Need_Data['ub_list'])
    ub_list[substrate_name]=substrate_value
    kcat_dict=Concretemodel_Need_Data['kcat_dict']
    mw_dict=Concretemodel_Need_Data['mw_dict']
    enzyme_rxns_dict=Concretemodel_Need_Data['enzyme_rxns_dict']
    e0=Concretemodel_Need_Data['e0']
    totle_E=Concretemodel_Need_Data['totle_E']

    EcoETM=FBA_template(set_obj_sum_e=True,reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,enzyme_rxns_dict=enzyme_rxns_dict,\
        totle_E=totle_E,e0=e0,metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
        obj_target=obj_target,set_obj_value_e=False,set_substrate_ini=True,\
        substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
        set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,set_constr10=True,set_constr11=True,K_value=K_value,B_value=B_value,\
        set_enzyme_constraint=True,set_enzyme_value=True,kcat_dict=kcat_dict,mw_dict=mw_dict,v0_biomass=v0_biomass,biomass_name=biomass_name,\
        v1_product_max=v1_product_max,product_name=product_name)
    # #EcoETM=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
    #     metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,reaction_kcat_MW=reaction_kcat_MW,lb_list=lb_list,ub_list=ub_list,\
    #     obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,\
    #     substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True,\
    #     set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,K_value=K_value,B_value=B_value,\
    #     set_enzyme_constraint=True,E_total=E_total)
    Model_Solve(EcoETM,solver)
    #EcoETM.write('./results/GEFBA.lp',io_options={'symbolic_solver_labels': True})

    return EcoETM

def showflux(ECM_EGFBA):
    flux_positive={i:"{:.3e}".format(value(ECM_EGFBA.reaction[i])) for i in ECM_EGFBA.reaction if value(ECM_EGFBA.reaction[i]) >0 and ('EX_' in i or 'DM_' in i)}
    return flux_positive


def TGFBA(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,K_value,B_value,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=copy.deepcopy(Concretemodel_Need_Data['ub_list'])
    ub_list[substrate_name]=substrate_value
 
    EcoTCM=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,\
        lb_list=lb_list,ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,substrate_name=substrate_name,\
        substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True, set_Df=True,set_integer=True,set_metabolite_ratio=True,\
        set_thermodynamics=True,K_value=K_value,B_value=B_value)
    
    Model_Solve(EcoTCM,solver)
    #EcoETM.write('./results/GEFBA.lp',io_options={'symbolic_solver_labels': True})

    return EcoTCM

# e'k
def TGFBA_wild(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,K_value,B_value,solver,v0_biomass,biomass_name):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=copy.deepcopy(Concretemodel_Need_Data['ub_list'])
    ub_list[substrate_name]=substrate_value


    EcoTCM=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,\
        lb_list=lb_list,ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True, set_constr9=True,v0_biomass=v0_biomass,\
        biomass_name=biomass_name,substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True, \
        set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,K_value=K_value,B_value=B_value)
    
    Model_Solve(EcoTCM,solver)
    #EcoETM.write('./results/GEFBA.lp',io_options={'symbolic_solver_labels': True})

    return EcoTCM


def TGFBA_product(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,K_value,B_value,solver,v0_biomass,biomass_name):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=copy.deepcopy(Concretemodel_Need_Data['ub_list'])
    ub_list[substrate_name]=substrate_value

    EcoTCM=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,\
            lb_list=lb_list,ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True, set_constr10=True,v0_biomass=v0_biomass,\
            biomass_name=biomass_name,substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True, \
            set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,K_value=K_value,B_value=B_value)
        
    Model_Solve(EcoTCM,solver)
    #EcoETM.write('./results/GEFBA.lp',io_options={'symbolic_solver_labels': True})

    return EcoTCM

def TGFBA_over(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,K_value,B_value,solver,v0_biomass,biomass_name,v1_product_max,product_name):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    reaction_g0=Concretemodel_Need_Data['reaction_g0']
    metabolites_lnC=Concretemodel_Need_Data['metabolites_lnC']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=copy.deepcopy(Concretemodel_Need_Data['ub_list'])
    ub_list[substrate_name]=substrate_value


    EcoTCM=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,metabolites_lnC=metabolites_lnC,reaction_g0=reaction_g0,\
        lb_list=lb_list,ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True, set_constr10=True,v0_biomass=v0_biomass,\
        biomass_name=biomass_name,set_constr11=True,v1_product_max=v1_product_max,product_name=product_name,substrate_name=substrate_name,substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_metabolite=True, \
        set_Df=True,set_integer=True,set_metabolite_ratio=True,set_thermodynamics=True,K_value=K_value,B_value=B_value)
    
    Model_Solve(EcoTCM,solver)
    #EcoETM.write('./results/GEFBA.lp',io_options={'symbolic_solver_labels': True})

    return EcoTCM


def GFBA(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=copy.deepcopy(Concretemodel_Need_Data['ub_list'])
    ub_list[substrate_name]=substrate_value
 
    EcoGEM=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,\
        lb_list=lb_list,ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,substrate_name=substrate_name,\
        substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True)
    
    Model_Solve(EcoGEM,solver)
    #EcoETM.write('./results/GEFBA.lp',io_options={'symbolic_solver_labels': True})

    return EcoGEM
def GFBA_wild(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,v0_biomass,biomass_name,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=copy.deepcopy(Concretemodel_Need_Data['ub_list'])
    ub_list[substrate_name]=substrate_value
 
    EcoGEM=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,v0_biomass=v0_biomass,biomass_name=biomass_name,\
        lb_list=lb_list,ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,substrate_name=substrate_name,\
        substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_constr9=True)
    
    Model_Solve(EcoGEM,solver)
    #EcoETM.write('./results/GEFBA.lp',io_options={'symbolic_solver_labels': True})

    return EcoGEM
def GFBA_product(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,v0_biomass,biomass_name,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=copy.deepcopy(Concretemodel_Need_Data['ub_list'])
    ub_list[substrate_name]=substrate_value
 
    EcoGEM=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,v0_biomass=v0_biomass,biomass_name=biomass_name,\
        lb_list=lb_list,ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,substrate_name=substrate_name,\
        substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_constr10=True)
    
    Model_Solve(EcoGEM,solver)
    #EcoETM.write('./results/GEFBA.lp',io_options={'symbolic_solver_labels': True})

    return EcoGEM
def GFBA_over(Concretemodel_Need_Data,obj_name,obj_target,substrate_name,substrate_value,v0_biomass,biomass_name,v1_product_max,product_name,solver):
    reaction_list=Concretemodel_Need_Data['reaction_list']
    metabolite_list=Concretemodel_Need_Data['metabolite_list']
    coef_matrix=Concretemodel_Need_Data['coef_matrix']
    lb_list=Concretemodel_Need_Data['lb_list']
    ub_list=copy.deepcopy(Concretemodel_Need_Data['ub_list'])
    ub_list[substrate_name]=substrate_value
 
    EcoGEM=FBA_template(reaction_list=reaction_list,metabolite_list=metabolite_list,coef_matrix=coef_matrix,v0_biomass=v0_biomass,biomass_name=biomass_name,\
        lb_list=lb_list,ub_list=ub_list,obj_name=obj_name,obj_target=obj_target,set_obj_value=True,set_substrate_ini=True,substrate_name=substrate_name,\
        substrate_value=substrate_value, set_stoi_matrix=True,set_bound=True,set_constr10=True,set_constr11=True,v1_product_max=v1_product_max,product_name=product_name)
    
    Model_Solve(EcoGEM,solver)
    #EcoETM.write('./results/GEFBA.lp',io_options={'symbolic_solver_labels': True})

    return EcoGEM