# This code is used to introduce enzyme concentration constraint in GEMs
# by COBRApy and to calculate the parameters that need to be entered
# during the construction of the enzyme-constrained model.
#from warnings import warn

import pandas as pd
import numpy as np
import json
import cobra
import math
import re
from cobra.core import Reaction
from cobra.io.dict import model_to_dict
from cobra.util.solver import set_objective
from xml.dom import minidom

def convert_to_irreversible(model):
    """Split reversible reactions into two irreversible reactions

    These two reactions will proceed in opposite directions. This
    guarentees that all reactions in the model will only allow
    positive flux values, which is useful for some modeling problems.

    Arguments
    ----------
    *model: cobra.Model ~ A Model object which will be modified in place.
 
    """
    #warn("deprecated, not applicable for optlang solvers", DeprecationWarning)
    reactions_to_add = []
    coefficients = {}
    for reaction in model.reactions:
        # If a reaction is reverse only, the forward reaction (which
        # will be constrained to 0) will be left in the model.
        if reaction.lower_bound < 0 and reaction.upper_bound >0:
            reverse_reaction = Reaction(reaction.id + "_reverse")
            reverse_reaction.lower_bound = max(0, -reaction.upper_bound)
            reverse_reaction.upper_bound = -reaction.lower_bound
            coefficients[
                reverse_reaction] = reaction.objective_coefficient * -1
            reaction.lower_bound = max(0, reaction.lower_bound)
            reaction.upper_bound = max(0, reaction.upper_bound)
            # Make the directions aware of each other
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
            reverse_reaction._gene_reaction_rule = reaction._gene_reaction_rule
            reactions_to_add.append(reverse_reaction)
    model.add_reactions(reactions_to_add)
    set_objective(model, coefficients, additive=True)


def get_genes_and_gpr(model):
    """Retrieving genes and gene_reaction_rule from GEM.

    Arguments
    ----------
    *model: cobra.Model ~ A genome scale metabolic network model for
        constructing the enzyme-constrained model.

    :return: all genes and gpr in model.
    """
    model_dict = model_to_dict(model, sort=False)
    genes = pd.DataFrame(model_dict['genes']).set_index(['id'])
    genes.to_csv("./analysis/genes.csv")
    all_gpr = pd.DataFrame(model_dict['reactions']).set_index(['id'])
    all_gpr.to_csv("./analysis/all_reaction_GPR.csv")
    return [genes,all_gpr]


def calculate_reaction_mw(reaction_gene_subunit_MW):
    """Calculate the molecular weight of the enzyme that catalyzes each
    reaction in GEM based on the number of subunits and
    molecular weight of each gene.

    Arguments
    ----------
    *reaction_gene_subunit_MW: A CSV file contains the GPR relationship
     for each reaction in the GEM model,the number of subunit components 
     of each gene expressed protein, and the molecular weight of each 
     gene expressed protein.
     
    :return: The molecular weight of the enzyme that catalyzes each reaction
     in the GEM model.
    """
    reaction_gene_subunit_MW = pd.read_csv(reaction_gene_subunit_MW, index_col=0)
    reaction_mw = pd.DataFrame()
    for reaction_id in reaction_gene_subunit_MW.index:
        subunit_mw_list = reaction_gene_subunit_MW.loc[reaction_id, 'subunit_mw'].\
            replace('(', '').replace(")", '').replace(" ", '').split('or')
        subunit_num_list = reaction_gene_subunit_MW.loc[reaction_id, 'subunit_num'].\
            replace('(', '').replace(")", '').replace(" ", '').split('or')

        mw_s = ''
        for mw_i in range(0, len(subunit_mw_list)):
            mw_list = np.array(subunit_mw_list[mw_i].split('and'))
            num_list = np.array(subunit_num_list[mw_i].split('and'))
            mw_list = list(map(float, mw_list))
            num_list = list(map(float, num_list))   
            mw_s = mw_s + str(round(np.sum(np.multiply(mw_list,num_list)), 4)) + ' or '

        mw_s = mw_s.rstrip(' or ')
        reaction_mw.loc[reaction_id, 'MW'] = mw_s
    reaction_mw.to_csv("./analysis/reaction_MW.csv")
    return reaction_mw

def calculate_reaction_mw_not_consider_subunit(reaction_gene_subunit_MW,save_file):
    """Calculate the molecular weight of the enzyme that catalyzes each
    reaction in GEM based on the number of subunits and
    molecular weight of each gene.

    Arguments
    ----------
    *reaction_gene_subunit_MW: A CSV file contains the GPR relationship
     for each reaction in the GEM model,the number of subunit components 
     of each gene expressed protein, and the molecular weight of each 
     gene expressed protein.
     
    :return: The molecular weight of the enzyme that catalyzes each reaction
     in the GEM model.
    """
    reaction_gene_subunit_MW = pd.read_csv(reaction_gene_subunit_MW, index_col=0)
    reaction_mw = pd.DataFrame()
    for reaction_id in reaction_gene_subunit_MW.index:
        subunit_mw_list = reaction_gene_subunit_MW.loc[reaction_id, 'subunit_mw'].\
            replace('(', '').replace(")", '').replace(" ", '').split('or')
        subunit_num_list = reaction_gene_subunit_MW.loc[reaction_id, 'subunit_num'].\
            replace('(', '').replace(")", '').replace(" ", '').split('or')

        mw_s = ''
        for mw_i in range(0, len(subunit_mw_list)):
            mw_list = np.array(subunit_mw_list[mw_i].split('and'))
            num_list = np.array(subunit_num_list[mw_i].split('and'))
            mw_list = list(map(float, mw_list))
            mw_s = mw_s + str(round(np.sum(np.multiply(mw_list,1)), 4)) + ' or '

        mw_s = mw_s.rstrip(' or ')
        reaction_mw.loc[reaction_id, 'MW'] = mw_s
    reaction_mw.to_csv(save_file)
    return reaction_mw

def calculate_reaction_kcat_mw_old(reaction_kcat_file, reaction_mw,save_file):
    """Calculating kcat/MW

    When the reaction is catalyzed by several isozymes,
    the maximum was retained.

    Arguments
    ----------
    *reaction_kcat_file: A CSV file contains the kcat values for each
    reaction in the model.
    *reaction_mw: The molecular weight of the enzyme that catalyzes
     each reaction in the GEM model.

    :return: The kcat/MW value of the enzyme catalyzing each reaction
     in the GEM model.
    """
    reaction_kcat = pd.read_csv(reaction_kcat_file, index_col=0)
    reaction_kcat_mw = pd.DataFrame()
    for reaction_id in reaction_kcat.index:
        if reaction_id in reaction_mw.index:
            mw = reaction_mw.loc[reaction_id, 'MW'].split('or')
            min_mw = min(map(float, mw))
            kcat_mw = reaction_kcat.loc[reaction_id, 'kcat'] / min_mw
            reaction_kcat_mw.loc[reaction_id, 'kcat'] = reaction_kcat.loc[reaction_id, 'kcat']
            reaction_kcat_mw.loc[reaction_id, 'MW'] = min_mw
            reaction_kcat_mw.loc[reaction_id, 'kcat_MW'] = kcat_mw
    reaction_kcat_mw.to_csv(save_file)
    return reaction_kcat_mw

def calculate_reaction_kcat_mw(reaction_kcat_file, reaction_mw,save_file):
    """Calculating kcat/MW

    When the reaction is catalyzed by several isozymes,
    the maximum was retained.

    Arguments
    ----------
    *reaction_kcat_file: A CSV file contains the kcat values for each
    reaction in the model.
    *reaction_mw: The molecular weight of the enzyme that catalyzes
     each reaction in the GEM model.

    :return: The kcat/MW value of the enzyme catalyzing each reaction
     in the GEM model.
    """
    reaction_kcat = pd.read_csv(reaction_kcat_file, index_col=0)
    reaction_kcat_mw = pd.DataFrame()
    for reaction_idmw in reaction_mw.index:
        reaction_id = reaction_idmw.split('_num')[0]
        if reaction_id in reaction_kcat.index:
            mw = reaction_mw.loc[reaction_idmw, 'MW'].split('or')
            min_mw = min(map(float, mw))
            kcat_mw = reaction_kcat.loc[reaction_id, 'kcat'] / min_mw
            reaction_kcat_mw.loc[reaction_idmw, 'kcat'] = reaction_kcat.loc[reaction_id, 'kcat']
            reaction_kcat_mw.loc[reaction_idmw, 'MW'] = min_mw
            reaction_kcat_mw.loc[reaction_idmw, 'kcat_MW'] = kcat_mw
    reaction_kcat_mw.to_csv(save_file)
    return reaction_kcat_mw

def calculate_f(genes, gene_abundance_file, subunit_molecular_weight_file):
    """Calculating f (the mass fraction of enzymes that are accounted
    in the model out of all proteins) based on the protein abundance
    which can be obtained from PAXdb database.

    Arguments
    ----------
    *genes: All the genes in the model.
    *gene_abundance_file: The protein abundance of each gene
     in the E. coli genome.
    *subunit_molecular_weight_file: The molecular weight of the
     protein subunit expressed by each gene.

    :return: The enzyme mass fraction f.
    """
    gene_abundance = pd.read_csv(gene_abundance_file, index_col=0)
    subunit_molecular_weight = pd.read_csv(subunit_molecular_weight_file, index_col=0)
    enzy_abundance = 0
    pro_abundance = 0
    for gene_i in gene_abundance.index:
        abundance=gene_abundance.loc[gene_i, 'abundance'] * subunit_molecular_weight.loc[gene_i, 'mw']
        pro_abundance += abundance     
        if gene_i in genes.index:
            enzy_abundance += abundance
    f = enzy_abundance/pro_abundance
    return f


def set_enzyme_constraint(model, reaction_kcat_mw, lowerbound, upperbound):
    """Introducing enzyme concentration constraint
    by COBRApy using the calculated parameters.

    Arguments
    ----------
    *model: cobra.Model ~ A genome scale metabolic network model for
        constructing the enzyme-constrained model.
    *reaction_kcat_mw: The kcat/MW value of the enzyme catalyzing each
     reaction in the GEM model.
    *lowerbound: The lower bound of enzyme concentration constraint in
     the enzyme-constrained model.
    *upperbound: The upper bound of enzyme concentration constraint in
     the enzyme-constrained model.

    :return: Construct an enzyme-constrained model.
    """
    coefficients = dict()
    for rxn in model.reactions:
        if rxn.id in reaction_kcat_mw.index:
            coefficients[rxn.forward_variable] = 1/float(reaction_kcat_mw.loc[rxn.id, 'kcat_MW'])
    constraint = model.problem.Constraint(0, lb=lowerbound, ub=upperbound)
    model.add_cons_vars(constraint)
    model.solver.update()
    constraint.set_linear_coefficients(coefficients=coefficients)
    return model

def json_load(path) :
    """Loads the given JSON file and returns it as dictionary.

    Arguments
    ----------
    * path: str ~ The path of the JSON file
    """
    with open(path) as f:
        dictionary = json.load(f)
    return dictionary

def json_write(path, dictionary):
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

def trans_model2enz_json_model_split_isoenzyme(model_file, reaction_kcat_mw_file, f, ptot, sigma, lowerbound, upperbound, json_output_file):
    """Tansform cobra model to json mode with  
    enzyme concentration constraintat.

    Arguments
    ----------
    * model_file: str ~  The path of 
    * reaction_kcat_mw_file: str ~  The path of 
    *f:
    * ptot:  ~  
    * sigma:  ~  
    *lowerbound:   
    *upperbound:  

    """

    model = cobra.io.read_sbml_model(model_file)
    #model = isoenzyme_split(model)
    convert_to_irreversible(model)
    model = isoenzyme_split(model)
    #model = isoenzyme_split2(model)
    model_name=model_file.split('/')[-1].split('.')[0]
    json_path="./model/%s_irreversible.json"%model_name
    cobra.io.save_json_model(model, json_path)

    dictionary_model = json_load(json_path)
    dictionary_model['enzyme_constraint']={'enzyme_mass_fraction': f, 'total_protein_fraction': ptot,\
        'average_saturation': sigma, 'lowerbound': lowerbound, 'upperbound': upperbound} 
    # Reaction-kcat_mw file.
    # eg. AADDGT,49389.2889,40.6396,1215.299582180927
    reaction_kcat_mw=pd.read_csv(reaction_kcat_mw_file, index_col=0)

    reaction_kcay_mw_dict={}
    for eachreaction in  range(len(dictionary_model['reactions'])): 
        reaction_id=dictionary_model['reactions'][eachreaction]['id']
        if reaction_id in reaction_kcat_mw.index:
            dictionary_model['reactions'][eachreaction]['kcat']=reaction_kcat_mw.loc[reaction_id,'kcat']
            dictionary_model['reactions'][eachreaction]['kcat_MW']=reaction_kcat_mw.loc[reaction_id,'kcat_MW']
            reaction_kcay_mw_dict[reaction_id]=reaction_kcat_mw.loc[reaction_id,'kcat_MW']
        else:
            dictionary_model['reactions'][eachreaction]['kcat']=''
            dictionary_model['reactions'][eachreaction]['kcat_MW']=''  

    dictionary_model['enzyme_constraint']['kcat_MW']=reaction_kcay_mw_dict

    json_write(json_output_file, dictionary_model)

def trans_model2enz_json_model_split_isoenzyme_only(model_file, reaction_kcat_mw_file, f, ptot, sigma, lowerbound, upperbound, json_output_file):
    """Tansform cobra model to json mode with  
    enzyme concentration constraintat.

    Arguments
    ----------
    * model_file: str ~  The path of 
    * reaction_kcat_mw_file: str ~  The path of 
    *f:
    * ptot:  ~  
    * sigma:  ~  
    *lowerbound:   
    *upperbound:  

    """

    model = cobra.io.read_sbml_model(model_file)
    #model = isoenzyme_split(model)
    convert_to_irreversible(model)
    model = isoenzyme_split_only(model)
    #model = isoenzyme_split2(model)
    model_name=model_file.split('/')[-1].split('.')[0]
    json_path="./model/%s_irreversible.json"%model_name
    cobra.io.save_json_model(model, json_path)

    dictionary_model = json_load(json_path)
    dictionary_model['enzyme_constraint']={'enzyme_mass_fraction': f, 'total_protein_fraction': ptot,\
        'average_saturation': sigma, 'lowerbound': lowerbound, 'upperbound': upperbound} 
    # Reaction-kcat_mw file.
    # eg. AADDGT,49389.2889,40.6396,1215.299582180927
    reaction_kcat_mw=pd.read_csv(reaction_kcat_mw_file, index_col=0)

    reaction_kcay_mw_dict={}
    for eachreaction in  range(len(dictionary_model['reactions'])): 
        reaction_id=dictionary_model['reactions'][eachreaction]['id']
        if reaction_id in reaction_kcat_mw.index:
            dictionary_model['reactions'][eachreaction]['kcat']=reaction_kcat_mw.loc[reaction_id,'kcat']
            dictionary_model['reactions'][eachreaction]['kcat_MW']=reaction_kcat_mw.loc[reaction_id,'kcat_MW']
            reaction_kcay_mw_dict[reaction_id]=reaction_kcat_mw.loc[reaction_id,'kcat_MW']
        else:
            dictionary_model['reactions'][eachreaction]['kcat']=''
            dictionary_model['reactions'][eachreaction]['kcat_MW']=''  

    dictionary_model['enzyme_constraint']['kcat_MW']=reaction_kcay_mw_dict

    json_write(json_output_file, dictionary_model)
def trans_model2enz_json_model(model_file, reaction_kcat_mw_file, f, ptot, sigma, lowerbound, upperbound, json_output_file):
    """Tansform cobra model to json mode with  
    enzyme concentration constraintat.

    Arguments
    ----------
    * model_file: str ~  The path of 
    * reaction_kcat_mw_file: str ~  The path of 
    *f:
    * ptot:  ~  
    * sigma:  ~  
    *lowerbound:   
    *upperbound:  

    """

    model = cobra.io.read_sbml_model(model_file)
    convert_to_irreversible(model)
    model_name=model_file.split('/')[-1].split('.')[0]
    json_path="./model/%s_irreversible.json"%model_name
    cobra.io.save_json_model(model, json_path)

    dictionary_model = json_load(json_path)
    dictionary_model['enzyme_constraint']={'enzyme_mass_fraction': f, 'total_protein_fraction': ptot,\
        'average_saturation': sigma, 'lowerbound': lowerbound, 'upperbound': upperbound} 
    # Reaction-kcat_mw file.
    # eg. AADDGT,49389.2889,40.6396,1215.299582180927
    reaction_kcat_mw=pd.read_csv(reaction_kcat_mw_file, index_col=0)

    reaction_kcay_mw_dict={}
    for eachreaction in  range(len(dictionary_model['reactions'])): 
        reaction_id=dictionary_model['reactions'][eachreaction]['id']
        if reaction_id in reaction_kcat_mw.index:
            dictionary_model['reactions'][eachreaction]['kcat']=reaction_kcat_mw.loc[reaction_id,'kcat']
            dictionary_model['reactions'][eachreaction]['kcat_MW']=reaction_kcat_mw.loc[reaction_id,'kcat_MW']
            reaction_kcay_mw_dict[reaction_id]=reaction_kcat_mw.loc[reaction_id,'kcat_MW']
        else:
            dictionary_model['reactions'][eachreaction]['kcat']=''
            dictionary_model['reactions'][eachreaction]['kcat_MW']=''  

    dictionary_model['enzyme_constraint']['kcat_MW']=reaction_kcay_mw_dict

    json_write(json_output_file, dictionary_model)

def trans_model2enz_json_model_PDH(model_file, reaction_kcat_mw_file, f, ptot, sigma, lowerbound, upperbound, json_output_file):
    """Tansform cobra model to json mode with  
    enzyme concentration constraintat.

    Arguments
    ----------
    * model_file: str ~  The path of 
    * reaction_kcat_mw_file: str ~  The path of 
    *f:
    * ptot:  ~  
    * sigma:  ~  
    *lowerbound:   
    *upperbound:  

    """

    model = cobra.io.read_sbml_model(model_file)
    convert_to_irreversible(model)
    model = isoenzyme_split_PDH(model)
    model_name=model_file.split('/')[-1].split('.')[0]
    json_path="./model/%s_irreversible.json"%model_name
    cobra.io.save_json_model(model, json_path)

    dictionary_model = json_load(json_path)
    dictionary_model['enzyme_constraint']={'enzyme_mass_fraction': f, 'total_protein_fraction': ptot,\
        'average_saturation': sigma, 'lowerbound': lowerbound, 'upperbound': upperbound} 
    # Reaction-kcat_mw file.
    # eg. AADDGT,49389.2889,40.6396,1215.299582180927
    reaction_kcat_mw=pd.read_csv(reaction_kcat_mw_file, index_col=0)

    reaction_kcay_mw_dict={}
    for eachreaction in  range(len(dictionary_model['reactions'])): 
        reaction_id=dictionary_model['reactions'][eachreaction]['id']
        if reaction_id in reaction_kcat_mw.index:
            dictionary_model['reactions'][eachreaction]['kcat']=reaction_kcat_mw.loc[reaction_id,'kcat']
            dictionary_model['reactions'][eachreaction]['kcat_MW']=reaction_kcat_mw.loc[reaction_id,'kcat_MW']
            reaction_kcay_mw_dict[reaction_id]=reaction_kcat_mw.loc[reaction_id,'kcat_MW']
        else:
            dictionary_model['reactions'][eachreaction]['kcat']=''
            dictionary_model['reactions'][eachreaction]['kcat_MW']=''  

    dictionary_model['enzyme_constraint']['kcat_MW']=reaction_kcay_mw_dict

    json_write(json_output_file, dictionary_model)

def get_enzyme_constraint_model(json_model_file):
    """using enzyme concentration constraint
    json model to create a COBRApy model.

    Arguments
    ----------
    *json_model_file: json Model file.

    :return: Construct an enzyme-constrained model.
    """

    dictionary_model = json_load(json_model_file)
    model=cobra.io.json.load_json_model(json_model_file) 

    coefficients = dict()
    for rxn in model.reactions:
        if rxn.id in dictionary_model['enzyme_constraint']['kcat_MW'].keys():
            coefficients[rxn.forward_variable] = 1/float(dictionary_model['enzyme_constraint']['kcat_MW'][rxn.id])

    lowerbound=dictionary_model['enzyme_constraint']['lowerbound']
    upperbound=dictionary_model['enzyme_constraint']['upperbound']
    constraint = model.problem.Constraint(0, lb=lowerbound, ub=upperbound)
    model.add_cons_vars(constraint)
    model.solver.update()
    constraint.set_linear_coefficients(coefficients=coefficients)
    return model

#This file may be used for preprocessing the SVG file of a pathway map, e.g. replace gene/compound name, standardize the CSS styles
#for interactive visualization, d3 may be used
def draw_svg(cb_df,col_name,insvg,outsvg,rclass):
    doc = minidom.parse(insvg)  
    c=[]
    svgct=[]
    modelct=[]

    for path in doc.getElementsByTagName('text'):
        if path.getAttribute('class') in rclass:
            for n in path.childNodes:
                #print(n)
                if n.nodeName=="#text":
                    rid=n.data
                    #print(rid)
                    path.setAttribute("id",n.data)
                    if rid in cb_df.index:
                        #print(rid)
                        #print(str(cb_df.loc[rid,col_name]))
                        n.data=str(cb_df.loc[rid,col_name])
                        c.append(str(rid))
                    else:
                        #print(str(str(rid)))#反应
                        svgct.append(str(str(rid)))
    l=[]     
    for line in cb_df.keys():
        l.append(line)
    cs=set(c)
    ls=set(l)
    modelct=ls-cs
    with open(outsvg,'w',encoding='UTF-8') as fw:
        doc.writexml(fw,encoding='UTF-8')    

def select_calibration_reaction(reaction_kcat_mw_file, json_model_path, enzyme_amount, percentage, reaction_biomass_outfile, select_value):
    reaction_kcat_mw = pd.read_csv(reaction_kcat_mw_file, index_col=0)
    norm_model=cobra.io.json.load_json_model(json_model_path)
    norm_biomass=norm_model.slim_optimize() 
    df_biomass = pd.DataFrame()
    df_biomass_select = pd.DataFrame()
    for r in norm_model.reactions:
        with norm_model as model:
            if r.id in list(reaction_kcat_mw.index):
                r.bounds = (0, reaction_kcat_mw.loc[r.id,'kcat_MW']*enzyme_amount*percentage)
                df_biomass.loc[r.id,'biomass'] = model.slim_optimize()
                biomass_diff = norm_biomass-model.slim_optimize()
                biomass_diff_ratio = (norm_biomass-model.slim_optimize())/norm_biomass
                df_biomass.loc[r.id,'biomass_diff'] = biomass_diff
                df_biomass.loc[r.id,'biomass_diff_ratio'] = biomass_diff_ratio
                if biomass_diff_ratio > select_value: #select difference range
                    df_biomass_select.loc[r.id,'biomass_diff'] = biomass_diff
                    df_biomass_select.loc[r.id,'biomass_diff_ratio'] = biomass_diff_ratio

    df_biomass = df_biomass.sort_values(by="biomass_diff_ratio",axis = 0,ascending = False)
    df_biomass.to_csv(reaction_biomass_outfile)

    if df_biomass_select.empty:
        pass
    else:
        df_reaction_select = df_biomass_select.sort_values(by="biomass_diff_ratio",axis = 0,ascending = False)
        return(df_reaction_select)

def calibration_kcat(need_change_reaction, reaction_kcat_mw_file, json_model_path, adj_kcat_title, change_kapp_file, reaction_kapp_change_file):
    reaction_kappori = pd.read_csv(reaction_kcat_mw_file, index_col=0)
    kcat_data_colect_file="./data/kcat_data_colect.csv"
    kcat_data_colect = pd.read_csv(kcat_data_colect_file, index_col=0)
    norm_model=cobra.io.json.load_json_model(json_model_path)
    norm_biomass=norm_model.slim_optimize() 
    round_1_reaction_kapp_change = pd.DataFrame()
    for eachreaction in need_change_reaction:
        kcat_ori = reaction_kappori.loc[eachreaction,'kcat']
        kcat_smoment_adj = kcat_data_colect.loc[eachreaction, adj_kcat_title] *2 * 3600
        if kcat_ori < kcat_smoment_adj:
            reaction_kappori.loc[eachreaction,'kcat'] = kcat_smoment_adj
        reaction_kappori.loc[eachreaction, 'kcat_MW'] = reaction_kappori.loc[eachreaction, 'kcat'] / reaction_kappori.loc[eachreaction,'MW']
        for r in norm_model.reactions:
            with norm_model as model:
                if r.id == eachreaction:
                    r.bounds = (0, reaction_kappori.loc[eachreaction, 'kcat_MW']*0.0228)
                    round_1_reaction_kapp_change.loc[eachreaction,'kcat_ori'] = kcat_ori
                    round_1_reaction_kapp_change.loc[eachreaction,'kcat_change'] = reaction_kappori.loc[eachreaction,'kcat']
                    round_1_reaction_kapp_change.loc[eachreaction,'MW'] = reaction_kappori.loc[eachreaction,'MW']
                    round_1_reaction_kapp_change.loc[eachreaction,'kcat_mw_new'] = reaction_kappori.loc[eachreaction, 'kcat_MW']
                    round_1_reaction_kapp_change.loc[eachreaction,'norm_biomass'] = norm_biomass
                    round_1_reaction_kapp_change.loc[eachreaction,'new_biomass'] = model.slim_optimize()
    round_1_reaction_kapp_change.to_csv(change_kapp_file)
    reaction_kappori.to_csv(reaction_kapp_change_file)
    
def get_enzyme_usage(enz_total,reaction_flux_file,reaction_kcat_mw_file,reaction_enz_usage_file):
    reaction_fluxes = pd.read_csv(reaction_flux_file, index_col=0)
    reaction_kcat_mw = pd.read_csv(reaction_kcat_mw_file, index_col=0)

    reaction_enz_usage_df = pd.DataFrame()
    for index,row in reaction_kcat_mw.iterrows():
        if index in reaction_fluxes.index:
            reaction_enz_usage_df.loc[index,'kcat_mw'] = row['kcat_MW']
            reaction_enz_usage_df.loc[index,'flux'] = reaction_fluxes.loc[index,'fluxes']
            reaction_enz_usage_df.loc[index,'enz useage'] = reaction_fluxes.loc[index,'fluxes']/row['kcat_MW']
            reaction_enz_usage_df.loc[index,'enz ratio'] = reaction_fluxes.loc[index,'fluxes']/row['kcat_MW']/enz_total

    reaction_enz_usage_df = reaction_enz_usage_df.sort_values(by="enz ratio",axis = 0,ascending = False)
    reaction_enz_usage_df.to_csv(reaction_enz_usage_file)
    return reaction_enz_usage_df
def change_reaction_kcat_by_autopacmen(select_reaction,reaction_kcat_mw_file,reaction_kapp_change_file):
    reaction_kcat_mw = pd.read_csv(reaction_kcat_mw_file, index_col=0)
    kcat_data_colect_file="./data/kcat_data_colect.csv"
    kcat_data_colect = pd.read_csv(kcat_data_colect_file, index_col=0)

    reaction_change_accord_fold=[]
    for eachreaction in select_reaction:
        if reaction_kcat_mw.loc[eachreaction,'kcat'] < kcat_data_colect.loc[eachreaction, 'smoment_adj_kcat'] *2 * 3600:
            reaction_kcat_mw.loc[eachreaction,'kcat'] = kcat_data_colect.loc[eachreaction, 'smoment_adj_kcat'] *2 * 3600
            reaction_kcat_mw.loc[eachreaction,'kcat_MW'] = kcat_data_colect.loc[eachreaction, 'smoment_adj_kcat'] *2 * 3600/reaction_kcat_mw.loc[eachreaction,'MW']
            reaction_change_accord_fold.append(eachreaction)
        else:
            pass
    reaction_kcat_mw.to_csv(reaction_kapp_change_file)
    return(reaction_change_accord_fold)

def change_reaction_kcat_part_by_fold(select_reaction,change_fold,reaction_kcat_mw_file,reaction_kapp_change_file):
    reaction_kcat_mw = pd.read_csv(reaction_kcat_mw_file, index_col=0)
    kcat_data_colect_file="./data/kcat_data_colect.csv"
    kcat_data_colect = pd.read_csv(kcat_data_colect_file, index_col=0)

    reaction_change_accord_fold=[]
    for eachreaction in select_reaction:
        if reaction_kcat_mw.loc[eachreaction,'kcat'] < kcat_data_colect.loc[eachreaction, 'smoment_adj_kcat'] *2 * 3600:
            reaction_kcat_mw.loc[eachreaction,'kcat'] = kcat_data_colect.loc[eachreaction, 'smoment_adj_kcat'] *2 * 3600
            reaction_kcat_mw.loc[eachreaction,'kcat_MW'] = kcat_data_colect.loc[eachreaction, 'smoment_adj_kcat'] *2 * 3600/reaction_kcat_mw.loc[eachreaction,'MW']
            reaction_change_accord_fold.append(eachreaction)
        else:
            reaction_kcat_mw.loc[eachreaction,'kcat'] = reaction_kcat_mw.loc[eachreaction,'kcat'] * change_fold
            reaction_kcat_mw.loc[eachreaction,'kcat_MW'] = reaction_kcat_mw.loc[eachreaction,'kcat_MW'] * change_fold
    reaction_kcat_mw.to_csv(reaction_kapp_change_file)
    return(reaction_change_accord_fold)

def change_reaction_kcat_by_fold(select_reaction,change_fold,reaction_kcat_mw_file,reaction_kapp_change_file):
    reaction_kcat_mw = pd.read_csv(reaction_kcat_mw_file, index_col=0)
    reaction_change_accord_fold=[]
    for eachreaction in select_reaction:
        reaction_kcat_mw.loc[eachreaction,'kcat'] = reaction_kcat_mw.loc[eachreaction,'kcat'] * change_fold
        reaction_kcat_mw.loc[eachreaction,'kcat_MW'] = reaction_kcat_mw.loc[eachreaction,'kcat_MW'] * change_fold
        reaction_change_accord_fold.append(eachreaction)
    reaction_kcat_mw.to_csv(reaction_kapp_change_file)
    return(reaction_change_accord_fold)

def change_reaction_kcat_by_foldlist(select_reaction,change_fold,reaction_kcat_mw_file,reaction_kapp_change_file):
    reaction_kcat_mw = pd.read_csv(reaction_kcat_mw_file, index_col=0)
    reaction_change_accord_fold=[]
    i=0
    for eachreaction in select_reaction:
        reaction_kcat_mw.loc[eachreaction,'kcat'] = reaction_kcat_mw.loc[eachreaction,'kcat'] * change_fold[i]
        reaction_kcat_mw.loc[eachreaction,'kcat_MW'] = reaction_kcat_mw.loc[eachreaction,'kcat_MW'] * change_fold[i]
        reaction_change_accord_fold.append(eachreaction)
        i=i+1
    reaction_kcat_mw.to_csv(reaction_kapp_change_file)
    return(reaction_change_accord_fold)
    
def draw_calibration_kcat_figure(Orimodel_solution_frame,ECMpy_solution_frame,ECMpy_adj_solution_frame,insvg,outsvg):
    model_data=pd.DataFrame()
    for eachreaction in ECMpy_solution_frame.index:
        model_data.loc[eachreaction,'model_ori_fluxes']=Orimodel_solution_frame.loc[eachreaction,'fluxes']
        model_data.loc[eachreaction,'ECMpy_fluxes']=ECMpy_solution_frame.loc[eachreaction,'fluxes']
        model_data.loc[eachreaction,'ECMpy_adj_fluxes']=ECMpy_adj_solution_frame.loc[eachreaction,'fluxes']  

    cb_df=pd.DataFrame()
    for index, row in model_data.iterrows():
        index_reverse=index+'_reverse'
        if re.search('reverse',index):
            pass
        elif index_reverse in model_data.index:
            model_ori_reverse_fluxes=float(model_data.loc[index_reverse,'model_ori_fluxes'])
            ECMpy_reverse_fluxes=float(model_data.loc[index_reverse,'ECMpy_fluxes'])
            ECMpy_adj_reverse_fluxes=float(model_data.loc[index_reverse,'ECMpy_adj_fluxes'])

            if math.isnan(model_ori_reverse_fluxes):
                model_ori_fluxes=model_data.loc[index,'model_ori_fluxes']
            else:
                model_ori_fluxes=np.max([model_data.loc[index,'model_ori_fluxes'],model_data.loc[index_reverse,'model_ori_fluxes']])
                
            if math.isnan(ECMpy_reverse_fluxes):
                ECMpy_fluxes=model_data.loc[index,'ECMpy_fluxes']
            else:
                ECMpy_fluxes=np.max([model_data.loc[index,'ECMpy_fluxes'],model_data.loc[index_reverse,'ECMpy_fluxes']])
                
            if math.isnan(ECMpy_adj_reverse_fluxes):
                ECMpy_adj_fluxes=model_data.loc[index,'ECMpy_adj_fluxes']
            else:
                ECMpy_adj_fluxes=np.max([model_data.loc[index,'ECMpy_adj_fluxes'],model_data.loc[index_reverse,'ECMpy_adj_fluxes']])
                            
            flux_cb=str(round(model_ori_fluxes,2))+' # '+' # '+str(round(ECMpy_fluxes,2))+' # ' \
        +' # '+str(round(ECMpy_adj_fluxes,2))
            cb_df.loc[index,'flux_cb']=flux_cb
        else:
            flux_cb=str(round(row['model_ori_fluxes'],2))+' # '+' # '+str(round(row['ECMpy_fluxes'],2))+' # ' \
            +' # '+str(round(row['ECMpy_adj_fluxes'],2))
            cb_df.loc[index,'flux_cb']=flux_cb

    rclass=['st6','st15','st18','st16']
    draw_svg(cb_df,'flux_cb',insvg,outsvg,rclass)

def draw_calibration_kcat_figure_g(Orimodel_solution_frame,ECMpy_solution_frame,ECMpy_adj_solution_frame,insvg,outsvg):
    model_data=pd.DataFrame()
    for eachreaction in ECMpy_solution_frame.index:
        if eachreaction in Orimodel_solution_frame.index:
            model_data.loc[eachreaction,'flux_cb']=str(round(Orimodel_solution_frame.loc[eachreaction,'Flux norm'],2))+\
                ' # '+' # '+str(round(ECMpy_solution_frame.loc[eachreaction,'fluxes'],2))+' # ' \
                +' # '+str(round(ECMpy_adj_solution_frame.loc[eachreaction,'fluxes'],2))
        else:
             model_data.loc[eachreaction,'flux_cb']='Not give'+\
                ' # '+' # '+str(round(ECMpy_solution_frame.loc[eachreaction,'fluxes'],2))+' # ' \
                +' # '+str(round(ECMpy_adj_solution_frame.loc[eachreaction,'fluxes'],2))           
    rclass=['st6','st15','st18','st16']
    draw_svg(model_data,'flux_cb',insvg,outsvg,rclass)

def draw_calibration_kcat_figure_g2(Orimodel_solution_frame,ECMpy_solution_frame,ECMpy_adj_solution_frame,insvg,outsvg):
    model_data=pd.DataFrame()
    flux2_list=pd.DataFrame()
    flux3_list=pd.DataFrame()
    for reaction in ECMpy_solution_frame.index:
        if re.search('_num',reaction):
            eachreaction = reaction.split('_num')[0]
            if eachreaction in flux2_list.index:
                flux2_list.loc[eachreaction,'fluxes'] = np.max([flux2_list.loc[eachreaction,'fluxes'],ECMpy_solution_frame.loc[reaction,'fluxes']])
            else:
                flux2_list.loc[eachreaction,'fluxes'] = ECMpy_solution_frame.loc[reaction,'fluxes']
            if eachreaction in flux3_list.index:
                flux3_list.loc[eachreaction,'fluxes'] = np.max([flux3_list.loc[eachreaction,'fluxes'],ECMpy_adj_solution_frame.loc[reaction,'fluxes']])
            else:
                flux3_list.loc[eachreaction,'fluxes'] = ECMpy_adj_solution_frame.loc[reaction,'fluxes']
        else:
            flux2_list.loc[reaction,'fluxes'] = ECMpy_solution_frame.loc[reaction,'fluxes']
            flux3_list.loc[reaction,'fluxes'] = ECMpy_adj_solution_frame.loc[reaction,'fluxes']

    for eachreaction in flux2_list.index:            
        if eachreaction in Orimodel_solution_frame.index:
            model_data.loc[eachreaction,'flux_cb']=str(round(Orimodel_solution_frame.loc[eachreaction,'Flux norm'],2))+\
                    ' # '+' # '+str(round(flux2_list.loc[eachreaction,'fluxes'],2))+' # ' \
                    +' # '+str(round(flux3_list.loc[eachreaction,'fluxes'],2))
        else:
            model_data.loc[eachreaction,'flux_cb']='Not give'+\
                    ' # '+' # '+str(round(flux2_list.loc[eachreaction,'fluxes'],2))+' # ' \
                    +' # '+str(round(flux3_list.loc[eachreaction,'fluxes'],2))    
                        
    rclass=['st6','st15','st18','st16']
    draw_svg(model_data,'flux_cb',insvg,outsvg,rclass)

def draw_calibration_kcat_figure_g3(Orimodel_solution_frame,ECMpy_solution_frame,ECMpy_adj_solution_frame,insvg,outsvg):
    model_data=pd.DataFrame()
    flux2_list=pd.DataFrame()
    flux3_list=pd.DataFrame()
    for reaction in ECMpy_adj_solution_frame.index:
        if re.search('_num',reaction):
            eachreaction = reaction.split('_num')[0]
            if eachreaction in flux2_list.index:
                flux2_list.loc[eachreaction,'fluxes'] = np.max([flux2_list.loc[eachreaction,'fluxes'],ECMpy_solution_frame.loc[eachreaction,'fluxes']])
            else:
                flux2_list.loc[eachreaction,'fluxes'] = ECMpy_solution_frame.loc[eachreaction,'fluxes']
            if eachreaction in flux3_list.index:
                flux3_list.loc[eachreaction,'fluxes'] = np.max([flux3_list.loc[eachreaction,'fluxes'],ECMpy_adj_solution_frame.loc[reaction,'fluxes']])
            else:
                flux3_list.loc[eachreaction,'fluxes'] = ECMpy_adj_solution_frame.loc[reaction,'fluxes']
        else:
            flux2_list.loc[reaction,'fluxes'] = ECMpy_solution_frame.loc[reaction,'fluxes']
            flux3_list.loc[reaction,'fluxes'] = ECMpy_adj_solution_frame.loc[reaction,'fluxes']

    for eachreaction in flux2_list.index:            
        if eachreaction in Orimodel_solution_frame.index:
            model_data.loc[eachreaction,'flux_cb']=str(round(Orimodel_solution_frame.loc[eachreaction,'Flux norm'],2))+\
                    ' # '+' # '+str(round(flux2_list.loc[eachreaction,'fluxes'],2))+' # ' \
                    +' # '+str(round(flux3_list.loc[eachreaction,'fluxes'],2))
        else:
            model_data.loc[eachreaction,'flux_cb']='Not give'+\
                    ' # '+' # '+str(round(flux2_list.loc[eachreaction,'fluxes'],2))+' # ' \
                    +' # '+str(round(flux3_list.loc[eachreaction,'fluxes'],2))    
                        
    rclass=['st6','st15','st18','st16']
    draw_svg(model_data,'flux_cb',insvg,outsvg,rclass)

def draw_different_model_cb_figure(model_data,insvg,outsvg):
    cb_df=pd.DataFrame()
    for index, row in model_data.iterrows():
        index_reverse=index+'_reverse'
        if re.search('reverse',index):
            pass
        elif index_reverse in model_data.index:
            model_ori_reverse_fluxes=float(model_data.loc[index_reverse,'model_ori_fluxes'])
            ECMpy_reverse_fluxes=float(model_data.loc[index_reverse,'ECMpy_fluxes'])
            model_gecko_adj_subunit_reverse_fluxes=float(model_data.loc[index_reverse,'model_gecko_adj_subunit_fluxes'])
            model_smoment_adj_subunit_reverse_fluxes=float(model_data.loc[index_reverse,'model_smoment_adj_subunit_fluxes'])
            
            if math.isnan(model_ori_reverse_fluxes):
                model_ori_fluxes=model_data.loc[index,'model_ori_fluxes']
            else:
                model_ori_fluxes=np.max([model_data.loc[index,'model_ori_fluxes'],model_data.loc[index_reverse,'model_ori_fluxes']])
                
            if math.isnan(ECMpy_reverse_fluxes):
                ECMpy_fluxes=model_data.loc[index,'ECMpy_fluxes']
            else:
                ECMpy_fluxes=np.max([model_data.loc[index,'ECMpy_fluxes'],model_data.loc[index_reverse,'ECMpy_fluxes']])
                
            if math.isnan(model_gecko_adj_subunit_reverse_fluxes):
                model_gecko_adj_subunit_fluxes=model_data.loc[index,'model_gecko_adj_subunit_fluxes']
            else:
                model_gecko_adj_subunit_fluxes=np.max([model_data.loc[index,'model_gecko_adj_subunit_fluxes'],\
                                                    model_data.loc[index_reverse,'model_gecko_adj_subunit_fluxes']])
                
            if math.isnan(model_smoment_adj_subunit_reverse_fluxes):
                model_smoment_adj_subunit_fluxes=model_data.loc[index,'model_smoment_adj_subunit_fluxes']
            else:
                model_smoment_adj_subunit_fluxes=np.max([model_data.loc[index,'model_smoment_adj_subunit_fluxes'],\
                                                        model_data.loc[index_reverse,'model_smoment_adj_subunit_fluxes']])
            flux_cb=str(round(model_ori_fluxes,2))+' # '+' # '+str(round(ECMpy_fluxes,2))+' # ' \
        +' # '+str(round(model_gecko_adj_subunit_fluxes,2))+' # '+str(round(model_smoment_adj_subunit_fluxes,2))
            cb_df.loc[index,'flux_cb2']=flux_cb
        else:
            flux_cb=str(round(row['model_ori_fluxes'],2))+' # '+' # '+str(round(row['ECMpy_fluxes'],2))+' # ' \
            +' # '+str(round(row['model_gecko_adj_subunit_fluxes'],2))+' # '+str(round(row['model_smoment_adj_subunit_fluxes'],2))
            cb_df.loc[index,'flux_cb2']=flux_cb

    rclass=['st6','st15','st18','st16']
    draw_svg(cb_df,'flux_cb2',insvg,outsvg,rclass)

def draw_different_model_cb_figure_g(model_data,insvg,outsvg):
    cb_df=pd.DataFrame()
    for index, row in model_data.iterrows():
        flux_cb=str(round(row['model_ori_fluxes'],2))+' # '+' # '+str(round(row['ECMpy_fluxes'],2))+' # ' \
            +' # '+str(round(row['model_gecko_adj_subunit_fluxes'],2))+' # '+str(round(row['model_smoment_adj_subunit_fluxes'],2))
        cb_df.loc[index,'flux_cb2']=flux_cb

    rclass=['st6','st15','st18','st16']
    draw_svg(cb_df,'flux_cb2',insvg,outsvg,rclass)

def get_fluxes_detail_in_model(model,fluxes_outfile,reaction_kcat_mw_file):
    model_pfba_solution = cobra.flux_analysis.pfba(model)
    model_pfba_solution = model_pfba_solution.to_frame()
    reaction_kcat_mw = pd.read_csv(reaction_kcat_mw_file, index_col=0)
    model_pfba_solution_detail = pd.DataFrame()
    for index, row in model_pfba_solution.iterrows():
        reaction_detail = model.reactions.get_by_id(index)
        model_pfba_solution_detail.loc[index,'fluxes'] = row['fluxes']
        if index in reaction_kcat_mw.index:
            model_pfba_solution_detail.loc[index,'kcat'] = reaction_kcat_mw.loc[index,'kcat']
            model_pfba_solution_detail.loc[index,'MW'] = reaction_kcat_mw.loc[index,'MW']
            model_pfba_solution_detail.loc[index,'kcat_MW'] = reaction_kcat_mw.loc[index,'kcat_MW']
            if 'source' in reaction_kcat_mw.columns:
                model_pfba_solution_detail.loc[index,'source'] = reaction_kcat_mw.loc[index,'source']
        model_pfba_solution_detail.loc[index,'equ'] = reaction_detail.reaction
    model_pfba_solution_detail.to_csv(fluxes_outfile) 
    return model_pfba_solution

def change_reaction_kcat_by_database(select_reaction,kcat_data_colect_file,reaction_kcat_mw_file,reaction_kapp_change_file):
    reaction_kcat_mw = pd.read_csv(reaction_kcat_mw_file, index_col=0)
    kcat_data_colect = pd.read_csv(kcat_data_colect_file, index_col=0)

    reaction_change_accord_fold=[]
    for eachreaction in select_reaction:
        if eachreaction in kcat_data_colect.index:
            if reaction_kcat_mw.loc[eachreaction,'kcat'] < kcat_data_colect.loc[eachreaction, 'kcat'] * 3600:
                reaction_kcat_mw.loc[eachreaction,'kcat'] = kcat_data_colect.loc[eachreaction, 'kcat']  * 3600
                reaction_kcat_mw.loc[eachreaction,'kcat_MW'] = kcat_data_colect.loc[eachreaction, 'kcat'] * 3600/reaction_kcat_mw.loc[eachreaction,'MW']
                reaction_kcat_mw.loc[eachreaction,'source'] = kcat_data_colect.loc[eachreaction, 'SOURCE']
                reaction_change_accord_fold.append(eachreaction)

    reaction_kcat_mw.to_csv(reaction_kapp_change_file)
    return(reaction_change_accord_fold)

def change_reaction_kcat_by_database_kapp(select_reaction,kcat_data_colect_file,reaction_kcat_mw_file,reaction_kapp_change_file):
    reaction_kcat_mw = pd.read_csv(reaction_kcat_mw_file, index_col=0)
    kcat_data_colect = pd.read_csv(kcat_data_colect_file, index_col=0)

    reaction_change_accord_fold=[]
    for eachreaction in select_reaction:
        if eachreaction in kcat_data_colect.index:
            if reaction_kcat_mw.loc[eachreaction,'kcat'] < kcat_data_colect.loc[eachreaction, 'kcat'] *2 * 3600:
                reaction_kcat_mw.loc[eachreaction,'kcat'] = kcat_data_colect.loc[eachreaction, 'kcat']  *2 * 3600
                reaction_kcat_mw.loc[eachreaction,'kcat_MW'] = kcat_data_colect.loc[eachreaction, 'kcat']  *2* 3600/reaction_kcat_mw.loc[eachreaction,'MW']
                reaction_kcat_mw.loc[eachreaction,'source'] = kcat_data_colect.loc[eachreaction, 'SOURCE']
                reaction_change_accord_fold.append(eachreaction)

    reaction_kcat_mw.to_csv(reaction_kapp_change_file)
    return(reaction_change_accord_fold)

def change_reaction_kcat_by_database_kapp_g(select_reaction,kcat_data_colect_file,reaction_kcat_mw_file,reaction_kapp_change_file):
    reaction_kcat_mw = pd.read_csv(reaction_kcat_mw_file, index_col=0)
    kcat_data_colect = pd.read_csv(kcat_data_colect_file, index_col=0)

    reaction_change_accord_fold=[]
    for reaction in select_reaction:
        if re.search('_num',reaction):
            eachreaction = reaction.split('_num')[0]
        else:
            eachreaction = reaction
        if eachreaction in kcat_data_colect.index:
            if reaction_kcat_mw.loc[reaction,'kcat'] < kcat_data_colect.loc[eachreaction, 'kcat'] *2 * 3600:
                reaction_kcat_mw.loc[reaction,'kcat'] = kcat_data_colect.loc[eachreaction, 'kcat']  *2 * 3600
                reaction_kcat_mw.loc[reaction,'kcat_MW'] = kcat_data_colect.loc[eachreaction, 'kcat']  *2* 3600/reaction_kcat_mw.loc[reaction,'MW']
                reaction_kcat_mw.loc[reaction,'source'] = kcat_data_colect.loc[eachreaction, 'SOURCE']
                reaction_change_accord_fold.append(reaction)

    reaction_kcat_mw.to_csv(reaction_kapp_change_file)
    return(reaction_change_accord_fold)
def change_reaction_kcat_by_database_g(select_reaction,kcat_data_colect_file,reaction_kcat_mw_file,reaction_kapp_change_file):
    reaction_kcat_mw = pd.read_csv(reaction_kcat_mw_file, index_col=0)
    kcat_data_colect = pd.read_csv(kcat_data_colect_file, index_col=0)

    reaction_change_accord_fold=[]
    for reaction in select_reaction:
        if re.search('_num',reaction):
            eachreaction = reaction.split('_num')[0]
        else:
            eachreaction = reaction
        if eachreaction in kcat_data_colect.index:
            if reaction_kcat_mw.loc[reaction,'kcat'] < kcat_data_colect.loc[eachreaction, 'kcat'] * 3600:
                reaction_kcat_mw.loc[reaction,'kcat'] = kcat_data_colect.loc[eachreaction, 'kcat']  * 3600
                reaction_kcat_mw.loc[reaction,'kcat_MW'] = kcat_data_colect.loc[eachreaction, 'kcat'] * 3600/reaction_kcat_mw.loc[reaction,'MW']
                reaction_kcat_mw.loc[reaction,'source'] = kcat_data_colect.loc[eachreaction, 'SOURCE']
                reaction_change_accord_fold.append(reaction)

    reaction_kcat_mw.to_csv(reaction_kapp_change_file)
    return(reaction_change_accord_fold)

def select_calibration_reaction_by_biomass(reaction_kcat_mw_file, json_model_path, enzyme_amount, percentage, reaction_biomass_outfile, select_value):
    reaction_kcat_mw = pd.read_csv(reaction_kcat_mw_file, index_col=0)
    norm_model=cobra.io.json.load_json_model(json_model_path)
    norm_biomass=norm_model.slim_optimize() 
    df_biomass = pd.DataFrame()
    df_biomass_select = pd.DataFrame()
    for r in norm_model.reactions:
        with norm_model as model:
            if r.id in list(reaction_kcat_mw.index):
                r.bounds = (0, reaction_kcat_mw.loc[r.id,'kcat_MW']*enzyme_amount*percentage)
                df_biomass.loc[r.id,'biomass'] = model.slim_optimize()
                biomass_diff = norm_biomass-model.slim_optimize()
                biomass_diff_ratio = (norm_biomass-model.slim_optimize())/norm_biomass
                df_biomass.loc[r.id,'biomass_diff'] = biomass_diff
                df_biomass.loc[r.id,'biomass_diff_ratio'] = biomass_diff_ratio
                if model.slim_optimize() < select_value*norm_biomass: #select difference range
                    df_biomass_select.loc[r.id,'biomass_enz'] = model.slim_optimize()
                    df_biomass_select.loc[r.id,'biomass_orimodel'] = norm_biomass
                    df_biomass_select.loc[r.id,'biomass_diff'] = biomass_diff
                    df_biomass_select.loc[r.id,'biomass_diff_ratio'] = biomass_diff_ratio

    df_biomass = df_biomass.sort_values(by="biomass_diff_ratio",axis = 0,ascending = False)
    df_biomass.to_csv(reaction_biomass_outfile)

    if df_biomass_select.empty:
        return('no change')
    else:
        df_reaction_select = df_biomass_select.sort_values(by="biomass_diff_ratio",axis = 0,ascending = False)
        return(df_reaction_select)

def select_calibration_reaction_by_c13(reaction_kcat_mw_file, c13reaction_file, enzyme_amount, percentage, sigma):
    reaction_kcat_mw = pd.read_csv(reaction_kcat_mw_file, index_col=0)
    c13reaction = pd.read_csv(c13reaction_file, index_col=0)
    c13reaction_selecet =[]      
    for index,row in c13reaction.iterrows():
        if  index in reaction_kcat_mw.index:
            ECMpy_c13_reaction_flux = reaction_kcat_mw.loc[index,'kcat_MW']*enzyme_amount*percentage*sigma
            if ECMpy_c13_reaction_flux < row['Flux norm']:
                c13reaction_selecet.append(index)   
    return(c13reaction_selecet)

# 将所有的gpr关系为or的反应拆开
def reaction_gene_subunit_MW_split(reaction_gene_subunit_MW):
    reaction_gene_subunit_MW_new = pd.DataFrame()
    for reaction, data in reaction_gene_subunit_MW.iterrows():
        #PDH酶是复合体，分子量太大，进行拆分
        if reaction =='PDH':
            gene = enumerate(data['gene_reaction_rule'].split(" and "))
            subunit_mw= data['subunit_mw'].split(" and ")
            subunit_num = data['subunit_num'].split(" and ")
            for index, value in gene:
                if index == 0:
                    reaction_new = reaction + "_num1"
                    reaction_gene_subunit_MW_new.loc[reaction_new,'name'] = data['name']
                    reaction_gene_subunit_MW_new.loc[reaction_new,'gene_reaction_rule'] = value
                    reaction_gene_subunit_MW_new.loc[reaction_new,'subunit_mw'] = subunit_mw[index]
                    reaction_gene_subunit_MW_new.loc[reaction_new,'subunit_num'] = subunit_num[index]
                else:
                    reaction_new = reaction + "_num" + str(index+1)
                    reaction_gene_subunit_MW_new.loc[reaction_new,'name'] = data['name']
                    reaction_gene_subunit_MW_new.loc[reaction_new,'gene_reaction_rule'] = value
                    reaction_gene_subunit_MW_new.loc[reaction_new,'subunit_mw'] = subunit_mw[index]
                    reaction_gene_subunit_MW_new.loc[reaction_new,'subunit_num'] = subunit_num[index] 
        elif re.search(" or ", data['gene_reaction_rule']):
            gene = enumerate(data['gene_reaction_rule'].split(" or "))
            subunit_mw= data['subunit_mw'].split(" or ")
            subunit_num = data['subunit_num'].split(" or ")
            for index, value in gene:
                if index == 0:
                    reaction_new = reaction + "_num1"
                    reaction_gene_subunit_MW_new.loc[reaction_new,'name'] = data['name']
                    reaction_gene_subunit_MW_new.loc[reaction_new,'gene_reaction_rule'] = value
                    reaction_gene_subunit_MW_new.loc[reaction_new,'subunit_mw'] = subunit_mw[index]
                    reaction_gene_subunit_MW_new.loc[reaction_new,'subunit_num'] = subunit_num[index]
                else:
                    reaction_new = reaction + "_num" + str(index+1)
                    reaction_gene_subunit_MW_new.loc[reaction_new,'name'] = data['name']
                    reaction_gene_subunit_MW_new.loc[reaction_new,'gene_reaction_rule'] = value
                    reaction_gene_subunit_MW_new.loc[reaction_new,'subunit_mw'] = subunit_mw[index]
                    reaction_gene_subunit_MW_new.loc[reaction_new,'subunit_num'] = subunit_num[index] 
        else:
            reaction_gene_subunit_MW_new.loc[reaction,'name'] = data['name']
            reaction_gene_subunit_MW_new.loc[reaction,'gene_reaction_rule'] = data['gene_reaction_rule']
            reaction_gene_subunit_MW_new.loc[reaction,'subunit_mw'] = data['subunit_mw']
            reaction_gene_subunit_MW_new.loc[reaction,'subunit_num'] = data['subunit_num']
    reaction_gene_subunit_MW_new.to_csv("./data/reaction_gene_subunit_MW.csv") 

def reaction_gene_subunit_MW_split_only(reaction_gene_subunit_MW):
    reaction_gene_subunit_MW_new = pd.DataFrame()
    for reaction, data in reaction_gene_subunit_MW.iterrows():
        if re.search(" or ", data['gene_reaction_rule']):
            gene = enumerate(data['gene_reaction_rule'].split(" or "))
            subunit_mw= data['subunit_mw'].split(" or ")
            subunit_num = data['subunit_num'].split(" or ")
            for index, value in gene:
                if index == 0:
                    reaction_new = reaction + "_num1"
                    reaction_gene_subunit_MW_new.loc[reaction_new,'name'] = data['name']
                    reaction_gene_subunit_MW_new.loc[reaction_new,'gene_reaction_rule'] = value
                    reaction_gene_subunit_MW_new.loc[reaction_new,'subunit_mw'] = subunit_mw[index]
                    reaction_gene_subunit_MW_new.loc[reaction_new,'subunit_num'] = subunit_num[index]
                else:
                    reaction_new = reaction + "_num" + str(index+1)
                    reaction_gene_subunit_MW_new.loc[reaction_new,'name'] = data['name']
                    reaction_gene_subunit_MW_new.loc[reaction_new,'gene_reaction_rule'] = value
                    reaction_gene_subunit_MW_new.loc[reaction_new,'subunit_mw'] = subunit_mw[index]
                    reaction_gene_subunit_MW_new.loc[reaction_new,'subunit_num'] = subunit_num[index] 
        else:
            reaction_gene_subunit_MW_new.loc[reaction,'name'] = data['name']
            reaction_gene_subunit_MW_new.loc[reaction,'gene_reaction_rule'] = data['gene_reaction_rule']
            reaction_gene_subunit_MW_new.loc[reaction,'subunit_mw'] = data['subunit_mw']
            reaction_gene_subunit_MW_new.loc[reaction,'subunit_num'] = data['subunit_num']
    reaction_gene_subunit_MW_new.to_csv("./data/reaction_gene_subunit_MW.csv") 

def reaction_gene_subunit_MW_split_PDH(reaction_gene_subunit_MW):
    reaction_gene_subunit_MW_new = pd.DataFrame()
    for reaction, data in reaction_gene_subunit_MW.iterrows():
        #PDH酶是复合体，分子量太大，进行拆分
        if reaction =='PDH':
            gene = enumerate(data['gene_reaction_rule'].split(" and "))
            subunit_mw= data['subunit_mw'].split(" and ")
            subunit_num = data['subunit_num'].split(" and ")
            for index, value in gene:
                if index == 0:
                    reaction_new = reaction + "_num1"
                    reaction_gene_subunit_MW_new.loc[reaction_new,'name'] = data['name']
                    reaction_gene_subunit_MW_new.loc[reaction_new,'gene_reaction_rule'] = value
                    reaction_gene_subunit_MW_new.loc[reaction_new,'subunit_mw'] = subunit_mw[index]
                    reaction_gene_subunit_MW_new.loc[reaction_new,'subunit_num'] = subunit_num[index]
                else:
                    reaction_new = reaction + "_num" + str(index+1)
                    reaction_gene_subunit_MW_new.loc[reaction_new,'name'] = data['name']
                    reaction_gene_subunit_MW_new.loc[reaction_new,'gene_reaction_rule'] = value
                    reaction_gene_subunit_MW_new.loc[reaction_new,'subunit_mw'] = subunit_mw[index]
                    reaction_gene_subunit_MW_new.loc[reaction_new,'subunit_num'] = subunit_num[index] 
        else:
            reaction_gene_subunit_MW_new.loc[reaction,'name'] = data['name']
            reaction_gene_subunit_MW_new.loc[reaction,'gene_reaction_rule'] = data['gene_reaction_rule']
            reaction_gene_subunit_MW_new.loc[reaction,'subunit_mw'] = data['subunit_mw']
            reaction_gene_subunit_MW_new.loc[reaction,'subunit_num'] = data['subunit_num']
    reaction_gene_subunit_MW_new.to_csv("./data/reaction_gene_subunit_MW.csv") 

# 将所有的gpr关系为or的反应拆开
def isoenzyme_split(model):
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
                    model.add_reaction(r_add)     
        #PDH酶是复合体，分子量太大，进行拆分
        if r.id =='PDH':
            rea = r.copy()
            gene = r.gene_reaction_rule.split(" and ")
            for index, value in enumerate(gene):
                if index == 0:
                    r.id = r.id + "_num1"
                    r.gene_reaction_rule = value
                else:
                    r_add = rea.copy()
                    r_add.id = rea.id + "_num" + str(index+1)
                    r_add.gene_reaction_rule = value
                    model.add_reaction(r_add)        
    for r in model.reactions:
        r.gene_reaction_rule = r.gene_reaction_rule.strip("( )")
    return model

def isoenzyme_split_only(model):
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
                    model.add_reaction(r_add)           
    for r in model.reactions:
        r.gene_reaction_rule = r.gene_reaction_rule.strip("( )")
    return model

def isoenzyme_split_PDH(model):
    for r in model.reactions:
        #PDH酶是复合体，分子量太大，进行拆分
        if r.id =='PDH':
            rea = r.copy()
            gene = r.gene_reaction_rule.split(" and ")
            for index, value in enumerate(gene):
                if index == 0:
                    r.id = r.id + "_num1"
                    r.gene_reaction_rule = value
                else:
                    r_add = rea.copy()
                    r_add.id = rea.id + "_num" + str(index+1)
                    r_add.gene_reaction_rule = value
                    model.add_reaction(r_add)        
    return model

def get_diff_reaction_use_c13(c13reaction_file,model_fluxes):
    c13reaction = pd.read_csv(c13reaction_file, index_col=0)
    c13reaction = list(c13reaction.index)
    enz_model_pfba_solution_select = model_fluxes[model_fluxes['fluxes']>0]
    enz_model_pfba_solution_select_id = []
    for eachreaction in enz_model_pfba_solution_select.index:
        if re.search('_num',eachreaction):
            enz_model_pfba_solution_select_id.append(eachreaction.split('_num')[0])
        else:
            enz_model_pfba_solution_select_id.append(eachreaction)
    c13reaction_2_enz_model_diff=list(set(c13reaction).difference(set(enz_model_pfba_solution_select_id)))   
    return(c13reaction_2_enz_model_diff)

def get_enz_model_use_biomass_diff(reaction_kcat_mw_file, json_model_path, percentage, reaction_biomass_outfile, select_percentage,kcat_data_colect_file, model_file, f, ptot, sigma, lowerbound, upperbound, json_output_file):

    df_reaction_select = select_calibration_reaction_by_biomass(reaction_kcat_mw_file, json_model_path, upperbound, percentage, reaction_biomass_outfile, select_percentage)

    if isinstance(df_reaction_select, pd.DataFrame):
        need_change_reaction=list(df_reaction_select.index)
        reaction_kapp_change_file = "./analysis/reaction_change_by_biomass.csv"
        change_reaction_list_round1=change_reaction_kcat_by_database_kapp_g(need_change_reaction,kcat_data_colect_file,reaction_kcat_mw_file,reaction_kapp_change_file)
        print(change_reaction_list_round1)
        reaction_kcat_mw_file="./analysis/reaction_change_by_biomass.csv"

    trans_model2enz_json_model_split_isoenzyme_only(model_file, reaction_kcat_mw_file, f, ptot, sigma, lowerbound, upperbound, json_output_file)
    enz_model=get_enzyme_constraint_model(json_output_file)
    return [df_reaction_select,enz_model]

def get_enz_model_use_c13(reaction_kcat_mw_file, c13reaction_file, percentage, df_reaction_select,kcat_data_colect_file,model_file, f, ptot, sigma, lowerbound, upperbound, json_output_file):

    c13reaction_selecet=select_calibration_reaction_by_c13(reaction_kcat_mw_file, c13reaction_file, upperbound, percentage, sigma)
    print(c13reaction_selecet)

    if isinstance(df_reaction_select, pd.DataFrame):    
        reaction_kcat_mw_file="./analysis/reaction_change_by_biomass.csv"

    reaction_kapp_change_file = "./analysis/reaction_change_by_c13.csv"
    #c13reaction_selecet=['CS','ACONTa','ACONTb','ICDHyr','MALS', 'MDH', 'ICL', 'SUCOAS_reverse', 'SUCDi', 'AKGDH']
    change_reaction_list_round1=change_reaction_kcat_by_database_kapp_g(c13reaction_selecet,kcat_data_colect_file,reaction_kcat_mw_file,reaction_kapp_change_file)
    print(change_reaction_list_round1)

    reaction_kcat_mw_file = "./analysis/reaction_change_by_c13.csv"
    trans_model2enz_json_model_split_isoenzyme_only(model_file, reaction_kcat_mw_file, f, ptot, sigma, lowerbound, upperbound, json_output_file)
    enz_model=get_enzyme_constraint_model(json_output_file)
    return enz_model

def get_enz_model_use_enz_usage(enz_ratio,reaction_flux_file,reaction_kcat_mw_file,reaction_enz_usage_file,kcat_data_colect_file,model_file, f, ptot, sigma, lowerbound, upperbound, json_output_file):

    reaction_enz_usage_df = get_enzyme_usage(upperbound,reaction_flux_file,reaction_kcat_mw_file,reaction_enz_usage_file)

    select_reaction = list(reaction_enz_usage_df[reaction_enz_usage_df['enz ratio']>enz_ratio].index)#more than 1%
    print(select_reaction)
    reaction_kapp_change_file = "./analysis/reaction_change_by_enzuse.csv"
    change_reaction_list_round1=change_reaction_kcat_by_database_kapp_g(select_reaction,kcat_data_colect_file,reaction_kcat_mw_file,reaction_kapp_change_file)
    print(change_reaction_list_round1)

    reaction_kcat_mw_file="./analysis/reaction_change_by_enzuse.csv"
    trans_model2enz_json_model_split_isoenzyme_only(model_file, reaction_kcat_mw_file, f, ptot, sigma, lowerbound, upperbound, json_output_file)

    enz_model=get_enzyme_constraint_model(json_output_file)
    return enz_model
