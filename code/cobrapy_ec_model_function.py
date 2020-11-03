# This code is used to introduce enzyme concentration constraint in GEMs
# by COBRApy and to calculate the parameters that need to be entered
# during the construction of the enzyme-constrained model.
#from warnings import warn

import pandas as pd
import numpy as np
import json
import cobra
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
        if reaction.lower_bound < 0:
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
                if biomass_diff > select_value: #select difference range
                    df_biomass_select.loc[r.id,'biomass_diff'] = biomass_diff
                    df_biomass_select.loc[r.id,'biomass_diff_ratio'] = biomass_diff_ratio

    df_biomass = df_biomass.sort_values(by="biomass_diff",axis = 0,ascending = False)
    df_biomass.to_csv(reaction_biomass_outfile)

    if df_biomass_select.empty:
        pass
    else:
        df_reaction_select = df_biomass_select.sort_values(by="biomass_diff",axis = 0,ascending = False)
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
    