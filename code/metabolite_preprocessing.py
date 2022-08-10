import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
import shutil
import pickle
import os
from os.path import join

import warnings
warnings.filterwarnings("ignore")
CURRENT_DIR = os.getcwd()
CURRENT_DIR = join(CURRENT_DIR, "..")


#df_metabolites = pd.read_pickle(join(CURRENT_DIR, "..", "data", "additional_data", "all_substrates.pkl"))

def reaction_preprocessing(substrate_list, product_list):
	#removing duplicated entries and creating a pandas DataFrame with all metabolites
    df_reaction = pd.DataFrame(data = {"substrates" : substrate_list, "products": product_list})
    df_reaction["ID"], df_reaction["reaction_message"] = np.nan, np.nan
    df_reaction["difference_fp"] = ""
    #each metabolite should be either a KEGG ID, InChI string, or a SMILES:
    for ind in df_reaction.index:
        df_reaction["ID"][ind] = "reaction_" + str(ind)
        left_site = get_reaction_site_smarts(df_reaction["substrates"][ind])
        right_site = get_reaction_site_smarts(df_reaction["products"][ind])
        if pd.isnull(left_site) or pd.isnull(right_site):
            df_reaction["reaction_message"][ind] = "invalid"
        elif "invalid" in left_site or "invalid" in right_site:
            df_reaction["reaction_message"][ind] = "invalid"
        else:
            rxn_forward = AllChem.ReactionFromSmarts(left_site + ">>" + right_site)
            difference_fp = Chem.rdChemReactions.CreateDifferenceFingerprintForReaction(rxn_forward)
            difference_fp = convert_fp_to_array(difference_fp.GetNonzeroElements())
            df_reaction["difference_fp"][ind] = difference_fp
            df_reaction["reaction_message"][ind] = "complete"
    return(df_reaction)


def get_metabolite_type(met):
    if is_KEGG_ID(met):
        return("KEGG")
    elif is_InChI(met):
        return("InChI")
    elif is_SMILES(met):
        return("SMILES")
    else:
        return("invalid")

def get_reaction_site_smarts(metabolites):
    reaction_site = ""
    metabolites = metabolites.split(";")
    for met in metabolites:
        met_type = get_metabolite_type(met)

        if met_type == "KEGG":
            Smarts = Chem.MolToSmarts(Chem.MolFromMolFile(join(CURRENT_DIR, "data", "mol-files",  met + ".mol")))
        elif met_type == "InChI":
            Smarts = Chem.MolToSmarts(Chem.inchi.MolFromInchi(met))
        elif met_type == "SMILES":
            Smarts = Chem.MolToSmarts(Chem.MolFromSmiles(met))
        else:
            Smarts = "invalid"

        reaction_site = reaction_site + "." + Smarts
    return(reaction_site[1:])


def is_KEGG_ID(met):
    #a valid KEGG ID starts with a "C" or "D" followed by a 5 digit number:
    if len(met) == 6 and met[0] in ["C", "D"]:
        try:
            int(met[1:])
            return(True)
        except: 
            pass
    return(False)

def is_SMILES(met):
    m = Chem.MolFromSmiles(met,sanitize=False)
    if m is None:
      return(False)
    else:
      try:
        Chem.SanitizeMol(m)
      except:
        print('.......Metabolite string "%s" is in SMILES format but has invalid chemistry' % met)
        return(False)
    return(True)

def is_InChI(met):
    m = Chem.inchi.MolFromInchi(met,sanitize=False)
    if m is None:
      return(False)
    else:
      try:
        Chem.SanitizeMol(m)
      except:
        print('.......Metabolite string "%s" is in InChI format but has invalid chemistry' % met)
        return(False)
    return(True)

def convert_fp_to_array(difference_fp_dict):
    fp = np.zeros(2048)
    for key in difference_fp_dict.keys():
        fp[key] = difference_fp_dict[key]
    return(fp)