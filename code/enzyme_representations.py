import numpy as np
import pandas as pd
import shutil
import pickle
import torch
import esm
import os
from os.path import join


CURRENT_DIR = os.getcwd()
CURRENT_DIR = join(CURRENT_DIR, "..")

aa = set("abcdefghiklmnpqrstxvwyzv".upper())

def validate_enzyme(seq, alphabet=aa):
    "Checks that a sequence only contains values from an alphabet"
    leftover = set(seq.upper()) - alphabet
    return not leftover

def calcualte_esm1b_ts_vectors(enzyme_list):

	#creating model input:
	df_enzyme = preprocess_enzymes(enzyme_list)
	model_input = [(df_enzyme["ID"][ind], df_enzyme["model_input"][ind]) for ind in df_enzyme.index]
	seqs = [model_input[i][1] for i in range(len(model_input))]
	#loading ESM-1b model:
	print(".....2(a) Loading ESM-1b model.")
	#model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
	model_location = join(CURRENT_DIR, "data", "saved_models", "ESM1b", "esm1b_t33_650M_UR50S.pt")
	
	model_data = torch.load(model_location, map_location='cpu')
	regression_location = model_location[:-3] + "-contact-regression.pt"
	
	regression_data = torch.load(regression_location, map_location='cpu')
	model, alphabet = esm.pretrained.load_model_and_alphabet_core(model_data, regression_data)
	#model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_location = model_location)

	
	batch_converter = alphabet.get_batch_converter()
	print(".....2(b) Loading model parameters for task-specific model.")
	model.eval()
	PATH = join(CURRENT_DIR, "data", "saved_models", "ESM1b", 'model_ESM_binary_A100_epoch_1_new_split.pkl')
	model_dict = torch.load(PATH, map_location=torch.device('cpu'))
	model_dict_V2 = {k.split("model.")[-1]: v for k, v in model_dict.items()}

	for key in ["module.fc1.weight", "module.fc1.bias", "module.fc2.weight", "module.fc2.bias", "module.fc3.weight", "module.fc3.bias"]:
		del model_dict_V2[key]
	model.load_state_dict(model_dict_V2)

	#convert input into batches:
	#Calculate ESM-1b representations
	print(".....2(c) Calculating enzyme representations.")
	df_enzyme["enzyme rep"] = ""

	for ind in df_enzyme.index:
		if validate_enzyme(seq = df_enzyme["model_input"][ind]):

			batch_labels, batch_strs, batch_tokens = batch_converter([(df_enzyme["ID"][ind], df_enzyme["model_input"][ind])])
			with torch.no_grad():
			    results = model(batch_tokens, repr_layers=[33])
			df_enzyme["enzyme rep"][ind] = results["representations"][33][0][0].numpy() 
	return(df_enzyme)



def preprocess_enzymes(enzyme_list):
	df_enzyme = pd.DataFrame(data = {"amino acid sequence" : list(set(enzyme_list))})
	df_enzyme["ID"] = ["protein_" + str(ind) for ind in df_enzyme.index]
	#if length of sequence is longer than 1020 amino acids, we crop it:
	df_enzyme["model_input"] = [seq[:1022] for seq in df_enzyme["amino acid sequence"]]
	return(df_enzyme)
