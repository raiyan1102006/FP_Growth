##################
# Import libraries
##################

import pandas as pd
import numpy as np
import math
import itertools
from operator import itemgetter
import json



###########
# Load data
###########

adult = pd.read_csv('data/adult.data', header=None)
adult.columns = ['age', 'workclass','fnlwgt', 'education', 
                'education-num','marital-status','occupation', 'relationship','race',
                'sex','capital-gain','capital-loss','hours-per-week','native-country','income_category']


################
# Pre-processing
################

#drop redundant column
adult = adult.drop("education-num", axis=1) 

#deal with ? entries
columns_with_blank_entries=["workclass","occupation","native-country"] 
for column in columns_with_blank_entries:
    adult["converted_"+column] = adult[column].astype(str).replace(" ", "")+"_"+column
    adult = adult.drop(column, axis=1)
    
#deal with numeric attribures
numeric_columns = ["age","fnlwgt","capital-gain","capital-loss","hours-per-week"] 
for column in numeric_columns:
    bins = np.histogram(adult[column])
    bins = list(bins[1]) #generate 10 equal-width bins
    if(bins[0]==0.0):
        bins[0]=-0.1
    category = pd.cut(adult[column],bins)
    category = category.to_frame()
    category.columns = ['converted_'+column]
    adult = pd.concat([adult,category],axis = 1)
    adult["converted_"+column] = column+"_"+adult["converted_"+column].astype(str).replace(" ", "")
    adult = adult.drop(column, axis=1)
    
#convert to dictionary format for using as an input to the program
dict_table = {}
temp_list=[]
for index, data in adult.iterrows():
    dict_table[str(index)] = data.tolist()
    
    
###########
# Functions
###########

def generate_pattern_base(fp_tree,beta): #generates pattern base
    temp_dict = {}
    keys = [key for key,value in fp_tree.items() if value['item']==beta[0]]
    
    for key in keys: #each key of fp_tree is a beta[0] leaf
        #find ancestors from tree, might be inefficient
        list_ancestors = []
        parent_index = fp_tree[key]['parent']
        while(parent_index):
            list_ancestors.append(fp_tree[parent_index]['item']) 
            parent_index = fp_tree[parent_index]['parent']
        list_ancestors.reverse()
        #make sure it HAS ancestors
        if(list_ancestors):
            for i in range(fp_tree[key]['count']):
                temp_dict[len(temp_dict)] = list_ancestors
    return temp_dict
    

def FP_mining(fp_tree,alpha,count_itemset,min_sup_count,final_itemset):
    first_gen_children = [key for key,value in fp_tree.items() if value['parent'] == 0] # find how many branches are connected to root
    
    #tree contains single path
    if(len(first_gen_children)==1): 
        nodes_in_path = [value['item'] for key,value in fp_tree.items()]

        #create subsets
        subset_list = []
        for cardinality in range(1,len(nodes_in_path) + 1):
            temp_list = list(itertools.combinations(nodes_in_path,cardinality))
            for x in temp_list:
                subset_list.append(list(x))
        
        #generate itemsets
        for x in subset_list:
            temp_itemset=x
            temp_itemset.extend(alpha)
            temp_sup_count = min([value['count'] for key,value in fp_tree.items()])
            final_itemset.append({'itemset':temp_itemset,'support':temp_sup_count})
     
    #tree contains multiple paths
    else: 
        collection_a = list(set([value['item'] for key,value in fp_tree.items()])) #finding header entries in the tree
        sorted_itemset = get_flattened_list_from_dict(count_itemset)
        collection_a = sorted(collection_a, key=lambda x: sorted_itemset.index(x), reverse=True) #sorting in reverse order of frequency
        
        for a_i in collection_a: #each entry in the header
            beta = [a_i]
            beta.extend(alpha)

            #append to final itemsets
            temp_sup_count = [x['support'] for x in count_itemset if x['itemset']==[a_i]]
            final_itemset.append({'itemset':beta,'support':temp_sup_count[0]})
            
            #generate pattern base
            beta_pattern_base_dict = generate_pattern_base(fp_tree,beta) 
            candidates = generate_candidates(beta_pattern_base_dict)
            pruned_candidates = [x for x in candidates if x["support"] >= min_sup_count]
            flattened_L = get_flattened_list_from_dict(pruned_candidates)
            
            #generate tree
            beta_fp_tree={}
            for transaction_list in beta_pattern_base_dict.values(): 
                clean_transaction_list = [x for x in transaction_list if x in flattened_L] #removing entries in transaction list that didn't make it through to frequent 1-itemset
                ordered_transaction_list =  sorted(clean_transaction_list, key=lambda x: flattened_L.index(x)) #ordering by L
                beta_fp_tree = update_tree(beta_fp_tree,ordered_transaction_list)
  
            #recursive calling of the mining function
            if(beta_fp_tree):
                final_itemset = FP_mining(beta_fp_tree,beta,pruned_candidates,min_sup_count,final_itemset)
        
    return final_itemset

def update_tree(fp_tree,ordered_transaction_list):  #adds a transaction to the FP-tree
    address_id = 0
    for item_index, item in enumerate(ordered_transaction_list):
        temp_key = [key for key,value in fp_tree.items() if value['parent'] == address_id and value['item']==item]
        
        if(not temp_key):   #create a new node
            temp_dict = {'item' : item, 'count':1,'parent':address_id}
            address_id = len(fp_tree)+1
            fp_tree[address_id]=temp_dict
        
        else:               #update count of an existing node
            fp_tree[temp_key[0]]['count']+=1
            address_id = temp_key[0]
        
    return fp_tree

def get_flattened_list_from_dict(count_dict_list):
    list_ = [x['itemset'] for x in count_dict_list]
    flattened_list = [x1 for [x1] in list_]
    return flattened_list
    
def generate_candidates(dict_table):
    temp_C=[]
    items = [item for dict_list in dict_table.values() for item in dict_list]
    values, counts = np.unique(items, return_counts=True)
    for x in range(len(values)):
        temp_C.append({"itemset":[values[x]],"support":counts[x]})
    return temp_C

def prune_and_sort(candidates, min_sup_count):
    pruned_candidates = [x for x in candidates if x["support"] >= min_sup_count]
    sorted_pruned_L_count = sorted(pruned_candidates, key=itemgetter("itemset"), reverse=False) #first sort the items alphabetically
    sorted_pruned_L_count = sorted(sorted_pruned_L_count, key=itemgetter("support"), reverse=True) #then sort by sup_count
    return sorted_pruned_L_count


def FP_growth(dict_table,support):
    min_sup_count = len(dict_table)*support
    
    print("First scan of the database")
    candidates = generate_candidates(dict_table) #list of candidate dicts
    sorted_pruned_L_count = prune_and_sort(candidates, min_sup_count) #prune by min_sup, sort by min_sup

    flattened_L = get_flattened_list_from_dict(sorted_pruned_L_count) #used for sorting in order of support counts

    print("Second scan of the database: Creating the FP-tree")
    #creating the tree
    fp_tree={}
    for transaction_list in dict_table.values(): 
        clean_transaction_list = [x for x in transaction_list if x in flattened_L] #removing entries in transaction list that didn't make it through to frequent 1-itemset
        ordered_transaction_list =  sorted(clean_transaction_list, key=lambda x: flattened_L.index(x)) #ordering by L
        fp_tree = update_tree(fp_tree,ordered_transaction_list) #updating the transaction to the tree

    print("Mining the patterns")
    #mining patterns
    frequent_patterns=[]
    frequent_patterns = FP_mining(fp_tree,[],sorted_pruned_L_count,min_sup_count,frequent_patterns) 
    
    for x in frequent_patterns: #converting support count to support
        x['support']=(float(x['support'])/float(len(dict_table)))
    
    return frequent_patterns


################
# Initialization
################

if __name__=='__main__':
    support = 0.23
    frequent_patterns = FP_growth(dict_table,support=support)
    print(" ")
    print("Number of frequent itemsets:")
    print(len(frequent_patterns))
    print(" ")
    print("Frequent itemsets with support:")
    print(json.dumps(frequent_patterns, indent=4))

