#!/usr/local/bin/python2.7
# encoding: utf-8
'''
Created on 2017年5月15日

@author: lenovo
'''
def write_data(v_data, file_path='data/ml-100k/user_latent_factors_data.data'):
    output = open(file_path, "w")
    for line in v_data:
        reslt = ""
        for field in line:
            reslt += "%s," %field
        output.write(reslt[:-1] + "\n")
    output.close()
    
def write_pred_data(v_data, file_path = ''):
    output = open(file_path, "w")
    for line in v_data:
        output.write("%s\n" %line)
    output.close()
    
    
