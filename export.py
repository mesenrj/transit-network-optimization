import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import geopandas as gpd
import itertools
import time
import random


def return_xpress_int_txt(array, filename=None):
    string = ''
    
    for line in array:
        string += str(round(line, 2)) + ' '
            
        # string += '\n'
    
    xpress_input = string
    
    # If has no filename, print results
    if filename == None:
        return print(xpress_input)
    
    # If has filename, save to disk and print
    else:
        
        # Create file
        f = open(filename, "w")
        f.write(xpress_input)
        f.close()
        
        return print(f'DONE | File {filename} saved to disk.')


def return_xpress_str_txt(array, filename=None):
    string = ''
    
    for line in array:
        string += '"' + str(line) + '" '
    
    xpress_input = string
    
    # If has no filename, print results
    if filename == None:
        return print(xpress_input)
    
    # If has filename, save to disk and print
    else:
        
        # Create file
        f = open(filename, "w")
        f.write(xpress_input)
        f.close()
        
        return print(f'DONE | File {filename} saved to disk.')


def return_DELTA_xpress_array(array):
    start = '['
    string = ''
    end = ']'
    
    for line in array:
        for item in line:
            string += str(item) + ', '
            
        # string += '\n'
    
    string = string[:-2]
    xpress_input = start + string + end
    
    return print(xpress_input)


def return_delta_xpress_array(array):
    start = '['
    string = ''
    end = ']'
    
    for od in array:
        for path in od:
            for edge in path:
                string += str(edge) + ', '
            
        # string += '\n'
    
    string = string[:-2]
    xpress_input = start + string + end
    
    return print(xpress_input)


def return_xpress_array(array):
    start = '['
    string = ''
    end = ']'
    
    for line in array:
        string += '"' + str(line) + '", '
            
        # string += '\n'
    
    string = string[:-2]
    xpress_input = start + string + end
    
    return print(xpress_input)


def return_xpress_int_array(array):
    start = '['
    string = ''
    end = ']'
    
    for line in array:
        string += '' + str(round(line, 2)) + ', '
            
        # string += '\n'
    
    string = string[:-2]
    xpress_input = start + string + end
    
    return print(xpress_input)