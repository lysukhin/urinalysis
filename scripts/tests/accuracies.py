import numpy as np
import pandas as pd


def nice_print(accs,indicators):
    for i in indicators:
        print i+"   ",
    print 
    for x in accs:
        print "%.2f"%x," ",
    print '\n'


def hard_accuracy(x,y,indicators):
    try:
        accs = np.mean(x[indicators].values==y[indicators].values,axis=0)
        nice_print(accs,indicators)
        return accs
    except:
        print "Something bad happened"
        
def soft_accuracy(x,y,indicators):
    try:
        accs = np.mean(np.abs(x[indicators].values-y[indicators].values)<=1,axis=0)
        nice_print(accs,indicators)
        return accs
    except:
        print "Something bad happened"