import numpy as np
import pandas as pd


def small_nice_print(accs):
    for x in accs:
        print "%.2f"%x," ",
    print '\n'

def soft_accuracy_report(ref,algo,indicators):
    try:
        accs = np.mean((ref[indicators].values-algo[indicators].values)==1,axis=0)
        for i in indicators:
            print i+"   ",
        print 
        print "ref - algo == 1"
        small_nice_print(accs)
        
        print "ref - algo == 0"
        accs = np.mean((ref[indicators].values-algo[indicators].values)==0,axis=0)
        small_nice_print(accs)
        
        print "ref - algo == -1"
        accs = np.mean((ref[indicators].values-algo[indicators].values)==-1,axis=0)
        small_nice_print(accs)
        
        accs = np.mean(np.abs(ref[indicators].values-algo[indicators].values)<=1,axis=0)
        print "Total soft-accuracy:"
        nice_print(accs,indicators)
        return accs
    except:
        print "Something bad happened"
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
        