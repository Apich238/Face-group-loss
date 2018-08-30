import numpy as np
import scipy.io as sio
import os.path
import os


def ReadLFWList(folder,blufr_list_file):
    with open(blufr_list_file,'r') as f:
        flines=f.readlines()
    reslist=[]
    for i,l in enumerate(flines):
        pl=os.path.join(folder,l[:l.rfind('_')],l[:l.rfind('\n')])
        reslist.append(pl)
    return reslist

def SaveEmbeddings(fname,embeddings):
    dict={'Descriptors':embeddings}
    sio.savemat(fname,dict,True,'4')


