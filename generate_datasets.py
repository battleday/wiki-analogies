
# coding: utf-8

# In[1]:
import os
import matplotlib.pyplot as plt
import math
import numpy as np

# In[2]:


def draw_multinomial(num_entities, zeta):
    draws = np.random.multinomial(1, pvals=zeta, size=num_entities)
    z = np.argmax(draws, axis = 1)
    return z

def draw_interaction(eta):
    return np.random.binomial(1, eta)

def draw_system_parameters(config):
    print('drawing system parameters')
    systemParams = np.load(config['load_path'], allow_pickle=True)
    eta = systemParams['intMat']
    zeta = systemParams['freqVec']
    return eta, zeta

def create_dataset(num_entities, interactionMatrix, frequencyVector):
    """Samples num_entities start points from frequencyVector
    Samples num_entities end points from frequencyVector
    Derives edges between them exhaustively from interactionMatrix
    Shuffles and splits into test and train
    Partitions train according to percentage of data observed."""
    dataMatrix = np.empty((num_entities**2, 3))
    startPoints = draw_multinomial(num_entities, frequencyVector)
    endPoints = draw_multinomial(num_entities, frequencyVector)
    dataMatrix[:, :2] = np.array(np.meshgrid(np.arange(num_entities), np.arange(num_entities))).T.reshape(-1,2) # matrix indexing
    dataMatrix = np.int_(dataMatrix)
    for row_num, (i, j, _) in enumerate(dataMatrix):
        classI = startPoints[i]
        classJ = endPoints[j]
        dataMatrix[row_num, -1] = draw_interaction(interactionMatrix[classI, classJ])
    return dataMatrix

def split_dataset(num_entities, dataMatrix, num_splits=10):
    split_size = np.int((num_entities**2) / num_splits)
    print('working with a split of {} size'.format(split_size))
    incomplete = True
    while incomplete:
        try:
            dataPerm = np.random.permutation(dataMatrix)
            testSet = dataPerm[:split_size]
            firstSet = dataPerm[split_size:(2*split_size)]
            assert testSet.shape[0] == split_size, "testSet not split_size rows: {}".format(testSet.shape[0])
            assert firstSet.shape[0] == split_size, "firstSet not split_size rows: {}".format(firstSet.shape[0])
            testElements = np.unique(testSet[:, :2])
            firstElements = np.unique(firstSet[:, :2])
            assert len(testElements) == num_entities, "not enough test entities"
            assert len(firstElements) == num_entities, "not enough first entities"
            assert (testElements == np.arange(num_entities)).all(), "not all elements included in testSet"
            assert (firstElements == np.arange(num_entities)).all(), "not all elements included in firstSet"
            
        except Exception as e:
            print(e)
            continue
        incomplete = False
        
        return dataPerm, split_size

def save_dataset(dataPerm, config, split_size, num_splits=10):
    trainRanges = np.array([(split_size, x*split_size) for x in np.arange(2, num_splits+1)])
    testRange = np.array([0, split_size])
    np.savez('{}.npz'.format(config['save_path']), fullData = dataPerm, testRange=testRange, 
            trainRanges=trainRanges)

num_entities = 100 # [30, 100]
loadDir = '/tigress/ruairidh/cogsci2021analogy/WIKIdata/matrices2020'
saveDir = '/tigress/ruairidh/cogsci2021analogy/data/domains'
domains = os.listdir(loadDir)
for file in domains:
    print(file)
    filename = file.replace('.npz', '')
    #if '.DS' in file:
    if ('.DS' in filename) or ('wiki' in filename):
        print("wiki or DS; skipping")
        continue
    try:
        os.makedirs('{}/{}'.format(saveDir, filename))
    except Exception as e:
        print(e)
        
    loadPath = '{}/{}'.format(loadDir, file)
    savePath = '{}/{}/{}_entities_data'.format(saveDir, filename, num_entities)
    config = {'num_entities': num_entities, 'save_path': savePath, 'load_path': loadPath}
    
    if os.path.exists('{}.npz'.format(savePath)):
        print("path already exists; moving on")
        continue
    else:
        print("drawing params")
        eta, zeta = draw_system_parameters(config)
        dM = create_dataset(config['num_entities'], eta, zeta)
        dP, split_size = split_dataset(config['num_entities'], dM)
        save_dataset(dP, config, split_size)

