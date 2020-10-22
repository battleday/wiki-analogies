
# coding: utf-8

# ## Blockmodel library
# This notebook contains the modules necessary to perform hill climbing on stochastic blockmodels.

# In[1]:


import numpy as np
import random
import gc, os, sys, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
script, gamma = sys.argv
gamma = np.int(gamma)


# To do:
# 
# Relation matrix set at start
# Maintain m and mNeg
# Update partition vector, m, and mNeg if ll increased
# Create function to use m and mNeg to recover parameter matrix.

# In[2]:


def extract_class_info(partitionVector):
    """Basic module for extracting class information from 
    partition vector"""
    numEntities = np.shape(partitionVector)[0]
    classes = np.unique(partitionVector)
    numClasses = np.shape(classes)[0]
    classIdxs = [np.where(partitionVector==x)[0] for x in classes]
    classSizes = [np.shape(x)[0] for x in classIdxs]
    classInfo = {'iden' : classes, 'num': numClasses, 'idxs': classIdxs,
            'ents': numEntities, 'sizes': classSizes}
    
    return classInfo

#zTest = np.array([0, 0, 1, 1])
#classInfo = extract_class_info(zTest)


# In[3]:


def reduce_to_class_interactions(relationMatrix, partitionVector, 
                                 classInfo):
    """This function takes a matrix of inter-entity observed interactions 
    and a vector assigning entities to classes,
    and returns two matrices of interclass counts and noncounts"""
    
    # returns sorted (ascending) unique classes
     
        
    m = np.empty((classInfo['num'], classInfo['num']))
    mNeg = m.copy()
        
    # goes through all pairs of class assignments and counts them
    for i, a in enumerate(classInfo['iden']):
        aIdx = classInfo['idxs'][i]
        
        # select out class A rows
        classRelMat = relationMatrix[aIdx, :]
        for j, B in enumerate(classInfo['iden']):
            bIdx = classInfo['idxs'][j]
            
            # subselect out class B columns
            tempRelMat = classRelMat[:, bIdx]
            
            # count number of interactive entities in two classes
            total = tempRelMat.shape[0] * tempRelMat.shape[1]
            counts = np.sum(tempRelMat)
            m[i, j] = counts
            mNeg[i, j] = total - counts
            
    return m, mNeg

# test

#rTest = np.array([[0, 0, 1, 0],[0, 0, 0, 1],[1, 0, 1, 0],[0, 1, 0, 0]])

#m, mNeg = reduce_to_class_interactions(rTest, zTest, classInfo)
#print(m, '\n', mNeg)
#print('\n test results: does function work as expected?')
#print((m == np.array([[0, 2],[2, 1]])).all())


# In[4]:


# pdfs / functions
# work through conjugacy stuff again later

def gamma_function(n):
    """Returns gamma function on positive integers. May need to 
    extend to positive reals"""
    try:
        g = np.math.factorial(n-1)
        return g
    except:
         print("Oops!  Gamma input invalid: {0}.  Try again...".format(n)) 
        

def beta_function(alpha = 1, beta = 1):
    """Just returns beta function"""
    numer = gamma_function(alpha) * gamma_function(beta)
    denom = gamma_function(alpha + beta)
    return  numer / denom


# In[5]:


def relation_score(m, mNeg, classInfo, alpha = 1, beta = 1):
    """For the given inter-class interactions, return 
    factorized score of clustered relational matrix."""
    #calculate factors in p(R|z)
    
    denom = beta_function(alpha, beta)
    m_t = m + alpha 
    mNeg_t = mNeg + beta
    
    # n^2 pairs
    pairScores = []
    # create pair array
    for index in np.ndindex(classInfo['num'], classInfo['num']):
        numer = beta_function(m_t[index], mNeg_t[index])
        calc = numer / denom
        
        pairScores.append(calc)
    
    # use product for likelihood or sum for log likelihood
    return np.array(pairScores)


#print(relation_score(m, mNeg, classInfo))


# In[6]:


# Assess prob of distribution under CRP
def crp_probability(partitionVector, classInfo, gamma=1):
    """Gives the factorized probability of partition
    under CRP with gamma parameter.
    
    https://en.wikipedia.org/wiki/Chinese_restaurant_process
    """
    
    firstTermNumer = (gamma_function(gamma) * (gamma**classInfo['num'])) 
    #print(firstTermNumer)
    firstTermDenom = gamma_function(gamma + classInfo['ents'])
    #print(firstTermDenom)
    firstTerm = firstTermNumer / firstTermDenom
    
    # each term here should be gamma function of size of class
    classTerms = []
    classTerms.append(firstTerm)
    for cS in classInfo['sizes']:
        # indices for each class
        classTerms.append(gamma_function(cS))
        
    # use product for likelihood or sum for log likelihood
    return np.array(classTerms)

#print(crp_probability(zTest, classInfo))
#zTest2 = np.array([0, 1, 2, 2, 2])
#classInfo2 = extract_class_info(zTest2)
#print(crp_probability(zTest2, classInfo2)) 


# In[7]:


# make sure these are only declared once
alpha = 1
beta = 1
#gamma = 1

def calculate_log_likelihood(relationMatrix, partitionVector, gamma):
    """Function to be called each evaluation. Calculates
    log likelihood of partition vector wrt relation matrix
    and crp prior"""
    
    classInfo = extract_class_info(partitionVector)
    m, mNeg = reduce_to_class_interactions(relationMatrix, 
                                           partitionVector,
                                          classInfo)
    
    relationScoreTerms = relation_score(m, mNeg, classInfo)
    partitionScoreTerms = crp_probability(partitionVector, classInfo, gamma)
    scoreTerms = np.append(relationScoreTerms, partitionScoreTerms)

    # sometimes scoreTerms has type 'object' instead of 'float', which
    # causes error. Not sure why it happens, but only on merge; below 
    # fixes.
    try:
        logLike = np.sum(np.log(scoreTerms))
    except:
        logLike = np.sum(np.log(scoreTerms.astype('float')))
    return {'partitionVector': partitionVector.copy(), 'score': logLike, 
            'classInfo': classInfo, 
            'm': m, 'mNeg': mNeg, 'gamma': gamma}

#print(calculate_log_likelihood(rTest, zTest))


# In[8]:


def inner_loop(intervention, hill_fn, relationMatrix, currentBest, 
               fig, 
               ticks, scores,
               gamma,
               n = 1000):
    
    for it in np.arange(n):
        
        candPartitionVector = hill_fn(currentBest)
        #try:
        candDict = calculate_log_likelihood(relationMatrix, 
                                                candPartitionVector, gamma)
        #except:
        #    continue
        if (candDict['score'] > currentBest['score']): #or ((candDict['score'] == currentBest['score']) and np.random.randint(1+1)):
            
            print('{0} {1} succeeded. New score: {2}'.format(intervention, ticks, candDict['score']))
            currentBest = candDict
            
            newIdxs = np.argsort(candPartitionVector)
            tempRelationMatrix = relationMatrix[:, newIdxs][newIdxs]
            ax = plt.gca()
            ax.matshow(tempRelationMatrix, cmap='gray', vmax=1, vmin=0)
            plt.savefig('scores_{0}/Shuffled_matrix_at_{0}.png'.format(gamma, ticks))
            
        ticks += 1
        scores.append(currentBest['score'])
    return currentBest, ticks, scores


# In[9]:


def move_fn(currentBest):
    """Picks a random entity and assigns it to a random
    class; outputs new partition vector."""
    elem = np.random.randint(currentBest['classInfo']['ents'])
    #print(elem)
    c = np.random.choice(currentBest['classInfo']['iden'])
    currentBest['partitionVector'][elem] = c
    
    return currentBest['partitionVector']

#move_fn(currentBest)


# In[10]:


def merge_fn(currentBest):
    """Picks a random class and assigns all
    entities within it to a random
    class; outputs new partition vector. 
    Needs to be more than one class."""
    elems1 = random.choice(currentBest['classInfo']['idxs'])
    elems2 = random.choice(currentBest['classInfo']['idxs'])
    c1 = currentBest['partitionVector'][elems1[0]]
    c2 = currentBest['partitionVector'][elems2[0]]
 
    if c1 < c2:
        currentBest['partitionVector'][elems2] = c1
    else:
        currentBest['partitionVector'][elems1] = c2
    
    return currentBest['partitionVector']

#currentBest = calculate_log_likelihood(relationMatrix, currentBest['partitionVector'])
#merge_fn(currentBest)


# In[11]:


def split_fn(currentBest):
    """Picks a random class and assigns a random set of
    entities within it to a new
    class, and the remainder to another new class; 
    outputs new partition vector.
    Needs to be more than one element in chosen group."""
    elems = np.array(random.choice(currentBest['classInfo']['idxs']))
    elems = np.random.permutation(elems)
    splitPoint = np.random.randint(len(elems))
    
    c1 = np.max(currentBest['classInfo']['iden']) + 1
    c2 = c1 + 1
    currentBest['partitionVector'][elems[:splitPoint]] = c1
    currentBest['partitionVector'][elems[splitPoint:]] = c2
    
    return currentBest['partitionVector']

#currentBest = calculate_log_likelihood(relationMatrix, currentBest['partitionVector'])
#split_fn(currentBest)


# In[12]:


def recover_parameter_matrix(currentBest, alpha = 1, beta = 1):
    """Recover parameter matrix using closed-form maximum
    a posteriori approach."""
    m_t = currentBest['m'] + alpha 
    mNeg_t = currentBest['mNeg'] + beta
    mu = np.empty(m_t.shape)
    for index in np.ndindex(currentBest['classInfo']['num'], 
                            currentBest['classInfo']['num']):
        numer = m_t[index]
        denom = m_t[index] + mNeg_t[index]
        calc = numer / denom
        mu[index] = calc
        
    return mu
    


# In[13]:


# test inner loop
relationMatrix = np.load("networkRelationMatrixNew.npy")
numEntities = relationMatrix.shape[0]
partitionVector = np.ones(numEntities)
finalBest = calculate_log_likelihood(relationMatrix, partitionVector, 1)
#plt.matshow(relationMatrix, cmap='gray', vmax=1, vmin=0)


# In[16]:



loops = 100000
sampleFunctionVector = np.ones(100)
sampleFunctionVector[:10] = 2
sampleFunctionVector[10:20] = 3
#for gamma in np.arange(1, 11):

for iii in np.arange(10):
    plt.close("all")
    if not os.path.exists('finalScores_{0}'.format(gamma)):
        os.mkdir('finalScores_{0}'.format(gamma))
    if os.path.exists('scores_{0}'.format(gamma)):
        shutil.rmtree('scores_{0}'.format(gamma))
        
    os.mkdir('scores_{0}'.format(gamma))
    print('\n New loop with gamma:', gamma)
    
    currentBest = calculate_log_likelihood(relationMatrix, partitionVector, gamma)
    scores = [currentBest['score']]
    ticks = 0
    # movie stuff
    fig = plt.figure()    
    for i in np.arange(loops):
        if i % (loops//10) == 0:
            print('Iter {0} out of {1}'.format(i, loops))
            gc.collect()
            gc.collect()
            gc.collect()

                # samples a type of intervention at random
        choice = np.random.choice(sampleFunctionVector)
        if choice == 1:
                    #print('\n Moving for iter: {0}'.format(i))
            currentBest, ticks, scores = inner_loop('Move', move_fn, relationMatrix, currentBest, fig, ticks, scores, gamma, n = 100)
        elif choice == 2:
                    #print('\n Merging for iter: {0}'.format(i))
            currentBest, ticks, scores = inner_loop('Merge', merge_fn, relationMatrix, currentBest, fig, ticks, scores, gamma, n = 10)
        elif choice == 3:
                  #print('\n Splitting for iter: {0}'.format(i))
            currentBest, ticks, scores = inner_loop('Split', split_fn, relationMatrix, currentBest, fig, ticks, scores, gamma, n = 10)

        #print('\n done! Final partition vector: \n {0}'.format(currentBest))
        
        
        if currentBest['score'] > finalBest['score']:
            print('New top score: {0}'.format(currentBest['score']))
            finalBest = currentBest
            np.save('finalScores_gamma_{0}.npy'.format(gamma, iii), np.array(scores))
            
            plt.gca()
        
            plt.savefig('finalShuffled_gamma_{0}.png'.format(gamma, iii))
            plt.close()

            shutil.rmtree('finalScores_{0}'.format(gamma))
            shutil.copytree('scores_{0}'.format(gamma), 'finalScores_{0}'.format(gamma))
        
            recoveredParameterMatrix = recover_parameter_matrix(finalBest)
            plt.matshow(recoveredParameterMatrix, cmap='gray', vmax=1, vmin=0)
            plt.savefig('recoveredParameter_gamma_{0}.png'.format(gamma))

            with open('finalBest_{0}.p'.format(gamma), 'wb') as pfile:
                pickle.dump(finalBest, pfile)
    
print('loops done!')
