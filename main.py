# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 09:58:57 2019

@author: weishi
"""

#methods for BPCACS
#Input parameters:
    #X: Design matrix. X is supposed to have the shape:n*m. where n is the number of data instance and m is the number of features.
    #Y: Label matrix. Y is supposed to have the shapre:n*k. where k is the number of all potential labels. If the i th data x_i has label j then Y[i,j]=1 otherwise Y[i,j]=0;
        #note: The random data split is not built in. You might want to shuffle the data before calling this method to have a more credible result.
    #trainRate,poolRate,testRate: The percentages of the data/labels that will be used for initial training, sampling and testing Takes real number that <1 and the sum of the three should also be <=1.
    #sampleSize the number of active learning iterations.
    #sampleBatch: Tested for batch model compress sensing based AL. It belongs to the future work, please ignore this(fix the value to 1) if you want to replicate the work proposed in ICML paper.
    #bestRecov: how many labels per instance you want the classifier to predict. The adaptive recover of labels is the future work. This papameter here is global. The suggested value is the cardinality of the dataset.
    #compressRate: the compress rate
    #Up date the kernel parameters every how many active learning iterations. The update method uses bayesian optimization by default. To change the optimization method to simplex, change the method call bayesianHyperOpt() to directSearchHyperOpt(). To change to traditional max marginal likelihood optimization, just fit() the model again, it will automatically be done by sklearn.
    #alpha: The parameter in Gaussian Process that keeps the diagonal of the covariance matrix non-zero.
    #PCAWrapper: Boolean. If true, then pca is used to ensure orthogonal. 
    #PCARate: If 'full', all components will be used. Else if <1, the corresponding percentage of the component will be used.
    #To switch to use BPCA and simply replace PCA() with BPCA() by importing bpca.py in this folder.
    #eigenWeigthed: use the eigen value to imporve sampling process.
    #sharedSearchSpace: Whether use a global search space in bayesian opt or use individual search space for each label's kernel optimization.
    #sw:search range for each parameter in kernel opt. Adopt log scale
    #cg:search step(grids) for direct search.
    #fixPhi: For future test purpose. You can pass a fixed compress matrix,Phi or make it adaptive to each AL iteration. If set to None, a random Phi generated from Gaussian distribution is used by default
    #ProvidedIndex: a list of indecis of train test and pool. If you want to use some prefered index rather than a randomly split one, you can pass the index through this parameter.
#Return the Macro F1 for each AL iteration. and (for test purpose) history of updated kernel parameters.            
    
def AL_In_Compress_GPR_BayesianOPT(X,Y,trainRate,poolRate,testRate,sampleSize,sampleBatch=1,bestRecov=2,compressRate=0.5,kernelUpdate=1,alpha=1e-5,PCAWrapper=False,PCARate='full',eigenWeigthed=False,sharedSearchSpace=True,sw=0.6,cg=20,fixPhi=None,provideIndex=None):
    if(provideIndex is None):
        #split the data using the train,pool test rate.
        trainIndex=range(int(X.shape[0]*trainRate));
        poolIndex=range(int(X.shape[0]*trainRate),int(X.shape[0]*(trainRate+poolRate)));
        testIndex=range(int(X.shape[0]*(trainRate+poolRate)),int(X.shape[0]*(trainRate+poolRate+testRate)));
    else:
        #use the pre-defined indecis to split the data.
        trainIndex=list(provideIndex[0])
        poolIndex=list(provideIndex[1])
        testIndex=list(provideIndex[2])
    #this list stores the final learning result       
    ALres=[];
    #record k.T for both test pool and candidate pool so the prediction step can be optimized
    outer_shared_mat=[]
    outer_shared_mat_test=[]
    outer_shared_mat_candidate=[]
    lastSample=0;
    '''
    #if PCA wrapper does not remove any col from compressed Y space,then we have int(Y.shape[1]*compressRate+1) GP to train which means the same number of hyperparameter sets need to be stored. Since hyperparameter sets are bundled with their kernel function, we just store kernel objects.
    if(PCARate=='full'):#if we gonna optimize hyperParameter every iteration, we do not even need to preserve different kernels for different GPs.
        kernels=[customKernel() for x in range(int(Y.shape[1]*compressRate+1))];
    else:
        kernels=[customKernel() for x in range(PCARate)];
        '''
    #thetas is the search space for hyper parameter tunning        
    thetas=[]
    bestParaCandi=[]
    bestPara=[]
    #iniThetas=[];
    parameterHistory=[]
    parameterHistory2=[[] for x in range(sampleSize)];
    bestPhi=[]
    #The main AL loop:
    for i in range(sampleSize):
        tt1=time.time()
        print("start "+str(i)+" th iteration");
        if(fixPhi is None):
            #compress the label space according to compressRate
            res=compress(Y[trainIndex,:],compressRate);
            Phi=res[0];
            U=res[1];
        elif(fixPhi=='auto'):
            if(i%50==0):#generate new Phi
                #print('will update compres mat')
                bestPhi=findBestPhi(X,Y,trainIndex,testIndex,compressRate=compressRate,bestRecov=bestRecov,NumOfIter=5)
                U=np.dot(bestPhi,np.transpose(Y[trainIndex,:]))
                Phi=bestPhi
            else:
                print('will use the current opt compress mat')
                Phi=bestPhi
                U=np.dot(Phi,np.transpose(Y[trainIndex,:]))
        #if using PCA wrapper
        tt2=time.time()
        #print('compress labels in:  '+str(tt2-tt1))
        if(PCAWrapper==True):
            #choose the PCs so that PCARate of the variance is preserved
            if(PCARate=='full'):
                pcaSolver=PCA(n_components=U.shape[0])
            else:    
                pcaSolver=PCA(n_components=PCARate)
            tem=U.transpose()
            tem=sklearn.preprocessing.normalize(tem)
            transformedU=pcaSolver.fit_transform(tem)
            U=transformedU.transpose()
        tt3=time.time()
        print('PCA of the compressed label in :'+str(tt3-tt2))
        
        #save the predicted variance(sum of all vars from all GPs) for each candidate(pool) data
        var=np.zeros([len(poolIndex),1]);
        compressedPrediction=np.zeros([len(testIndex),U.shape[0]]);
        #The logic about this part sucks, mainly because the incremental computation of gram matrix is coded in GP, if we separate that as a individual function K_new=f（K_old,data_new,kernel） and pass K_new directly to GP, things will be much easier.
        sharedMats=[x for x in outer_shared_mat]#all GPs share the same [A,B](dot product and distance mat) 
        sharedMats_test=[x for x in outer_shared_mat_test]
        sharedMats_candidate=[x for x in outer_shared_mat_candidate]

        if(i==0):
            for j in range(U.shape[0]):
                #gpr=GaussianProcessRegressor(kernel=customKernel(),alpha=alpha,optimizer=None);
                optKer,thetaIni=iniSearchConfig(X[trainIndex,:], np.transpose(U)[:,j],customKernel(),nstart=3)
                bestPara.append(thetaIni)
                print(thetaIni)
            gpr=GaussianProcessRegressor(kernel=customKernel(),alpha=alpha,optimizer=None);
            gpr.fit(X[trainIndex,:], np.transpose(U)[:,0])
            mats=gpr.mats_return
            bestParameter,returnParameter,thetas=bayesianHyperOpt(gpr,searchIteration=8,previousCandidate=None,thetaIni=thetaIni,candiGrid=20,searchWidth=0.6,thetas=None,pre_mats=mats)
            parameterHistory.append([x for x in bestPara])
        tretrain=0
        ttune=0
        tupdate=0
        teval=0
        tsample=0
        for j in range(U.shape[0]):
            #print('deal with the'+str(j)+'th GP')
            #create the model  
            #we do not use the default hyperParameter opimization here
            gpr=GaussianProcessRegressor(kernel=customKernel(np.exp(bestPara[j])),alpha=alpha,optimizer=None);
            #gpr2=GaussianProcessRegressor(kernel=customKernel(),alpha=alpha)
            if(i==0):#every GP trains in opt way at start
                gpr.fit(X[trainIndex,:], np.transpose(U)[:,j]);
                if(len(sharedMats)==0):#then compute [A,B] used for next sampling iteration
                    sharedMats=[x for x in gpr.mats_return]
                    outer_shared_mat=[x for x in sharedMats]
                #now tune the hyper parameter
                
                bestParameter,returnParameter,searchSpace=bayesianHyperOpt(gpr,searchIteration=8,previousCandidate=None,thetaIni=bestPara[j],candiGrid=cg,searchWidth=sw,thetas=None,pre_mats=gpr.mats_return)
                thetas=searchSpace
                bestParaCandi.append(returnParameter)
                bestPara[j]=bestParameter
                #retain the model with opt parameter
                gpr=GaussianProcessRegressor(kernel=customKernel(np.exp(bestParameter)),alpha=alpha,optimizer=None);
                gpr.fit(X[trainIndex,:], np.transpose(U)[:,j]);
                #finish hyperParameter train
                
                tem=gpr.predict(X[testIndex,:]);
                compressedPrediction[:,j]=tem;
                if(len(sharedMats_test)==0):
                    sharedMats_test=[x for x in gpr.test_mats_return]
                    outer_shared_mat_test=[x for x in sharedMats_test]
                #predict again over candidate pool. this time the GP return [candidate_A,candidate_B]
                predictMean,predictVar=gpr.predict(X[poolIndex,:],return_std=True);                
                if(len(sharedMats_candidate)==0):
                    sharedMats_candidate=[x for x in gpr.candidate_mats_return]
                    outer_shared_mat_candidate=[x for x in sharedMats_candidate]
            else:
                t1=time.time()
                gpr.fit(X[trainIndex,:], np.transpose(U)[:,j],previous_mats=sharedMats);
                t2=time.time()
                tretrain=tretrain+t2-t1
                #start hyper parameter tuning
                if(sharedSearchSpace):
                    bestParameter,returnParameter,thetas=bayesianHyperOpt(gpr,searchIteration=8,previousCandidate=bestParaCandi[j],thetaIni=None,candiGrid=cg,searchWidth=sw,thetas=thetas,pre_mats=gpr.mats_return)
                else:#recreate search space based on last fine tuned hyperparameter
                    bestParameter,returnParameter,thetas=bayesianHyperOpt(gpr,searchIteration=8,previousCandidate=bestParaCandi[j],thetaIni=bestPara[j],candiGrid=cg,searchWidth=sw,thetas=None,pre_mats=gpr.mats_return)
                    bestPara[j]=bestParameter
                t3=time.time()
                ttune=ttune+t3-t2
                bestParaCandi[j]=returnParameter
                #retrain the model 
                gpr=GaussianProcessRegressor(kernel=customKernel(np.exp(bestParameter)),alpha=alpha,optimizer=None);
                
                gpr.fit(X[trainIndex,:], np.transpose(U)[:,j],previous_mats=sharedMats);
                #gpr.fit(X[trainIndex,:], np.transpose(U)[:,j]);
                t4=time.time()
                tupdate=tupdate+t4-t3
                #finish hyperParameter train
                tem=gpr.predict(X[testIndex,:]);
                #tem=gpr.predict(X[testIndex,:]);
                compressedPrediction[:,j]=tem;
                t5=time.time()
                #print('evaluate in: '+ str(t5-t4))
                teval=teval+t5-t4
                predictMean,predictVar=gpr.predict(X[poolIndex,:],return_std=True,pre_candidate_mats=sharedMats_candidate,lastSampleIndex=lastSample);       
                t6=time.time()
                tsample=tsample+t6-t5
                #since using sharedMats[A,B] to train GP will return a updated [A,B],when the last GP is reached, update the sharedMat for next iteration
                #so does [test_A,test_B]
                if(j==range(U.shape[0])[-1]):
                    outer_shared_mat=[x for x in gpr.mats_return]
                    outer_shared_mat_test=[x for x in gpr.test_mats_return]  
                    outer_shared_mat_candidate=[x for x in gpr.candidate_mats_return]
            
               
            if(eigenWeigthed is True):
                #if each GP is iid with var_i, the det of cov of the joint should be the sum of log(var_i) (cause the cov is diagonal)
                var[:,0]=var[:,0]+pcaSolver.singular_values_[j]*np.log(predictVar)
            else:
                var[:,0]=var[:,0]+np.log(predictVar);
        parameterHistory.append([x for x in bestPara]);
        #print(parameterHistory[0])
        tt4=time.time()
        if(PCAWrapper==True):
            compressedPrediction=pcaSolver.inverse_transform(compressedPrediction)
        compressedPrediction=np.transpose(compressedPrediction);
        #compute current performance
        res,entropy,lams=recoverARD(Phi,compressedPrediction);
        sres=sparsify(res,bestRecov); 
        score=F1(sres,Y[testIndex,:]);
        tt5=time.time()
        ALres.append(score);
        print ("current F1"+str(score[0]));
        print('sample time:'+str(time.time()-tt1))
        #print('traintime: '+str(trainTime)+ 'testTime: '+str(testTime)+'TestCandidateTime: '+str(testCandiTime))                
#select sampleBatch samples at once
        sampleIndex=np.argsort(var,axis=0)[::-1][0:sampleBatch][:,0];
        candidateIndex=[poolIndex[x] for x in sampleIndex];
        lastSample=sampleIndex[0];
        #temIndex=[poolIndex[x] for x in sampleIndex]
        trainIndex=trainIndex+candidateIndex;
        testIndex=[x for x in range(X.shape[0]) if x not in trainIndex]
        print('trainIndex:'+str(len(trainIndex)))
        poolIndex=[x for x in poolIndex if x not in candidateIndex];
    return ALres,parameterHistory,parameterHistory2;


#simplex optimizatino
def directSearchHyperOpt(modelGP):
    def func(thetas):
        #since direct search is an unconstrained method expand the def of obj.
        if(np.all(thetas>0)):    
            thetas=np.log(thetas);
            print(thetas)
            return modelGP.log_marginal_likelihood(thetas)*(-1);
        else:
            return np.inf
    #move this out! the range should be specified w.r.t datasets!
    theta0=np.exp(modelGP.kernel_.theta)
    res = minimize(func, theta0, method='nelder-mead',options={'xtol': 1e-6, 'disp': True})
    return res.x
        
def bayesianHyperOpt(modelGP,searchIteration=5,previousCandidate=None,thetaIni=[-8,6,11,-3,-8],candiGrid=20,searchWidth=0.6,thetas=None,pre_mats=None,tol=1e-4,optHyperGP=True):#directly pass the fitted model in. This model is used for computing likelihood
    #previousCandidate stores the hyperParameter combinations that found previously.They can be used to
    #1)train the GP. implemented
    #2)adaptively adjust the searching range.  Not implemented
    #3)candiGrid:for teach theta[i],how manycandidate values will be searched
    #4)thetaIni:from the current optimized theta, determine the search range for next opt theta.   
    #5)thetas is the search matrix for all candidate theta values
    #testCase=[[1,3,1],[1,4,1],[3,8,1],[6,8,1]]
    #import a dataset(diabetes)
    '''
    data=datasets.load_diabetes()
    X=data.data
    Y=data.target
    '''
    #normalize data
    #X=sklearn.preprocessing.normalize(X)
    #define search sapce
    #notice that the third parameter is a fixed one#move this out! the range should be specified w.r.t datasets!
    '''
    if(thetaIni is None):
        config=[[1000,5000,100],[1e-5,1e-4,1e-5],[1e-5,(1e-5)*2,1e-5],[500,2000,100],[10000,25000,300]]
    else:
        config=[[x-searchWidth,x+searchWidth,2*searchWidth/candiGrid] for x in thetaIni]      
    '''          
    
    #generate candidate parameter searching space. Notice that we assume all search in log scale
    if(thetas is None):
        thetas=genCandidate(thetaIni,itertool=True,SW=searchWidth,CG=candiGrid)
    #randomly choose 5 samples from parameter searching space to form a training:
    sample_index=np.random.randint(0,thetas.shape[0],5)
    candidate_index=np.array([x for x in range(thetas.shape[0]) if x not in sample_index])
    
    #fit the GP(data model) with data and kernel
    '''
    #generate the kernel function
    customKer=customKernel();
    modelGP = GaussianProcessRegressor(kernel=customKer,optimizer=None);
    #use gradient to get the good start
    #modelGP = GaussianProcessRegressor(kernel=customKer);
    modelGP.fit(X,Y);
    '''
    #a=modelGP.kernel_
    x_train=thetas[sample_index,:]
    if(previousCandidate is not None):
        x_train=np.append(x_train,previousCandidate,axis=0)
    #compute the lg likelihood given the randomly choosen samples(thetas)
    #b=modelGP.log_marginal_likelihood(x_train[4,:])
    scores=likelihood(x_train,modelGP,pre_mats=pre_mats)
    pre_y_max=-999999;    
    #hyper GP also need to be fast.
    #hyper_mats=[]
    iterCount=0
    for i in range(searchIteration):
        iterCount=iterCount+1
        #those 5 random samples are used to train another GP for hyperparameter tunning
        
        #compute the lg likelihood given the randomly choosen samples(thetas)
        #b=modelGP.log_marginal_likelihood(x_train[4,:])
        #scores=likelihood(x_train,modelGP)
        #Train the gp for hyperParameter searching
        if(not optHyperGP):
            hyperGP=GaussianProcessRegressor(optimizer=None);
        else:
            hyperGP=GaussianProcessRegressor();
        #print('--------------------------logL below are GPs with thetas---------------------')
        hyperGP.fit(x_train,scores)
        y_max=max(scores)
        if(abs(y_max-pre_y_max)<tol):
            break
        else:
            pre_y_max=y_max
        #print('current max LogL: '+ str(y_max))
        #
        #t1=time.time()
        y_mean,y_std=hyperGP.predict(thetas[candidate_index,:],return_std=True)
        #t2=time.time()
        #note that this index is wrt candidate_index, not the index of the row of thetas        
        nextSample_index=next_parameter_by_ei(y_max, y_mean, y_std, thetas[candidate_index,:])
        #append new sample index
        sample_index=np.append(sample_index,candidate_index[nextSample_index])
        scores.append(likelihood(thetas[[candidate_index[nextSample_index]],:],modelGP)[0])
        candidate_index=np.delete(candidate_index,nextSample_index)
        if(previousCandidate is not None):
            x_train=np.append(thetas[sample_index,:],previousCandidate,axis=0)
        else:
            x_train=thetas[sample_index,:]
        #print('-------------------------logL above are GPs with thetas----------------------')
    bestParameter=x_train[np.argmax(scores),:]
    #get the parameters and their likelihoods
    selectedIndex=range(x_train.shape[0])[::-1][0:iterCount]
    returnParameter=x_train[selectedIndex,:]                   
    #note that we need returnParameter to be in the scale of log so when pass to the function at next iteration, it can be directely added to the training set
    return bestParameter,returnParameter,thetas;

def iniSearchConfig(X,Y,kernelFun,alpha=0.5,nstart=None):
    if(nstart is not None):
        model=GaussianProcessRegressor(kernel=kernelFun,alpha=alpha,n_restarts_optimizer=nstart)    
    else:
        model=GaussianProcessRegressor(kernel=kernelFun,alpha=alpha)    
    model.fit(X,Y)
    theta=model.kernel_.theta
    return model.kernel_, theta

#sparsity of the label space
def compress(Ytrain,compressRate):
    #return Phi and compressed mat 
    K=np.ceil(np.sum(Ytrain)*1.0/(Ytrain.shape[0])); 
#------- traget dimension of label space after compressing------------------
    m=int(np.floor(compressRate*Ytrain.shape[1]));
#------number of labels-------
    d=int(Ytrain.shape[1]);
    adjustm=np.ceil(K*np.log(d/K));
    Phi=np.random.randn(m,d)*(1/adjustm);
    return [Phi,np.dot(Phi,np.transpose(Ytrain))];    

def likelihood(thetas,model,pre_mats=None):
    y_train=[]#this called y_train because it is used later in the 
    #t1=time.time()
    if (thetas.shape[0]>1):
        for i in range(thetas.shape[0]):
            y_train.append(model.log_marginal_likelihood(thetas[i,:],pre_K=pre_mats))
    else:
        y_train.append(model.log_marginal_likelihood(thetas[0],pre_K=pre_mats))
    #print('compute likelihood in :' +str(time.time()-t1))    
    return y_train;

def next_parameter_by_ei(y_max, y_mean, y_std, x_choices):
    # Calculate expecte improvement from 95% confidence interval
    expected_improvement =  (y_mean + 1.96 * y_std) - y_max
    expected_improvement[expected_improvement < 0] = 0

    max_index = expected_improvement.argmax()
    # Select next choice
    #next_parameter = x_choices[max_index]

    return max_index        

def customKernel(thetas=[1.001,0.5,1,1.001],kernelType=1):
    
    k1= ConstantKernel(constant_value=thetas[0],constant_value_bounds=(0.0001, 10.0)) * RBF(length_scale=thetas[1],length_scale_bounds=(0.0001, 100.0));
    #dot product kernelnpp
    #note that sigma=0=>homogeneous linear kernel, However, in sklearn, thetas are transformed into log. So give exact 0 will cause -inf.
    if (kernelType==1):
        k2= DotProduct(sigma_0=thetas[2],sigma_0_bounds=(1e-5, 1e5))*ConstantKernel(constant_value=thetas[3],constant_value_bounds=(0.0001, 10.0))
    else:
        k2= DotProduct(sigma_0=thetas[2],sigma_0_bounds=(1e-5, 1e5))+ConstantKernel(constant_value=thetas[3],constant_value_bounds=(0.0001, 10.0))
   
        #constant kernel
    return k1+k2

def F1(predicted,Y_test,analysis=False):
    predicted=np.transpose(predicted);
    macro_p=[]
    macro_r=[]
    macro_f=[]
    label_tp=[]
    TP=0
    for k in range(Y_test.shape[1]):
        label_TP=np.sum(np.logical_and(predicted[:,k],Y_test[:,k]))*1.0;
        label_tp.append(label_TP)
        #in case divided by 0:
        if(label_TP==0):
            label_P=0
            label_R=0
        else:
            label_P=label_TP/np.sum(predicted[:,k]);
            label_R=label_TP/np.sum(Y_test[:,k]);
        #in case divided by 0:    
        if((label_P+label_R)==0):
            label_F=0
        else:
            label_F=2*label_P*label_R/(label_P+label_R)
        macro_p.append(label_P)
        macro_r.append(label_R)
        macro_f.append(label_F)
        TP=TP+label_TP
    if (analysis is True):
        return [[np.mean(macro_f),np.mean(macro_r),np.mean(macro_p),TP],[macro_f,macro_r,macro_p],label_tp]    ;
    else:
        return [np.mean(macro_f),np.mean(macro_r),np.mean(macro_p),TP];