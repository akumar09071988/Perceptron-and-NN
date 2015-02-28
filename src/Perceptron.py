'''
Created on Feb 25, 2015

@author: abhishekkumar
'''
from PreProcess import PreProcess;
from Utility import Utility;
class Perceptron(object):
    '''
    classdocs
    '''
    def _initiaLizeWeights(self,size):
        for i in range(0,size):
            self.weightList.append(0);
        
    def _getTrainPredList(self,input):
        if(input>0):return 1;
        else: return 0;
        

    
    def _getError(self,predList,actualList):
        diffList =map(lambda pair: pair[0]-pair[1], zip(actualList, predList));
        return diffList;
    
    def _updateWeights(self,index,diffList,xList):
        constDiff = self.learningRate*diffList;
        deltaWeights = map(lambda x: x*constDiff, xList[index] ); 
        self.weightList =map(lambda pair: pair[0]+pair[1], zip(self.weightList, deltaWeights));
        #print "after update";      
            
             
    
    def _train(self):
        for i in range(0,self.numIter):
            for j in range(0,len(self.trainObj.xList)):
                outPut =map(lambda pair: pair[0]*pair[1], zip(self.weightList, self.trainObj.xList[j]));
                pred = self._getTrainPredList(sum(outPut));
                diff = self.trainObj.yList[j] - pred;
                self._updateWeights(j, diff, self.trainObj.xList);
            #print self.weightList;
                
            
    def _pred(self):
        multList =Utility.multFeatureWeights(self.weightList, self.predObj.xList);
        outPut =[];
        for x in multList:
            if(x>0): outPut.append(1);
            else: outPut.append(0);
        return outPut;
    
    def _calcAccuracy(self,outList):
        correct =0;
        for i in range(0,len(outList)):
            if(outList[i]== self.predObj.yList[i]): correct+=1;
        self.accuracy = float(correct)/float(len(self.predObj.yList));
        self.accuracy *=100.00;
        print correct;
        print self.accuracy;

    def __init__(self, trainObj,predObj,learningRate,numIter):
        self.learningRate=learningRate;
        self.numIter =numIter;
        self.trainObj = trainObj;
        self.predObj = predObj;
        self.weightList =[];
        self._initiaLizeWeights(len(trainObj.vocabDict));
        self._train();
        outPut =self._pred();
        self._calcAccuracy(outPut);

        