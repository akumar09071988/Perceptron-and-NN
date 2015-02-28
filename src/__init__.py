from Utility import Utility;
from PreProcess import PreProcess;
from Perceptron import Perceptron;



def main():
    percepTronTrainDict = {};
    percepTronTrainDict['spam'] = 'train/spam';
    percepTronTrainDict['ham'] = 'train/ham';
    percepTronTestDict = {};
    percepTronTestDict['spam'] = 'test/spam';
    percepTronTestDict['ham'] = 'test/ham';
    classDict={};
    classDict['spam'] =1;
    classDict['ham']=0;
    preProcessTrainObj = PreProcess(percepTronTrainDict,classDict);
    preProcessTestObj = PreProcess(percepTronTestDict,classDict,preProcessTrainObj.vocabList);
    iterArray =[300,10,30,50,100,200];
    learnRateArray=[0.01,0.03,0.06,0.09,0.1,0.4,0.7,1];
    
    for i in range(0,len(iterArray)):
        for j in range(0,len(learnRateArray)):
            iter = iterArray[i];
            learnRate = learnRateArray[j]
            percepObj =Perceptron(preProcessTrainObj,preProcessTestObj,learnRate,iter);
            print "Accuarcy " +str(percepObj.accuracy) +" with iterations "+ str(iter)+ " learning rate "+str(learnRate)+" without stop words";
            
    #Perceptron(preProcessTrainObj,preProcessTestObj,0.1,10);
    
    


#8641     8641
#14036     8405







if __name__ == '__main__':
    main();