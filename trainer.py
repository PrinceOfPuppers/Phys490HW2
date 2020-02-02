import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

class Trainer:
    def __init__(self,trainingData,testingData):
        self.trainingAccuracy=[]
        self.trainingLoss=[]

        self.testingAccuracy=[]
        self.testingLoss=[]

        self.trainingImgs=torch.from_numpy(np.stack(trainingData[:,0]))
        self.trainingAnswers=torch.from_numpy(np.stack(trainingData[:,1]))

        self.testingImgs=torch.from_numpy(np.stack(testingData[:,0]))
        self.testingAnswers=torch.from_numpy(np.stack(testingData[:,1]))
    
    def trainAndTest(self,neuralNet,hyp):
        for i in range(0,hyp["epochs"]):
            print("Training Epoch {}:".format(i+1))
            trainAcc,trainLoss=neuralNet.train(self.trainingImgs,self.trainingAnswers,hyp["trainBatchSize"])
            self.trainingAccuracy.append(trainAcc)
            self.trainingLoss.append(trainLoss)
            
            print("Testing Epoch {}:".format(i+1))
            testAcc,testLoss=neuralNet.test(self.testingImgs,self.testingAnswers,hyp["testBatchSize"])
            self.testingAccuracy.append(testAcc)
            self.testingLoss.append(testLoss)
