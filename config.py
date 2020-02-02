class Config:
    def __init__(self):
        self.inputPhotoDim=(14,14)
        #note order in this list will be used to determine which one hot vector corrisponds
        #to which label 
        self.allLabels=[0,2,4,6,8]

        self.dataPath="even_mnist.csv"
        self.numLabels=len(self.allLabels)

        self.inputLen=self.inputPhotoDim[0]*self.inputPhotoDim[1]

        #number of fully connected layers (includes input layer, excludes output layer)
        self.fcLayers=2

        #fcNodes only contains the number of nodes in hidden layers
        #number of input nodes/output nodes are determined by input size/number of labels respectivly
        self.fcNodes=[512]
        
        