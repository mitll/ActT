import tensorflow as tf
import numpy as np
import sklearn as sk
import pandas as pd




class _Dataset():
    """
    Docstring for Dataset class
    used interanlly for managing our data

    class variables:
        X : features array


    """
    def __init__(self,classification=True,y_dim=1, classes=[]):
        #self.feature_shape = x_dim
        #self.y_dim=y_dim
        #self.classification =classification

        #self.labels_per_example=labels_per_example

        self.X = None
        self.Y = {}

        self.vetted = []
        self.ground_truth = []
        self.noisy = []
        self.y_dim = y_dim
        self.num_experts = 0
        self.classes = classes

        self.random = None
        self.count = 0
        self.sample_index = 0

    def _reset_sampling(self):
        self.random = np.random.permutation(self.count)
        self.sample_index = 0

    def _updateCounts(self):
        self.count = len(self.X)
        assert self.count==len(self.noisy)
        assert self.count==len(self.ground_truth)
        #assert self.count==len(self.Y_gt)
        self._reset_sampling()

    def addFromPandas(self,X,Y_ground_truth=None,Y_vetted=None,Y_noisy=None):
        X = pd.DataFrame(X)
        X = X.to_numpy()

        Y_ground_truth = pd.DataFrame(Y_ground_truth)
        Y_ground_truth = Y_ground_truth.to_numpy()

        Y_vetted = pd.DataFrame(Y_vetted)
        Y_vetted = Y_vetted.to_numpy()

        Y_noisy = pd.DataFrame(Y_noisy)
        Y_noisy = Y_noisy.to_numpy()

        self.addFromNumpy(X,Y_ground_truth=None,Y_vetted=None,Y_noisy=None)

    def addFromNumpy(self,X,Y_ground_truth=None,Y_vetted=None,Y_noisy=None):
        """
        addFromNumpy will add the data to the dataset from numpy arrays
        ------
        param1: numpy array, of features
        param2: numpy array, ground truth labels if any
        param3: numpy array, vetted labels if any, prefereable an array of -1 if no vetted
        param4: numpy array, noisy labels if any
        """

        # handle features
        if self.X is not None:
            self.X = np.append(self.X,X,axis=0)
        else:
            self.X = X
        new_items = len(X)
        indicies = np.arange(self.count,self.count+new_items)
        #print(indicies)
        offset = self.count
        self.count+=new_items

        vetted = []
        ground_truth = []
        noisy = []

        for i in indicies:
            self.Y[i]={}
            if Y_ground_truth is not None:
                self.Y[i]['gt']=Y_ground_truth[i-offset]
                ground_truth.append(i)
            if Y_vetted is not None:
                self.Y[i]['v']=Y_vetted[i-offset]
                vetted.append(i)
            if Y_noisy is not None:
                # TODO: Add conditions for different array data types; currently only numpy supported
                # TODO: If adding more noisy labels; may need to adjust pass noisy labels
                if self.num_experts==0:
                    self.num_experts=Y_noisy.shape[1]
                self.Y[i]['noisy']=Y_noisy[i-offset]
                noisy.append(i)

        self.vetted = np.append(self.vetted,vetted).astype(int)
        self.ground_truth = np.append(self.ground_truth,ground_truth).astype(int)
        self.noisy = np.append(self.noisy,noisy).astype(int)


    def sample(self,num=-1,rearrange=True,without_replacement=False,labels=[]):
        """
        Where labels can be any, vetted, noisy, ground_truth
        any returns ground truth, then vetted, then noisy
        """
        data = {}
        if num == -1:
            num = self.count
        if rearrange:
            indicies = np.random.choice(np.arange(self.count),num, replace=False)
        else:
            indicies = np.arange(self.count)[:num]
        if 'ground_truth' in labels:
            data['ground_truth'] = np.array([self.Y[i].get('gt',-1) for i in indicies])
        if 'vetted' in labels:
            data['vetted'] = np.array([self.Y[i].get('v',-1) for i in indicies])
        # TODO: Assumes noisy will always be in labels array; breaks if not
        if 'noisy' in labels:
            data['noisy'] = np.array([self.Y[i]['noisy'] for i in indicies])

        data['indices'] = indicies
        data['features'] = np.take(self.X,indicies,axis=0)

        return data


    def getTFBatch(self):
        pass

    def getNPBatch(self):
        pass


def createDataset(features_array,classes=[],y_dim=1,num_experts=1,Y_ground_truth=None,Y_vetted=None,Y_noisy=None):
    """
    To create a dataset, you must pass atleast a features_array, y_dim, and num_experts
    y_dim and num_experts are 1 by default (and the min value) but if your noisy labels have
    more than 1 expert you will need to pass the max number of experts
    y_dim should be the dimentionality of the label, for classification this would
    typically be 1 (with the integer number of the class). For regression or other problems
    this could be more than 1, such as a bounding box estimating model, that would have a dimention of 4

    """
    dataset = _Dataset(y_dim=y_dim, classes=classes)
    if isinstance(features_array, pd.DataFrame):
        dataset.addFromPandas(X=features_array,Y_ground_truth=Y_ground_truth,Y_vetted=Y_vetted,Y_noisy=Y_noisy)
    else:
        dataset.addFromNumpy(X=features_array,Y_ground_truth=Y_ground_truth,Y_vetted=Y_vetted,Y_noisy=Y_noisy)
    return dataset
