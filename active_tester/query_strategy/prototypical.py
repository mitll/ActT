from .base import BaseQueryStrategy
import numpy as np
import scipy

def getlowertriag(a):
    return a[np.tril_indices(a.shape[0],-1)]

class Prototypes():
    """
    This is a class for storing and computing prototypes for use within Prototypical query strategy
    vettings. It may be called directly if you wanted to compute those outside of the
    query strategy
    """
    def __init__(self,embeddings,p_class):
        self.num_prototypes=len(p_class[0])
        self.dim = len(embeddings[0])
        self.example_counts=np.zeros(self.num_prototypes)
        self.prototypes=None

    def _estimate_prototypes(self,embeddings,labels):
        """
        Estimate prototypes given some embeddings and labels
        embeddings and labels must be the same length (same number of examples)

        If a label is -1 that should indicate that it is not given, and thus will not be
        used for computing prototypes in this function

        """
        prototypes=np.zeros((self.num_prototypes,self.dim))
        for i in range(self.num_prototypes):
            indices=np.where(labels==i)[0]
            self.example_counts = len(indices)
            prototype = np.average(embeddings[indices],axis=0)
            # may add std
            prototypes[i,:]=prototype
        return prototypes

    def update_prototypes_batch(self,embeddings,labels):
        """
        This is a depracated function but kept in here as there are potential
        additions to this codebase that may make use of it (also testing)
        """
        for i in range(self.num_prototypes):
            indicies=np.where(labels==i)[0]
            update_count = len(indicies)
            if update_count==0:
                continue
            self.example_counts[i]+=update_count
            prototype = np.average(embeddings[indicies],axis=0)
            self.prototypes[i,:]=np.average([prototype,self.prototypes[i,:]],axis=0,
                                            weights=[update_count,self.example_counts[i]])

    def update_prototypes(self,vetted_embeddings,vetted_labels,update_weight=[.5,.5,0],unvetted_embeddings=None,unvetted_labels=None):
        """
        This function is called by query strategy to update the prototypes based
        on new labels

        The update weight is used for weighting how much to favor vetted, old prototypes, and unvetted
        """
        # add check for out of bound
        estimate_prototypes=self._estimate_prototypes(vetted_embeddings,vetted_labels)
        if unvetted_embeddings is not None and update_weight[2] != 0:
            unvetted_prototypes=self._estimate_prototypes(unvetted_embeddings,unvetted_labels)
        else:
            unvetted_prototypes=np.zeros((self.num_prototypes,self.dim))
            update_weight[2] = 0

        if self.prototypes is None:
            self.prototypes=np.zeros((self.num_prototypes,self.dim))
            weights = update_weight
            weights[1] = 0
        else:
            weights=update_weight
        for i in range(self.num_prototypes):
            self.prototypes[i,:] = np.average([estimate_prototypes[i,:],
                                               self.prototypes[i,:],unvetted_prototypes[i,:]],axis=0,weights=weights)


    def compute_dist_labels(self,embeddings, distance_metric='euclidean'):
        distance = scipy.spatial.distance.cdist(embeddings,self.prototypes,'euclidean')
        proto_labels = np.argmin(distance,axis=1)
        return {'distance':distance,'proto_labels':proto_labels}

    def compute_prototype_dist(self,metric='euclidean'):
        dist = scipy.spatial.distance.pdist(self.prototypes,metric=metric)
        dist = scipy.spatial.distance.squareform(dist)
        triag = getlowertriag(dist)
        avg = np.average(triag)
        std = np.std(triag)
        return {'dist':dist,'avg':avg,'std':std}


class Prototypical(BaseQueryStrategy):
    """
    Prototypical Query Strategy

    weights: how much to weight the prototypes at each batch update
    weights[0]: vetted samples
    weights[1]: previous prototypes
    weights[2]: unvetted samples

    min_examples: number of samples need for each class to sample randomly before
    using the distance based strategy (more is typicaly better)

    sample_window: minimum=2, how wide the sample window should be when sampling
    either randomly or distance based, this is to help expand the potential sampling range, which
    does improve performance

    use_unvetted: defualt=False, deciding to use the unvetted prediction from the model

    """
    def __init__(self,weights=[.6,.2,.2], min_examples=5, sample_window=2, use_unvetted=False,X=None,
                 y_noisy=None, p_classifier=None):
        super().__init__(X, y_noisy, p_classifier)
        self.min_examples=min_examples
        self.sample_window=sample_window
        self.update_weights = weights
        self.use_unvetted=use_unvetted

        if self.sample_window<2:
            self.sample_window=2

    def get_dependencies(self):
        return ['embedding','p_classifier']

    def set_args(self,embeddings,p_classifier):
        self.embeddings=embeddings
        self.p_classifier = p_classifier
        self.num_classes = len(p_classifier[0])
        self.Prototypes=Prototypes(self.embeddings,self.p_classifier)

    def choose_indices(self, y_vetted, k):
        unvetted = y_vetted==-1
        vetted = y_vetted!=-1

        unique, counts =np.unique(y_vetted,return_counts=True)
        vetted_counts=dict(zip(unique,counts))
        count = []
        for i in range(self.num_classes):
            count.append(vetted_counts.get(i,0))

        if min(count)<self.min_examples:
            best = np.array([],int)
            n = -1*self.sample_window*self.min_examples
            for i in range(self.num_classes):
                best = np.append(best,np.argpartition(self.p_classifier[:,i],n)[n:])
            best = np.unique(best)
            selection = best[unvetted[best]]

        else:
            # find max distance from label
            # estimate vetted embeddings
            if self.use_unvetted:
                self.Prototypes.update_prototypes(self.embeddings[vetted],
                                                  y_vetted[vetted],
                                                  self.update_weights,
                                                  self.embeddings[unvetted],
                                                  np.argmax(self.p_classifier, axis=1)[unvetted])
            else:
                self.Prototypes.update_prototypes(self.embeddings[vetted],y_vetted[vetted],self.update_weights)
            dist_labels=self.Prototypes.compute_dist_labels(self.embeddings)
            proto_labels=dist_labels['proto_labels']
            distance =dist_labels['distance']
            dist_from_label = np.zeros(len(distance))
            for i in range(len(proto_labels)):
                dist_from_label[i]=distance[i][proto_labels[i]]
            indices=np.argsort(dist_from_label)
            valid_indices = unvetted[indices]
            worst = indices[valid_indices][-k*2:]
            best = indices[valid_indices][:k]
            selection = np.append(worst,best)
        return np.random.permutation(selection)[:k]
