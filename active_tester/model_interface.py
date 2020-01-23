import tensorflow as tf
import numpy as np
import sklearn as sk
import pandas as pd



class ModelInterface():
    """
    This class provides a basic interface for the Active Tester to be able to call
    any type of supported model without having to have that logic in its code


    """
    def __init__(self,model,num_classes,model_type='sklearn',outputs=None,atts=None):
        """
        outputs: should be a list of the outputs the model generates where each one is a
        tuple (name,dim)
        examples : (class/prediction,1): would be a class number
                    (probabilities,10): probabilities for a 10 class model
                    (embedding,25) internal represenation

        valid_options : class, prediction, onehot, probs, probabilities, logits,
                        embedding, represenation, std, dist, distance, similarity
        if outputs is empty or length is 1, it is assumed the output from the model
        will be a single tensor

        outputs is needed when having a tensorflow model otherwise we don't know how to name it

        certain vetting strategies require a particular naming schema, so probabilities
        and embeddings are the best names to use
        """
        self.model=model
        self.num_classes=num_classes
        self.outputs = outputs
        self.atts=atts
        self.model_type = model_type
        # atts will be used for getting additional infomation from the model if needed


    def _predict_class(self,x):
        return self.model.predict(x)

    def _predict_proba(self,x):
        return self.model.predict_proba(x)

    def _predict_sklearn(self,x):
        pred = self.model.predict(x)
        probs = self.model.predict_proba(x)
        return {'predictions':pred,'probabilities':probs}

    def _predict_tf(self,x):
        moutput = self.model.predict(x)
        data = {}
        if len(self.outputs)==1:
            if len(moutput)>1:
                data[self.outputs[0][0]]=moutput[0]
            else:
                data[self.outputs[0][0]]=moutput
        else:
            assert len(moutput)==len(self.outputs)
            for i in range(len(moutput)):
                data[self.outputs[i][0]]=moutput[i]
        return data


    def predict(self,x,probs=True,as_onehot=False,atts=None):
        """
        x must be a 2d array (or if your model expects more than a 1d array a
        number of samples)
        x must also be in batch format
        Probs true means the model has a probabilities function and will return that response
        if it is false, it will just take the predicted classes and make an array
        with the values for the predicted class being 1

        as_onehot, if true returns the predicted classes as a one hot encoding
        atts will be for future versions to add functionality
        """
        if self.model_type == 'sklearn':
            data = self._predict_sklearn(x)
        elif self.model_type == 'tf':
            data = self._predict_tf(x)
        elif self.model_type == 'keras':
            data = self._predict_tf(x)

        # maybe some post processing
        if as_onehot:
            data['predictions'] = np.eye(self.num_classes)[data['predictions']]
        return data
