"""Reusable frozen-feature fusion head; no split/CV/test/threshold-search code."""
import numpy as np


def create_fusion_head(components=64,C=1.):
    """Seven clinical fields followed by CFP and OCTA patch-mean descriptors.

    Fits preprocessing on caller-supplied training vectors only. C/PCA settings
    are arguments, not selected using a test set. The score is not calibrated risk.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    clinical=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())])
    images=Pipeline([('scale',StandardScaler()),('pca',PCA(n_components=components,whiten=True))])
    return Pipeline([('features',ColumnTransformer([('clinical',clinical,slice(0,7)),('retinal',images,slice(7,2055))])),
        ('classifier',SVC(C=C,kernel='rbf',probability=False))])


def screening_scores(head,vectors):
    from scipy.special import expit
    vectors=np.asarray(vectors)
    if vectors.ndim!=2 or vectors.shape[1]!=2055:raise ValueError('Seven clinical and two 1024-wide descriptors required')
    return expit(head.decision_function(vectors))
