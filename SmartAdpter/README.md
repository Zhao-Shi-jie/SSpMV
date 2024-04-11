# Smart Adapter for SpMV


**Baseline.py**:  This is a model proposed by paper ["Prediction of Optimal Solvers for Sparse Linear Systems Using Deep Learning"](https://epubs.siam.org/doi/abs/10.1137/1.9781611977141.2), it's using the fully-connected neural network with handcrafted features.

**new_wide_deep_add.py**, **new_wide_deep_concat.py**, **new_wide.py**, and **new_wide_deep.py**: Using different ways to implement wide&deep model: 

(1) new_wide_deep_add.py: Combine model in *WideAndDeepAdd* by adding.

(2) new_wide_deep_concat.py: Combine model in *NewWideAndDeepConcat* by concat. 

(3) new_wide_deep.py: Without bn layer, don't do Batch Normalization.

(4) new_wide.py: Without CNN part, only have Wide model.

(5) wide_and_deep_com_concat.py: using the original model given by paper: ["Wide & Deep Learning for Recommender Systems"](https://arxiv.org/abs/1606.07792). Concatenated method: element-wise add.
