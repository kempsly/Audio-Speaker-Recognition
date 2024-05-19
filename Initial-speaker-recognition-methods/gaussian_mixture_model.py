import sklearn 



# Initializing a GMM model

model = sklearn.mixture.GaussianMixture(
    n_components=1, covariance_type='full', max_iter=1 n_init=1, init_params='kmeans'
)

# fitting the model
model.fit(X)

# Log-likelyhood
s = model.score(y)