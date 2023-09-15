import pandas as pd 
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import statsmodels.api as sm

def find_elbow_point(errors):
# 변화율을 계산합니다.
	deltas = np.diff(errors)
#변화율의 변화율을 계산합니다.
	double_deltas = np.diff(deltas)
#변화율의 변화율이 가장 큰 지점을 찾습니다.
	elbow_point = np.argmax(double_deltas) + 1
	return elbow_point

data_matrix = df_rf_5.values  
errors = []
component_range = range(1,51)
for n in component_range:
	nmf = NMF(n_components=n, init='random', random_state=1, max_iter=10000)
	W = nmf.fit_transform(data_matrix)
	H = nmf.components_
	reconstructed_matrix = np.dot(W, H)
	error = np.linalg.norm(data_matrix - reconstructed_matrix, 'fro')
	errors.append(error)

elbow_point = find_elbow_point(errors)
elbow_point

nmf = NMF(n_components=50, init='random', random_state=1,max_iter=10000)
W = nmf.fit_transform(df_rf_5.values)
H = nmf.components_

reconstructed = np.dot(W, H)
error = np.linalg.norm(df_rf_5.values - reconstructed, 'fro')
W_df = pd.DataFrame(W, columns=[f'Feature_{i}' for i in range(W.shape[1])])
H_df = pd.DataFrame(H, columns=[f'Original_Feature_{i}' for i in range(H.shape[1])])


W_val = nmf.transform(val_rf_5.values)
similarity = cosine_similarity(W_old, W_new)
print("Similarity:", np.mean(similarity))

X= sm.add_constant(W_df['Component_3'])
y=W_df['case']
model = sm.Logit(y,X)
result = model.fit()
result.summary()
