from sklearn.datasets import load_diabetes
import pandas as pd
data = load_diabetes(as_frame=True,return_X_y=True)
pd.merge(data[0],data[1],left_index=True,right_index=True).to_csv("dia.csv",index=False)