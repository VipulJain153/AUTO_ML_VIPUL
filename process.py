from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
def preprocess(df,target):
    attribs =  []
    strAttribs = []
    for c in df.columns:
        if c!=target:
            dtype = df.loc[:,c].dtype
            if dtype == object or dtype == pd.CategoricalDtype:
                attribs.append(c)
            elif dtype == pd.StringDtype:
                strAttribs.append(c)
    if strAttribs:
        df.drop(strAttribs,axis=1,inplace=True)
    try:
        corrs = list(df.corr()[target].sort_values(ascending=False)[1:4].index)
        df["new1"] = df[corrs[0]]+df[corrs[1]]
        df["new2"] = df[corrs[2]]+df[corrs[0]]
    except:
        pass
    pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("std_scler", StandardScaler())
    ])
    otherAttribs=[]
    for i in df.columns:
        if i not in attribs and i!=target:
            otherAttribs.append(i)
    transformer = ColumnTransformer([
    ("attribs",OneHotEncoder(),attribs),
    ("other",pipeline,otherAttribs),
    ])
    encoder=OneHotEncoder()
    try:
        x = df.drop(target,axis=1)
        X=transformer.fit_transform(x)
        y=df[target].to_numpy() if target not in attribs else encoder.fit_transform(df[target])
        return X,y
    except:
        print("ok")