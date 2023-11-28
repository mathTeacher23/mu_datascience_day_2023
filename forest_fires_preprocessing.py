import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def feature_engineer(df):
    df['size_category'] = np.where(df['area'] > 6, '1', '0')
    df['size_category']= pd.to_numeric(df['size_category'])
    return(df)

def preprocess(df):
    # converting to is weekend
    df['day'] = ((df['day'] == 'sun') | (df['day'] == 'sat'))
    # renaming column
    df = df.rename(columns = {'day' : 'is_weekend'})
    return(df)

def scale_features(df):
    # natural logarithm scaling (+1 to prevent errors at 0)
    df.loc[:, ['rain', 'area']] = df.loc[:, ['rain', 'area']].apply(lambda x: np.log(x + 1), axis = 1)
    return(df)

def make_train_test_split_scaled(df):
    features = df.drop(['size_category'], axis = 1)
    labels = df['size_category'].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size = 0.2, random_state = 42)
    # fitting scaler
    sc_features = StandardScaler()
    # transforming features
    X_test = sc_features.fit_transform(X_test)
    X_train = sc_features.transform(X_train)
    # features
    X_test = pd.DataFrame(X_test, columns = features.columns)
    X_train = pd.DataFrame(X_train, columns = features.columns)
    # labels
    y_test = pd.DataFrame(y_test, columns = ['size_category'])
    y_train = pd.DataFrame(y_train, columns = ['size_category'])
    return(X_train, X_test, y_train, y_test)

def main():
    df = pd.read_csv("data/forestfires.csv")
    df = feature_engineer(df)
    df = preprocess(df)
    df = scale_features(df)
    
    X_train, X_test, y_train, y_test = make_train_test_split_scaled(df)
    
    return(X_train, X_test, y_train, y_test)

    
if __name__ == "__main__":
    main()
    

    