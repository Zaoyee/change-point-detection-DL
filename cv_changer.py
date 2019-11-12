import pandas as pd

def cv_change(odFile, cvFile):
    df = pd.read_csv(odFile)
    cvs = pd.read_csv(cvFile)
    cvs = cvs[cvs['sequenceID'].isin(df['sequenceID'])]
    cvs

    df['foldID'] = cvs['fold'].values
    return df


odFile = './data/processed/cv2/systemic/datamat.csv'
cvFile = './data/processed/cv2/systemic/folds.csv'
ndf = cv_change(odFile, cvFile)
ndf.to_csv(odFile)