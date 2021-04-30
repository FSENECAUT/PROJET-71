import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import collections
import seaborn as sns; sns.set()
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef

# Function to calculate missing values by column 
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    # Return the dataframe with missing information
    return mis_val_table_ren_columns

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Regroupement des colonnes selon leur type
def separation_des_variables_selon_type(df):
    # Regroupement des colonnes selon leur type
    cat_columns = [col for col in df.columns if df[col].dtype == 'object']
    num_columns = [col for col in df.columns if (df[col].dtype == 'int'|df[col].dtype == 'float')]
    return cat_columns, num_columns

# regroupement des variables numériques et des modalités des variables catégorielles
def noms_variables_et_modalites(X):
    cat_cols = X.columns[X.dtypes == 'object']
    num_cols = X.columns[X.dtypes == 'float64']|X.columns[X.dtypes == 'int64']
    # regroupement des catégories et gestion des manquants chez les variables catégorielles
    X[cat_cols]=X[cat_cols].fillna(value='manquant')
    # regroupement des features numérique et catégoriel
    features_noms=[]
    for col in num_cols:
        features_noms.append(col)
    for col in cat_cols:
        for i in range(len(X[col].unique())):
            modalite=(col+'_'+X[col].unique()[i])
            features_noms.append(modalite)               
    return features_noms

# sous-fonction préparation des variables (pipeline de transformation)
def preparation_des_variables(X):                           
    #regroupement des variables catégorielles et des variables quantitatives
    cat_cols = X.columns[X.dtypes == 'object']
    num_cols = X.columns[X.dtypes == 'float64']|X.columns[X.dtypes == 'int64']
    # regroupement des catégories et gestion des manquants chez les variables catégorielles
    categories = [X[column].unique() for column in X[cat_cols]]
    for cat in categories:
        cat[cat == None] = 'manquant'
    # création pipeline de transformation des quantitatives avec variables quanti standardisées
    transfo_quanti_std = Pipeline(steps=[('imputation', SimpleImputer(strategy='median')),
                                        ('standard', StandardScaler())])
    # création pipeline de transformation des qualitatives 
    transfo_quali = Pipeline(steps=[
    ('imputation', SimpleImputer(strategy='constant', fill_value='manquant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore',categories=categories))])
    # on définit l'objet de la classe ColumnTransformer
    # qui va permettre d'appliquer toutes les étapes de transformation
    preparation = ColumnTransformer(transformers=[('quanti_std', transfo_quanti_std , num_cols),
                                                  ('quali', transfo_quali , cat_cols)])
    X_prep = make_pipeline(preparation, 'passthrough').fit_transform(X) # X_rep est une sparse matrix
    #df_X = pd.DataFrame(X_prep) # si nrows grand .toarray()

    return X_prep
    
# création d'un pipe de préparation des données entrainé sur le jeu de training
def création_pipe_preparation_des_variables(X):                           
    #regroupement des variables catégorielles et des variables quantitatives
    cat_cols = X.columns[X.dtypes == 'object']
    num_cols = X.columns[X.dtypes == 'float64']|X.columns[X.dtypes == 'int64']
    # regroupement des catégories et gestion des manquants chez les variables catégorielles
    categories = [X[column].unique() for column in X[cat_cols]]
    for cat in categories:
        cat[cat == None] = 'manquant'
    # création pipeline de transformation des quantitatives avec variables quanti standardisées
    transfo_quanti_std = Pipeline(steps=[('imputation', SimpleImputer(strategy='median')),
                                        ('standard', StandardScaler())])
    # création pipeline de transformation des qualitatives 
    transfo_quali = Pipeline(steps=[
    ('imputation', SimpleImputer(strategy='constant', fill_value='manquant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore',categories=categories))])
    # on définit l'objet de la classe ColumnTransformer
    # qui va permettre d'appliquer toutes les étapes de transformation
    preparation = ColumnTransformer(transformers=[('quanti_std', transfo_quanti_std , num_cols),
                                                  ('quali', transfo_quali , cat_cols)])
    pipe_prep = make_pipeline(preparation, 'passthrough').fit(X) # X_rep est une sparse matrix
    return pipe_prep, preparation

# sous-fonction créant un pipe de modèle avec recherche des meilleurs hyperparamètres
def definition_des_pipes_de_modelisation(max_iter, oversampling=True, undersampling=True, class_weight='balanced') :
    # pipeline modèle regression logistique
    model_reglog=LogisticRegressionCV(cv=5, random_state=0, max_iter=max_iter, class_weight=class_weight)
    #LogisticRegression(penalty='l1', solver='liblinear' ,max_iter=100, random_state=0, class_weight=class_weight) # !!!!!!jouer avec param C
    # grid_search_rf= GridSearchCV(estimator = model_rf, param_grid = param_grid_rf, 
                         # cv = 5, n_jobs = -1, verbose = 2)
    # Augmentation de l'échantillon sous_représenté (imbalance classification) par méthode SMOTE
    over = SMOTE(sampling_strategy=0.9) # 10 percent the number of examples of the majority class
    under = RandomUnderSampler(sampling_strategy=0.5) # 50 percent more than the minority class
    if oversampling:
        pipe_reglog=make_pipeline(over, model_reglog)
    elif undersampling:
        pipe_reglog=make_pipeline(under, model_reglog)
    elif (oversampling and undersampling) :
        pipe_reglog=make_pipeline(over, under, model_reglog)
    elif class_weight == 'balanced':
        pipe_reglog=make_pipeline(model_reglog)
    else : pipe_reglog=make_pipeline(model_reglog)
    return pipe_reglog

def entrainement_prediction_modele(pipe_model, X, Y):
    # sous_fonction entrainement et prediction du modèle
    pipe_model.fit(X, Y)
    pred_model = pipe_model.predict_proba(X)
    Y_pred = pipe_model.predict(X)
    return pred_model, Y_pred

def prediction_modele(pipe_model, X):
    # sous_fonction entrainement et prediction du modèle
    pred_model = pipe_model.predict_proba(X)
    Y_pred = pipe_model.predict(X)
    return pred_model, Y_pred

def calcul_des_metriques(Y, Y_pred, pred_model):
    # sous-fonction élaborant les métriques 
    accurancy = accuracy_score(Y, Y_pred, normalize=True)
    F_mesure=f1_score(Y, Y_pred, average=None)
    MCC = matthews_corrcoef(Y, Y_pred)
    AUC=roc_auc_score(Y, pred_model[:, 1])
    Tx_bon_pred = collections.Counter((Y_pred+Y)/2)[1]/collections.Counter(Y)[1]
    cm = confusion_matrix(Y, Y_pred)
    return accurancy, AUC, F_mesure, MCC, Tx_bon_pred, cm


def data_sampling(X, Y, k, oversampling=True, undersampling=True, class_weight='balanced'):
    over = SMOTE(sampling_strategy=0.55, k_neighbors=k) # environ 55% du jeu de données
    under = RandomUnderSampler(sampling_strategy=1.) # les effectifs de la classe  minoritaire sont 50% de ceux de la classe majoritaire 
    if (oversampling and undersampling):
        pipe=make_pipeline(over, under)
        X1, Y1 = pipe.fit_resample(X, Y)
    elif oversampling:
        pipe=make_pipeline(over)
        X1, Y1 = pipe.fit_resample(X, Y)
    elif undersampling:
        pipe=make_pipeline(under)
        X1, Y1 = pipe.fit_resample(X, Y)
    elif (class_weight=='balanced' or class_weight==None):
        (X1, Y1)=(X, Y)
    return X1, Y1


def matrice_de_confusion(cm):
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in
                cm.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in
                     cm.flatten()/np.sum(cm)]
    group_percentages_class = ['{0:.2%}'.format(cm[0][0]/np.sum(cm[0])), '{0:.2%}'.format(cm[0][1]/np.sum(cm[0])),'{0:.2%}'.format( cm[1][0]/np.sum(cm[1])),'{0:.2%}'.format( cm[1][1]/np.sum(cm[1]))] 
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages_class)]
    labels = np.asarray(labels).reshape(2,2)
    plt.figure(figsize = (10, 10))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
    plt.show()

# sous-fonction enregistrant les résultats
def mise_a_jour_resultat(cpt_essai, modelisation, donnees, echantillonage, accuracy, AUC, F_mesure, MCC, Tx_bon_pred):   
    #incrémentation du compteur d'essai
    cpt_essai+=1
    # definition du tableau intermédiaire
    resultat_essai=pd.DataFrame(columns = ['num_essai','modelisation', 'donnees','echan.(Ov ,Und, Bal)','accurancy', 'AUC','F_mesure', 'MCC', 'Tx_bon_pred'])
    # alimentation du tableau intermediaire
    resultat_essai=resultat_essai.append({'num_essai':cpt_essai,'modelisation':modelisation, 'donnees':donnees, 'echan.(Ov ,Und, Bal)':echantillonage,'accurancy':np.round(accuracy, 4),  'AUC': np.round(AUC, 4),'F_mesure':np.round(F_mesure, 4), 'MCC':np.round(MCC, 4), 'Tx_bon_pred': np.round(Tx_bon_pred, 4)}, ignore_index=True)                   
    return cpt_essai, resultat_essai