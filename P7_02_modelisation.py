import script.P7_01_features_preprocessing as feat  # script de features engineering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # script contenant des fonctions et sous_fonctions
import script.P7_03_fonctions as fct
from importlib import reload # pour recharger les scripts feat et fct après modification
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from joblib import dump, load  # export d'objet python ex: modèle
import matplotlib.patches as mpatches  # pour créer une légende matplotlib


# Création du dataframe resultat
cpt_essai = 0
resultat_campagne = pd.DataFrame(columns=['num_essai', 'modelisation', 'donnees',
                                          'echan.(Ov ,Und, Bal)', 'accurancy', 'AUC', 'F_mesure', 'MCC', 'Tx_bon_pred'])
# importation des jeux de données (features engineering effectué)
nrows = None
train_3 = pd.read_csv('data/train_3.csv', index_col=0, nrows=nrows)
train_3.shape  # (307507 lignes , 247 colonnes)
test_3 = pd.read_csv('data/test_3.csv', index_col=0, nrows=nrows)
test_3.shape  # (48744 lignes , 246 colonnes)


# ANALYSE EXPLORATOIRE
# Types des colonnes
train_3.info()  # float64(171), int64(39), object(37)
# Distribution de la variable 'TARGET'
taux_defaut_paiement = (train_3['TARGET'].value_counts()[
                        1]/train_3['TARGET'].count())
print('le taux de défaut de paiement est : ', round(taux_defaut_paiement, 2))
train_3['TARGET'].plot.hist()
plt.show()
# Detection des valeurs manquantes
valeurs_manquantes_par_colonne = fct.missing_values_table(train_3)
print(valeurs_manquantes_par_colonne.head(30))
taux_de_valeurs_manquantes = valeurs_manquantes_par_colonne['Missing Values'].sum(
)/(307507*247)  # taux de valeurs manquantes de 24%
# Histogramme du dataframe complet
nv = train_3.count()
ax = nv.plot(kind="bar", figsize=(20, 6))
ax.set_xlabel("variable explicatives", fontsize=25)
ax.set_ylabel("nombre de clients", fontsize=20)
ax.set_title('nombre de clients par variable explicative', fontsize=25)
ax.legend().set_visible(False)
ax.xaxis.set_ticklabels([])
plt.show()
# Détection des valeurs abberantes
train_3.describe(include='all')
# Recherche des correlations avec la cible. Tri hiérarchique
correlations = train_3.corr()['TARGET'].sort_values()
# Présentation des résultats des correlations
print('Les correlations les plus positives :\n', correlations.tail(15))


# DETERMINATION DU MEILLEUR MODELE (REGRESSION LOGISTIQUE) SUR LE TRAINING SET
# Preparation des donnees (limitée aux 10000 premières lignes)
# imputation, standardisation et discrétisation des variables de train_3(retrait id_client et TARGET). X_train_prep comporte 548 colonnes
train_prep = fct.preparation_des_variables(train_3.iloc[:10000, 2:])
# Sauvegarde du pipe de préparation des données fitté sur le jeu de training
pipe_prep, preparation = fct.création_pipe_preparation_des_variables(
    train_3.iloc[:10000, 2:])
dump(pipe_prep, 'data/pipe_preparation_variable1.joblib') # enregistrer le modèle de préparation

# Séparation du jeu de données labelisées train_3 en un jeu d'entrainement et un jeu de validation
X_train, X_val, y_train, y_val = train_test_split(
    train_prep, train_3.iloc[:10000, 1], test_size=0.15, random_state=42)  # X_train : 261380 lignes, X_val : 46127 lignes

# Recherche du meilleur hyperparamètre C en appliquant la fonction regressionlogisticCV sur le training set X_train
# selon 5 configurations d'échantillonage des observations et de pénalisation de la classification
bilan_scores_CV = {}  # pour enregistrement des scores
bilan_Cs = {}  # pour enregistrement de l'hyperparamètre C
# Entrainement du modèle par cross validation (3 folds) pour les 5 configurations d'échantillonage des observations et de pénalisation de la classification
for (oversampling, undersampling, class_weight, name) in [(True, False, None, 'over'), (False, True, None, 'under'), (True, True, None, 'hybride'), (False, False, 'balanced', 'pondération'), (False, False, None, 'normal')]:
    train_sampling, train_labels_sampling = fct.data_sampling(
        X_train, y_train, k=5, oversampling=oversampling, undersampling=undersampling, class_weight=class_weight)  # échantillonage selon la configuration choisie
    # définition du modèle - class_weight est le paramétre de pénalisation de la fonction coût - métrique adaptée : AUROC
    model_reglog = LogisticRegressionCV(
        cv=3, scoring='roc_auc', Cs=10, penalty='l2', solver='lbfgs', random_state=0, max_iter=4000, class_weight=None)
    model_reglog.fit(train_sampling, train_labels_sampling)
    bilan_scores_CV[name] = model_reglog.scores_  # enregistrement des scores
    bilan_Cs[name] = model_reglog.C_  # enregistrement de l'hyperparamètre C

# Entrainement du modèle de régression logistique avec le meilleur hyperparamètre C pour chacune des 5 configurations
# Prédiction sur le jeu de validation et calcul des métriques d'évaluation pour chacune des 5 configurations
model = {}
bilan_scores = {}
confusion_mat = {}
for (oversampling, undersampling, class_weight, name) in [(True, False, None, 'over'), (False, True, None, 'under'), (True, True, None, 'hybride'), (False, False, 'balanced', 'pondération'), (False, False, None, 'normal')]:
    # Echantillonage selon la configuration choisie
    train_sampling, train_labels_sampling = fct.data_sampling(
        X_train, y_train, k=5, oversampling=oversampling, undersampling=undersampling, class_weight=class_weight)
    # Entrainement du modèle pour chaque configuration avec le meilleur hyperparamètre C sur le X_train préparé
    model[name] = LogisticRegression(C=float(bilan_Cs[name]), penalty='l2', solver='lbfgs',
                                     random_state=0, max_iter=4000, class_weight=class_weight)  # définition du modèle
    model[name].fit(train_sampling, train_labels_sampling)
    bilan_scores[name] = model[name].score(
        train_sampling, train_labels_sampling)
    # Enregistrement du meilleur modèle entrainé dans chaque configuration
    dump(model[name], 'data/reglog1_%s.joblib' % name)
    # prediction du modèle sur le jeu de validation
    # Y_pred donne les labels, pred_model donne la probailité d'appartenance à chaque classe
    pred_model, Y_pred = fct.prediction_modele(model[name], X_val)
    # Calcul des métriques choisies pour leur pertinences
    accuracy, AUC, F_mesure, MCC, Tx_bon_pred, cm = fct.calcul_des_metriques(
        y_val, Y_pred, pred_model)  # 5 métriques et matrice de confusion
    # Affichage et enregistrement de la matrice de confusion
    confusion_mat[name] = fct.matrice_de_confusion(cm)
    # Enregistrement des paramètres de la modélisation
    cpt_essai, resultat_essai = fct.mise_a_jour_resultat(
        cpt_essai, 'reglog'+'_'+name, '307507', (oversampling, undersampling, class_weight), accuracy, AUC, F_mesure, MCC, Tx_bon_pred)
    resultat_campagne = pd.concat([resultat_campagne, resultat_essai])
resultat_campagne.iloc[:, 3:]
resultat_campagne.to_csv(
    'data/resultat_campagne_entr10000_predval46000', index=False)


# APPLICATION DU MEILLEUR MODELE SUR LE JEU DE TEST
# Rapppel du modèle exporté (le meilleur est "pondération" qui pénalise la classification)
reglog_ponderation = load('data/reglog_pondération.joblib', mmap_mode=None)
# Preparation des donnees (les datasets ne sont pas limités en nbre de lignes)
# Rappel du pipe de préparation des données
pipe_prep = load('data\\pipe_preparation_variable1.joblib', mmap_mode=None)
# retrait de la colonne "SK_ID_CURR"
test_prep = pipe_prep.transform(test_3.iloc[:, 1:])
# retrait des colonnes "SK_ID_CURR" et "TARGET"
train_prep = pipe_prep.transform(train_3.iloc[:, 2:])
# prediction du jeu de test
# Y_pred donne les labels, pred_model donne la probailité d'appartenance à chaque classe
pred_model, Y_pred = fct.prediction_modele(reglog_ponderation, test_prep)
test_3['score_proba'] = pred_model[:, 1]
test_3['scoring'] = Y_pred
# affichage des résultats sur le jeu de test puis sauvegarde de celui-ci pour utilisation par Dash
test_3[['SK_ID_CURR', 'score_proba', 'scoring']]
test_3.to_csv('data/test_3_scoring.csv')
# histogramme du scoring
plt.hist(test_3['scoring'], bins=100)
plt.show()
test_3['scoring'].value_counts()


# IMPORTANCE DES COEFFICIENTS ET TOP DES VARIABLES
# Obtention des coefficients
importance = reglog_ponderation.coef_[0]
# Noms des colonnes aprés préparation des données
noms_var = fct.noms_variables_et_modalites(train_3.iloc[:, 2:])
# Mise en tableau des coeff, création des colonnes valeur absolue et signe des coeff
tab_coef = pd.DataFrame({'coeff': importance}, index=noms_var)
tab_coef['|coeff|'] = tab_coef['coeff'].apply(lambda x: abs(x))
tab_coef['signe_coeff'] = tab_coef['coeff'].apply(lambda x: (x > 0) - (x < 0))
tab_coef.to_csv('data/coefs.csv')
#pd.read_csv('data/coefs.csv', sep=',', index_col=None)
# Barplot du top50 des coeff en valeur absolue avec couleur selon le signe
tab_coef = tab_coef.sort_values(by='|coeff|', ascending=False)
colors = ['red' if x > 0 else 'green' for x in list(
    tab_coef.iloc[:10, 2])]  # choix des couleurs selon signe coeff
tab_coef.iloc[:50, 1].plot(kind='barh', figsize=(9, 7), color=colors)
red_patch = mpatches.Patch(color='red', label='risque augmenté')
green_patch = mpatches.Patch(color='green', label='risque diminué')
plt.legend(handles=[red_patch, green_patch])
plt.title('Importance des variables - Regression logistic')
plt.axvline(x=0, color='.5')
plt.subplots_adjust(left=.4)
plt.show()
# création d'un masque des 20 premières variables (pour faciliter réalisation script Dash)
mask_top20_coeff = list(tab_coef[:20].index)