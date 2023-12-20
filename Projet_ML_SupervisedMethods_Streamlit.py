import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle

# Fonction pour afficher la première page
def Regression_Lineaire():
    st.title(" Regression Linéaire")
    



    df = pd.read_csv("Student_Performance.csv")
    pages = ["Contexte du projet", "Exploration des données", "Analyse de données", "Modélisation"]

    page = st.sidebar.radio("Aller vers la page :", pages)

    if page == pages[0] : 
        
        st.write("### Contexte du projet")
        
        st.write("Ce projet s'inscrit dans un contexte scolaire. L'objectif est de prédire la performance académique ou scolaire potentielles d'étudiants. Ce type de jeu de données est souvent utilisé pour analyser et prédire les performances des étudiants en fonction de divers facteurs tels que le temps d'étude, les habitudes de sommeil, les scores précédents, etc. L'objectif peut être de construire un modèle prédictif pour estimer ou prédire l'indice de performance d'un étudiant en fonction de ces paramètres.")
        
        st.write("Nous avons à notre disposition le fichier Student_Performance.csv qui contient des données académiques. Chaque observation en ligne correspond à un étudiant. Chaque variable en colonne est une caractéristique des performances .")
        
        st.write("Dans un premier temps, nous explorerons ce dataset. Puis nous l'analyserons visuellement pour en extraire des informations selon certains axes d'étude. Finalement nous implémenterons des modèles de Machine Learning pour prédire la performance.")
        
        st.image("performance.jpeg")
        
    elif page == pages[1]:
        st.write("### Exploration des données")
        
        st.dataframe(df.head())
        
        st.write("Dimensions du dataframe :")
        
        st.write(df.shape)
        
        st.write("Statistique descriptive", df.describe())
        
        if st.checkbox("Afficher les valeurs manquantes") : 
            st.dataframe(df.isna().sum())
            
        if st.checkbox("Afficher les doublons") : 
            st.write(df.duplicated().sum())
        
    elif page == pages[2]:
        st.write("### Analyse de données")
        
        fig = sns.displot(x='Performance Index', data=df, kde=True)
        plt.title("Distribution de la variable cible performance")
        st.pyplot(fig)
        
        fig2 = px.scatter(df, x="Performance Index", y="Hours Studied", title="Evolution de la performance en fonction des heures étudiées")
        st.plotly_chart(fig2)
        
        fig3 = px.scatter(df, x="Performance Index", y="Sleep Hours", title="Evolution de la performance en fonction du nombre d'heures de sommeil")
        st.plotly_chart(fig3)
        
        fig4 = px.scatter(df, x="Performance Index", y="Previous Scores", title="Evolution de la performance en fonction des scores obtenus dernièrement")
        st.plotly_chart(fig4)
        
        fig5, ax = plt.subplots()
        sns.heatmap(df.corr(), ax=ax)
        plt.title("Matrice de corrélation des variables du dataframe")
        st.write(fig5)
        

    elif page == pages[3]:
        st.write("### Modélisation")
        
        
        from sklearn.preprocessing import LabelEncoder
        df['Extracurricular Activities'] = LabelEncoder().fit_transform(df['Extracurricular Activities'])
        
        df_prep = pd.read_csv("df_preprocessed.csv")
        
        Y = df_prep["Performance_Index"]
        X= df_prep.drop("Performance_Index", axis=1)
        
        
        
        #Normaliser les données (X)
        scaler = StandardScaler()
        scaler.fit_transform(X) 
        
        
        #Splitter les données 
        x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
        #Splitter les données en val et test
        x_val, x_test, y_val, y_test= train_test_split(x_test, y_test, test_size=0.5, random_state=42)
        
        linear_model = joblib.load('LinearRegression_model.pkl')
        ridge_model = joblib.load('RidgeRegression_model.pkl')
        lasso_model = joblib.load('Lasso_model.pkl')
        
        
        y_pred_lr= linear_model.predict(x_val)
        y_pred_rr=ridge_model.predict(x_val)
        y_pred_lass=lasso_model.predict(x_val)
        
        model_choisi = st.selectbox(label="Modèle", options=['Regression Lineaire', 'Ridge Regression', 'Lasso'])

        def accu(y_true, y_pred):
            R2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            return R2, mse

        if model_choisi == 'Regression Lineaire':
        # Effectuer la prédiction avec le modèle de régression linéaire
        
            R2, mse = accu(y_val, y_pred_lr)
            st.write(f"La précision du modèle de régression linéaire est de :  {np.round(R2, 7)}, {np.round(mse, 7)}")

        elif model_choisi == 'Ridge Regression':
        # Effectuer la prédiction avec le modèle de régression Ridge
      
            R2, mse = accu(y_val, y_pred_rr)
            st.write(f"La précision du modèle est de : {np.round(R2, 7)}, {np.round(mse, 7)}")

        elif model_choisi == 'Lasso':
        # Effectuer la prédiction avec le modèle Lasso
      
            R2, mse = accu(y_val, y_pred_lass)
            st.write(f"La précision du modèle Lasso est de :  {np.round(R2, 7)}, {np.round(mse, 7)}")

        
        st.success("La ridge regression est le meilleur modèle d'après les scores obtenus")
        
        params = {
        'alpha': np.logspace(-8, 8, 100)
        }
        models = {
        'LinearRegression': LinearRegression(),
        'RidgeRegression': GridSearchCV(Ridge(), params, cv=5),
        'Lasso': GridSearchCV(Lasso(), params, cv=5)
        }
        # Entraînement du meilleur modèle (Linear Regression dans ce cas)
        best_model = models['RidgeRegression']
        best_model.fit(x_train, y_train)

        # Prédiction sur les données de test (les 3 premières observations)
        x_test_3 = x_test[:3]
        y_pred_3 = best_model.predict(x_test_3)

        # Affichage des prédictions sur Streamlit
        st.write("Affichage des prédictions pour les 3 premiers étudiants :")
        for i in range(3):
            st.write(f'Étudiant {i} - Performance prédite : {np.round(y_pred_3[i], 2)} --- {y_test.iloc[i]} Performance réelle')


        
        

        # Charger le modèle
        model = joblib.load('best_model_ridge_regression.pkl')
        scaler = joblib.load('scaler.pkl')  # Chargez le scaler

    


        # Interface utilisateur Streamlit
        st.title('Prédiction des performances des étudiants')

        # Ajouter des champs pour saisir les données nécessaires à la prédiction
        hours_studied = st.slider('Nombre d\'heures d\'étude', min_value=1, max_value=9)
        sleep_hours = st.slider('Nombre d\'heures de sommeil', min_value=4, max_value=9)
        previous_scores = st.slider('Scores précédents', min_value=40, max_value=99)
        Extracurricular_Activities = st.slider('Activités extra-scolaire', min_value=0, max_value=1)
        Sample_Question_Papers_Practiced = st.slider('Exemples de questions pratiquées', min_value=0, max_value=9)

        # Bouton pour lancer la prédiction
        if st.button('Prédire'):
            # Organiser les données dans un tableau pour la prédiction
            user_data = np.array([[hours_studied, sleep_hours, previous_scores, Extracurricular_Activities, Sample_Question_Papers_Practiced]])
            

            
        
            
            user_data_normalized = scaler.transform(user_data)  # Normaliser les données utilisateur
            
            
        
            
            # Faire la prédiction avec le modèle chargé
            prediction = model.predict(user_data_normalized)
            
            
            
            
        
            
            
            # Afficher la prédiction
            st.write(f"La performance académique prédite de l'étudiant est de: {np.round(prediction[0], 2)}")
        
        



# Fonction pour afficher la deuxième page
def Classification():
    st.title("Classification")
    

    df = pd.read_csv("diabetes.csv")

    st.dataframe(df.head())

    st.sidebar.title("Sommaire🎉")
    pages = ["Contexte du projet❄️", "Exploration des données❄️", "Analyse de données❄️", "Modélisation 🎈"]

    page = st.sidebar.radio("Aller vers la page :", pages)

    if page == pages[0]:
        st.write("### Contexte du projet")

        st.write(" L'ensemble de données provient de l'Institut national du diabète et des maladies digestives et rénales (NIDDK) et vise à résoudre un problème crucial dans le domaine médical : la prédiction diagnostique du diabète. Le diabète est une maladie chronique affectant la régulation du glucose dans le sang et a des implications significatives sur la santé publique.")

        st.write("Nous disposons d'un ensemble de données stocké dans le fichier diabet.csv. Cet ensemble de données comprend plusieurs caractéristiques médicales telles que l'âge, la pression artérielle, le taux de glucose, l'indice de masse corporelle (IMC) et d'autres indicateurs importants. Ces caractéristiques seront utilisées comme variables d'entrée pour le modèle de prédiction.")

        st.write("Dans un premier temps, nous explorerons ce dataset. Puis nous l'analyserons visuellement pour en extraire des informations selon certains axes d'étude. Finalement nous implémenterons des modèles de Machine Learning pour prédire si un patient souffre du diabète ou non .")

        st.image("ldiabete.jpg")

    elif page == pages[1]:
        st.write("### Exploration des données")

        st.dataframe(df.head())

        st.write("Dimensions du dataframe :")

        st.write(df.shape)
        st.write("Description des donnees:")
        st.write(df.describe())
        st.write("le nombre d'observation de la variable cible:")
        st.write(df["Outcome"].value_counts())
        if st.checkbox("Afficher les valeurs manquantes"):
            st.dataframe(df.isna().sum())

        if st.checkbox("Afficher les doublons"):
            st.write(df.duplicated().sum())

    elif page == pages[2]:
        st.write("### Analyse de données")

        fig3, ax = plt.subplots()
        sns.heatmap(df.corr(), ax=ax, annot=True)
        plt.title("Matrice de corrélation des variables du dataframe")
        st.write(fig3)

        #with pd.option_context():
        fig = sns.displot(x='Outcome', data=df, kde=True)
        plt.title("Distribution de la variable cible Outcome")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(15, 10))
        sns.boxplot(data=df, width=0.5, ax=ax, fliersize=10)
        plt.title("visualisation des valeurs aberrantes de chaque variable")
        st.pyplot(fig)

        fig = sns.pairplot(df)
        plt.title("visualisation de la relation entre les variable")
        st.pyplot(fig)

    elif page == pages[3]:
        st.write("### Modélisation")

        #df_prep = pd.read_csv("diabetes.csv")

        x = df.drop("Outcome", axis=1).values
        y = df.Outcome.values

        # Normaliser les données
        standard = StandardScaler()
        x= standard.fit_transform(x)
        joblib.dump(standard, 'standard.pkl')

        # spliter les donnees
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
        print("C:\\Users\\hp\\Desktop\\ML\\Group 2\\Le-d-ploiement_ML_Streamlit\\model_logisticR:", "model_logisticR.pkl")
        reg = joblib.load("model_logisticR.pkl")
        svm = joblib.load("model_svm.pkl")
        standard= joblib.load('standard.pkl')  
        KNN =  joblib.load("neigh.pkl")
      

        st.write("Modèles chargés avec succès.")

        y_pred_reg = reg.predict(x_val)
        y_pred_rf = svm.predict(x_val)
        y_pred_kn = KNN.predict(x_val)
        #y_pred_knn = knn.predict(x_val)

        model_choisi = st.selectbox("Modèle", options=['Logistique Regression', 'SVM','KNN'])


        def train_model(model_choisi):
            if model_choisi == 'Logistique Regression':
                y_pred = y_pred_reg
            elif model_choisi == 'SVM':
                y_pred = y_pred_rf
            elif model_choisi== 'KNN':
                y_pred = y_pred_kn  
            f1 = f1_score(y_pred, y_val)
            acc = accuracy_score(y_pred, y_val)
            return f1, acc

        st.write("Le Score F1  et le taux de Précision (accuracy)", train_model(model_choisi))
        st.success("Le KNN est le modèle le plus performant 🎉")
        st.text(" prédictions sur les 10 premières lignes du jeu de test à l'aide du KNN")
    # Prédictions
        x_test_3 = x_test[:10]
        y_test_3 = KNN.predict(x_test_3)

    # Créer un DataFrame pour les prédictions
        predictions_df = pd.DataFrame({
            'Personne': [f"Personne {i}" for i in range(0, 10)],
            'Statut': ['Diabétique' if status == 1 else 'Non-diabétique' for status in y_test_3]
    })

    # Afficher le DataFrame dans Streamlit
        st.dataframe(predictions_df)

    if page == pages[3]:      
        st.title("Prédiction du Diabète")
        # Champs de saisie pour l'utilisateur
        Pregnancies = st.slider('Nombre de grossesses', min_value=0, max_value=20)
        Glucose = st.slider('Glucose', min_value=0, max_value=199)
        BloodPressure = st.slider('Pression Arterielle', min_value=0, max_value=122)
        SkinThickness = st.slider('Epaisseur de la peau', min_value=0, max_value=99)
        Insulin = st.slider('Insulline', min_value=0, max_value=846)
        BMI = st.number_input('IMC', min_value=0.0, max_value=67.1)
        DiabetesPedigreeFunction = st.number_input('Pourcentage du diabete', min_value=0.0, max_value=2.5)
        Age = st.slider('Âge', min_value=21, max_value=86)
        
        # Créer un tableau NumPy avec toutes les caractéristiques
        user_input = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        # Vérifier si l'utilisateur veut faire une prédiction
        if st.button("Faire une prédiction"):
            try:
                # Normaliser les caractéristiques de l'utilisateur avec le même StandardScaler
                user_input_array = standard.transform(user_input)

                # Faire la prédiction avec le modèle de régression logistique
                prediction = KNN.predict(user_input_array)
        
                # Afficher le résultat de la prédiction pour cette personne
                st.write(f"Résultat de la prédiction : {'Diabétique' if prediction == 1 else 'Non-diabétique'}")
            except ValueError as e:
                st.write(f"Erreur lors de la prédiction : {e}")














pages = {
    "Regression Linéaire": Regression_Lineaire,
    "Classification": Classification,
}

# Barre de sélection pour naviguer entre les pages
page_selectionnee = st.sidebar.radio("Sélectionnez une page", tuple(pages.keys()))

# Affichage de la page sélectionnée
pages[page_selectionnee]()
