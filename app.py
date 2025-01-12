import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Όνομα εφαρμογής
st.title("Web-Based Application for Data Mining and Analysis")

# Αρχικοποίηση μεταβλητών
df = None
X = None  # Αρχικοποίηση του X
y = None  # Αρχικοποίηση του y

uploaded_file = st.file_uploader("Ανέβασε ένα CSV ή Excel αρχείο", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error during upload: {e}")
            st.stop()

    st.write("Data Preview:")
    st.write(df.head())

    # Ορισμός X και y εδώ, αφού έχει φορτωθεί το dataframe
    X = df[df.columns[:-1]]  # Όλες οι στήλες εκτός από την τελευταία
    y = df[df.columns[-1]]  # Η τελευταία στήλη (ετικέτα)


# Δημιουργία tabs
tab1, tab2, tab3, tab4 = st.tabs(["Οπτικοποίηση", "Κατηγοριοποίηση", "Ομαδοποίηση", "Πληροφορίες"])

# Tab 1: Οπτικοποίηση
with tab1:
    st.header('2D Visualization')
    if df is not None:  # Έλεγχος αν έχει φορτωθεί dataframe
        # Επιλογή στηλών για οπτικοποίηση
        columns = st.multiselect('Διάλεξε στήλες για οπτικοποίηση', df.columns[:-1])  # Εξαιρούμε την στήλη ετικέτας

        if columns:
            if len(columns) < 2:  # Έλεγχος για επαρκή αριθμό στηλών
                st.error("Please select at least 2 columns for PCA.")
            else:
                X_vis = df[columns]  # Χρησιμοποιούμε X_vis για την οπτικοποίηση

                # Μετατροπή ετικετών σε αριθμούς
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)

                # PCA
                st.subheader('PCA')
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_vis)
                st.write(f"Variance: {np.sum(pca.explained_variance_ratio_):.2f}")
                fig, ax = plt.subplots()

                # Χρήση y_encoded στο scatter plot
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded)

                legend1 = ax.legend(*scatter.legend_elements(),
                                    loc="lower left", title="Classes")
                ax.add_artist(legend1)
                st.pyplot(fig)

                # t-SNE
                st.subheader('t-SNE')
                tsne = TSNE(n_components=2, random_state=42)
                X_tsne = tsne.fit_transform(X)
                fig, ax = plt.subplots()
                scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_encoded)
                legend1 = ax.legend(*scatter.legend_elements(),
                                    loc="lower left", title="Classes")
                ax.add_artist(legend1)
                st.pyplot(fig)

                # EDA
                st.subheader('Exploratory Data Analysis (EDA)')

                # Pairplot
                if st.checkbox("Show Pairplot"):
                    fig = sns.pairplot(df[columns + [df.columns[-1]]], hue=df.columns[-1])
                    st.pyplot(fig)

                # Histogram
                if st.checkbox("Show Histogram"):
                    for col in columns:
                        st.write(f"**{col}**")
                        fig, ax = plt.subplots()
                        sns.histplot(df[col], ax=ax)
                        st.pyplot(fig)

# Tab 2: Κατηγοριοποίηση
with tab2:
    st.header('Μηχανική Μάθηση - Κατηγοριοποίηση')
    if df is not None:  # Έλεγχος αν έχει φορτωθεί dataframe
        # Διαχωρισμός δεδομένων σε train και test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # KNN
        k = st.slider('Επίλεξε τιμή για k (KNN)', 1, 20, 5)  # παράμετρος k
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)

        # Αποτελέσματα ΚΝΝ
        st.write("**Αποτελέσματα KNN:**")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.2f}")
        st.write(f"Precision: {precision_score(y_test, y_pred_knn, average='weighted'):.2f}")
        st.write(f"Recall: {recall_score(y_test, y_pred_knn, average='weighted'):.2f}")

        # Decision Tree
        max_depth = st.slider('Επίλεξε μέγιστο βάθος για το Decision Tree', 1, 10, 5)  # παράμετρος max_depth Ελέγχει το μέγιστο βάθος του Decision Tree
        dt = DecisionTreeClassifier(random_state=42, max_depth=max_depth)  # Προσθήκη max_depth
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)
        st.write("**Decision Tree**")

        # Έλεγχος για NaN τιμές στις προβλέψεις
        if np.isnan(y_pred_dt).any():
            st.error("Decision Tree predictions contain NaN values. Please check your data.")
        else:
            # Αποτελέσματα Decision Tree
            st.write("**Αποτελέσματα Decision Tree:**")
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.2f}")
            st.write(f"Precision: {precision_score(y_test, y_pred_dt, average='weighted'):.2f}")
            st.write(f"Recall: {recall_score(y_test, y_pred_dt, average='weighted'):.2f}")

        # Σύγκριση αλγορίθμων κατηγοριοποίησης
        knn_metrics = (accuracy_score(y_test, y_pred_knn) +
                       precision_score(y_test, y_pred_knn, average='weighted') +
                       recall_score(y_test, y_pred_knn, average='weighted')) / 3

        dt_metrics = (accuracy_score(y_test, y_pred_dt) +
                      precision_score(y_test, y_pred_dt, average='weighted') +
                      recall_score(y_test, y_pred_dt, average='weighted')) / 3

        if knn_metrics > dt_metrics:
            st.success("Ο KNN έχει την καλύτερη απόδοση στην κατηγοριοποίηση.")
        elif dt_metrics > knn_metrics:
            st.success("Ο Decision Tree έχει την καλύτερη απόδοση στην κατηγοριοποίηση.")
        else:
            st.warning("Ο KNN και ο Decision Tree έχουν παρόμοια απόδοση στην κατηγοριοποίηση.")

# Tab 3: Ομαδοποίηση
with tab3:
    st.header('Μηχανική Μάθηση - Ομαδοποίηση')
    if df is not None:
        # K-means
        k_means = st.slider('Επίλεξε τιμή για k (K-means)', 2, 10, 3)  # παράμετρος k
        kmeans = KMeans(n_clusters=k_means, random_state=42)
        kmeans_labels = kmeans.fit_predict(X)

        # Αποτελέσματα K-means
        st.write("**Αποτελέσματα K-means:**")
        st.write(f"Silhouette Score: {silhouette_score(X, kmeans_labels):.2f}")

        # Scatter plot για K-means
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels)
        st.pyplot(fig)

        # DBSCAN
        # Επιλογή παραμέτρων για DBSCAN
        eps = st.slider('Επίλεξε τιμή για eps (DBSCAN)', 0.1, 2.0, 0.5, step=0.1)  # παράμετρος eps Η μέγιστη απόσταση μεταξύ δύο δειγμάτων για να θεωρηθούν γείτονες στο DBSCAN.
        min_samples = st.slider('Επίλεξε τιμή για min_samples (DBSCAN)', 1, 7, 5)  # παράμετρος min_samples Ο ελάχιστος αριθμός δειγμάτων σε μια γειτονιά για να θεωρηθεί πυρήνας στο DBSCAN.
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(X)
        st.write("**DBSCAN**")
        st.write(f"Silhouette Score: {silhouette_score(X, dbscan_labels):.2f}")

        # Scatter plot για DBSCAN
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels)
        st.pyplot(fig)

        # Σύγκριση αλγορίθμων ομαδοποίησης
        if silhouette_score(X, kmeans_labels) > silhouette_score(X, dbscan_labels):
            st.success("Ο K-means έχει την καλύτερη απόδοση στην ομαδοποίηση.")
        elif silhouette_score(X, dbscan_labels) > silhouette_score(X, kmeans_labels):
            st.success("Ο DBSCAN έχει την καλύτερη απόδοση στην ομαδοποίηση.")
        else:
            st.warning("Ο K-means και ο DBSCAN έχουν παρόμοια απόδοση στην ομαδοποίηση.")

        # Έλεγχος για NaN τιμές στις ετικέτες
        if np.isnan(dbscan_labels).any():
            st.error("DBSCAN labels contain NaN values. Please check your data and parameters.")

# Tab 4: Πληροφορίες
with tab4:
    st.subheader("Σχετικά με την Εφαρμογή")
    st.write(
    """
    Η εφαρμογή ενσωματώνει αριθμητικά δεδομένα με τη μορφή πίνακα και βοηθάει στην ανάλυση και οπτικοποίηση αυτών, 
    χρησιμοποιώντας διάφορες τεχνικές εξόρυξης δεδομένων και μηχανικής μάθησης.
    """
    )
    st.write(
        """
        **Λειτουργίες:**

        * Φόρτωση δεδομένων CSV ή Excel.
        * Οπτικοποίηση δεδομένων με PCA και t-SNE.
        * Exploratory data analysis (EDA) με pairplot και histogram.
        * Κατηγοριοποίηση με KNN και Decision Tree.
        * Ομαδοποίηση με K-means και DBSCAN.
        * Αξιολόγηση απόδοσης με μετρικές όπως accuracy, precision, recall και silhouette score.
        """
    )

    st.subheader("Τρόπος Λειτουργίας")
    st.write(
        """
        1. Ανεβάστε ένα αρχείο CSV ή Excel.
        2. Επιλέξτε τις στήλες που θέλετε να χρησιμοποιήσετε για την οπτικοποίηση με PCA(τουλάχιστον 2) ή t-SNE.
        3. Επιλέξτε τα αντίστοιχα κουτάκια για την εμφάνιση των αντίστοιχων EDA διαγραμμάτων (pairplot και histogram).
        4. Επιλέξτε τιμή για k (KNN) και βάθος για το Decision Tree για τους αλγόριθμους κατηγοριοποίησης.
        5. Επιλέξτε τιμή για k (K-means) και για eps και min_samples για DBSCAN.
        6. Σύγκριση αλγορίθμων κατηγοριοποίησης και ομαδοποίησης και εξερεύνηση των αποτελεσμάτων.
        """
    )
    st.subheader("Περιγραφή Αλγορίθμων")
    st.write(
        """
        **PCA (Principal Component Analysis):** Μια τεχνική μείωσης διάστασης που χρησιμοποιείται για την 
        οπτικοποίηση δεδομένων σε χαμηλότερες διαστάσεις.

        **t-SNE (t-distributed Stochastic Neighbor Embedding):** Μια τεχνική μείωσης διάστασης που 
        χρησιμοποιείται για την οπτικοποίηση δεδομένων σε 2 ή 3 διαστάσεις.

        **KNN (K-Nearest Neighbors):** Ένας αλγόριθμος κατηγοριοποίησης που βασίζεται στην εύρεση των 
        k πλησιέστερων γειτόνων ενός σημείου.

        **Decision Tree:** Ένας αλγόριθμος κατηγοριοποίησης που δημιουργεί ένα δέντρο αποφάσεων για την 
        ταξινόμηση των δεδομένων.

        **K-means:** Ένας αλγόριθμος ομαδοποίησης που χωρίζει τα δεδομένα σε k clusters.

        **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** Ένας αλγόριθμος 
        ομαδοποίησης που βασίζεται στην πυκνότητα των δεδομένων.
        """
    )
    st.subheader("Μετρικές Απόδοσης")
    st.write(
        """
        **Κατηγοριοποίηση:**

        1. Accuracy: Το ποσοστό των σωστών προβλέψεων.

        2. Precision: Η αναλογία των σωστών θετικών προβλέψεων προς το συνολικό αριθμό των θετικών προβλέψεων.

        3. Recall: Η αναλογία των σωστών θετικών προβλέψεων προς το συνολικό αριθμό των πραγματικά θετικών δειγμάτων.

        **Ομαδοποίηση:**

        1. Silhouette Score: Μια μετρική που αξιολογεί την ποιότητα της ομαδοποίησης.
        """
    )
    st.subheader("Μέλη Ομάδας και Συμβολή τους στην εργασία")
    st.write(
        """ 
        **Μέλη:**

        1. Χρήστος Λίβας Π2019086
        2. Βασίλης Γκουλιώνης Π2019024
        3. Γκεντιάν Βόκα Π2019111

        **Συμβολή:** Όλα τα μέλη εργάστηκαν ισάξια για τη διεκπεραίωση των ερωτημάτων της εργασίας και υπήρξε καλή συνεργασία και συννενόηση.
        Πιο συγκεκριμένα ο Λίβας και ο Γκουλιώνης εργάστηκαν στο κομμάτι του κώδικα, του UML διαγράμματος και της Έκθεσης με Latex και ο Βόκα στην ανάπτυξη μέσω Docker 
        και τη διαχείριση του κώδικα μέσω Github.
        """)
