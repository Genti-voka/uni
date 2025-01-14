#Εργασία τεχνολογία λογισμικού 

Σχετικά με την Εφαρμογή
Η εφαρμογή ενσωματώνει αριθμητικά δεδομένα με τη μορφή πίνακα και βοηθάει στην ανάλυση και οπτικοποίηση αυτών, χρησιμοποιώντας διάφορες τεχνικές εξόρυξης δεδομένων και μηχανικής μάθησης.

Λειτουργίες:

Φόρτωση δεδομένων CSV ή Excel.
Οπτικοποίηση δεδομένων με PCA και t-SNE.
Exploratory data analysis (EDA) με pairplot και histogram.
Κατηγοριοποίηση με KNN και Decision Tree.
Ομαδοποίηση με K-means και DBSCAN.
Αξιολόγηση απόδοσης με μετρικές όπως accuracy, precision, recall και silhouette score.
Τρόπος Λειτουργίας
Ανεβάστε ένα αρχείο CSV ή Excel.
Επιλέξτε τις στήλες που θέλετε να χρησιμοποιήσετε για την οπτικοποίηση με PCA(τουλάχιστον 2) ή t-SNE.
Επιλέξτε τα αντίστοιχα κουτάκια για την εμφάνιση των αντίστοιχων EDA διαγραμμάτων (pairplot και histogram).
Επιλέξτε τιμή για k (KNN) και βάθος για το Decision Tree για τους αλγόριθμους κατηγοριοποίησης.
Επιλέξτε τιμή για k (K-means) και για eps και min_samples για DBSCAN.
Σύγκριση αλγορίθμων κατηγοριοποίησης και ομαδοποίησης και εξερεύνηση των αποτελεσμάτων.
Περιγραφή Αλγορίθμων
PCA (Principal Component Analysis): Μια τεχνική μείωσης διάστασης που χρησιμοποιείται για την οπτικοποίηση δεδομένων σε χαμηλότερες διαστάσεις.

t-SNE (t-distributed Stochastic Neighbor Embedding): Μια τεχνική μείωσης διάστασης που χρησιμοποιείται για την οπτικοποίηση δεδομένων σε 2 ή 3 διαστάσεις.

KNN (K-Nearest Neighbors): Ένας αλγόριθμος κατηγοριοποίησης που βασίζεται στην εύρεση των k πλησιέστερων γειτόνων ενός σημείου.

Decision Tree: Ένας αλγόριθμος κατηγοριοποίησης που δημιουργεί ένα δέντρο αποφάσεων για την ταξινόμηση των δεδομένων.

K-means: Ένας αλγόριθμος ομαδοποίησης που χωρίζει τα δεδομένα σε k clusters.

DBSCAN (Density-Based Spatial Clustering of Applications with Noise): Ένας αλγόριθμος ομαδοποίησης που βασίζεται στην πυκνότητα των δεδομένων.

Μετρικές Απόδοσης
Κατηγοριοποίηση:

Accuracy: Το ποσοστό των σωστών προβλέψεων.

Precision: Η αναλογία των σωστών θετικών προβλέψεων προς το συνολικό αριθμό των θετικών προβλέψεων.

Recall: Η αναλογία των σωστών θετικών προβλέψεων προς το συνολικό αριθμό των πραγματικά θετικών δειγμάτων.

Ομαδοποίηση:

Silhouette Score: Μια μετρική που αξιολογεί την ποιότητα της ομαδοποίησης.
Μέλη Ομάδας και Συμβολή τους στην εργασία
Μέλη:

Χρήστος Λίβας Π2019086
Βασίλης Γκουλιώνης Π2019024
Γκεντιάν Βόκα Π2019111
Συμβολή: Όλα τα μέλη εργάστηκαν ισάξια για τη διεκπεραίωση των ερωτημάτων της εργασίας και υπήρξε καλή συνεργασία και συννενόηση.
Πιο συγκεκριμένα ο Λίβας(Π2019086) και ο Γκουλιώνης(Π2019024) εργάστηκαν στο κομμάτι του κώδικα, του UML διαγράμματος και της Έκθεσης με  και ο Βόκα(Π2019111) στην ανάπτυξη μέσω Docker και τη διαχείριση του κώδικα μέσω Github.


#Ακολουθούν απλά βήματα για να τρέξεις έναν έτοιμο κώδικα από ένα GitHub repository τοπικά στον υπολογιστή σου:

#Εγκατάσταση των Εργαλείων:
    -Git: Κατέβασε και εγκατέστησε το Git.
    -Docker Desktop: Κατέβασε και εγκατέστησε το Docker Desktop.
#Κλωνοποίηση του Repository:
    -Άνοιξε το τερματικό ή το Git Bash.
    -git clone <URL-του-Repository>
    -(π.χ. git clone https://github.com/username/repository.git).
#Μετάβαση στο Project Folder:
    -Πήγαινε στον φάκελο του project:
    -cd repository
#Έλεγχος για Docker Configuration:
    -Βεβαιώσου ότι υπάρχει ένα αρχείο Dockerfile ή docker-compose.yml στο project.
#Εκτέλεση με Docker:
    -Αν υπάρχει docker-compose.yml: Εκτέλεσε την εντολή:
    -docker-compose up
    -Αυτό θα δημιουργήσει και θα τρέξει τα containers της εφαρμογής.
#Αν υπάρχει μόνο Dockerfile: Εκτέλεσε τις εντολές:
    -docker build -t my-app .
    -docker run -p 8000:8000 my-app
    -(Αντικατέστησε το 8000:8000 με τις θύρες που χρησιμοποιεί η εφαρμογή σου).
#Πρόσβαση στην Εφαρμογή:

#Άνοιξε τον browser και πήγαινε στη διεύθυνση:
    -http://localhost:8000 (ή την αντίστοιχη θύρα που χρησιμοποιεί η εφαρμογή).
#Εκτέλεση χωρίς Docker (προαιρετικά):
  Αν το project δεν χρησιμοποιεί Docker, δες το αρχείο README.md για οδηγίες.
  Συνήθως οι εντολές είναι:
  pip install -r requirements.txt  # Για Python projects
  python app.py

#Μετά την κλωνοποίηση υπάρχει και η επιλογή μέσω vscode terminal , 
αφου βρισκόμαστε στον φακελο όπου έγινε η κλωνοποίηση 
τότε τρεχουμε την εντολή -streamlit run app.py 
οπου app.py είναι το αρχείο όπου έχει τον κώδικα
