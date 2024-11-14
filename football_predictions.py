# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Sample dataset with match outcomes based on past matches outcomes
# outcomes 1x=1, 2x=2,bts=3, >1.5=4, home>0.5=6, away>0.5=7
data = {
                             'Outcome': [1,     2,     3,     4,    6,   3,    4,     1,    3,    2,      1, 2],
                'home_pointswon_ratio': [3.0,  1.0,  1.0,   2.6,  2.5,   1.3,  0.8, 2.6,  1.6,  1.3,    2.5, 0.5],  
                'Away_pointswon_ratio': [1.0,  3.0,  2.2,   3.0,  2.8,   1.9,  2.3, 0.8,  2.0,  2.6,    0.9, 1.8], 
   'home_goalscoredhome_pergame_ratio': [2.8,  0.7,  1.8,   2.6,  2.1,   1.7,  1.4, 2.1,  1.9,  0.8,    2.4, 0.8], 
   'away_goalscoredaway_pergame_ratio': [0.8,  1.8,  2.4,   2.4,  2.0,   1.8,  2.4, 0.9,  1.7,  1.7,    1.1, 1.7],
    'home_goalconceeded_pergame_ratio': [1.0,  1.1,  2.0,   1.0,  1.4,   1.9,  2.2, 0.9,  1.7,  1.4,    0.8, 1.3],
    'away_goalconceeded_pergame_ratio': [1.8,  0.6,  1.5,   2.0,  1.8,   1.7,  0.8, 2.0,  1.9,  0.8,    2.0, 1.6],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Features (X) and Target (y)
X = df[['home_pointswon_ratio', 'Away_pointswon_ratio' , 
         'home_goalscoredhome_pergame_ratio', 
         'away_goalscoredaway_pergame_ratio',
         'home_goalconceeded_pergame_ratio',  'away_goalconceeded_pergame_ratio']]

y = df['Outcome']

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the  model
model =  DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict the results on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# New match with the following attributes

#juventus 2-0 torino prediction was 2x with gaussian nahh but home>0.5 with decision tree
juventus_match_features = [[2.0,1.1,1.1,1.6,0.5,1.8]]

#inter 1-1 napoli prediction is inter one goal with decision tree
inter_match_features = [[2.1,2.0,2.5,1.4,1.4,0.6]]

#brentford 3-2 bournemouth prediction is
brentford_match_features = [[2.1,2.0,2.5,1.4,1.4,0.6]]

#totham 1-2 ipswich prediction is 1x totham  bu they lost 
totham_match_features = [[2.4,0.4,3.0,1.2,0.8,2.6]]

#valence 2-3 laspalmas prediction is 2x and laspalmas actually won wow
valence_match_features = [[1.6,0,1.0,0.8,0.8,2.3]]

#zalgiris 2-3 panev prediction is 1x and bts and actually both teams scored but 1x lost
zalgiris_match_features = [[2.3,1.0,2.3,1.1,0.8,1.4]]

#denmark - spain prediction is 1x or bts lets see
#new_match_features = [[3,2,2,2,0,0.5]]

#romania - kosovo prediction is
#new_match_features = [[3,1.5,3,3,1,0.5]]

#slovenien - norvegen prediction is
new_match_features = [[3,1.5,2.5,0.5,0.5,1.5]]

# Predict whether this person will pay on time (1) or not (0)
prediction_juv = model.predict(juventus_match_features)
prediction_inter = model.predict(inter_match_features)
prediction_totham = model.predict(totham_match_features)
prediction_valence = model.predict(valence_match_features)
prediction_zalgiris = model.predict(zalgiris_match_features)
prediction_spain = model.predict(new_match_features)

print(f"Predicted outcome: {prediction_spain}")


# Create and train the Decision Tree model
#model = DecisionTreeClassifier()
#model.fit(X_train, y_train)

# Predict the results on the test set
#y_pred = model.predict(X_test)

# Evaluate the model
#print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
#print("\nClassification Report:")
#print(classification_report(y_test, y_pred))
#print("\nConfusion Matrix:")
#print(confusion_matrix(y_test, y_pred))