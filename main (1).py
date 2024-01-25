import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
from keras import regularizers
from keras.optimizers import RMSprop

# Load the dataset
data = pd.read_csv('ADPD (1).csv')

# Convert the "Disease" column to numeric before dropping it
data.loc[data["Disease"] == "AD", "Disease"] = 0
data.loc[data["Disease"] == "PD", "Disease"] = 1
data.loc[data["Disease"] == "Common", "Disease"] = 2

# Drop non-numeric columns
data = data.drop(["Genes"], axis=1)

# Display the correlation matrix
print(data.corr())

# Standardize the data
sc = StandardScaler()
x = data
x = pd.DataFrame(sc.fit_transform(x))

# Convert "y" to a NumPy array before indexing
y = data["Disease"].values
enc = OneHotEncoder()
Y = enc.fit_transform(y[:, np.newaxis]).toarray()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.2)

# Create a neural network model
model = Sequential()
model.add(Dense(64, input_dim=1438, activation='relu'))
model.add(LeakyReLU(alpha=0.05))
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.09))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.09)))
model.add(Dropout(0.3))
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

# Display a summary of the model
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=1000, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
