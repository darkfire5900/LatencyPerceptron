!wget https://github.com/fferegrino/yu-gi-oh/raw/main/data/cards.csv

import pandas as pd
df = pd.read_csv('cards.csv')
dsc = df[df['type']=='Effect Monster']
df2 = df[df['type']=='Effect Monster']
dsc['desc'].to_csv("desc.csv")
dsc = dsc['desc'].to_list()

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text data
texts = dsc
# Convert texts to numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts).toarray()

class LatencyPerceptron:
    def __init__(self, n_input, n_hidden, latencies, refractory_periods):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.weights = np.random.uniform(-1, 1, (n_input, n_hidden))
        self.latencies = latencies
        self.refractory_periods = refractory_periods
        
    def simulate(self, X, steps):
        n_samples = X.shape[0]
        activations = np.zeros((n_samples, self.n_hidden))
        
        for i in range(n_samples):
            for j in range(self.n_hidden):
                activation_time = np.dot(X[i], self.weights[:, j]) + self.latencies[j]
                if activation_time > self.refractory_periods[j]:
                    activations[i, j] = activation_time
        
        return activations

# Initialize the latency perceptron model
n_input = X.shape[1]
n_hidden = 500
latencies = np.random.uniform(0, 1, n_hidden)
refractory_periods = np.random.randint(0, 10, n_hidden)
model = LatencyPerceptron(n_input, n_hidden, latencies, refractory_periods)

# Generate activation patterns
activations = model.simulate(X, steps=5)


import networkx as nx
from scipy.spatial.distance import correlation

# print out cards that are similar to each other

g2 = nx.Graph()

#vectors = np.array(first_order_rotation).reshape(-1, 50*50)

for n1, a in zip(df2['name'],activations):
  for n2,b in zip(df2['name'],activations):
    #print(b)
    if n1 != n2:
        t = 1-correlation(a,b)
        #print(t)
        if t >= .8:
          print("CARD NAME 1: "+n1,"CARD NAME 2: "+n2,"SIMILARITY: "+str(t))
          g2.add_edge(n1,n2,weight=t)
