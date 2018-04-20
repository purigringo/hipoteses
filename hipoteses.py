from sklearn import tree

features = [[499, 1], [130, 0], [250, 0], [570, 1]]
labels = [0, 0, 1, 1] # 0 é fesi e 1 é fesimn

# treinamento
clf = tree.DecisionTreeClassifier() # classificando
clf = clf.fit(features, labels) # treino

# nova classificação com base no treino
print(clf.predict([[150, 0]]))
