
def vCf(name):
			
			import pandas as pd
			import numpy as np
			#import os
			import time
			from sklearn.feature_selection import SelectKBest
			from sklearn.feature_selection import chi2
			import numpy
			path='C:\\Users\\kadgi\\Downloads\\'
			wine=pd.read_csv('td_clean')
			wi=pd.read_csv(name)
			x1=wi.iloc[:,0:19]


			# In[2]:


			X=wine.iloc[:,:-1]
			y=wine.iloc[:,-1]


			# In[3]:


			from sklearn.model_selection import train_test_split
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 20)


			# In[4]:


			from sklearn.preprocessing import StandardScaler
			sc=StandardScaler()
			sc.fit(X_train)


			# In[5]:


			X_train = sc.transform(X_train)
			X_test = sc.transform(X_test)
			x1 = sc.transform(x1)


			# In[6]:


			from sklearn.linear_model import LogisticRegression
			l1 = LogisticRegression()
			l1.fit(X_train, y_train)
			pred=l1.predict(X_train)
			predt=l1.predict(X_test)




			# In[7]:


			#print(sum((y_train-pred)==0)/len(y_train))
			#print(sum((y_test-predt)==0)/len(y_test))
			#predt


			# In[8]:


			from sklearn.svm import SVC
			svm = SVC( kernel = 'poly' )
			svm.fit(X_train, y_train)
			pred = svm.predict(X_train)
			predt=svm.predict(X_test)


			# In[9]:


			#print(sum((y_train-pred)==0)/len(y_train))
			#print(sum((y_test-predt)==0)/len(y_test))
			#predt


			# In[10]:


			from sklearn.neural_network import MLPClassifier
			clf = MLPClassifier(learning_rate = 'constant', activation = 'relu', alpha=1, batch_size = 300)
			clf.fit(X_train, y_train)
			predt = clf.predict(X_test)
			pred = clf.predict(X_train)


			# In[11]:


			#print(sum((y_train-pred)==0)/len(y_train))
			#print(sum((y_test-predt)==0)/len(y_test))
			#predt


			# In[12]:


			from sklearn.ensemble import VotingClassifier
			eclf1 = VotingClassifier(estimators=[('lr', l1), ('rf', svm), ('gnb', clf)], voting='hard')
			eclf1.fit(X_train, y_train)
			predt = eclf1.predict(X_test)
			pred = eclf1.predict(X_train)


			# In[13]:


			#print(sum((y_train-pred)==0)/len(y_train))
			#print(sum((y_test-predt)==0)/len(y_test))
			#predt


			# In[14]:


			from sklearn.ensemble import RandomForestClassifier
			from sklearn.model_selection import cross_val_score
			rc=RandomForestClassifier(n_estimators=10, criterion='entropy', max_features=3)
			rc.fit(X_train,y_train)
			pred=rc.predict(X_train)
			predt=rc.predict(X_test)
			scores=cross_val_score(rc, X_train, y_train, cv=5)
			scores


			# In[15]:


			#print(sum((y_train-pred)==0)/len(y_train))
			#print(sum((y_test-predt)==0)/len(y_test))


			# In[16]:


			from sklearn.ensemble import VotingClassifier
			eclf2 = VotingClassifier(estimators=[('lr', eclf1), ('rf', svm), ('gnb', rc)], voting='hard')
			eclf2.fit(X_train, y_train)
			predt = eclf2.predict(x1)
			pred = eclf2.predict(X_train)


			# In[17]:


			print(sum((y_train-pred)==0)/len(y_train))
			#print(sum((y_test-predt)==0)/len(y_test))
			#predt


			# In[19]:


			df=pd.DataFrame(predt)
			df1=pd.DataFrame(x1)
			df1.append(df)
			gui=df1.head(5)
			df.columns=['churn']*len(df.columns)
			df.to_csv(path+'target.csv')
			
			from sklearn.feature_selection import SelectKBest
			from sklearn.feature_selection import chi2
			import numpy
			test = SelectKBest(score_func=chi2, k=19)
			fit = test.fit(X,y)
			#numpy.set_printoptions(precision=3)
			print(fit.scores_)
			
			
				
			return df
		
			
			
			
			


			

# In[20]:



