from flask import Flask, render_template, request
from werkzeug import secure_filename
import csv
import os.path
from os.path import join
import io
import pandas as pd
import numpy as np
global r
global df
import ProjectC as temp1
app = Flask(__name__)

@app.route('/')
def upload_file():
	return render_template('upload.html')
	
@app.route('/', methods = ['GET', 'POST'])
def upload_file1():
	if request.method == 'POST':
		f = request.files['file']
		f.save(secure_filename(f.filename))
		
		opencsv(f.filename)
		print (f.filename)
		
		

		return render_template('upload.html',output='FILE UPLOADED SUCCESSFULLY')
		

		


def opencsv(name):
	global df
	df=temp1.vCf(name)
	q=df.to_html('abcde.html')
	

@app.route('/result')
def result():
	return render_template('abcde.html')
	
@app.route('/route')
def graph():
			import matplotlib.pyplot as plt
			from pylab import title
			labels = 'Tenure', 'Contract', 'Monthly Charges', 'Total Charges'
			fracs = [70, 20, 55, 84]
			plt.pie(fracs, labels=labels,radius = 1)
			title('Driving Features', bbox={'facecolor':'0.8', 'pad':4})
			plt.axis('equal')
			plt.show()
			return render_template('upload.html')
@app.route('/route1')
def graph1():
			import pandas as pd
			import numpy as np
			from sklearn import preprocessing
			import matplotlib.pyplot as plt
			from sklearn import cluster, decomposition
			sample=pd.read_csv('td_clean')
			num_list=['InternetService','Dependents','MonthlyCharges','TotalCharges','Churn']
			city_data = sample.dropna(axis=0)[num_list + ['tenure']]
			city_data.head()
			city_groups = city_data.groupby('tenure').mean().reset_index().dropna(axis=0)
			city_groups.head()
			city_groups_std = city_groups.copy()
			for i in num_list:
				city_groups_std[i] = preprocessing.scale(city_groups_std[i])
			city_groups_std.head() 
			km = cluster.KMeans(n_clusters=4, max_iter=300, random_state=None)
			city_groups_std['cluster'] = km.fit_predict(city_groups_std[num_list])
			count=0
			for i in city_groups_std['cluster']:
				if i==0:
					count=count+1;
			print(count)       
			pca = decomposition.PCA(n_components=2, whiten=True)
			pca.fit(city_groups[num_list])
			city_groups_std['x'] =pca.fit_transform(city_groups_std[num_list])[:, 0]
			city_groups_std['y'] = pca.fit_transform(city_groups_std[num_list])[:, 1]
			plt.scatter(city_groups_std['x'], city_groups_std['y'], c=city_groups_std['cluster'])
			plt.show()
			return render_template('upload.html')
			
@app.route('/info')
def info():
	return render_template('info.html')


		
if __name__ == '__main__':
	app.run(debug = True)