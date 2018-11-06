from flask import Flask, render_template, request, redirect
from bokeh.plotting import figure
from bokeh.embed import components
import pandas as pd
from io import StringIO
from datetime import date
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn import neighbors
from bokeh.io import show, output_file
from bokeh.models import LogColorMapper
from bokeh.models import LinearColorMapper
from bokeh.palettes import Spectral6 as palette
#from bokeh.palettes import Viridis6 as palette
from bokeh.plotting import figure
import math
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression
import heapq
from bokeh.models import LogColorMapper, LogTicker, ColorBar
palette=['#ff0000','#ff1a1a','#ff3333','#ff4d4d','#ff6666','#ff8080','#ff9999','#ffb3b3','#ffcccc','#ffe6e6','#bfbfbf','#e6e6ff','#ccccff','#b3b3ff','#9999ff','#8080ff','#6666ff','#4d4dff','#3333ff','#1a1aff','#0000ff']
from bokeh.sampledata.unemployment import data as unemployment

app = Flask(__name__)

def plot(location, predictions,state_name):
	from bokeh.sampledata.us_counties import data as counties
	#print(counties.keys())
	counties = {code: county for code, county in counties.items() if county["state"] in [state_name]}
	print(len(counties))
	#print(type(counties))
	county_xs = [county["lons"] for county in counties.values()]
	county_ys = [county["lats"] for county in counties.values()]
	county_names = [county['name'] for county in counties.values()]#Make sure names match with data
	print(county_names)
	color_mapper = LinearColorMapper(palette=palette,low=25, high=75)
	data=dict(x=county_xs,y=county_ys,name=county_names,rate=predictions,)
	TOOLS = "pan,wheel_zoom,reset,hover,save"

	p = figure(title="Vote Preference Predictions", tools=TOOLS, plot_width=900,
	x_axis_location=None, y_axis_location=None,tooltips=[("Name", "@name"), ("Yes vote (percent)", "@rate%")])

	color_bar = ColorBar(color_mapper=color_mapper, ticker=LogTicker(),label_standoff=12, border_line_color=None, location=(0,0))
	p.add_layout(color_bar, 'right')

	p.grid.grid_line_color = None
	p.hover.point_policy = "follow_mouse"
	
	p.patches('x', 'y', source=data,fill_color={'field': 'rate', 'transform': color_mapper},fill_alpha=1.0, line_color="white", line_width=0.5)
	
	script, div = components(p)
	return script, div


def plot_corr(pcc):
	factors=['Clinton Margin Over Trump','Per Capita Income','Percent Uninsured','Percent With Some College','Percent African-American','Percent White','Percent Asian','Percent Hispanic','Percent Seniors','Poverty Rate','Income Inequality','Unemployment Rate','Rural Population','Percent Citizens']
	from bokeh.models import ColumnDataSource
	from bokeh.transform import factor_cmap
	x = [ [pcc[i], factors[i]] for i in range(len(factors)) ]
	#print(x)
	x=sorted(x, key=lambda x: -abs(x[0]))
	factors = [ i[1] for i in x ]
	pcc = [ i[0] for i in x ]
	source = ColumnDataSource(data=dict(factors=factors, pcc=pcc))
	p = figure(x_range=factors, plot_height=700, title="",toolbar_location=None, tools="")

	p.vbar(x=factors, top=pcc, width=0.9)
	p.xaxis.major_label_orientation = math.pi/2
	p.xgrid.grid_line_color = None

	p.yaxis.axis_label="PCC"
	
	p.yaxis.axis_label_text_font_size = "20pt"
	p.yaxis.major_label_text_font_size = "15pt"
	p.xaxis.major_label_text_font_size = "15pt"
	p.y_range.start = -1
	p.y_range.end = 1

	script, div = components(p)
	return script, div

@app.route('/')
def index():
  return render_template('options_menu.html')

@app.route('/graph', methods=['POST','GET'])
def about():
  result = request.form
  issue=result['dropdown']
  state=result['state']
  county='Harford'
  pcc, scc, pcc_p, scc_p, GLR_R2, GLR_predict, vict_marg, per_capita_income, uninsured_rate, some_college, N, script, div, script_cor, div_cor=analysis(issue,state,county)

  return render_template('graph.html',
	GLM_predict=str(float(GLR_predict))[0:5],
	GLM_R2=str(GLR_R2)[:4],
	vict_marg=str('%.1f'%vict_marg), per_capita_income=int(per_capita_income),
	uninsured_rate=str(float(uninsured_rate)*100.0)[:5],N=N,
	some_college=str(float(some_college)*100.0)[:5],script=script,div=div,script_cor=script_cor,div_cor=div_cor)

def analysis(issue, state, county):

	data=pd.read_csv('data_train.csv')
	data=shuffle(data)
	data['clinton_margin']=data['CLINTON_%']-data['TRUMP_%']
	max_income=data['PER_CAPITA_INCOME'].max()
	data['PER_CAPITA_INCOME']=data['PER_CAPITA_INCOME']/max_income
	data=data[data['Issue']==issue]
	data_train=data.ix[:, ['clinton_margin','PER_CAPITA_INCOME','UNINSURED_RATE','SOME_COLLEGE','AFAMERPER','WHITESPER','ASIANPER','HISPANICPER','PERSENIORS','POVERTY_RATE','INCINEQUALITY','UNEMP_RATE','RURAL_POP','CITIZENS']]
	data['PERCENT']=data['Yes'].astype(int)/(data['Yes'].astype(int)+data['No'].astype(int)).astype(float)
	y_train=data['PERCENT'].astype(float)

	pcc=[];scc=[];pcc_p=[];scc_p=[]
	#Are the features correlated with the margin of victory?
	for i in range(len(data_train.columns)):
		corr_p=pearsonr(data_train.ix[:,i],data['PERCENT'])
		corr_s=spearmanr(data_train.ix[:,i],data['PERCENT'])
		pcc.append(corr_p[0]);scc.append(corr_s[0])
		pcc_p.append(corr_p[1]);scc_p.append(corr_s[1])

	GLR = LinearRegression()
	GLR.fit(data_train, y_train)
	N=len(data_train)
	GLR_R2=GLR.score(data_train, y_train)
	CV_GLR = cross_val_score(GLR, data_train, y_train, cv=2)

	mod1=GLR.predict(data_train)

	LRR = Ridge(alpha=0.1)
	LRR.fit(data_train, y_train)
	LRR_R2=LRR.score(data_train, y_train)
	CV_LRR = cross_val_score(LRR, data_train, y_train, cv=2)
	mod2=LRR.predict(data_train)

	LLR = Lasso(alpha=0.0005)
	LLR.fit(data_train, y_train)
	LLR_R2=LLR.score(data_train, y_train)
	CV_LLR = cross_val_score(LLR, data_train, y_train, cv=2)

	ELN = ElasticNet(alpha=0.0001)
	ELN.fit(data_train, y_train)
	ELN_R2=ELN.score(data_train, y_train)
	CV_ELN = cross_val_score(ELN, data_train, y_train, cv=2)

	BRR = BayesianRidge(alpha_1=0.1,alpha_2=0.1)
	BRR.fit(data_train, y_train)
	BRR_R2=BRR.score(data_train, y_train)
	CV_BRR = cross_val_score(BRR, data_train, y_train, cv=2)

	KNN = neighbors.KNeighborsRegressor(n_neighbors=10)
	KNN.fit(data_train, y_train)
	KNN_R2=KNN.score(data_train, y_train)
	scoresKNN = cross_val_score(KNN, data_train, y_train, cv=2)

	RFR = RandomForestRegressor(n_estimators=500, max_depth=12, n_jobs=1)
	RFR.fit(data_train, y_train)
	scoresRFR = cross_val_score(RFR, data_train, y_train, cv=2)
	mod3=RFR.predict(data_train)


	#MLPR=MLPRegressor(hidden_layer_sizes=(5,5), activation='relu', solver='adam', alpha=0.0001)
	#MLPR.fit(data_train, y_train)
	#scoresMLPR = cross_val_score(MLPR, data_train, y_train, cv=2)
	#mod4=MLPR.predict(data_train)

	#plt.scatter((mod3)/3.0,y_train,s=1)
	#plt.scatter(mod3,y_train,s=1)
	#plt.xlim(0,1)
	#plt.ylim(0,1)
	#plt.plot(np.arange(0,1,0.1),np.arange(0,1,0.1))
	#plt.show()

	h=[]
	#heapq.heappush(h, (np.mean(CV_GLR),'GLR'))
	#heapq.heappush(h, (np.mean(CV_LRR),'LRR'))
	heapq.heappush(h, (np.mean(CV_LLR),'LLR'))
	#heapq.heappush(h, (np.mean(CV_ELN),'ELN'))
	#heapq.heappush(h, (np.mean(CV_BRR),'BRR'))
	#heapq.heappush(h, (np.mean(scoresKNN),'KNN'))
	heapq.heappush(h, (np.mean(scoresRFR),'RFR'))

	#print('GLR:',CV_GLR, np.mean(CV_GLR), GLR_R2, GLR.coef_)
	#print('Ridge: ',CV_LRR, np.mean(CV_LRR), LRR_R2, LRR.coef_)
	#print('Lasso: ',CV_LLR, np.mean(CV_LLR), LLR_R2, LLR.coef_)
	#print('ELN: ',CV_ELN, np.mean(CV_ELN), ELN_R2, ELN.coef_)
	#print('BRR: ',CV_BRR, np.mean(CV_BRR), BRR_R2, BRR.coef_)
	#print('KNN: ', scoresKNN, np.mean(scoresKNN), KNN_R2)
	#print('RFR: ', scoresRFR, np.mean(scoresRFR), RFR.feature_importances_)

	best_model=heapq.nlargest(1,h)
	model_code=best_model[0][1]
	#print(model_code)

	from bokeh.sampledata.us_counties import data as counties
	statename_to_abbr = {'District of Columbia': 'DC','Alabama': 'AL','Montana': 'MT','Alaska': 'AK','Nebraska': 'NE','Arizona': 'AZ','Nevada': 'NV','Arkansas': 'AR','NewHampshire': 'NH','California': 'CA','NewJersey': 'NJ','Colorado': 'CO','NewMexico': 'NM','Connecticut': 'CT','NewYork': 'NY','Delaware': 'DE','NorthCarolina': 'NC','Florida': 'FL','NorthDakota': 'ND','Georgia': 'GA','Ohio': 'OH','Hawaii': 'HI','Oklahoma': 'OK','Idaho': 'ID','Oregon': 'OR','Illinois': 'IL','Pennsylvania': 'PA','Indiana': 'IN','RhodeIsland': 'RI','Iowa': 'IA','SouthCarolina': 'SC','Kansas': 'KS','SouthDakota': 'SD','Kentucky': 'KY','Tennessee': 'TN','Louisiana': 'LA','Texas': 'TX','Maine': 'ME','Utah': 'UT','Maryland': 'MD','Vermont': 'VT','Massachusetts': 'MA','Virginia': 'VA','Michigan': 'MI','Washington': 'WA','Minnesota': 'MN','WestVirginia': 'WV','Mississippi': 'MS','Wisconsin': 'WI','Missouri': 'MO','Wyoming': 'WY'}
	keys=counties.keys()
	locations=[]
	state_id_abbr_conv={'AL':1,'AZ':4,'AR':5,'CA':6,'CO':8,'CT':9,'DE':10,'FL':12,'GA':13,'ID':16,'IL':17,'IN':18,'IA':19,'KS':20,'KY':21,'LA':22,'ME':23,'MD':24,'MA':25,'MI':26,'MN':27,'MS':28,'MO':29,'MT':30,'NE':31,'NV':32,'NH':33,'NJ':34,'NM':35,'NY':36,'NC':37,'ND':38,'OH':39,'OK':40,'OR':41,'PA':42,'RI':44,'SC':45,'SD':46,'TN':47,'TX':48,'UT':49,'VT':50,'VA':51,'WV':54,'WY':56,'WI':55,'WA':53}
	state_id=state_id_abbr_conv[state]
	for i in keys:
		if i[0]==state_id:#Need to add 2-AK, 11-DC (no data),15 HI - Maui, 51, 53 WA need data
			name=counties[i]['detailed name'].split(',')
			if state_id == 51:
				print(name[0])
				name[0]=name[0].replace(' ','').replace('County','').replace('city','')
				print(name[0])
			else:
				name[0]=name[0].replace(' ','').replace('County','').replace('Parish','')
			name[1]=name[1].replace(' ','')
			location=statename_to_abbr[name[1]]+'_'+name[0].replace('County','').replace('.','').upper()
			locations.append(location)

	#location=str(state).upper()+"_"+str(county).upper()
	county_lookup=pd.read_csv('data_county_lookup.csv')
	county_lookup['clinton_margin']=county_lookup['CLINTON_%']-county_lookup['TRUMP_%']
	county_lookup['PER_CAPITA_INCOME']=county_lookup['PER_CAPITA_INCOME']/max_income

	predictions=[];lrr_predictions=[]
	state_name=locations[0][:2].lower()
	for place in locations:
		print(place)
		PredictX=county_lookup[county_lookup['STATE_COUNTY']==place]
		PredictX=PredictX.ix[:, ['clinton_margin','PER_CAPITA_INCOME','UNINSURED_RATE','SOME_COLLEGE','AFAMERPER','WHITESPER','ASIANPER','HISPANICPER','PERSENIORS','POVERTY_RATE','INCINEQUALITY','UNEMP_RATE','RURAL_POP','CITIZENS']]
		#model_code='LLR'
		if model_code=='GLR': GLR_predict=GLR.predict(PredictX)*100.0
		elif model_code=='LRR': GLR_predict=LRR.predict(PredictX)*100.0
		elif model_code=='LLR': GLR_predict=LLR.predict(PredictX)*100.0
		elif model_code=='ELN': GLR_predict=ELN.predict(PredictX)*100.0
		elif model_code=='BRR': GLR_predict=BRR.predict(PredictX)*100.0
		elif model_code=='KNN': GLR_predict=KNN.predict(PredictX)*100.0
		elif model_code=='RFR': GLR_predict=RFR.predict(PredictX)*100.0
		predictions.append(round(GLR_predict[0]))
	print(locations, predictions)
	script, div=plot(locations, predictions, state_name)
	script_coor, div_corr=plot_corr(pcc)

	return pcc, scc, pcc_p, scc_p, GLR_R2, GLR_predict[0]*100.0, PredictX.iloc[0,0]*100.0, PredictX.iloc[0,1]*max_income, PredictX.iloc[0,2], PredictX.iloc[0,3], N, script, div, script_coor, div_corr



if __name__ == '__main__':
  app.run(port=33507)
