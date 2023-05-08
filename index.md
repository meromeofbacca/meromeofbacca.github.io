## Are There Fecal Bacteria In Your Water?
Using drinking water quality data, I hope to predict when there are coliform bacteria in your water based on water quality indicators. Coliform bacteria levels are an indicator of harmful bacteria in your water, and predictions can inform you when your water may be unsafe to drink.

The Objective:
Using drinking water quality data, I hope to predict when there are coliform bacteria in your water based on water quality indicators. Coliform bacteria levels are an indicator of harmful bacteria in your water, and predictions can inform you when your water may be unsafe to drink.

The Importance:
Outside of personal curiosity, my mom always tells me that drinking tap water is dangerous and that I’d die the moment I drink it. I want to refute that claim so I can go back to drinking tap water again and not have to wait to boil water, although my mom would still yell at me.

Terms:
Coliforms: are bacteria that live in the intestines of mammals and are excreted in feces. Their presence in water isn’t particularly harmful, however if they are present then coliforms are a good indicator that disease causing pathogens can be present in drinking water.

Turbidity: A measure of how clear water is. Measured by how much light is scattered by the liquid.

Residual Free Chlorine: It is the amount of chlorine leftover after a certain period of time after being applied to water. An initial amount is added and some chlorine will be used to oxidize inorganic and organic materials. The residual chlorine indicates that there was enough chlorine to inactivate bacteria, and still be left over to inactivate more bacteria in the event of recontamination.

MPN/100mL: A statistical approach to measuring the amount of coliform bacteria organisms are in 100mL of water.

Data Source:
https://data.cityofnewyork.us/Environment/Drinking-Water-Quality-Distribution-Monitoring-Dat/bkwf-xfky

Data Dictionary:
My data consists of information on the samples, like sample number, date, time, site, and class. The sample qualities are measured numerically. Chlorine and fluorine is measured in mg/L, and turbidity in NTU (turbidity units). Coliforms and E.Coli are measured in MPN/100mL and are measured numerically, except when it is less than 1, where the data is listed as <1.

Predicting Column:
I’m interested in predicting when the coliform levels are over 1 MPN/100mL, which shows that there are coliform colonies in the water level. An acceptable level of coliform bacteria in water established by the EPA (Environmental Protective Agency) is 0 MPN/100mL, so water measured at or above 1 MPN/100mL is considered at risk for pathogens.

Summary Statistic Plots:
![probablity density histogram](/plots/histogram.png)
![probablity density histogram](/plots/scatterplot.png)


Map Graphs:
<iframe src="nyc_water_site_map.html" height="500" width="500"></iframe>

You can explore this map [as its own web page here](nyc_water_site_map.html)

Model Performance Plots:
![probablity density histogram](/plots/tpr.png)
![probablity density histogram](/plots/precision_vs_recall.png)
![probablity density histogram](/plots/confusion_matrix.png)



