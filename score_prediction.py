# --------Linear Regression Problem----------------
#Author : Kartik Manguli , Intern Machine Learning Trainee, SachiSoft Solutions
#NoTe: Student_data.csv file is enclosed with this repository If you wish to run this module then change the path in read_csv()
#---------------------------------------------------
#More Info read readme.md file

#Import Libraries
import matplotlib.pyplot as plot
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
 
#Read .CSV file (Comma Separated Values)
data_set=pd.read_csv('C:\\Users\\karthik\\Desktop\\Student_data.csv')
X=data_set.iloc[:,:-1].values # X denotes Student's Aptitude Scores
Y=data_set.iloc[:,1].values #Y denotes Statistical Grades based on research

#Training and Testing the data model
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

#Linear Regression 
regression = LinearRegression()
regression.fit(x_train,y_train)

y_prediction=regression.predict(x_test)

#Plotting the Graph for the Training Model
plot.scatter(x_train,y_train,color='red')
plot.plot(x_train,regression.predict(x_train),color='black')
plot.title('Aptitute Vs Statistics Grade(Training)')
plot.xlabel('Aptitude Score')
plot.ylabel('Statistics')
plot.show()
#Ploting the Graph for the Testing Model
plot.scatter(x_test,y_test,color='red')
plot.plot(x_train,regression.predict(x_train),color='black')
plot.title('Aptitute Vs Statistics Grade(Testing)')
plot.xlabel('Aptitude Score')
plot.ylabel('Statistics')
plot.show()
accuracy=sum(y_test)/sum(y_prediction)
print("Accuracy of this model is",accuracy*100,'%') #Calculates Accuracy For this Model Accuracy is around 90%