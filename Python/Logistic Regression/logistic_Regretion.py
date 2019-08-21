import csv
import matplotlib.pyplot as plt
import math
import numpy as np

features = []
result = []
costs=[]

# Reading Training set

with open('Dataset_2.csv','rt')as f:
	data = csv.reader(f)
	for x in data:
		features.append([float(x[0]),float(x[1])])
		result.append(float(x[2]))


# z = x0 + x1theta1 + x2theta2 etc

def z(X,theta):
	return (X.dot(theta))

def sigmoid(X):
	X= 1/(1+np.exp(-1*X))
	return X

def cost(X,theta,y):
	posIndex = np.where(y[:,0]==1)
	pos =np.log( sigmoid(z(X[posIndex],theta)))
	negIndex = np.where(y[:,0]==0)
	neg = np.log(1-sigmoid(z(X[negIndex],theta)))
	reg = sum(np.square(theta[1:]))*(lamda/(X.shape[0]*2))
	predict = -1*((sum(pos)+sum(neg))/X.shape[0])+reg
	return (predict)


def plot(X,y,u,v,Z):
	fig, grph = plt.subplots(nrows=1, ncols=2) # dividing plot into two sub graphs
	posIndex = np.where(y[:,0]==1)
	negIndex = np.where(y[:,0]==0)
	black_dots = grph[0].scatter(X[posIndex,1:][0][:,0],X[posIndex,1:][0][:,1],s=10,c="black")
	red_dots = grph[0].scatter(X[negIndex,1:][0][:,0],X[negIndex,1:][0][:,1],s=10,c="red")
	CS = grph[0].contour(u, v, Z,[0],colors='blue')
	grph[1].plot(costs)

# Setting up all Labels 

	grph[0].set_xlabel('Microchip Test 1')
	grph[0].set_ylabel('Microchip Test 2')
	grph[1].set_ylabel('Cost')
	grph[1].set_xlabel('itterations')
	grph[0].legend([red_dots, (red_dots, black_dots)], ["0", "1"])

	plt.show()



def gradientDecent(theta,y):
	while True:
		temp = z(X,theta)
		temp = sigmoid(temp)
		temp = temp - y
		temp = (np.transpose(temp).dot(X))/X.shape[0]
		xxx = theta
		xxx = ((xxx)*lamda)/X.shape[0]
		predict = temp+np.transpose(xxx)
		theta -= np.transpose(learn*predict)
		costs.append(cost(X,theta,y))
		if(len(costs)>4 and round(costs[-1][0],6)==round(costs[-4][0],6)):
			break
	return(theta)


def normalize(X):
	mean = np.mean(X,axis=0).reshape(1,X.shape[1])
	X = X/mean[:,None]
	return X[0]


def map(X):
	degree = 6
	for a in range(1,degree+1):
		for b in range(0,a+1):
			col = (np.power(X[:,1],a-b))*(np.power(X[:,2],b))
			col = col.reshape(col.shape[0],1)
			X = np.append(X,col,axis=1)
	return X



learn = 0.001
lamda =1

X = np.array(features)
y = np.asarray(result)
y = y.reshape((X.shape[0],1))
one = np.ones([X.shape[0],1])
X=np.append(one,X,axis=1)
X = map(X)


# X=normalize(X) // Dataset_2 is already normalize so this is commented please uncomment for dataset_1

theta = np.zeros([X.shape[1],1])


print("Learning ...")

theta=gradientDecent(theta,y)

u = np.linspace(-1, 1, 50)
v = np.linspace(-1, 1, 50)

z = np.zeros([u.shape[0], v.shape[0]])

for  i in range(u.shape[0]):
	for  j in range(v.shape[0]):
		temp = np.array([1,u[i],v[j]])
		temp = temp.reshape(1,temp.shape[0])
		z[i][j] = map(temp).dot(theta)

z = np.transpose(z)

plot(X,y,u,v,z)
exit()
