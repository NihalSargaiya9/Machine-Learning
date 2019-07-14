import csv
import matplotlib.pyplot as plt

areas = []
prices = []

# Reading Training set

with open('training set.csv','rt')as f:
  data = csv.reader(f)
  next(data)            # skip header
  for x in data:
  	areas.append(float(x[0]))
  	prices.append(float(x[1]))

# Mean Normalizing

meanAreas = sum(areas)/len(areas)
rArea = max(areas)-min(areas)
areas = [((area- meanAreas)/rArea) for area in areas]


# Gradient Decent

def gradientDecent(m,c):
	sameIt=0
	for _ in range(itterations):
		temp0=0
		temp1=0
		for x,y in zip(areas,prices):
			temp0+=((x*m)+c-y)
			temp1+=((x*m)+c-y)*x
		mlast = m
		clast = c
		m=m-(learn*(temp1/len(areas)))
		c=c-(learn*(temp0/len(prices)))
		costFunction(m,c)
		
	return m,c

squareMeanErrors=[]


# Linear Regression function to find Sqare Mean Errors

def costFunction(m,c):
	squareMeanErrorsTemp=[]
	for x,y in zip(areas,prices):
		temp = (((x*m)+c)-y)**2
		squareMeanErrorsTemp.append(temp)
	squareMeanErrors.append(sum(squareMeanErrorsTemp)/(len(squareMeanErrorsTemp)*2))


# Prediction of Values

def prdict(val,m,c):
	val = (val- meanAreas)/rArea		# Mean Normalize The value (val = User Input)
	temp = m*val+c 						# Calculate Y cordinates
	if round(temp,0)<0:
		print("Please Enter larger Value")
		return
	print("Price in Rupees(RS) is :",round(temp,0))							# Print Answer




fig, grph = plt.subplots(nrows=1, ncols=2) # dividing plot into two sub graphs
grph[0].scatter(areas,prices,s=10) 		   # Plotting points on graph 0

m =1										# Slope (Can be any value)
c = 2										# Intercept (Can be any value)

learn = 0.007								# Learning Rate
itterations=6000

print("Learning ... (Please wait)")

m,c=gradientDecent(m,c)


pointsX = areas 							# X codinates for potting Line
pointsY = [((m*t)+c) for t in pointsX]		# Y cordiantes for Plotting Line

grph[0].plot(pointsX,pointsY,c="red")		# Plot Line 

grph[1].plot(squareMeanErrors)				#plot Square mean error for Debuging Alogorithm


# Setting up all Labels 

grph[0].set_xlabel('Size in ($ft^2$)')
grph[0].set_ylabel('Price in rupees')
grph[0].set_title("Flats in vijaynagar")
grph[1].set_ylabel('Square mean Error')
grph[1].set_xlabel('itterations')


# Take Inputs For Predictions

noTest = int(input("Please enter Number of Test Cases : ")) # No of test cases

for _ in range(noTest):
	test = float(input("Enter Area in (Sqare ft) : ")) #Test Case
	prdict(test,m,c)	 #Prediction





plt.show()
