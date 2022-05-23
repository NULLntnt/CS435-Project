import re
import numpy as np
import matplotlib.pyplot as plt
import random

def PerceptronTrain(dataArr, Instances, Features):
    learningRate = 0.05
    Scenario = np.array([[1,2],[1,3],[3,2]]) #This is for training which set? (it's either class1 and 2 or class1 and 3 or class 2 and 3)
    WeightFile = open('Weights.txt', 'w')#This is to save the optimum weights for a classifier.
    WeightFile.truncate(0)
    for i in range(3):
        print("Training Classifier{}...".format(i+1))
        A = Scenario[i][0] - 1#Index.
        B = Scenario[i][1] - 1#Index.
        Epochs = 0#Number of iterations.
        missesAllowed = 0#Every 200 iterations this increases by 1 to prevent permanent updating.
        Weights = np.array([0, 0, 0, 0])
        optWeights = []
        instancesArray = np.append(dataArr[A][:][:], dataArr[B][:][:])#Get the data only from the two classes of interset.
        instancesArray = instancesArray.reshape(Instances,Features)
        Missclassifications = np.array([])#Holds the amounts of missclassifications as an elements, while the index of that element is the iteration number.
        while(True):
            #Record missclassified instances!
            numOfMisses = 0
            minNumOfMisses = 999
            for j in random.sample(range(Instances),Instances):
                Activiation = np.dot(instancesArray[j, 0:4], Weights)
                if instancesArray[j][4] == A + 1:#If class equal 1 or 2.
                    if Activiation >= 0:
                        Weights = Weights - instancesArray[j,0:4] * learningRate
                        numOfMisses+=1
                elif instancesArray[j][4] == B + 1: #Elif equal class 2 or 3.
                    if Activiation < 0:
                        Weights = Weights + instancesArray[j,0:4] * learningRate
                        numOfMisses+=1
            # print(numOfMisses)
            Missclassifications = np.insert(Missclassifications,len(Missclassifications), numOfMisses)
            if numOfMisses < minNumOfMisses:
                minNumOfMisses = numOfMisses
                optWeights = Weights
            Epochs+=1
            if Epochs % 200 == 0:
                missesAllowed+=1#Sacrifice accuracy.
            if(numOfMisses <= missesAllowed):#Stopping criteria
                WeightFile.write('{0:.2f},{1:.2f},{2:.2f},{3:.2f}\n'.format(optWeights[0],optWeights[1],optWeights[2],optWeights[3]))
                print()
                print('Optimum Weights: ', optWeights)
                break
            else:
                print('Misses Allowed','                  Weights')
                print('      {0}'.format(missesAllowed),'             ',Weights)
        print()
        for j in range(80):
            print('instance{:3d}:'.format(j+1),instancesArray[j,0:4],'{0:.2f}'.format(np.dot(instancesArray[j, 0:4], optWeights)), ' Class{}'.format(int(instancesArray[j][4])))
        plotAccuracy(Missclassifications, i)
        print()

def PerceptronTest():

    try:#Check if weight are calculated.
        file = open('Weights.txt','r')
    except IOError:
        print("Train first!")
        return

    Weights = np.genfromtxt('Weights.txt', delimiter=',')
    Data = np.genfromtxt('test.data', delimiter=',')
    numOfInstances, numOfFeatures = Data.shape
    numOfChosenInstances = numOfInstances - int(numOfInstances/3)
    i = 0
    file = open('test.data', 'r')
    temp = file.readline()
    while (i < numOfInstances):
        Data[i][4] = temp[-2] #Class number.
        i += 1
        temp = file.readline()
    Data = Data.reshape(3,10,5)
    Scenario = np.array([[1, 2], [1, 3], [3, 2]])  # This is for training which set? (it's either class1 and 2 or class1 and 3 or class 2 and 3)

    print('       *****Perceptron Tester*****')
    choice = 0
    while(choice not in [1,2,3]):
        print('Enter the operation number that you want to perform.\n'
              '\t1-Test classifier1(Class1 and Class2).\n'
              '\t2-Test classifier2(Class1 and Class3).\n'
              '\t3-Test classifier3(Class2 and Class3).\n'
              'Your choice:')
        choice = int(input())
    i = choice - 1
    A = Scenario[i][0] - 1  # Index.
    B = Scenario[i][1] - 1  # Index.
    instancesArray = np.append(Data[A][:][:], Data[B][:][:])
    instancesArray = instancesArray.reshape(numOfChosenInstances, numOfFeatures)
    numOfMisses = 0
    Predictions = []
    print()
    for j in range(numOfChosenInstances):
        Activiation = np.dot(instancesArray[j, 0:4], Weights[i][:])

        if Activiation >= 0:
            Predictions.insert(j, B+1)
            if instancesArray[j][4] == A + 1:  # If class equal 1 or 2.
                numOfMisses+=1

        elif Activiation < 0:
            Predictions.insert(j, A+1)
            if instancesArray[j][4] == B + 1:  # Elif equal class 2 or 3.
                numOfMisses+=1

    outputArray = []
    print("Amount of Missclassifications: {}".format(numOfMisses))
    print("{}       {}              {}  {}  {}".format('Instance Number','Input', "Output", "Predicted Class", "Actual Class"))
    for j in range(20):
        Output = np.dot(instancesArray[j, 0:4], Weights[i][:])
        outputArray.insert(j,Output)
        print('      {:3d}'.format(j + 1),'      {}'.format(instancesArray[j,:4]), '        {:.2f}'.format(Output),'     Class{}'.format(Predictions[j]),'         Class{}'.format(int(instancesArray[j][4])))

def plotAccuracy(numOfMisses, num):
    numOfMisses = numOfMisses/80 * 100
    numOfMisses = numOfMisses.astype(int)
    a = list(range(1,len(numOfMisses) + 1))
    plt.bar(a,numOfMisses)
    plt.ylabel('Ratio')
    plt.xlabel('Iteration')
    plt.xlim()
    plt.title("Calssifier{} Error Ratio per Iteration".format(str(num+1)))
    plt.show()

def main():
    #Get data from train.data.
    Data = np.genfromtxt('train.data', delimiter=',')
    numOfInstances, numOfFeatures = Data.shape
    chosenInstances = numOfInstances - int(numOfInstances/3)
    i = 0
    file = open('train.data', 'r')
    temp = file.readline()

    #genfromtxt() gets only numbers, so I have to insert the class numbers at column4.
    while (i < numOfInstances):
        Data[i][4] = temp[-2]#Class number
        i+=1
        temp = file.readline()

    #Each class data is in a sperate diemention.
    Data = Data.reshape(3,int(numOfInstances/3),numOfFeatures)


    while(True):
        print('         *****Main Menu*****')
        choice = ''
        while(choice not in [1,2,3]):
            print('Enter the operation number that you want to perform.\n'
                  '\t1-Call the Perceptron Trainer.\n'
                  '\t2-Call the Percptron Tester.\n'
                  '\t3-Exit program.\n'
                  'Your choice:')
            choice = int(input())

        if choice == 1:
            PerceptronTrain(Data, chosenInstances, numOfFeatures)
        elif choice == 2:
            PerceptronTest()
        else:
            exit(0)

main()