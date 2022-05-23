The program contains a perceptron training method, a perceptron testing method, a method for plotting the error rate per iteration in the training that is called by the trainer, and a main method.

The program first asks the user to enter the number of one the operations he wants, which are:
	1-Call the perceptron trainer and then plot the error ration.
	2-Call the perceptron tester to test it on unseen data.
	3-Exit the program.

The perceptron trainer reads the data from the train.data file, distributes the data based on the classifier to 3 sets(meaning a classifier for classes 1 and 2, and another one for classes 2 and 3 and so on) then it shuffles the data and then updates the weights untill it reaches an optimum then it stores them a file so that the tester can access them, after that it displays the output for the classifier and the plot of error ratio.  

The perceptron tester asks which classifier the user would like to test, then reads the data related to the chosen classifier from test.data, then, using the already calculated weights by the trainer, it predicts the data classification and displays the result.
