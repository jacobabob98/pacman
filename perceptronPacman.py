# perceptron_pacman.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Perceptron implementation for apprenticeship learning
import util
from pacman import GameState
import random
import numpy as np
from pacman import Directions
import matplotlib.pyplot as plt
# from data_classifier_new import convertToArray
import math
import numpy as np

PRINT = True


class SingleLayerPerceptronPacman():

    def __init__(self, num_weights=5, num_iterations=20, learning_rate=1):

        # weight initialization
        # model parameters initialization

        # Xavier weight initialisation
        self.weights = np.random.randn(num_weights) * np.sqrt(2 / (num_weights))

        self.max_iterations = num_iterations
        self.learning_rate = learning_rate


    def predict(self, feature_vector):
        """
        This function should take a feature vector as a numpy array and compute
        the dot product of the weights of your perceptron with the values of features.

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.

        Then the result of this computation should be passed through your activation function

        For example if x=feature_vector, and ReLU is the activation function
        this function should compute ReLU(x dot weights)
        """
        return self.activation(np.dot(feature_vector, self.weights))

    def activation(self, x):
        """
        Implement your chosen activation function here.
        """
        #return np.maximum(0, x) # ReLU 
        return 1 / (1 + np.exp(-x)) # Sigmoid Activation Function
    

    def evaluate(self, data, labels):
        """
        This function should take a data set and corresponding labels and compute the performance of the perceptron.
        You might for example use accuracy for classification, but you can implement whatever performance measure
        you think is suitable.

        The data should be a 2D numpy array where the each row is a feature vector

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.

        The labels should be a list of 1s and 0s, where the value at index i is the
        corresponding label for the feature vector at index i in the appropriate data set. For example, labels[1]
        is the label for the feature at data[1]
        """
        true_positive = 0
        false_positive = 0
        false_negative = 0
        for feature_vector, truth_label in zip(data, labels):
            prediction = self.predict(feature_vector)
            if prediction >= 0.5 and truth_label == 1:
                true_positive += 1
            elif prediction >= 0.5 and truth_label == 0:
                false_positive += 1
            elif prediction < 0.5 and truth_label == 1:
                false_negative += 1
        
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
        return f1_score


    def train(self, trainingData, trainingLabels, validationData, validationLabels, batch_size=1000, patience = 10):
        """
        This function should take training and validation data sets and train the perceptron

        The training and validation data sets should be 2D numpy arrays where each row is a different feature vector

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.

        The training and validation labels should be a list of 1s and 0s, where the value at index i is the
        corresponding label for the feature vector at index i in the appropriate data set. For example, trainingLabels[1]
        is the label for the feature at trainingData[1]
        """
        
        initial_learning_rate = self.learning_rate
        num_samples = len(trainingData)
        
        best_validation_loss = float('inf')
        patience_counter = 0
        
        print(f'Starting weights:\n{self.weights}')

        for epoch in range(self.max_iterations):
            
            #shuffle data for batch training
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            trainingData = trainingData[indices]
            trainingLabels = np.array(trainingLabels)
            trainingLabels = trainingLabels[indices]
            
            total_loss = 0
            for i in range(0, num_samples, batch_size):
                batch_data = trainingData[i : i + batch_size]
                batch_labels = trainingLabels[i : i + batch_size]
                
                batch_loss = 0
                weight_update = np.zeros_like(self.weights)
                
                for feature_vector, truth_label in zip(batch_data, batch_labels):
                    
                    prediction = self.predict(feature_vector)
                    
                    epsilon = 1e-15
                    loss = - (truth_label * np.log(prediction + epsilon) + (1 - truth_label) * np.log(1 - prediction + epsilon))
                    
                    batch_loss += loss
                    
                    gradient = (prediction - truth_label) * feature_vector
                    
                    weight_update += gradient
                    
                avg_weight_update = weight_update / len(batch_data)
                self.weights -= self.learning_rate * avg_weight_update
                
                total_loss += batch_loss
            
            avg_loss = total_loss / num_samples
            # learning rate alogarithmic nnealing
            self.learning_rate = initial_learning_rate / np.log2(epoch + 2)
            
            f1_score = self.evaluate(validationData, validationLabels)
            # Calculate validation loss
            validation_loss = 0
            for feature_vector, truth_label in zip(validationData, validationLabels):
                prediction = self.predict(feature_vector)
                epsilon = 1e-15
                loss = - (truth_label * np.log(prediction + epsilon) + (1 - truth_label) * np.log(1 - prediction + epsilon))
                validation_loss += loss
            
            avg_validation_loss = validation_loss / len(validationData)
            print(f"Epoch {epoch+1}/{self.max_iterations}\tLoss: {avg_loss}\tValidation Loss: {avg_validation_loss}\tF1 Score: {f1_score}\tLearning Rate: {self.learning_rate}")
            
            if avg_validation_loss < best_validation_loss:
                best_validation_loss = avg_validation_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early Stopping")
                    break # early stop training process
        
        print(f'Ending weights:\n{self.weights}')
        return self.weights
                
                


