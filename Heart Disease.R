# Firstly I must read the data that I am going to deal with.
# "/Users/mertaltun/Desktop/Project/heart.csv" is the specific location of the heart.csv file in my PC.
# You must be using the specific location of heart.csv file in your PC inside the read.csv() in order to read the data.
heart<-read.csv("/Users/mertaltun/Desktop/Project/heart.csv")
# Analysis of the data could be done by summary() function.
summary(heart)

# The box and whisker plot of each variable could be visualized by boxplot() function.
boxplot(heart)


# PRE-PROCESSING STAGE
# First job to be done in pre-processing stage is checking for the existence of missing data.
sum(is.na(heart))

# I can deduce that there are no missing data since the returned value is 0 (zero) after running "sum(is.na(heart))".
# Hence, I will not fill the missing data with the average value of the column.


# NORMALIZATION
# The formula for normalization of the dataset x is: (x-min(x))/(max(x)-min(x))
# I will create normalization function by using above formula.
normalization<-function(x){return((x-min(x))/(max(x)-min(x)))}
# Normalization of the data will be done after applying normalization function to the data.
normalization(heart)


# OUTLIERS
# In order for a number to be outlier in a dataset, it should not belong in the range Q1-1.5*IQR and Q3+1.5*IQR.
# Maximum non-outlier value in a column is the maximum value which is smaller than Q3+1.5*IQR in the column.

# Function which replaces any outlier with maximum non outlier of the corresponding column
replace_outlier<-function(data_set,outlier,column){
  quartile<-quantile(heart[[column]],probs=c(.75))
  iqr<-IQR(heart[[column]])
  
    # code for finding maximum non outlier of the column
  max_non_outlier=0
  for(x in heart[[column]]){
    if(x<=quartile+1.5*iqr && x>max_non_outlier){
      max_non_outlier=x
    }
  }
  
  data_set[[column]][outlier]=max_non_outlier
  return(data_set)
}

# It is obvious from the output of the command "boxplot(heart)" that there are 7 columns which have outliers.
# The columns which have outliers are trestbps, chol, fbs, thalach, oldpeak, ca and thal.
# I am not going to apply replace_outlier function on the columns named "fbs" and "ca". 
# The value of 1 in column named "fbs" stands for true. Hence, it should not be considered as an outlier.
# The values of 3 and 4 in column named "ca "are stand for number of major vessels. Hence they should not be considered as outlier. 

for(column_name in c("trestbps","chol","thalach","oldpeak","thal")){
  outlier<-boxplot.stats(heart[[column_name]])$out
  outlier_indexes<-which(heart[[column_name]] %in% c(outlier))
  heart<-replace_outlier(heart,outlier_indexes,column_name)
}


# FEATURE EXTRACTION
# Factor analysis method should be used in order for the feature extraction to be done. 
# Data set should be standardized before making factor analysis.
# I am going to store the standardized data in the new data set named "hearth_s".
# I am going to use scale() function in order to standardize the data stored in data set named "hearth_s".
hearth_s<-heart
hearth_s<-scale(hearth_s)
# I am going to check for mean values of each features by summary() function.
# All mean values should be zero if the standardization was done.
summary(hearth_s)

# I am going to install the library named "psych" to use corr.test function in order for the correlation analysis.
install.packages('psych')
library(psych)

# Correlation matrix of the data set named "hearth_s" will be returned after running the code below.
corr.test(hearth_s)

# I am going to install the library named "nFactors" whose installation requires the installation of the package named "lattice".
# The reason why I installed the above libraries is that I am going to use the functions named nScree() and eigen().
install.packages('lattice')
install.packages('nFactors')
library(lattice)
library(nFactors)

# It is not possible to put a data set into the function named nScree() if the data set is not a vector.
# Hence, I am going to store the vector form of the data set named "hearth_s" inside the variable named "hearth_s_v".
# nScree() is a function returning different outputs from four different techniques where each output stands for the number of factors to be used
# as result of factor analysis.
# The number of factors should be 0, 239, 0 and 1942 according to noc, naf, nparallel and nkaiser methodologies respectively. The mean of those
# values is 545,25. 
hearth_s_v<-as.vector(hearth_s)
nScree(hearth_s_v)

# I am going the put the correlation matrix of the data set named "hearth_s" inside the eigen() function by using cor() function.
# eigen() function is used to make eigenvalue test in factor analysis.
# The first value returned is 3.3133869 after running below code.
eigen(cor(hearth_s))

# The results of scree test were not as I expected probably due to the outliers that I did not remove since they were exceptions. Hence, the 
# value of 545,25 found in scree test could be neglected. The value of 3.3133689 found in eigenvalue test will be rounded to 3 and the number
# of factors will be determined as 3.

# The final process of factor analysis is factor rotation. To do this, I am going to install the library named "GPArotation".
install.packages('GPArotation')
library(GPArotation)

# factanal() function will be used to perform factor analysis after determining the number of factors.
# The number of factors was determined as 3. Hence the value of the parameter named "factors" will be 3.
# The type parameter inside the factanal() function stands for the type of factor rotation and it will be "oblimin" below.
factanal(hearth_s,factors = 3,type="oblimin")


# CLASSIFICATION
# I will need "caret" library in order to split the data and show the confusion matrices. "ggplot2" library has to be installed in order
# for the installation of "caret" library.
# The libraries "rpart", "e1071" and "randomForest" are needed in order to build different classification models.
install.packages('ggplot2')
install.packages('caret')
install.packages('rpart')
install.packages('e1071')
install.packages('randomForest')
library(ggplot2)
library(caret)
library(rpart)
library(e1071)
library(randomForest)

# I am going to convert sex column into a factor since I will need categorized data in classification.
heart$sex<-factor(heart$sex)

# I want the seed number to be constant in order to get same results after running the model many times.
# Hence, I set the seed number as 10.
set.seed(10)

# I am going to make the 30 percent of the data to stand in Testing Set. The remaining data will be belong to Training Set.
testing_index<-createDataPartition(y=heart$sex,p=0.3,list = FALSE)
testing_set<-heart[testing_index,]
training_set<-heart[-testing_index,]

# First classifier will use logistic regression algorithm.
FirstModel<-glm(sex ~ .,data = training_set,family = "binomial")
exp(cbind(FirstModel$coefficients,confint(FirstModel)))
# Below command will show the confusion matrix and accuracy, sensitivity, specificity, F1-Score values of the results of the classifier.
# I can dedeuce from the output of below command that the accuracy of first classifier is 0.663
confusionMatrix(table(predict(FirstModel,testing_set,type = "response")>=0.5,testing_set$sex==1))

# Second classifier will use cross-validation.
SecondModel<-train(
  sex ~ ., 
  data=training_set,
  trControl=trainControl(method ="cv",number=10,savePredictions = TRUE),
  tuneLength=10,
  method='rpart'
)
# Below command will show the confusion matrix and accuracy, sensitivity, specificity, F1-Score values of the results of the classifier.
# I can dedeuce from the output of below command that the accuracy of second classifier is 0.6848
confusionMatrix(predict(SecondModel,testing_set),testing_set$sex)

# Third classifier will use random forest.
ThirdModel<-randomForest(sex ~ ., training_set)
# Below command will show the confusion matrix and accuracy, sensitivity, specificity, F1-Score values of the results of the classifier.
# I can dedeuce from the output of below command that the accuracy of third classifier is 0.7065
confusionMatrix(predict(ThirdModel,testing_set),testing_set$sex)

# Fourth calssifier will use Support Vector Machines algorithm.
# sex column in the Training Set and Testing Set should be in the form of categorized data.
training_set$sex<-factor(training_set$sex)
testing_set$sex<-factor(testing_set$sex)
FourthModel<-svm(sex ~ .,data=training_set)
# Below command will show the confusion matrix and accuracy, sensitivity, specificity, F1-Score values of the results of the classifier.
# I can dedeuce from the output of below command that the accuracy of fourth classifier is 0.7174
confusionMatrix(predict(FourthModel,testing_set),testing_set$sex)

# Fifth classifier will use Support Machines algorithm with only Training Set.
FifthModel<-train(
  sex ~ .,
  data=training_set,
  trControl= trainControl(method="none"),
  method = "svmPoly",
)
# Below command will show the confusion matrix and accuracy, sensitivity, specificity, F1-Score values of the results of the classifier.
# I can dedeuce from the output of below command that the accuracy of fifth classifier is 0.6825
confusionMatrix(predict(FifthModel,training_set),training_set$sex)

# The variable named "BestModel" will store the fourth classifier since fourth classifier is the classifier which has the best accuracy with a 
# value of 0.7174
# When new data comes in, prediction will be done with the model named "BestModel".
BestModel<-FourthModel
