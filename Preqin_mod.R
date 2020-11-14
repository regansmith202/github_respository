# Packages
#install.packages("neuralnet")
#install.packages("mlbench")

# Libraries
library(keras)
library(mlbench) 
library(dplyr)
library(magrittr)
library(neuralnet)
library(gapminder)

# Data
data <- read.csv(file.choose(),header=T)
str(data)

data %>% mutate_if(is.integer(), as.numeric)

# Neural Network Visualization
n <- neuralnet(IRR ~ Fundsize+ImpactDUM+Early_StageDUM+GrowthDUM+BuyoutDUM+Fund_of_FundsDUM+Venture_GeneralDUM+Early_Stage_SeedDUM+Co_InvestmentDUM+North_America_DUM+Europe_DUM+Asia_DUM+Diversified_Multi_Regional_DUM+Americas_DUM+Africa_DUM+Middle_East_and_Israel_DUM+Australasia_DUM,
               data = data,
               hidden = c(10,5),
               linear.output = F,
               lifesign = 'full',
               rep=1)

plot(n,
     col.hidden = 'darkgreen',
     col.hidden.synapse = 'darkgreen',
     show.weights = F,
     information = F,
     fill = 'lightblue')

# Matrix
data <- as.matrix(data)
dimnames(data) <- NULL
subset <- c(1:22,25:89)
data <- data[, subset]

# Partition
set.seed(1234)
ind <- sample(2, nrow(data), replace = T, prob = c(.7, .3))
training <- data[ind==1,3:87]
test <- data[ind==2, 3:87]
trainingtarget <- data[ind==1, 2]
testtarget <- data[ind==2, 2]

# Normalize
m <- colMeans(training)
s <- apply(training, 2, sd)
training <- scale(training, center = m, scale = s)
test <- scale(test, center = m, scale = s)

# Create Model
model <- keras_model_sequential()
model %>% 
         layer_dense(units = 5, activation = 'relu', input_shape = c(85)) %>%
         layer_dense(units = 5,activation = 'relu') %>%
         layer_dense(units = 1)

# Compile
model %>% compile(loss = 'mse',
                  optimizer = 'rmsprop',
                  metrics = 'mae')

# Fit Model
mymodel <- model %>%
         fit(training,
             trainingtarget,
             epochs = 100,
             batch_size = 32,
             validation_split = 0.2)

# Evaluate
model %>% evaluate(test, testtarget)
pred <- model %>% predict(test)
mean((testtarget-pred)^2)
plot(testtarget, pred)
cbind(testtarget, pred)

dt <- data.table(testtarget, round(pred,digits=4))
dt <- dt[900:909]
colnames(dt)<-c("Actual","Predicted")
write.csv(dt,'Machine_Learning_NetIRR(%)_Prediction.csv')

# Upload excel file with data you wish to make predictions on
prediction_inputs <- read.csv(file.choose(),header=T)
str(prediction_inputs)
prediction_inputs %>% mutate_if(is.integer, as.numeric)
prediction_inputs <- as.matrix(prediction_inputs)
dimnames(prediction_inputs) <- NULL
inputs <- prediction_inputs[,3:19]
inputs_test <- scale(inputs, center = m, scale = s)


