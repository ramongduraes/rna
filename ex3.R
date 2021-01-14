########### Exercícios 03 e 04 Ramon DUrães


########## Exercicio 3-a) ELM para XOR
rm(list=ls())
library("rgl"); 
library(e1071); 
library(ggplot2);
library(caret);
library(corpcor);

# Loading and Visualizing
setwd("C:/Users/ramon/git/rna")
load("data2classXOR.txt")
plot(X[,1], X[,2], col=Y+2)

# ELM Training function
train_ELM <- function(X, Y, n_hidden, augment=TRUE){
  
  if(augment){
    X <- cbind(replicate(dim(X)[1],1),X)
  }
  
  # Calculating H and Hinv
  Z <- replicate(n_hidden, runif(dim(X)[2],-0.5, 0.5))
  H <- tanh(X %*% Z)
  Hinv <- solve(t(H) %*% H) %*% t(H)
  #Hinv <- pseudoinverse(H)
  # Calculating W
  W <- Hinv %*% Y
  return(list("Z"=Z,"W"=W))
}

n_hidden <- 6
ELM <- train_ELM(X, Y, n_hidden = 6)

test_ELM <- function(ELM, Xnew, Ytrue, augment=TRUE){
  if(augment){
    Xnew <- cbind(replicate(dim(Xnew)[1],1),Xnew)
  }
  Yhat <- sign(tanh(Xnew %*% ELM$Z) %*% ELM$W)
  confMat <- confusionMatrix(as.factor(Yhat), as.factor(Ytrue))
  return(list("Yhat" = Yhat, "confMat" = confMat))
}

results <- test_ELM(ELM, X_t, Y_t)

# Analyzing classification results
print(results$confMat)

# Plotting results and decision function (contour)
xseq <- seq(-2, 10, 0.1)
lseq <- length(xseq)
MZ <- matrix(nrow=lseq, ncol=lseq)
for (i in 1:lseq){
  for(j in 1:lseq){
    x1x2 <- as.matrix((cbind(1,xseq[i], xseq[j])))
    MZ[i, j] <- sign(tanh(x1x2 %*% ELM$Z)%*%ELM$W)
  }
}

plot(X_t[,1], X_t[,2], col=results$Yhat+2, xlim=c(0,6), ylim=c(0,6))
par(new=T)
contour(x= xseq, y=xseq, z=MZ, xlim=c(0,6), ylim=c(0,6), lwd=0.5, drawlabels = F)

########## Exercicio 3-b) ELM para Breast Cancer

# Loading and preprocessing data
library(mlbench)
data(BreastCancer)
bc <- BreastCancer
# Removing NA
bc <- bc[complete.cases(bc), ]
# Removing Id and Class
bc <- bc[,!names(bc) %in% c("Id", "Class")]
# Defining benign as 0 as malign as 1
class <- 2 * (BreastCancer$Class == "benign") -1
# Converting to numeric matrix
X <- data.matrix(bc)
Y <- data.matrix(class)

# Train test splits with random sort
seq <- sample(1:dim(X)[1])
train_perc <- 0.7
ntrain <- round(train_perc * dim(X)[1])
Xtrain <- X[seq[1:ntrain],]
Ytrain <- Y[seq[1:ntrain],]
Xtest <- X[tail(seq, -ntrain),]
Ytest <- Y[tail(seq, -ntrain),]

# Training and testing ELM
n_hidden = 50
ELMbc <- train_ELM(Xtrain, Ytrain, n_hidden)
resultsbc <- test_ELM(ELMbc, Xtest, Ytest)
resultsbc$confMat


########## Exercício 4) RBFs 

########## Exercício 4-a) RBFs para XOR 

rm(list=ls())
library("stats")
library("rgl"); 
library(e1071); 
library(ggplot2);
library(caret)

# Loading and Visualizing
setwd("C:/Users/ramon/git/rna")
load("data2classXOR.txt")
plot(X[,1], X[,2], col=Y+2)

# RBF Training function
train_RBF <- function(X, Y, n_hidden, augment=FALSE){
  
  n <- dim(X)[1]
  if(augment){
    X <- cbind(replicate(dim(X)[1],1),X)
  }
  
  # Finding clusters using Kmeans
  clusters <- kmeans(X, centers = n_hidden)
  mu <- clusters$centers
  s <- matrix(1, nrow=n_hidden, ncol=1) # Using unit radius for now to test results
  
  # Calculating H
  H <- matrix(nrow=n ,ncol=n_hidden)
  for(i in 1:n){
    xi <- X[i,]
    for(j in 1:n_hidden){
      H[i,j] <- exp(- (1/(s[j]^2)) * dist(rbind(xi,mu[j,])))
    }
  }
  
  # Calculating Hinv
  Hinv <- solve(t(H) %*% H) %*% t(H)
  
  # Calculating W
  W <- Hinv %*% Y
  return(list("mu"= mu, "s" = s, "W" = W))
}

n_hidden <- 4
RBF <- train_RBF(X, Y, n_hidden)

# Plotting clusters
plot(X[,1], X[,2], col=Y+2, xlim=c(0,6), ylim=c(0,6))
par(new=T)
plot(RBF$mu[,1], RBF$mu[,2], xlim=c(0,6), ylim=c(0,6), pch=4, col="red", lwd=2)

# Evaluating RBF
test_RBF <- function(RBF, Xnew, Ytrue, augment=FALSE){
  if(augment){
    Xnew <- cbind(replicate(dim(Xnew)[1],1),Xnew)
  }
  
  
  # Calculating H
  n <- dim(Xnew)[1]
  n_hidden <- dim(RBF$mu)[1]
  H <- matrix(nrow=n ,ncol=dim(RBF$mu)[1])
  for(i in 1:n){
    xi <- Xnew[i,]
    for(j in 1:n_hidden){
      H[i,j] <- exp(- (1/(RBF$s[j]^2)) * dist(rbind(xi,RBF$mu[j,])))
    }
  }
  Yhat <- sign(H %*% RBF$W)
  confMat <- confusionMatrix(as.factor(Yhat), as.factor(Ytrue))
  return(list("Yhat" = Yhat, "confMat" = confMat))
}

results <- test_RBF(RBF, X_t, Y_t)

# Analyzing classification results
print(results$confMat)

# Plotting results and decision function (contour)
xseq <- seq(-2, 10, 0.1)
lseq <- length(xseq)
MZ <- matrix(nrow=lseq, ncol=lseq)
for (i in 1:lseq){
  for(j in 1:lseq){
    x1x2 <- as.matrix((cbind(xseq[i], xseq[j])))
    hi <- matrix(nrow=1, ncol=n_hidden)
    for(c in 1:n_hidden){
      hi[1,c] <-  exp(- (1/(RBF$s[c]^2)) * dist(rbind(x1x2,RBF$mu[c,])))
    }
    MZ[i, j] <- sign(hi%*%RBF$W)
  }
}

plot(X_t[,1], X_t[,2], col=results$Yhat+2, xlim=c(0,6), ylim=c(0,6))
par(new=T)
contour(x= xseq, y=xseq, z=MZ, xlim=c(0,6), ylim=c(0,6), lwd=0.5, drawlabels = F)


########## Exercicio 4-b) RBF para Breast Cancer

# Loading and preprocessing data
library(mlbench)
data(BreastCancer)
bc <- BreastCancer
# Removing NA
bc <- bc[complete.cases(bc), ]
# Removing Id and Class
bc <- bc[,!names(bc) %in% c("Id", "Class")]
# Defining benign as 0 as malign as 1
class <- 2 * (BreastCancer$Class == "benign") -1
# Converting to numeric matrix
X <- data.matrix(bc)
Y <- data.matrix(class)

# Train test splits with random sort
seq <- sample(1:dim(X)[1])
train_perc <- 0.7
ntrain <- round(train_perc * dim(X)[1])
Xtrain <- X[seq[1:ntrain],]
Ytrain <- Y[seq[1:ntrain],]
Xtest <- X[tail(seq, -ntrain),]
Ytest <- Y[tail(seq, -ntrain),]

# Training and testing ELM
n_hidden =30
RBFbc <- train_RBF(Xtrain, Ytrain, n_hidden)
resultsbc <- test_RBF(RBFbc, Xtest, Ytest)
resultsbc$confMat$overall
resultsbc$confMat$byClass















# Finding clusters using Kmeans
n_hidden <- 4
clusters <- kmeans(X, centers = n_hidden)
centers <- clusters$centers
sigma <- matrix(1, nrow=n_hidden, ncol=1) # Using unit radius for now to test results



gauss <- function (x, mu, sigma){
  return( -(1/sigma) * t(x-mu)%*%(x-mu) )
}



yhat <- b + sum(wj * gauss(x, mu, sigma))








Xaug <- cbind(replicate(dim(X)[1],1),X)


# clusters$centers



Z <- replicate(n_hidden, runif(dim(Xaug)[2],-0.5, 0.5))
H <- tanh(Xaug %*% Z)

Hinv <- solve(t(H) %*% H) %*% t(H)

# Calculating W
W <- Hinv %*% Y

# Classifying new data
Xnew <- cbind(replicate(nc*4,1),X_t)
Yhat <- tanh(Xnew %*% Z) %*% W
plot(X_t[,1], X_t[,2], col=sign(Yhat)+2)

# Generating surface
xseq <- seq(-2, 10, 0.1)
lseq <- length(xseq)
MZ <- matrix(nrow=lseq, ncol=lseq)
for (i in 1:lseq){
  for(j in 1:lseq){
    x1x2 <- as.matrix((cbind(1,xseq[i], xseq[j])))
    MZ[i, j] <- sign(tanh(x1x2 %*% Z)%*%W)
  }
}

plot(X_t[,1], X_t[,2], col=sign(Yhat)+2, xlim=c(0,6), ylim=c(0,6))
par(new=T)
contour(x= xseq, y=xseq, z=MZ, xlim=c(0,6), ylim=c(0,6), lwd=0.5, drawlabels = F)


func = function(x, c, r){
  return(exp(-(x-c)^2/r^2))
}

#pdfnvar <- function(x, m, K, n){
#  ((1/sqrt((2*pi)^n*(det(K)))))*exp(-0.5*(t(x-m) %*% 
#                                            (solve(K)) %*% (x-m))))
#}