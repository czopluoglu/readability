# Initial submission experiment

#require(psych)

#train_df <- read.csv('../input/commonlitreadabilityprize/train.csv')
#test_df  <- read.csv('../input/commonlitreadabilityprize/test.csv')

#describe(train_df$target)

#naive <- sqrt(mean((train_df$target - mean(train_df$target))^2))

#naive

#outcome <- data.frame(id = test_df$id,target=naive)

#write.csv(outcome,'submission.csv',row.names=FALSE,quote=FALSE)

###############################################################################

# Install packages

require(here)
require(gtools)
require(psych)
require(caret)
require(glmnet)
require(here)
require(xgboost)

require(udpipe)
require(quanteda)
require(quanteda.textstats)
  
model  <- udpipe_download_model(language = "english")
ud_eng <- udpipe_load_model(model$file_model)

#Import the datasets

#train_df <- read.csv('../input/commonlitreadabilityprize/train.csv')
#test_df  <- read.csv('../input/commonlitreadabilityprize/test.csv')

train_df <- read.csv(here('data','train.csv'))
test_df  <- read.csv(here('data','test.csv'))


# MORE INFORMATION
# Readibility scores 
# https://cran.r-project.org/web/packages/koRpus/vignettes/koRpus_vignette.html)
# https://www.cs.upc.edu/~nlp/SVMTool/PennTreebank.html

# https://universaldependencies.org/format.html

################################################################################
################################################################################
################################################################################
#
#  Create the text features based using udpipe, quanteda, and quanteda.textstats
#
################################################################################
################################################################################
################################################################################

### Enginner the Text Features for Training Data

list1 <- vector('list',nrow(train_df))

for(i in 1:nrow(train_df)){
  
  text <- as.character(train_df[i,]$excerpt)
  
  
  tokenized <- tokens(text)
  dm <- dfm(tokenized)
  
  
  annotated <- udpipe_annotate(ud_eng, x = text)
  annotated <- as.data.frame(annotated)
  
  # cbind_morphological(annotated, term = "feats", which = "lexical")
    
  # Morphological annotation (universal POS tags, https://universaldependencies.org/u/pos/index.html)
  
  temp     <- data.frame(table(annotated$upos))
  temp[,2] <- temp[,2]
  
  words <- annotated[!annotated$upos%in%c('PUNCT','SYS','X'),]$token
  
  temp[,1] <- as.character(temp[,1])
  temp <- rbind(temp,data.frame(table(annotated$xpos)))
  
  temp <- rbind(temp,data.frame(Var1 ='nwords',Freq=length(words)))
  temp <- rbind(temp,data.frame(Var1 ='nchars',Freq=sum(nchar(annotated$token))))
  temp <- rbind(temp,data.frame(Var1 ='nchars',Freq=sum(nchar(annotated$token))/length(words)))
  temp <- rbind(temp,data.frame(Var1 ='wdiv',Freq=length(unique(words))/length(words)))
  temp <- rbind(temp,data.frame(Var1 ='nsent',Freq=length(unique(annotated$sentence_id))))
  

  # Morphologicla features (https://universaldependencies.org/u/feat/index.html)
  
  feats  <- na.omit(annotated$feats)
  feats1 <- unlist(strsplit(feats,split='\\|'))
  feats2 <-  unlist(strsplit(feats1,split='='))[c(TRUE,FALSE)]
  
  feats1        <- table(feats1)
  names(feats1) <- gsub('=','.',names(feats1))
  
  feats2        <- table(feats2)
  names(feats2) <- names(feats2)
  

  temp <- rbind(temp,data.frame(feats1))
  temp <- rbind(temp,data.frame(feats2))
  

  # Syntactic Annotation (https://universaldependencies.org/u/dep/index.html)
  
  temp <- rbind(temp,data.frame(table(annotated$dep_rel)))
  
  # Word Length distribution
  
  wl <- table(nchar(tokens(text,
                           remove_punct = TRUE,
                           remove_numbers = TRUE,
                           remove_symbols = TRUE,
                           remove_separators = TRUE)[[1]])
  )
  
  names(wl) <- paste0('l',names(wl))
  
 
  label_let <- names(wl)
  
  ifelse('l1'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l1',Freq=wl['l1'])),
         temp <- rbind(temp,data.frame(Var1 ='l1',Freq=0)))
  
  ifelse('l2'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l2',Freq=wl['l2'])),
         temp <- rbind(temp,data.frame(Var1 ='l2',Freq=0)))
  
  ifelse('l3'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l3',Freq=wl['l3'])),
         temp <- rbind(temp,data.frame(Var1 ='l3',Freq=0)))
  
  ifelse('l4'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l4',Freq=wl['l4'])),
         temp <- rbind(temp,data.frame(Var1 ='l4',Freq=0)))
  
  ifelse('l5'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l5',Freq=wl['l5'])),
         temp <- rbind(temp,data.frame(Var1 ='l5',Freq=0)))
  
  ifelse('l6'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l6',Freq=wl['l6'])),
         temp <- rbind(temp,data.frame(Var1 ='l6',Freq=0)))
  
  ifelse('l7'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l7',Freq=wl['l7'])),
         temp <- rbind(temp,data.frame(Var1 ='l7',Freq=0)))
  
  ifelse('l8'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l8',Freq=wl['l8'])),
         temp <- rbind(temp,data.frame(Var1 ='l8',Freq=0)))
  
  ifelse('l9'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l9',Freq=wl['l9'])),
         temp <- rbind(temp,data.frame(Var1 ='l9',Freq=0)))
  
  ifelse('l10'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l10',Freq=wl['l10'])),
         temp <- rbind(temp,data.frame(Var1 ='l10',Freq=0)))
  
  ifelse('l11'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l11',Freq=wl['l11'])),
         temp <- rbind(temp,data.frame(Var1 ='l11',Freq=0)))
  
  ifelse('l12'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l12',Freq=wl['l12'])),
         temp <- rbind(temp,data.frame(Var1 ='l12',Freq=0)))
  
  ifelse('l13'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l13',Freq=wl['l13'])),
         temp <- rbind(temp,data.frame(Var1 ='l13',Freq=0)))
  
  ifelse('l14'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l14',Freq=wl['l14'])),
         temp <- rbind(temp,data.frame(Var1 ='l14',Freq=0)))
  
  ifelse('l15'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l15',Freq=wl['l15'])),
         temp <- rbind(temp,data.frame(Var1 ='l15',Freq=0)))
  
  # Measures of Lexical variety)
  
  lexical <- textstat_lexdiv(tokenized,measure = 'all')[1,2:16]
    
  temp <- rbind(temp,data.frame(Var1 = colnames(lexical),Freq = as.numeric(lexical[1,])))
    
  # Measures of Readability
  
  readable <- textstat_readability(text, measure = 'all')[1,2:49]
  temp <- rbind(temp,data.frame(Var1 = colnames(readable),Freq = as.numeric(readable[1,])))

  
  list1[[i]] <- data.frame(t(temp[,2]))
  colnames(list1[[i]]) <- temp[,1]
  
print(i)  
}








x <- dtrain[,105:149]
x <- as.data.frame(scale(x))

x$ari     <- rowMeans(x[,1:3])
x$coleman <- rowMeans(x[,5:9])
x$dale    <- rowMeans(x[,10:12])
x$fog    <- rowMeans(x[,22:24])
x$forc    <- rowMeans(x[,25:26])
x$sp    <- rowMeans(x[,40:41])




x <- x[,-c(1:3,5:9,10:12,22:24,25:26,40:41)]


corr <- cor(x)
eigen(corr)$values

pca(corr,2,rotate = 'promax')



forcast
dale
coleman
Traenkle.Bailer.2
smog
Scrabble
Flesch
Danielson.Bryan.2
################################################################################

### Engineer the Text Features for Test Data

list2 <- vector('list',nrow(test_df))

for(i in 1:nrow(test_df)){
  
  text <- as.character(test_df[i,]$excerpt)
  
  
  tokenized <- tokens(text)
  dm <- dfm(tokenized)
  
  
  annotated <- udpipe_annotate(ud_eng, x = text)
  annotated <- as.data.frame(annotated)
  
  # cbind_morphological(annotated, term = "feats", which = "lexical")
  
  # Morphological annotation (universal POS tags, https://universaldependencies.org/u/pos/index.html)
  
  temp     <- data.frame(table(annotated$upos))
  temp[,2] <- temp[,2]
  
  words <- annotated[!annotated$upos%in%c('PUNCT','SYS','X'),]$token
  
  temp[,1] <- as.character(temp[,1])
  temp <- rbind(temp,data.frame(table(annotated$xpos)))
  
  temp <- rbind(temp,data.frame(Var1 ='nwords',Freq=length(words)))
  temp <- rbind(temp,data.frame(Var1 ='nchars',Freq=sum(nchar(annotated$token))))
  temp <- rbind(temp,data.frame(Var1 ='nchars',Freq=sum(nchar(annotated$token))/length(words)))
  temp <- rbind(temp,data.frame(Var1 ='wdiv',Freq=length(unique(words))/length(words)))
  temp <- rbind(temp,data.frame(Var1 ='nsent',Freq=length(unique(annotated$sentence_id))))
  
  
  # Morphologicla features (https://universaldependencies.org/u/feat/index.html)
  
  feats  <- na.omit(annotated$feats)
  feats1 <- unlist(strsplit(feats,split='\\|'))
  feats2 <-  unlist(strsplit(feats1,split='='))[c(TRUE,FALSE)]
  
  feats1        <- table(feats1)
  names(feats1) <- gsub('=','.',names(feats1))
  
  feats2        <- table(feats2)
  names(feats2) <- names(feats2)
  
  
  temp <- rbind(temp,data.frame(feats1))
  temp <- rbind(temp,data.frame(feats2))
  
  
  # Syntactic Annotation (https://universaldependencies.org/u/dep/index.html)
  
  temp <- rbind(temp,data.frame(table(annotated$dep_rel)))
  
  # Word Length distribution
  
  wl <- table(nchar(tokens(text,
                           remove_punct = TRUE,
                           remove_numbers = TRUE,
                           remove_symbols = TRUE,
                           remove_separators = TRUE)[[1]])
  )
  
  names(wl) <- paste0('l',names(wl))
  
  
  label_let <- names(wl)
  
  ifelse('l1'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l1',Freq=wl['l1'])),
         temp <- rbind(temp,data.frame(Var1 ='l1',Freq=0)))
  
  ifelse('l2'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l2',Freq=wl['l2'])),
         temp <- rbind(temp,data.frame(Var1 ='l2',Freq=0)))
  
  ifelse('l3'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l3',Freq=wl['l3'])),
         temp <- rbind(temp,data.frame(Var1 ='l3',Freq=0)))
  
  ifelse('l4'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l4',Freq=wl['l4'])),
         temp <- rbind(temp,data.frame(Var1 ='l4',Freq=0)))
  
  ifelse('l5'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l5',Freq=wl['l5'])),
         temp <- rbind(temp,data.frame(Var1 ='l5',Freq=0)))
  
  ifelse('l6'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l6',Freq=wl['l6'])),
         temp <- rbind(temp,data.frame(Var1 ='l6',Freq=0)))
  
  ifelse('l7'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l7',Freq=wl['l7'])),
         temp <- rbind(temp,data.frame(Var1 ='l7',Freq=0)))
  
  ifelse('l8'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l8',Freq=wl['l8'])),
         temp <- rbind(temp,data.frame(Var1 ='l8',Freq=0)))
  
  ifelse('l9'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l9',Freq=wl['l9'])),
         temp <- rbind(temp,data.frame(Var1 ='l9',Freq=0)))
  
  ifelse('l10'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l10',Freq=wl['l10'])),
         temp <- rbind(temp,data.frame(Var1 ='l10',Freq=0)))
  
  ifelse('l11'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l11',Freq=wl['l11'])),
         temp <- rbind(temp,data.frame(Var1 ='l11',Freq=0)))
  
  ifelse('l12'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l12',Freq=wl['l12'])),
         temp <- rbind(temp,data.frame(Var1 ='l12',Freq=0)))
  
  ifelse('l13'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l13',Freq=wl['l13'])),
         temp <- rbind(temp,data.frame(Var1 ='l13',Freq=0)))
  
  ifelse('l14'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l14',Freq=wl['l14'])),
         temp <- rbind(temp,data.frame(Var1 ='l14',Freq=0)))
  
  ifelse('l15'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l15',Freq=wl['l15'])),
         temp <- rbind(temp,data.frame(Var1 ='l15',Freq=0)))
  
  # Measures of Lexical variety)
  
  lexical <- textstat_lexdiv(tokenized,measure = 'all')[1,2:16]
  
  temp <- rbind(temp,data.frame(Var1 = colnames(lexical),Freq = as.numeric(lexical[1,])))
  
  # Measures of Readability
  
  readable <- textstat_readability(text, measure = 'all')[1,2:49]
  temp <- rbind(temp,data.frame(Var1 = colnames(readable),Freq = as.numeric(readable[1,])))
  
  
  list2[[i]] <- data.frame(t(temp[,2]))
  colnames(list2[[i]]) <- temp[,1]
  
  print(i)  
}



################################################################################

# Combine the text feature for all data entries

dtrain <- smartbind(list = list1)
dtest  <- smartbind(list = list2)

################################################################################
################################################################################

train_df <- read.csv(here('data','train.csv'))
test_df  <- read.csv(here('data','test.csv'))

#dtrain  <- read.csv(here('data','dtrain.csv'))
#dtest  <- read.csv(here('data','dtest.csv'))

out         <- train_df$target


# There are some missing values in this data because every reading passage did not have all features. 
# So, any missing value in this matrix should be recoded as 0 (that feature doesn't exist in the text)

for(i in 1:ncol(dtrain)){
  if(sum(is.na(dtrain[,i]))>0){
    dtrain[is.na(dtrain[,i]),i] = 0
  }
}

# Same for test data

for(i in 1:ncol(dtest)){
  if(sum(is.na(dtest[,i]))>0){
    dtest[is.na(dtest[,i]),i] = 0
  }
}


# Check if there is any column in the training data with 0 variance
# and remove it if any

loc    <- which(psych::describe(dtrain)$sd==0)

if(length(loc)>0){
  dtrain <- dtrain[,-loc]
}

################################################################################

require(reshape2)

corr <- cor(dtrain)
corr[lower.tri(corr)] <- NA
diag(corr) <- NA

cormat <- reshape2::melt(corr)
cormat <- na.omit(cormat)

head(cormat)

cormat <- cormat[cormat[,1]!=cormat[,2],]

head(cormat[cormat[,3]>0.9,],30)
nrow(cormat[cormat[,3]>0.9,])


# Eliminate due to high correlation with other variables
# Some of these variables turn out to be identical

elim <- c('cc','CC','CD','DT','JJ','NNP','nsent','IN','PRP','RB','Degree.Pos',
          'Number.Plur','NumType.Card','PronType.Art','PronType.Prs','Tense.Past',
          'VerbForm.Fin','VerbForm.Ger','VerbForm.Inf','VerbForm.Part','Case',
          'Definite','Degree','Mood','NumType','Person','Voice',
          'advmod','Voice.Pass','case','det','nsubj:pass',
          'nummod','punct','root','wdiv','C',
          'CTTR','U','S','R',
          'D','Vm','lgV0','lgeV0','ARI.simple','ARI.NRI',
          'Coleman.C2','Coleman.Liau.ECP','Coleman.Liau.short','Dale.Chall.PSK',
          'Danielson.Bryan','Dickes.Steiwer',
          'Farr.Jenkins.Paterson','Flesch.Kincaid',
          'FOG.PSK','FOG',
          'FORCAST.RGL','Fucks','LIW','nWS','nWs.2','nWs.3','nWs.4',
          'SMOG.C','SMOG.simple','SMOG.de',
          'Spache.old','Strain','Wheeler.Smith','meanSentenceLength',
          'Traenkle.Bailer','\'\'',
          'UH','Poss.Yes','Poss','Reflex.Yes',
          'Abbr.Yes','-LRB-','Typo.Yes','Foreign.Yes')


dtrain <- dtrain[,!colnames(dtrain)%in%elim]


################################################################################
describe(dtrain)[,c('mean','sd','min','max','skew','kurtosis')]

dtrain[which(dtrain$Bormuth.GP<0),]$Bormuth.GP = mean(dtrain$Bormuth.GP)
dtrain$Bormuth.GP <- log(dtrain$Bormuth.GP)


dtest[which(dtest$Bormuth.GP<0),]$Bormuth.GP = mean(dtest$Bormuth.GP)
dtest$Bormuth.GP <- log(dtest$Bormuth.GP)


    
# Make sure that training data and test data includes identical features
    
    keep.var <- colnames(dtrain)[colnames(dtrain)%in%colnames(dtest)]
    
    dtrain <- dtrain[,keep.var]
    dtest  <- dtest[,keep.var]
    
    
    
    
    
    
    
################################################################################
################################################################################
################################################################################
#
#
#                     GLMNET - Main Effects only
#                      - Penalized Regression -
################################################################################
################################################################################
################################################################################
    #poly(as.matrix(dtrain),2)
    
cv_elastic <- caret::train(x            = as.matrix(dtrain),
                           y            = out,
                           method       = "glmnet",
                           trControl    = trainControl(method = "cv", number = 20),
                           type.measure = 'RMSE',
                           family       = 'gaussian',
                           tuneLength   = 10)


fit <- glmnet(x            = as.matrix(dtrain),
              y            = train_df$target,
              alpha        = cv_elastic$bestTune[1],
              lambda       = cv_elastic$bestTune[2],
              method       = "glmnet",                                                      
              family       = 'gaussian',
              standardize  = FALSE)

imp.ind <- which(coef(fit)!=0)
imp     <- colnames(dtrain)[which(coef(fit)!=0)-1]
imp

cv_elastic$bestTune

mse <- cv_elastic$results[,1:3]
mse <- mse[order(mse[,3]),]
head(mse)

predictions1 <- predict(cv_elastic,as.matrix(dtest),type = 'raw')
predictions1


pred1 <- predict(fit,as.matrix(dtrain))

sqrt(mean((pred1-train_df$target)^2))

cor(pred1,train_df$target)


plot(pred1,train_df$target)


plot(density(train_df$target),ylim=c(0,.5))
points(density(pred1),lty=2,type='l')

################################################################################
################################################################################
################################################################################
#
#
#                     GLMNET - Main Effects  + Quadratic Effects
#                      - Penalized Regression -
################################################################################
################################################################################
################################################################################


dtrain1 <- dtrain[,imp]
dtrain2 <- dtrain[,imp]^2

dim(dtrain1)
dim(dtrain2)

cv_elastic <- caret::train(x            = cbind(as.matrix(dtrain1),as.matrix(dtrain2)),
                           y            = train_df$target,
                           method       = "glmnet",
                           trControl    = trainControl(method = "cv", number = 20),                           
                           type.measure = 'MSE',
                           family       = 'gaussian',
                           tuneLength   = 10)

cv_elastic$bestTune

mse <- cv_elastic$results[,1:3]
mse <- mse[order(mse[,3]),]
head(mse)

dtest1 <- dtest[,imp]
dtest2 <- dtest[,imp]^2
predictions2 <- predict(cv_elastic,cbind(as.matrix(dtest1),as.matrix(dtest2)),type = 'raw')
predictions2



fit <- glmnet(x            = cbind(as.matrix(dtrain1),as.matrix(dtrain2)),
              y            = train_df$target,
              alpha        = cv_elastic$bestTune[1],
              lambda       = cv_elastic$bestTune[2],
              method       = "glmnet",                                                      
              family       = 'gaussian',
              standardize  = FALSE)

pred2 <- predict(fit,cbind(as.matrix(dtrain1),as.matrix(dtrain2)))

sqrt(mean((pred2-train_df$target)^2))

cor(pred2,train_df$target)


plot(pred2,train_df$target)


plot(density(train_df$target),ylim=c(0,.5))
points(density(pred2),lty=2,type='l')


################################################################################
################################################################################
################################################################################
#
#
#       GLMNET - Main Effects only + Quadratic Effects + 
#                        Multiple Outcomes
################################################################################
################################################################################
################################################################################

multiple <- matrix(nrow=nrow(train_df),ncol=10)

for(i in 1:nrow(train_df)){
  
  multiple[i,] = rnorm(10,train_df[i,]$target,train_df[i,]$standard_error)
  
}
  

model1     <- caret::train(x            = cbind(as.matrix(dtrain1),as.matrix(dtrain2)),
                           y            = multiple[,1],
                           method       = "glmnet",
                           trControl    = trainControl(method = "cv", number = 20),                           
                           type.measure = 'MSE',
                           family       = 'gaussian',
                           tuneLength   = 10)

model2     <- caret::train(x            = cbind(as.matrix(dtrain1),as.matrix(dtrain2)),
                           y            = multiple[,2],
                           method       = "glmnet",
                           trControl    = trainControl(method = "cv", number = 20),                           
                           type.measure = 'MSE',
                           family       = 'gaussian',
                           tuneLength   = 10)

model3     <- caret::train(x            = cbind(as.matrix(dtrain1),as.matrix(dtrain2)),
                           y            = multiple[,3],
                           method       = "glmnet",
                           trControl    = trainControl(method = "cv", number = 20),                           
                           type.measure = 'MSE',
                           family       = 'gaussian',
                           tuneLength   = 10)

model4     <- caret::train(x            = cbind(as.matrix(dtrain1),as.matrix(dtrain2)),
                           y            = multiple[,4],
                           method       = "glmnet",
                           trControl    = trainControl(method = "cv", number = 20),                           
                           type.measure = 'MSE',
                           family       = 'gaussian',
                           tuneLength   = 10)


model5     <- caret::train(x            = cbind(as.matrix(dtrain1),as.matrix(dtrain2)),
                           y            = multiple[,5],
                           method       = "glmnet",
                           trControl    = trainControl(method = "cv", number = 20),                           
                           type.measure = 'MSE',
                           family       = 'gaussian',
                           tuneLength   = 10)


model6     <- caret::train(x            = cbind(as.matrix(dtrain1),as.matrix(dtrain2)),
                           y            = multiple[,6],
                           method       = "glmnet",
                           trControl    = trainControl(method = "cv", number = 20),                           
                           type.measure = 'MSE',
                           family       = 'gaussian',
                           tuneLength   = 10)

model7     <- caret::train(x            = cbind(as.matrix(dtrain1),as.matrix(dtrain2)),
                           y            = multiple[,7],
                           method       = "glmnet",
                           trControl    = trainControl(method = "cv", number = 20),                           
                           type.measure = 'MSE',
                           family       = 'gaussian',
                           tuneLength   = 10)

model8     <- caret::train(x            = cbind(as.matrix(dtrain1),as.matrix(dtrain2)),
                           y            = multiple[,8],
                           method       = "glmnet",
                           trControl    = trainControl(method = "cv", number = 20),                           
                           type.measure = 'MSE',
                           family       = 'gaussian',
                           tuneLength   = 10)

model9     <- caret::train(x            = cbind(as.matrix(dtrain1),as.matrix(dtrain2)),
                           y            = multiple[,9],
                           method       = "glmnet",
                           trControl    = trainControl(method = "cv", number = 20),                           
                           type.measure = 'MSE',
                           family       = 'gaussian',
                           tuneLength   = 10)

model10     <- caret::train(x            = cbind(as.matrix(dtrain1),as.matrix(dtrain2)),
                           y            = multiple[,10],
                           method       = "glmnet",
                           trControl    = trainControl(method = "cv", number = 20),                           
                           type.measure = 'MSE',
                           family       = 'gaussian',
                           tuneLength   = 10)





fit1 <- glmnet(x           = cbind(as.matrix(dtrain1),as.matrix(dtrain2)),
              y            = multiple[,1],
              alpha        = model1$bestTune[1],
              lambda       = model1$bestTune[2],
              method       = "glmnet",                                                      
              family       = 'gaussian',
              standardize  = FALSE)

fit2 <- glmnet(x           = cbind(as.matrix(dtrain1),as.matrix(dtrain2)),
               y            = multiple[,2],
               alpha        = model2$bestTune[1],
               lambda       = model2$bestTune[2],
               method       = "glmnet",                                                      
               family       = 'gaussian',
               standardize  = FALSE)

fit3 <- glmnet(x           = cbind(as.matrix(dtrain1),as.matrix(dtrain2)),
               y            = multiple[,3],
               alpha        = model3$bestTune[1],
               lambda       = model3$bestTune[2],
               method       = "glmnet",                                                      
               family       = 'gaussian',
               standardize  = FALSE)


fit4 <- glmnet(x           = cbind(as.matrix(dtrain1),as.matrix(dtrain2)),
               y            = multiple[,4],
               alpha        = model4$bestTune[1],
               lambda       = model4$bestTune[2],
               method       = "glmnet",                                                      
               family       = 'gaussian',
               standardize  = FALSE)


fit5 <- glmnet(x           = cbind(as.matrix(dtrain1),as.matrix(dtrain2)),
               y            = multiple[,5],
               alpha        = model5$bestTune[1],
               lambda       = model5$bestTune[2],
               method       = "glmnet",                                                      
               family       = 'gaussian',
               standardize  = FALSE)


fit6 <- glmnet(x           = cbind(as.matrix(dtrain1),as.matrix(dtrain2)),
               y            = multiple[,6],
               alpha        = model6$bestTune[1],
               lambda       = model6$bestTune[2],
               method       = "glmnet",                                                      
               family       = 'gaussian',
               standardize  = FALSE)



fit7 <- glmnet(x           = cbind(as.matrix(dtrain1),as.matrix(dtrain2)),
               y            = multiple[,7],
               alpha        = model7$bestTune[1],
               lambda       = model7$bestTune[2],
               method       = "glmnet",                                                      
               family       = 'gaussian',
               standardize  = FALSE)



fit8 <- glmnet(x           = cbind(as.matrix(dtrain1),as.matrix(dtrain2)),
               y            = multiple[,8],
               alpha        = model8$bestTune[1],
               lambda       = model8$bestTune[2],
               method       = "glmnet",                                                      
               family       = 'gaussian',
               standardize  = FALSE)


fit9 <- glmnet(x           = cbind(as.matrix(dtrain1),as.matrix(dtrain2)),
               y            = multiple[,9],
               alpha        = model9$bestTune[1],
               lambda       = model9$bestTune[2],
               method       = "glmnet",                                                      
               family       = 'gaussian',
               standardize  = FALSE)


fit10 <- glmnet(x           = cbind(as.matrix(dtrain1),as.matrix(dtrain2)),
               y            = multiple[,10],
               alpha        = model10$bestTune[1],
               lambda       = model10$bestTune[2],
               method       = "glmnet",                                                      
               family       = 'gaussian',
               standardize  = FALSE)




pred1  <- predict(fit1,cbind(as.matrix(dtrain1),as.matrix(dtrain2)))
pred2  <- predict(fit2,cbind(as.matrix(dtrain1),as.matrix(dtrain2)))
pred3  <- predict(fit3,cbind(as.matrix(dtrain1),as.matrix(dtrain2)))
pred4  <- predict(fit4,cbind(as.matrix(dtrain1),as.matrix(dtrain2)))
pred5  <- predict(fit5,cbind(as.matrix(dtrain1),as.matrix(dtrain2)))
pred6  <- predict(fit6,cbind(as.matrix(dtrain1),as.matrix(dtrain2)))
pred7  <- predict(fit7,cbind(as.matrix(dtrain1),as.matrix(dtrain2)))
pred8  <- predict(fit8,cbind(as.matrix(dtrain1),as.matrix(dtrain2)))
pred9  <- predict(fit9,cbind(as.matrix(dtrain1),as.matrix(dtrain2)))
pred10 <- predict(fit10,cbind(as.matrix(dtrain1),as.matrix(dtrain2)))


pred <- (pred1 + pred2 + pred3 + pred4 + pred5 + pred6 + pred7 + 
         pred8 + pred9 + pred10)/10


sqrt(mean((pred-train_df$target)^2))

cor(pred,train_df$target)


plot(pred,train_df$target)


plot(density(train_df$target),ylim=c(0,.5))
points(density(pred),lty=2,type='l')



################################################################################
################################################################################
################################################################################
#
#
#       GLMNET - Main Effects only + Quadratic Effects + 
#     PRedict Standadr Error and Piecewise function
################################################################################
################################################################################
################################################################################

ind <- which(train_df$standard_error < .4)

temp       <- dtrain[-ind,]
out.temp   <- train_df[-ind,]$target 
out.temp2  <- train_df[-ind,]$standard_error


temp1 <- temp

model_std <- caret::train(x             = cbind(as.matrix(temp1)),
                           y            = out.temp2,
                           method       = "glmnet",
                           trControl    = trainControl(method = "cv", number = 20),                           
                           type.measure = 'MSE',
                           family       = 'gaussian',
                           tuneLength   = 10)

fit_std <- glmnet(x             = cbind(as.matrix(temp1)),
                          y            = out.temp2,
                          method       = "glmnet",
                          alpha        = model_std$bestTune[1],
                          lambda       = model_std$bestTune[2],                           
                          family       = 'gaussian')

pr_std <- predict(fit_std,cbind(as.matrix(temp1)))

sqrt(mean((pr_std - out.temp2)^2))

cor(pr_std,out.temp2)


plot(pr_std,out.temp2)





model_bin <- caret::train(x            = cbind(as.matrix(temp1)),
                          y            = as.factor(ifelse(out.temp < (-0.96),0,1)),
                          method       = "glmnet",
                          trControl    = trainControl(method = "cv", number = 20),                           
                          type.measure = 'logloss',
                          family       = 'binomial',
                          tuneLength   = 10)

fit_bin <- glmnet(x            = cbind(as.matrix(temp1),as.matrix(temp2)),
                          y            = as.factor(ifelse(out.temp < (-0.96),0,1)),
                          method       = "glmnet",
                          alpha        = model_bin$bestTune[1],
                          lambda       = model_bin$bestTune[2],                           
                          family       = 'binomial')

pr_bin <- predict(fit_bin,cbind(as.matrix(temp1),as.matrix(temp2)),type='response')

sqrt(mean((pr_bin - ifelse(out.temp < (-0.96),0,1))^2))

mltools::auc_roc(preds   = pr_bin[,1],
                 actuals = ifelse(out.temp < (-0.96),0,1),
                 returnDT=FALSE)



##############################################################

# Append the predicted standard errors and binary target prediction (below or 
# above average)

temp$std <- pr_std
temp$bin <- pr_bin

temp1$std <- pr_std
temp1$bin <- pr_bin

  # Add the interaction

    temp$std_bin  <- temp$std*temp$bin
    temp1$std_bin <- temp1$std*temp1$bin
    

# Refit the model with the additional two variables

model_final <- caret::train(x          = cbind(as.matrix(temp1),as.matrix(temp2)),
                          y            = out.temp,
                          method       = "glmnet",
                          trControl    = trainControl(method = "cv", number = 20),                           
                          type.measure = 'MSE',
                          family       = 'gaussian',
                          tuneLength   = 10) 

fit_final <- glmnet(      x            = cbind(as.matrix(temp1),as.matrix(temp2)),
                            y            = out.temp,
                            method       = "glmnet",
                            alpha        = model_final$bestTune[1],
                            lambda       = model_final$bestTune[2], 
                            type.measure = 'MSE',
                            family       = 'gaussian',
                            tuneLength   = 10)


model_final$bestTune

model_final$results[order(model_final$results$RMSE),]

pr_final <- predict(fit_final,cbind(as.matrix(temp1),as.matrix(temp2)))

sqrt(mean((pr_final - out.temp)^2))

cor(pr_final,out.temp)


plot(pr_final,out.temp)



plot(density(out.temp),ylim=c(0,.5))
points(density(pr_final),lty=2,type='l')








################################################################################
################################################################################
################################################################################
#
#
#                    XGBOOST
#
################################################################################
################################################################################

dtrain <- xgb.DMatrix(data = data.matrix(dtrain), label=train_df$target)


set.seed(05092021)

myfolds <- createFolds(1:nrow(dtrain),10)
################################################################################

################################################################################
################################################################################
#Stage 1:  Tune eta
################################################################################
################################################################################

params <- list(booster           = "gblinear", 
               max_depth         = 5, 
               min_child_weight  = 1, 
               gamma             = 0, 
               subsample         = 1, 
               colsample_bytree  = 1,
               max_delta_step    = 0,
               lambda            = 1,
               alpha             = 1,
               scale_pos_weight  = 1,
               num_parallel_tree = 1)


grid <- expand.grid(etas = seq(1,1.2,.005))

grid$loss    <- NA
grid$iter    <- NA

for(i in 1:nrow(grid)){
  
  mod <-  xgb.cv(data                   = dtrain, 
                 params                 = params,
                 objective              = 'reg:squarederror',
                 eval_metric            = 'rmse',
                 showsd                 = TRUE,
                 nthread                = 10, 
                 predict                = TRUE,
                 folds                  = myfolds,
                 nrounds                = 10000,
                 eta                    = grid[i,]$etas,
                 early_stopping_rounds  = 30)
  
  logs <- mod$evaluation_log
  
  grid[i,]$iter <- which.min(logs$test_rmse_mean)
  grid[i,]$loss <- min(logs$test_rmse_mean)
  
  print(i)
  
}

grid[which.min(grid$loss),]

#eta = 1.065
#iter = 105

################################################################################
################################################################################
#Stage 1:  Tune max_depth  min_child_weight
################################################################################
################################################################################

params <- list(booster           = "gblinear", 
               gamma             = 0, 
               subsample         = 1, 
               colsample_bytree  = 1,
               max_delta_step    = 0,
               lambda            = 1,
               alpha             = 1,
               scale_pos_weight  = 1,
               num_parallel_tree = 1)

max_dep <- seq(3,10,1)
min_child_weight <- seq(0,7,0.25)

grid <- expand.grid(depth = max_dep, weight = min_child_weight)

grid$loss    <- NA
grid$iter    <- NA

for(i in 1:nrow(grid)){
  
  mod <-  xgb.cv(data                   = dtrain, 
                 params                 = params,
                 objective              = 'reg:squarederror',
                 eval_metric            = 'rmse',
                 showsd                 = TRUE,
                 nthread                = 10, 
                 predict                = TRUE,
                 folds                  = myfolds,
                 nrounds                = 434,
                 eta                    = 1,
                 max_depth              = grid[i,]$depth, 
                 min_child_weight       = grid[i,]$weight)
  
  logs <- mod$evaluation_log
  
  grid[i,]$iter <- which.min(logs$test_rmse_mean)
  grid[i,]$loss <- min(logs$test_rmse_mean)
  
  print(i)
  
}

grid[which.min(grid$loss),]

#depth = 2
#weight = 2






