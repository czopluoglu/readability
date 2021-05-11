###############################################################################
###############################################################################
###############################################################################
#
#                 THIS CODE RELIES ON KORPUR PACKAGE
#          RUNS FINE ON WINDOWS IF YOU CAN MAKE TREETAGGER WORK
#
#    DOESN'T WORK ON KAGGLE BECAUSE TREETAGGER DOESN'T WORK ON KAGGLE KERNEL
#
###############################################################################
###############################################################################
###############################################################################

# Initial submission experiment

#require(psych)

#train_df <- read.csv('../input/commonlitreadabilityprize/train.csv')
#test_df  <- read.csv('../input/commonlitreadabilityprize/test.csv')

#describe(train_df$target)

#naive <- sqrt(mean((train_df$target - mean(train_df$target))^2))

#naive

#outcome <- data.frame(id = test_df$id,target=naive)

#write.csv(outcome,'submission.csv',row.names=FALSE,quote=FALSE)

#https://www.rdocumentation.org/packages/koRpus/versions/0.13-5/topics/flesch.kincaid

###############################################################################

# Install packages

require(tm)
require(syuzhet)
require(here)
require(gtools)
require(psych)
require(caret)
require(glmnet)
require(here)
require(xgboost)
require(koRpus)
library(koRpus.lang.en)

LCC.en <- read.corp.LCC(here('data',
                             'eng_news_2020_1M'),
                        fileEncoding = 'UCS-2LE')


#Import the datasets

#train_df <- read.csv('../input/commonlitreadabilityprize/train.csv')
#test_df  <- read.csv('../input/commonlitreadabilityprize/test.csv')

train_df <- read.csv(here('data','train.csv'))
test_df  <- read.csv(here('data','test.csv'))

################################################################################

### Features for Training Data

list1 <- vector('list',nrow(train_df))


for(i in 1:nrow(train_df)){
  
  text <- as.character(train_df[i,]$excerpt)
  
  # Readibility scores 
  # https://cran.r-project.org/web/packages/koRpus/vignettes/koRpus_vignette.html)
  # https://www.cs.upc.edu/~nlp/SVMTool/PennTreebank.html
  
  tagged.text <- treetag(file = text,
                         treetagger="manual",
                         format = 'obj',
                         lang="en",
                         TT.options=list(path=here('data','treetagger'),
                                         preset="en"))
  
  
  temp     <- data.frame(table(tagged.text@tokens$tag))
  temp[,2] <- temp[,2]/sum(temp[,2])
  temp[,1] <- as.character(temp[,1])
  
  desc              <- sylly::describe(tagged.text)
  
  freq.analysis.res <- freq.analysis(tagged.text, corp.freq=LCC.en)
  classes           <- sylly::describe(freq.analysis.res)$classes
  #taggedText(freq.analysis.res)
  
  
  hyph.txt.en       <- hyphen(tagged.text)
  
  readbl.txt        <- readability(tagged.text, hyphen=hyph.txt.en)
  
  read_mes <- data.frame(Var1 = names(summary(readbl.txt, flat=TRUE)),
                         Freq = summary(readbl.txt, flat=TRUE),
                         row.names = NULL)
  
  
  temp <- rbind(temp,data.frame(Var1 ='all.chars',Freq=desc$all.chars))
  temp <- rbind(temp,data.frame(Var1 ='lines',Freq=desc$lines))
  temp <- rbind(temp,data.frame(Var1 ='space1',Freq=desc$normalized.space))
  temp <- rbind(temp,data.frame(Var1 ='space2',Freq=desc$chars.no.space))
  temp <- rbind(temp,data.frame(Var1 ='punct',Freq=desc$punct))
  temp <- rbind(temp,data.frame(Var1 ='digits',Freq=desc$digits))
  temp <- rbind(temp,data.frame(Var1 ='letters',Freq=desc$letters[1]))
  
  label_let <- names(desc$letters)
  
  ifelse('l1'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l1',Freq=desc$letters['l1'])),
         temp <- rbind(temp,data.frame(Var1 ='l1',Freq=0)))
  
  ifelse('l2'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l2',Freq=desc$letters['l2'])),
         temp <- rbind(temp,data.frame(Var1 ='l2',Freq=0)))
  
  ifelse('l3'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l3',Freq=desc$letters['l3'])),
         temp <- rbind(temp,data.frame(Var1 ='l3',Freq=0)))
  
  ifelse('l4'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l4',Freq=desc$letters['l4'])),
         temp <- rbind(temp,data.frame(Var1 ='l4',Freq=0)))
  
  ifelse('l5'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l5',Freq=desc$letters['l5'])),
         temp <- rbind(temp,data.frame(Var1 ='l5',Freq=0)))
  
  ifelse('l6'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l6',Freq=desc$letters['l6'])),
         temp <- rbind(temp,data.frame(Var1 ='l6',Freq=0)))
  
  ifelse('l7'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l7',Freq=desc$letters['l7'])),
         temp <- rbind(temp,data.frame(Var1 ='l7',Freq=0)))
  
  ifelse('l8'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l8',Freq=desc$letters['l8'])),
         temp <- rbind(temp,data.frame(Var1 ='l8',Freq=0)))
  
  ifelse('l9'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l9',Freq=desc$letters['l9'])),
         temp <- rbind(temp,data.frame(Var1 ='l9',Freq=0)))
  
  ifelse('l10'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l10',Freq=desc$letters['l10'])),
         temp <- rbind(temp,data.frame(Var1 ='l10',Freq=0)))
  
  ifelse('l11'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l11',Freq=desc$letters['l11'])),
         temp <- rbind(temp,data.frame(Var1 ='l11',Freq=0)))
  
  temp <- rbind(temp,data.frame(Var1 ='ave.sentc',Freq=desc$avg.sentc.length))
  temp <- rbind(temp,data.frame(Var1 ='ave.word',Freq=desc$avg.word.length))
  temp <- rbind(temp,data.frame(Var1 ='n.sentc',Freq=desc$sentences))
  temp <- rbind(temp,data.frame(Var1 ='n.word',Freq=desc$words))
  temp <- rbind(temp,data.frame(Var1 ='ttr',Freq=TTR(tagged.text)@ TTR))
  temp <- rbind(temp,data.frame(Var1 ='msttr',Freq=MSTTR(tagged.text)@ MSTTR$ MSTTR))
  temp <- rbind(temp,data.frame(Var1 ='mattr',Freq=MATTR(tagged.text)@ MATTR$MATTR))
  temp <- rbind(temp,data.frame(Var1 ='C',Freq=C.ld(tagged.text)@ C.ld))
  temp <- rbind(temp,data.frame(Var1 ='R',Freq=R.ld(tagged.text)@ R.ld))
  temp <- rbind(temp,data.frame(Var1 ='cttr',Freq=CTTR(tagged.text)@ CTTR))
  temp <- rbind(temp,data.frame(Var1 ='U',Freq=U.ld(tagged.text)@ U.ld))
  temp <- rbind(temp,data.frame(Var1 ='maas1',Freq=maas(tagged.text)@Maas))
  temp <- rbind(temp,data.frame(Var1 ='maas2',Freq=maas(tagged.text)@lgV0))
  temp <- rbind(temp,data.frame(Var1 ='maas3',Freq=maas(tagged.text)@lgeV0))
  temp <- rbind(temp,data.frame(Var1 ='maas4',Freq=maas(tagged.text)@Maas.grw))
  temp <- rbind(temp,data.frame(Var1 ='mtld1',Freq=MTLD(tagged.text)@MTLD$MTLD))
  temp <- rbind(temp,data.frame(Var1 ='mtld2',Freq=MTLD(tagged.text)@MTLD$factors['mean']))
  temp <- rbind(temp,data.frame(Var1 ='mtld3',Freq=MTLD(tagged.text)@MTLD$lengths$sd))
  temp <- rbind(temp,data.frame(Var1 ='mtld4',Freq=MTLD(tagged.text)@MTLD$lengths$sd.comp))
  temp <- rbind(temp,data.frame(Var1 ='hdd1',Freq=HDD(tagged.text)@ HDD$HDD))
  temp <- rbind(temp,data.frame(Var1 ='hdd2',Freq=HDD(tagged.text)@ HDD$ATTR))
  temp <- rbind(temp,data.frame(Var1 ='nadj',Freq=length(classes$adjective)))
  temp <- rbind(temp,data.frame(Var1 ='namb',Freq=length(classes$ambiguous)))
  temp <- rbind(temp,data.frame(Var1 ='nart',Freq=length(classes$article)))
  temp <- rbind(temp,data.frame(Var1 ='npar',Freq=length(classes$particle)))
  temp <- rbind(temp,data.frame(Var1 ='nconj',Freq=length(classes$conjunction)))
  temp <- rbind(temp,data.frame(Var1 ='number',Freq=length(classes$ambiguous)))
  temp <- rbind(temp,data.frame(Var1 ='ndet',Freq=length(classes$determiner)))
  temp <- rbind(temp,data.frame(Var1 ='nexist',Freq=length(classes$existential)))
  temp <- rbind(temp,data.frame(Var1 ='nfor',Freq=length(classes$preposition)))
  temp <- rbind(temp,data.frame(Var1 ='nintj',Freq=length(classes$interjection)))
  temp <- rbind(temp,data.frame(Var1 ='nmod',Freq=length(classes$modal)))
  temp <- rbind(temp,data.frame(Var1 ='nnoun',Freq=length(classes$noun)))
  temp <- rbind(temp,data.frame(Var1 ='nname',Freq=length(classes$name)))
  temp <- rbind(temp,data.frame(Var1 ='npredet',Freq=length(classes$predeterminer)))
  temp <- rbind(temp,data.frame(Var1 ='npronoun',Freq=length(classes$pronoun)))
  temp <- rbind(temp,data.frame(Var1 ='npos',Freq=length(classes$possessive)))
  temp <- rbind(temp,data.frame(Var1 ='nsym',Freq=length(classes$symbol)))
  temp <- rbind(temp,data.frame(Var1 ='nto',Freq=length(classes$to)))
  temp <- rbind(temp,data.frame(Var1 ='nclass',Freq=length(classes$unclassified)))
  temp <- rbind(temp,data.frame(Var1 ='nverb',Freq=length(classes$verb)))
  temp <- rbind(temp,data.frame(Var1 ='nneg',Freq=length(classes$negation)))
  temp <- rbind(temp,data.frame(Var1 ='nword',Freq=length(classes$word)))
  temp <- rbind(temp,data.frame(Var1 ='nabbr',Freq=length(classes$abbreviation)))
  temp <- rbind(temp,data.frame(Var1 ='nunk',Freq=length(classes$unknown)))
  temp <- rbind(temp,data.frame(Var1 ='nadpos',Freq=length(classes$adposition)))
  temp <- rbind(temp,data.frame(Var1 ='naux',Freq=length(classes$auxiliary)))
  temp <- rbind(temp,data.frame(Var1 ='nother',Freq=length(classes$other)))
  temp <- rbind(temp,data.frame(Var1 ='npunc',Freq=length(classes$punctuation)))
  temp <- rbind(temp,data.frame(Var1 ='ncomm',Freq=length(classes$comma)))
  temp <- rbind(temp,data.frame(Var1 ='nlett',Freq=length(classes$letter)))
  
  temp <- rbind(temp,read_mes)
  
  list1[[i]] <- data.frame(t(temp[,2]))
  colnames(list1[[i]]) <- temp[,1]
  
  print(i)  
}

################################################################################

### Features for Test

list2 <- vector('list',nrow(test_df))


for(i in 1:nrow(test_df)){
  
  text <- as.character(test_df[i,]$excerpt)
  
  # Readibility scores 
  # https://cran.r-project.org/web/packages/koRpus/vignettes/koRpus_vignette.html)
  # https://www.cs.upc.edu/~nlp/SVMTool/PennTreebank.html
  
  tagged.text <- treetag(file = text,
                         treetagger="manual",
                         format = 'obj',
                         lang="en",
                         TT.options=list(path=here('data','treetagger'),
                                         preset="en"))
  
  
  temp     <- data.frame(table(tagged.text@tokens$tag))
  temp[,2] <- temp[,2]/sum(temp[,2])
  temp[,1] <- as.character(temp[,1])
  
  desc              <- sylly::describe(tagged.text)
  
  freq.analysis.res <- freq.analysis(tagged.text, corp.freq=LCC.en)
  classes           <- sylly::describe(freq.analysis.res)$classes
  #taggedText(freq.analysis.res)
  
  
  hyph.txt.en       <- hyphen(tagged.text)
  
  readbl.txt        <- readability(tagged.text, hyphen=hyph.txt.en)
  
  read_mes <- data.frame(Var1 = names(summary(readbl.txt, flat=TRUE)),
                         Freq = summary(readbl.txt, flat=TRUE),
                         row.names = NULL)
  
  
  temp <- rbind(temp,data.frame(Var1 ='all.chars',Freq=desc$all.chars))
  temp <- rbind(temp,data.frame(Var1 ='lines',Freq=desc$lines))
  temp <- rbind(temp,data.frame(Var1 ='space1',Freq=desc$normalized.space))
  temp <- rbind(temp,data.frame(Var1 ='space2',Freq=desc$chars.no.space))
  temp <- rbind(temp,data.frame(Var1 ='punct',Freq=desc$punct))
  temp <- rbind(temp,data.frame(Var1 ='digits',Freq=desc$digits))
  temp <- rbind(temp,data.frame(Var1 ='letters',Freq=desc$letters[1]))
  
  label_let <- names(desc$letters)
  
  ifelse('l1'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l1',Freq=desc$letters['l1'])),
         temp <- rbind(temp,data.frame(Var1 ='l1',Freq=0)))
  
  ifelse('l2'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l2',Freq=desc$letters['l2'])),
         temp <- rbind(temp,data.frame(Var1 ='l2',Freq=0)))
  
  ifelse('l3'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l3',Freq=desc$letters['l3'])),
         temp <- rbind(temp,data.frame(Var1 ='l3',Freq=0)))
  
  ifelse('l4'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l4',Freq=desc$letters['l4'])),
         temp <- rbind(temp,data.frame(Var1 ='l4',Freq=0)))
  
  ifelse('l5'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l5',Freq=desc$letters['l5'])),
         temp <- rbind(temp,data.frame(Var1 ='l5',Freq=0)))
  
  ifelse('l6'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l6',Freq=desc$letters['l6'])),
         temp <- rbind(temp,data.frame(Var1 ='l6',Freq=0)))
  
  ifelse('l7'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l7',Freq=desc$letters['l7'])),
         temp <- rbind(temp,data.frame(Var1 ='l7',Freq=0)))
  
  ifelse('l8'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l8',Freq=desc$letters['l8'])),
         temp <- rbind(temp,data.frame(Var1 ='l8',Freq=0)))
  
  ifelse('l9'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l9',Freq=desc$letters['l9'])),
         temp <- rbind(temp,data.frame(Var1 ='l9',Freq=0)))
  
  ifelse('l10'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l10',Freq=desc$letters['l10'])),
         temp <- rbind(temp,data.frame(Var1 ='l10',Freq=0)))
  
  ifelse('l11'%in%label_let,
         temp <- rbind(temp,data.frame(Var1 ='l11',Freq=desc$letters['l11'])),
         temp <- rbind(temp,data.frame(Var1 ='l11',Freq=0)))
  
  temp <- rbind(temp,data.frame(Var1 ='ave.sentc',Freq=desc$avg.sentc.length))
  temp <- rbind(temp,data.frame(Var1 ='ave.word',Freq=desc$avg.word.length))
  temp <- rbind(temp,data.frame(Var1 ='n.sentc',Freq=desc$sentences))
  temp <- rbind(temp,data.frame(Var1 ='n.word',Freq=desc$words))
  temp <- rbind(temp,data.frame(Var1 ='ttr',Freq=TTR(tagged.text)@ TTR))
  temp <- rbind(temp,data.frame(Var1 ='msttr',Freq=MSTTR(tagged.text)@ MSTTR$ MSTTR))
  temp <- rbind(temp,data.frame(Var1 ='mattr',Freq=MATTR(tagged.text)@ MATTR$MATTR))
  temp <- rbind(temp,data.frame(Var1 ='C',Freq=C.ld(tagged.text)@ C.ld))
  temp <- rbind(temp,data.frame(Var1 ='R',Freq=R.ld(tagged.text)@ R.ld))
  temp <- rbind(temp,data.frame(Var1 ='cttr',Freq=CTTR(tagged.text)@ CTTR))
  temp <- rbind(temp,data.frame(Var1 ='U',Freq=U.ld(tagged.text)@ U.ld))
  temp <- rbind(temp,data.frame(Var1 ='maas1',Freq=maas(tagged.text)@Maas))
  temp <- rbind(temp,data.frame(Var1 ='maas2',Freq=maas(tagged.text)@lgV0))
  temp <- rbind(temp,data.frame(Var1 ='maas3',Freq=maas(tagged.text)@lgeV0))
  temp <- rbind(temp,data.frame(Var1 ='maas4',Freq=maas(tagged.text)@Maas.grw))
  temp <- rbind(temp,data.frame(Var1 ='mtld1',Freq=MTLD(tagged.text)@MTLD$MTLD))
  temp <- rbind(temp,data.frame(Var1 ='mtld2',Freq=MTLD(tagged.text)@MTLD$factors['mean']))
  temp <- rbind(temp,data.frame(Var1 ='mtld3',Freq=MTLD(tagged.text)@MTLD$lengths$sd))
  temp <- rbind(temp,data.frame(Var1 ='mtld4',Freq=MTLD(tagged.text)@MTLD$lengths$sd.comp))
  temp <- rbind(temp,data.frame(Var1 ='hdd1',Freq=HDD(tagged.text)@ HDD$HDD))
  temp <- rbind(temp,data.frame(Var1 ='hdd2',Freq=HDD(tagged.text)@ HDD$ATTR))
  temp <- rbind(temp,data.frame(Var1 ='nadj',Freq=length(classes$adjective)))
  temp <- rbind(temp,data.frame(Var1 ='namb',Freq=length(classes$ambiguous)))
  temp <- rbind(temp,data.frame(Var1 ='nart',Freq=length(classes$article)))
  temp <- rbind(temp,data.frame(Var1 ='npar',Freq=length(classes$particle)))
  temp <- rbind(temp,data.frame(Var1 ='nconj',Freq=length(classes$conjunction)))
  temp <- rbind(temp,data.frame(Var1 ='number',Freq=length(classes$ambiguous)))
  temp <- rbind(temp,data.frame(Var1 ='ndet',Freq=length(classes$determiner)))
  temp <- rbind(temp,data.frame(Var1 ='nexist',Freq=length(classes$existential)))
  temp <- rbind(temp,data.frame(Var1 ='nfor',Freq=length(classes$preposition)))
  temp <- rbind(temp,data.frame(Var1 ='nintj',Freq=length(classes$interjection)))
  temp <- rbind(temp,data.frame(Var1 ='nmod',Freq=length(classes$modal)))
  temp <- rbind(temp,data.frame(Var1 ='nnoun',Freq=length(classes$noun)))
  temp <- rbind(temp,data.frame(Var1 ='nname',Freq=length(classes$name)))
  temp <- rbind(temp,data.frame(Var1 ='npredet',Freq=length(classes$predeterminer)))
  temp <- rbind(temp,data.frame(Var1 ='npronoun',Freq=length(classes$pronoun)))
  temp <- rbind(temp,data.frame(Var1 ='npos',Freq=length(classes$possessive)))
  temp <- rbind(temp,data.frame(Var1 ='nsym',Freq=length(classes$symbol)))
  temp <- rbind(temp,data.frame(Var1 ='nto',Freq=length(classes$to)))
  temp <- rbind(temp,data.frame(Var1 ='nclass',Freq=length(classes$unclassified)))
  temp <- rbind(temp,data.frame(Var1 ='nverb',Freq=length(classes$verb)))
  temp <- rbind(temp,data.frame(Var1 ='nneg',Freq=length(classes$negation)))
  temp <- rbind(temp,data.frame(Var1 ='nword',Freq=length(classes$word)))
  temp <- rbind(temp,data.frame(Var1 ='nabbr',Freq=length(classes$abbreviation)))
  temp <- rbind(temp,data.frame(Var1 ='nunk',Freq=length(classes$unknown)))
  temp <- rbind(temp,data.frame(Var1 ='nadpos',Freq=length(classes$adposition)))
  temp <- rbind(temp,data.frame(Var1 ='naux',Freq=length(classes$auxiliary)))
  temp <- rbind(temp,data.frame(Var1 ='nother',Freq=length(classes$other)))
  temp <- rbind(temp,data.frame(Var1 ='npunc',Freq=length(classes$punctuation)))
  temp <- rbind(temp,data.frame(Var1 ='ncomm',Freq=length(classes$comma)))
  temp <- rbind(temp,data.frame(Var1 ='nlett',Freq=length(classes$letter)))
  
  temp <- rbind(temp,read_mes)
  
  list2[[i]] <- data.frame(t(temp[,2]))
  colnames(list2[[i]]) <- temp[,1]
  
  print(i)  
}


################################################################################

dtrain <- smartbind(list = list1)
dtest  <- smartbind(list = list2)

write.csv(dtrain,'dtrain.csv',row.names=FALSE,quote=FALSE)
write.csv(dtest,'dtest.csv',row.names=FALSE,quote=FALSE)

################################################################################

train_df <- read.csv(here('data','train.csv'))
test_df  <- read.csv(here('data','test.csv'))

dtrain  <- read.csv(here('data','dtrain.csv'))
dtest  <- read.csv(here('data','dtest.csv'))

out         <- train_df$target


loc <- which(describe(dtrain)$sd==0)


dtrain <- dtrain[,-loc]
dtest  <- dtest[,-loc]

loc2 <- which(colSums(is.na(dtrain))==nrow(dtrain))


dtrain <- dtrain[,-loc2]
dtest  <- dtest[,-loc2]



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


ridge <- cv.glmnet(x = as.matrix(dtrain),
                   y = out,
                   alpha = 0,
                   family = gaussian(),
                   nfold = 20,
                   type.measure = 'mse')

plot(ridge, main = "Ridge penalty")

mse <- cbind(ridge$lambda,ridge$cvm)

mse <- mse[order(mse[,2]),]

head(mse)

ridge$lambda.min
coef(ridge,ridge$lambda.min)

ridge.fit <- glmnet(x = as.matrix(dtrain), 
                    y = out, 
                    alpha = 0, 
                    lambda = ridge$lambda.min,
                    family = gaussian())


pridge <- predict(ridge.fit,as.matrix(dtest))

################################################################

lasso <-  cv.glmnet(x = as.matrix(dtrain),
                    y = out,
                    alpha = 1,
                    family = gaussian(),
                    nfold = 20,
                    type.measure = 'mse')

plot(lasso, main = "Lasso penalty")

mse <- cbind(lasso$lambda,lasso$cvm)

mse <- mse[order(mse[,2]),]

head(mse)


lasso$lambda.min
coef(lasso,lasso$lambda.min)

lasso.fit <- glmnet(x = as.matrix(dtrain), 
                    y = out, 
                    alpha = 1, 
                    lambda = lasso$lambda.min,
                    family = 'gaussian')


plasso <- predict(lasso.fit,as.matrix(dtest))

################################################################


cv_elastic <- caret::train(x = as.matrix(dtrain),
                           y = out,
                           method = "glmnet",
                           trControl = trainControl(method = "cv", number = 20),
                           type.measure = 'MSE',
                           family = 'gaussian',
                           tuneLength = 10)

cv_elastic$bestTune

mse <- cv_elastic$results[,1:3]
mse[,3] <- mse[,3]^2
mse <- mse[order(mse[,3]),]
mse


elastic.fit <- glmnet(x = as.matrix(dtrain), 
                      y = out, 
                      alpha  = mse[1,1], 
                      lambda = mse[1,2],
                      family = 'gaussian')

pelastic <- predict(elastic.fit,as.matrix(dtest))


################################################################################
################################################################################
################################################################################
#
#
#                    XGBOOST
#
################################################################################
################################################################################

dtrain <- xgb.DMatrix(data = data.matrix(dtrain), label=out1)
dtest  <- xgb.DMatrix(data = data.matrix(tdtest),  label=out2)

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


grid <- expand.grid(etas = seq(0.5,1,.05))

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

#eta = 1
#iter = 434

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






