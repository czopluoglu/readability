################################################################################
#
#             Readability Prediction by RobertaTF via R
#
################################################################################

# Blog post
# https://blogs.rstudio.com/ai/posts/2020-07-30-state-of-the-art-nlp-models-from-r/

# https://arxiv.org/pdf/1907.11692.pdf

# https://towardsdatascience.com/bert-roberta-distilbert-xlnet-which-one-to-use-3d5ab82ba5f8

# https://towardsdatascience.com/hyperparameter-optimization-for-optimum-transformer-models-b95a32b70949


#                    THIS ONLY RUNS ON GPU
# Most likely it will run on a Kaggle notebook, PCs will have issues


# Code is modified according to the example at the blow link

# https://blogs.rstudio.com/ai/posts/2020-07-30-state-of-the-art-nlp-models-from-r/

################################################################################

# Load the libraries

reticulate::py_install('transformers', pip = TRUE)
library(keras)
library(tensorflow)
library(dplyr)
library(tfdatasets)

transformer <- reticulate::import('transformers')

  # If running TensorFlow on GPU one could specify the following parameters 
  # in order to avoid memory issues.
    
    physical_devices = tf$config$list_physical_devices('GPU')
    tf$config$experimental$set_memory_growth(physical_devices[[1]],TRUE)
    
    tf$keras$backend$set_floatx('float32')

# Download the tokenizer, model, and weights

  # Tokenizer

      transformer$RobertaTokenizer$from_pretrained('roberta-base', do_lower_case=TRUE)

  # Model with weights

  transformer$TFRobertaModel$from_pretrained('roberta-base')

# Read the datasets
  
  df_train <- read.csv('../input/commonlitreadabilityprize/train.csv')
  df_test  <- read.csv('../input/commonlitreadabilityprize/test.csv')
  
  df_train <- data.table::as.data.table(df_train)
  df_test  <- data.table::as.data.table(df_test)
  
  dim(df_train)
  dim(df_test)
  
  str(df_train)
  str(df_test)
  
  head(df_train,3)


# Split the training data into a training set and validation set
  
  ind <- sample.int(nrow(df_train)*0.8)
  
  train_ <- df_train[ind,]
  test_  <- df_train[-ind,]
  
  dim(train_)
  dim(test_)

# parameters
  
  max_len <- 50L
  epochs  <- 5
  batch_size <- 32
  

# tokenizer
  
ai_m <- c('TFRobertaModel',    'RobertaTokenizer',    'roberta-base') 
  
  tokenizer <- glue::glue("transformer${ai_m[2]}$from_pretrained('{ai_m[3]}',
                        do_lower_case=TRUE)") %>% 
    rlang::parse_expr() %>% 
    eval()
  
  tokenizer
  
# model
  
  model_ <- glue::glue("transformer${ai_m[1]}$from_pretrained('{ai_m[3]}')") %>% 
    rlang::parse_expr() %>% 
    eval()
  
  model_



# Data preparation
  
  # inputs
  
  text = list()
  
  # outputs
  
  label = list()
  
  data_prep = function(data) {
    
    for (i in 1:nrow(data)) {
      
      tokens <- tokenizer$encode(data[i,]$excerpt,max_length = max_len,truncation=T) 
      txt    <- list(as.matrix(t(tokens)))
      
      lbl    <- t(data[i,]$target)
      
      text  <- append(text,txt)
      label <- append(label,lbl)
    }
    
    list(do.call(plyr::rbind.fill.matrix,text), do.call(plyr::rbind.fill.matrix,label))
  }
  
  train <- data_prep(train_)
  test  <- data_prep(test_)
  
  str(train)
  str(test)
  
  head(train[[1]])
  head(test[[1]])


# slice datasets
  
  train_slices  <- tensor_slices_dataset(list(train[[1]],train[[2]]))
  train_batch   <- dataset_batch(dataset = train_slices, batch_size = batch_size, drop_remainder = TRUE)
  train_shuffle <- dataset_shuffle(train_batch,128)
  train_repeat  <- dataset_repeat(train_shuffle,epochs)
  train_prefetch <- dataset_prefetch(train_repeat,tf$data$experimental$AUTOTUNE)
  
  tf_train <- train_prefetch
  
  str(tf_train)
  
  test_slices  <- tensor_slices_dataset(list(test[[1]],test[[2]]))
  test_batch   <- dataset_batch(dataset = test_slices, batch_size = batch_size, drop_remainder = TRUE)
  test_shuffle <- dataset_shuffle(test_batch,128)
  test_repeat  <- dataset_repeat(test_shuffle,epochs)
  test_prefetch <- dataset_prefetch(test_repeat,tf$data$experimental$AUTOTUNE)
  
  tf_test <- test_prefetch
  
  str(tf_test)

# Fit the model
  
  # https://keras.io/api/metrics/regression_metrics/#rootmeansquarederror-class
  # https://keras.io/api/metrics/classification_metrics/
  # https://keras.io/api/optimizers/
  
  #set.seed(05112021)
  
  # create layers
  
  input  <- layer_input(shape=c(max_len), dtype='int32')
  
  hidden <- tf$reduce_mean(model_(input)[[1]], axis=1L) %>% 
    layer_dense(units=64, activation='relu') %>%
    layer_dense(units=64, activation='relu')
  
  
  output = hidden %>% layer_dense(units=1, activation='linear',dtype = 'float32')
  
  model <- keras_model(inputs=input, outputs = output)
  
  # compile with RMSE score
  
  model %>% compile(optimizer= tf$keras$optimizers$SGD(learning_rate=0.0004),
                    loss = tf$losses$MeanSquaredError(),
                    metrics = tf$metrics$RootMeanSquaredError())
  
  
  # train the model
  
  history <- model %>% keras::fit(tf_train, 
                                  epochs=5,
                                  validation_data=tf_test)
  
  str(history)
  
  data.frame(epochs = 1:5,
             loss=history$metrics$loss,
             mse=history$metrics$root_mean_squared_error,
             test_loss = history$metrics$val_loss,
             test_mse  = history$metrics$val_root_mean_squared_error)
  
  plot(history)
  
  
# Data preparation for the validation set
  
  data_prep = function(data) {
    
    text = list()
    
    for (i in 1:nrow(data)) {
      
      tokens <- tokenizer$encode(data[i,]$excerpt,max_length = max_len,truncation=T) 
      txt    <- list(as.matrix(t(tokens)))
      
      text  <- append(text,txt)
    }
    
    list(do.call(plyr::rbind.fill.matrix,text))
  }
  
  test.final  <- data_prep(df_test)
  
  str(test.final)
  
  head(test.final[[1]])
  
  # PRedictions for the validation set
  
  pred <- predict(object = model,x = test.final)
  
  str(pred)
  
  # Submission file
  
  outcome <- data.frame(id = df_test$id,target=pred)
  
  outcome
  
  write.csv(outcome,'submission.csv',row.names=FALSE,quote=FALSE)
  
  pred
  