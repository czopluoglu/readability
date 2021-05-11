text <- as.character(train_df[i,]$excerpt)

docs <- Corpus(VectorSource(text))

# Convert the text to lower case

docs <- tm_map(docs, content_transformer(tolower))

# Remove numbers

docs <- tm_map(docs, removeNumbers)

# Remove punctuations

docs <- tm_map(docs, removePunctuation)


# Eliminate extra white spaces

docs <- tm_map(docs, stripWhitespace)



dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
m <- as.matrix(m[nchar(rownames(m))!=1,])

# Remove english common stopwords

docs2 <- tm_map(docs, removeWords, stopwords("english"))

dtm    <- TermDocumentMatrix(docs2)
m.stop <- as.matrix(dtm)
m.stop <- as.matrix(m.stop[nchar(rownames(m.stop))!=1,])

a <- data.frame(n1 = nrow(m),
                n2 = nrow(m.stop),
                n3 = sum(nchar(rownames(m))),
                n4 = sum(nchar(rownames(m.stop))))

# Add frequency of first letters for all words

a <- cbind(a,
           data.frame(matrix(NA,
                             nrow = 1,
                             ncol = 26,
                             dimnames = list(c(1),letters)
           )
           )) 


nletter <- table(substr(rownames(m),1,1))
nletter <- nletter[names(nletter)%in%letters]

a[,colnames(a)%in%names(nletter)] <- nletter

colnames(a)[5:30] <- paste0(letters,'1')

# Add frequency of first letters for words excluding stop words

a <- cbind(a,
           data.frame(matrix(NA,
                             nrow = 1,
                             ncol = 26,
                             dimnames = list(c(1),letters)
           )
           )) 


nletter2 <- table(substr(rownames(m.stop),1,1))
nletter2 <- nletter2[names(nletter2)%in%letters]

a[,colnames(a)%in%names(nletter2)] <- nletter2

colnames(a)[31:56] <- paste0(letters,'2')

a[is.na(a)] = 0

# Sentiment Analysis

d  <- get_nrc_sentiment(text)
td <- data.frame(t(d))
td_new <- data.frame(rowSums(td))
names(td_new)[1] <- "count"
td_new <- cbind("sentiment" = rownames(td_new), td_new)
rownames(td_new) <- NULL
td_new <- td_new[order(td_new$count,decreasing=T),]

td_new$perc <- td_new$count/sum(td_new$count)

a <- cbind(a,
           data.frame(matrix(NA,
                             nrow = 1,
                             ncol = 10,
                             dimnames = list(c(1),
                                             c('positive','joy','trust',
                                               'negative','anticipation',
                                               'anger','disgust','fear',
                                               'surprise','sadness'))
           )
           )) 


a[,colnames(a)%in%td_new[,1]] <- td_new[,3]

