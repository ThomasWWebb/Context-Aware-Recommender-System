import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
import scipy
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def getAllUserMeans(contextualRatings):
    return contextualRatings.groupby(by="UserID",as_index=True)['Rating'].mean()

def getRatingProportions(activeUserID, ratings, contexts):
    userRatings = getUserRatings(activeUserID, ratings)
    maxRatings = userRatings['Rating'].size
    ratingProportions = {}
    for context in contexts:
        ratingProportions[context] = userRatings[userRatings['mood'] == context]['Rating'].size / maxRatings
    return ratingProportions

def getUserRatings(activeUserID, ratings):
    userRatings = ratings[ratings['UserID'] == activeUserID]
    return userRatings[userRatings['Rating'] != 0]

def getUserSimilarities(userItemTable, activeUserID):
    similarities = []
    for index, row in userItemTable.iterrows():
        if index != activeUserID:
            rowArray = row.to_numpy()
            userArray = userItemTable.loc[activeUserID].to_numpy()
            newOtherArray = []
            newUserArray = []
            for rowIndex in range(len(rowArray)):
                if rowArray[rowIndex] != 0 and userArray[rowIndex] != 0:
                    newOtherArray.append(rowArray[rowIndex])
                    newUserArray.append(userArray[rowIndex])
            similarity = 0      
            if np.count_nonzero(row.to_numpy()) != 0:
                if len(newOtherArray) > 0:
                    similarity = 1 - scipy.spatial.distance.cosine(np.array(newUserArray), np.array(newOtherArray))
                    #if len(newOtherArray) == 1 and similarity == 1:
                        #similarity = 0.5
            similarities.append([index, similarity])
    return sorted(similarities, key = lambda x: x[1], reverse=True)[:30]

def getItemPrediction(itemIndex, similarities, userMeans, userItemTable, activeUserID):
    itemAdditionSum = 0
    similaritySum = 0
    normalisedPrediction = 0
    #gets the mean for the active user
    activeUserMean = userMeans[activeUserID]
    for user, userSimilarity  in similarities:
        #mean of the other user
        userMean = userMeans[user]
        #adds the weight of each item 
        if itemIndex in userItemTable.columns:
            otherUserRating = userItemTable[itemIndex][user]
            #only add the rating if the other user has rated and similarity is not 0
            if otherUserRating != 0 and userSimilarity != 0:
                itemAdditionSum += userSimilarity * (otherUserRating - userMean)
                similaritySum += userSimilarity
    if similaritySum != 0:
        normalisedPrediction = activeUserMean + (itemAdditionSum / similaritySum)
    return normalisedPrediction

def getUserPredictions(activeUserID, ratedMusicTracks, similarityTuples):
    predictions = []
    for itemIndex in ratedMusicTracks:
        weightSum = 0
        predictionSum = 0
        similarities, userMeans, userItemTable = similarityTuples
        individualPrediction = getItemPrediction(itemIndex, similarities, userMeans, userItemTable, activeUserID)
        predictions.append([itemIndex, individualPrediction])
    return sorted(predictions, key = lambda x: x[1], reverse=True)

def getWeights(activeUserID, ratings, contexts, userContext, contextValues):
    ratingProportions = getRatingProportions(activeUserID, ratings, contexts)
    weights = {}
    for contextIndex in range(len(contexts)):
        context = contexts[contextIndex]
        distance = abs(contextValues[contexts.index(userContext)][contextIndex])
        weights[context] = (1/distance)*ratingProportions[context]
    return weights

def recommend(activeUserID, ratings):
    #calculate context weights
    allItems = ratings['ItemID'].unique()
    userMeans = getAllUserMeans(ratings)
    ratedMusicTracks = ratings['ItemID'].unique()
    userItemTable = ratings.pivot_table(index='UserID', columns='ItemID', values='Rating').fillna(0)
    activeUserMeanRating = userMeans[activeUserID]

    similarities = getUserSimilarities(userItemTable, activeUserID)
    predictions = getUserPredictions(activeUserID, allItems, (similarities, userMeans, userItemTable))
    return predictions, userItemTable

def getLikedItems(ratingSet, likeThreshold):
    likedItemsSum = 0
    for index, row in ratingSet.iterrows():
        if row['Rating'] > 3.5:
            likedItemsSum += 1
    return likedItemsSum

def runMAE(originalRatings, listSize, likeThreshold):
    kf = KFold(n_splits=5, shuffle=True)
    foldCount = 0
    allMAESum = 0
    allPrecisionSum = 0
    allRecallSum = 0
    print("Running Evaluations")
    for train_index, test_index in kf.split(originalRatings):
        foldCount += 1
        print("Running fold {}".format(foldCount))
        train, test = originalRatings.iloc[train_index], originalRatings.iloc[test_index]
        #train, test = train_test_split(originalRatings, test_size=0.2)
        ratings = train

        #test data
        groupTest = test.groupby(by=["UserID"])

        #for MAE
        MAESum = 0
        ratingsNum = 0

        #for precision and recall
        truePositives = 0
        falsePositives = 0
        testLikedItems = getLikedItems(test, likeThreshold)

        #for printing
        userSum = 1

        for userID, row in groupTest:
            if getUserRatings(userID, ratings).size == 0:
                continue

            #group by mood
            userGroup = groupTest.get_group(userID).set_index('ItemID')
            userSum += 1
            #get the predictions
            predictions, userItemTable = recommend(userID, ratings)
            predictionIndex = 0
            currentTruePositives = 0
            for prediction in predictions:
                #if the prediction has a true value calculate the error
                if prediction[0] in userGroup.index:
                    trueRating = userGroup.loc[prediction[0]]["Rating"]
                    if trueRating != 0:
                        MAESum += abs(trueRating - prediction[1])
                        ratingsNum += 1
                        if predictionIndex < listSize:
                            if trueRating > likeThreshold and prediction[0] > likeThreshold:
                                truePositives += 1
                            elif trueRating < likeThreshold and prediction[0] > likeThreshold:
                                falsePositives += 1
                predictionIndex += 1
        MAE = MAESum / ratingsNum
        allMAESum += MAE
        allPrecisionSum += truePositives/(falsePositives + truePositives)
        allRecallSum += truePositives/testLikedItems
    trueMAE = allMAESum / 5
    truePrecision = allPrecisionSum / 5
    trueRecall = allRecallSum / 5
    print("the MAE is {}".format(trueMAE))
    print("Precision {} recall {}".format(truePrecision, trueRecall))

ratings = pd.read_excel('Data_InCarMusic.xlsx', usecols=[0,1,2], sheet_name='ContextualRating')
musicTracks = pd.read_excel('Data_InCarMusic.xlsx', usecols=[0, 2,3, 7], sheet_name='Music Track') 
originalRatings = ratings.drop_duplicates(subset=["UserID", "ItemID"])

runMAE(originalRatings, 10, 3.5)






