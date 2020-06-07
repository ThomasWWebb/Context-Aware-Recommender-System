import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
import scipy
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os

def getAllUserMeans(contextualRatings):
    return contextualRatings.groupby(by="UserID",as_index=True)['Rating'].mean()

def getRatingProportions(activeUserID, ratings, contexts):
    userRatings = getUserRatings(activeUserID, ratings)
    maxRatings = userRatings['Rating'].size
    ratingProportions = {}
    for context in contexts:
        ratingProportions[context] = userRatings[userRatings['DrivingStyle'] == context]['Rating'].size / maxRatings
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
                    if len(newOtherArray) == 1 and similarity == 1:
                        similarity = 0.5
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

def getUserPredictions(activeUserID, ratedMusicTracks, similarityTuples, weights):
    predictions = []
    for itemIndex in ratedMusicTracks:
        weightSum = 0
        predictionSum = 0
        for index in range(len(similarityTuples)):
            if similarityTuples[index] == ():
                continue
            similarities, userMeans, userItemTable = similarityTuples[index]
            individualPrediction = getItemPrediction(itemIndex, similarities, userMeans, userItemTable, activeUserID)
            predictionSum += weights[contexts[index]] * individualPrediction
            weightSum += weights[contexts[index]]
        prediction = predictionSum / weightSum
        predictions.append([itemIndex, prediction])
    return sorted(predictions, key = lambda x: x[1], reverse=True)

def getWeights(activeUserID, ratings, contexts, userContext, contextValues):
    ratingProportions = getRatingProportions(activeUserID, ratings, contexts)
    weights = {}
    for contextIndex in range(len(contexts)):
        context = contexts[contextIndex]
        distance = abs(contextValues[contexts.index(userContext)][contextIndex])
        weights[context] = (1/distance)*ratingProportions[context]
    return weights

def recommend(activeUserID, userContext, ratings, contexts, contextValues):
    
    #find the set of items to predict for
    allSeenItems = ratings[(ratings.UserID == activeUserID) & (ratings.DrivingStyle == userContext)]
    allSeenItems = allSeenItems['ItemID'].unique() 
    allUnseenItems = ratings[~ratings['ItemID'].isin(allSeenItems)]
    allUnseenItems = allUnseenItems['ItemID'].unique()

    #get the weights for each context
    weights = getWeights(activeUserID, ratings, contexts, userContext, contextValues)
    similarityTuples = []
    for context in contexts:
        contextRatings = ratings[ratings['DrivingStyle'] == context]
        userMeans = getAllUserMeans(contextRatings)
        if activeUserID not in userMeans.index:
            similarityTuples.append(())
            continue
        ratedMusicTracks = contextRatings['ItemID'].unique()
        userItemTable = contextRatings.pivot_table(index='UserID', columns='ItemID', values='Rating').fillna(0)

        activeUserMeanRating = userMeans[activeUserID]

        similarities = getUserSimilarities(userItemTable, activeUserID)
        similarityTuples.append((similarities, userMeans, userItemTable))
    predictions = getUserPredictions(activeUserID, allUnseenItems, similarityTuples, weights)
    return predictions

def addDrivingStyles(ratings, contexts):
    key = "DrivingStyle"
    for index in range(ratings['ItemID'].size):
        if ratings.at[index, 'sleepiness'] == "sleepy" and pd.isnull(ratings.at[index, key]):
            ratings.at[index, key] = "relaxed driving"
        elif ratings.at[index, 'sleepiness'] == "awake" and pd.isnull(ratings.at[index, key]):
            ratings.at[index, key] = "sport driving"
        elif ratings.at[index, 'mood'] == "lazy" and pd.isnull(ratings.at[index, key]):
            ratings.at[index, key] = "relaxed driving"
        elif ratings.at[index, 'mood'] == "active" and pd.isnull(ratings.at[index, key]):
            ratings.at[index, key] = "sport driving"
        elif ratings.at[index, 'trafficConditions'] == "traffic jam" and pd.isnull(ratings.at[index, key]):
            ratings.at[index, key] = "relaxed driving"
        elif ratings.at[index, 'trafficConditions'] == "free road" and pd.isnull(ratings.at[index, key]):
            ratings.at[index, key] = "sport driving"
        elif pd.isnull(ratings.at[index, key]):
            ratings.at[index, key] = random.choice(contexts)

    ratings = ratings.drop(columns=['sleepiness', 'mood', 'trafficConditions'])
    return ratings.dropna()

def getLikedItems(ratingSet, likeThreshold):
    likedItemsSum = 0
    for index, row in ratingSet.iterrows():
        if row['Rating'] > 3.5:
            likedItemsSum += 1
    return likedItemsSum

def runEvaluation(originalRatings, contexts, contextValues, listSize, likeThreshold):
    kf = KFold(n_splits=5, shuffle=True)
    foldCount = 0
    allMAESum = 0
    allPrecisionSum = 0
    allRecallSum = 0
    for train_index, test_index in kf.split(originalRatings):
        foldCount += 1
        print("Running against fold {}".format(foldCount))
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
            userGroup = groupTest.get_group(userID).groupby('DrivingStyle')
            userSum += 1
            for context in userGroup.groups.keys():
                #get the group for that context
                contextGroup = userGroup.get_group(context).set_index('ItemID')
                #get the predictions
                predictions = recommend(userID, context, ratings, contexts, contextValues)
                predictionIndex = 0
                currentTruePositives = 0
                for prediction in predictions:
                    #if the prediction has a true value calculate the error
                    if prediction[0] in contextGroup.index:
                        trueRating = contextGroup.loc[prediction[0]]['Rating']
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
    print("MAE {} Precision {} Recall {}".format(trueMAE, truePrecision, trueRecall))

def displayRecommendations(recommendations, musicTracks):
    topRecommendations = pd.DataFrame(recommendations[:10], columns=["ItemID", "Rating"])
    formattedTracks = pd.merge(musicTracks, topRecommendations, on='ItemID').sort_values(by=['Rating'], ascending=False).drop(columns=['ItemID', 'category_id'])
    formattedTracks.index = np.arange(1, len(formattedTracks) + 1)
    print(formattedTracks)
    print("Press enter to continue:")
    key = input()

def recommendForUser(userID, ratings, contexts, contextValues, musicTracks):
    choice = ""
    context = ""
    os.system('cls')
    while choice != "x":
        print("singed in as {}".format(userID))
        print("(Enter x to return)")
        print("What is your current driving style?")
        print("1. Sport")
        print("2. Relaxed")
        choice = input()
        if choice == "1":
            recommendations = recommend(userID, "sport driving" ,ratings, contexts, contextValues)
            displayRecommendations(recommendations, musicTracks)
        elif choice == "2":
            recommendations = recommend(userID, "relaxed driving" ,ratings, contexts, contextValues)
            displayRecommendations(recommendations, musicTracks)
        os.system('cls')
        
        

contexts = ["sport driving", "relaxed driving"]
contextValues = [[1, 2], 
                [2, 1]]
#ratings = pd.read_excel('Data_InCarMusicRandom.xlsx', usecols=[0,1,2,11], sheet_name='ContextualRating')
ratings = pd.read_excel('Data_InCarMusic.xlsx', usecols=[0,1,2,3,5, 8, 9], sheet_name='ContextualRating')
ratings = addDrivingStyles(ratings, contexts)
musicTracks = pd.read_excel('Data_InCarMusic.xlsx', usecols=[0, 2,3, 7], sheet_name='Music Track') 
genres = pd.read_excel('Data_InCarMusic.xlsx', usecols=[0,1], sheet_name='Music Category') 
musicTracks = pd.merge(musicTracks, genres, on='category_id')
originalRatings = ratings.drop_duplicates(subset=["UserID", "ItemID", "DrivingStyle"])
allUsers = ratings['UserID'].unique()

choice = ""
while choice != "3":
    os.system('cls')
    print("Welcome to the music CARS")
    print("Options:")
    print("1. Sign in and receive recommendations")
    print("2. Run Evaluations")
    print("3. Quit")
    choice = input()
    if choice == "1":
        userIDInput = ""
        signedIn = False
        os.system('cls')
        while userIDInput != "x":
            print("(Enter z to view all user IDs)")
            print("(Enter x to return to the main menu)")
            print("Please enter your user ID:")
            userIDInput = input()
            if userIDInput == "z":
                print(allUsers)
                print("Press enter to continue")
                input()
                os.system('cls')
            elif userIDInput.isnumeric():
                if int(userIDInput) in allUsers:
                    os.system('cls')
                    print("signed in as {}".format(userIDInput))
                    recommendForUser(int(userIDInput), ratings, contexts, contextValues, musicTracks)
                else:
                    os.system('cls')
                    print("please enter a valid user ID")
        os.system('cls')

    elif choice == "2":
        os.system('cls')
        runEvaluation(originalRatings, contexts, contextValues, 10, 3.5)
        print("Press enter to continue")
        input()






