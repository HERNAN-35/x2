# Libraries 
from fastapi import FastAPI
import uvicorn
import pandas as pd
import numpy as np
import json
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from collections import Counter


# Import all the datasets we need for the functions performance
df_F12 = pd.read_parquet('Dataset/F12.parquet')
df_F345 = pd.read_parquet('Dataset/F345.parquet')
df_FML1_for_Model = pd.read_parquet('Dataset/FML1.parquet')

# Functions and procedures that we need for the Endpoints 1 and 2

# Convert the column genre in df_F12 into list
def convert_to_list(genres_array):
    return [element.strip("'") for element in genres_array]
# Apply the function to the column 'genres' and replace the column
df_F12['genres'] = df_F12['genres'].apply(convert_to_list)

# Obtain the unique genres list from the dataframe df_F12
unique_genres = set()
for genres_list in df_F12['genres']:
    unique_genres.update(genres_list)
unique_genres_list = list(unique_genres)

# Function to normalize a list of words.
def normalize_list_of_words(sentences_list):
    # Function to normalize a list of sentences capitalizing the first letter of each word.
    normalized_sentences = []
    for sentence in sentences_list:
        words = sentence.split()  # We divide the sentence into words
        normalized_words = [word.capitalize() for word in words]  # We capitalize each word
        normalized_sentence = ' '.join(normalized_words)  # Join the words into a sentence
        normalized_sentences.append(normalized_sentence)
    return normalized_sentences

# Function to normalize a sentence
def capitalize_first_words_in_sentence(sentence):
    # Function to capitalize the first letter of each word in a sentence
    words = sentence.split()  # Split the sentence into words
    capitalized_words = [word[0].capitalize() + word.lower()[1:] for word in words]  # Capitalize the first letter of each word
    return ' '.join(capitalized_words)  # Join the words into a capitalized sentence

# Create an instance o the class FastAPI
app = FastAPI()




# Endpoint 1
@app.get("/PlayTimeGenre/{genre}")
def PlayTimeGenre(genre: str):

    """
    Returns the release year with the most played hours for the given genre.

    Parameters:
        genre (str): The genre for which the release year with the most played hours is requested.

    Returns:
        dict: A dictionary containing the release year with the most played hours for the specified genre.
              Example: {"Release year with the most played hours for Genre X": 2013}
    """

    # This function returns the release year with the most played hours for the given 'genre'.
    # Example of return: {"Release year with the most played hours for Genre X" : 2013}

    # We only use the variables that we need from the DataFrame df_F12
    df_f1 = df_F12[['genres','release_year','sum_playtime_forever']]

    # Normalize unique_genres_list
    normalized_genres = normalize_list_of_words(unique_genres_list)

    #We need to mke sure that the genre inserted by the user is a string
    if isinstance(genre, str):
        # The first thing that we need to do is to validate that the genre entered by the user is in the list of genres.
        # We need to normalize the genre entered by the user.
        norm_genre = capitalize_first_words_in_sentence(genre)

        #Now, let's look for that genre in the normalized list of genres:
        if norm_genre in normalized_genres:

            # Let's suppose that we are going to look for the genre 'norm_genre' as an input from the user.
            genre_to_find = norm_genre

            # There are two genres that need special attention: "Free to Play","RPG". Because they are not normalized in the
            # original dataframe where we are going to search later. So let's transform them:
            if norm_genre == 'Free To Play':
                genre_to_find = 'Free to Play'
            elif norm_genre == 'Rpg':
                genre_to_find = 'RPG'

            # We create a mask to select the rows that contain the desired genre and we take care of the NaN values
            mask = df_f1['genres'].apply(lambda x: genre_to_find in x)

            # We filtrate the DataFrame with the mask
            df_f1_by_genre = df_f1[mask]

            # We need to do a group by 'release_year' summing 'sum_playtime_forever'
            grouped_df_f1_by_year = df_f1_by_genre.groupby('release_year')['sum_playtime_forever'].sum().reset_index()

            # Now we only need the 'release_year' and 'sum_playtime_forever'
            df1genre = grouped_df_f1_by_year[['release_year','sum_playtime_forever']]

            # Now we sort by the 'sum_playtime_forever' to have the year with most played hours:
            df1genre.sort_values(by='sum_playtime_forever', ascending=False, inplace=True)

            # We reset the index
            df1genre.reset_index(drop=True, inplace=True)

            year_most_hours_played = df1genre.iloc[0,0]
            max_sum_playtime_forever = df1genre.iloc[0,1]

            # Response
            return {f"Release year with the most played hours for Genre {genre_to_find}": int(year_most_hours_played)}
            
        else:
            return "The genre entered is not valid. Please try again."
    else:
        return "Plese insert a genre as a string value (Text)."


# Endpoint 2
@app.get("/UseForGenre/{genre}")
def UseForGenre(genre:str):

    """
    Returns the user who has accumulated the most played hours for the given genre
    and a list of accumulated playtime by release year.

    Parameters:
        genre (str): The genre for which the user with the most played hours and playtime by year is requested.

    Returns:
        dict: A dictionary containing the user with the most played hours and a list of playtime by year.
              Example: {"User with the most played hours for Genre X": "us213ndjss09sdf",
                        "Playtime": [{"Year": 2013, "Hours": 203}, {"Year": 2012, "Hours": 100}, {"Year": 2011, "Hours": 23}]}

    """

    # This function return the user who has accumulated the most played hours for the given `genre` and a list of 
    # the accumulated playtime by release year.
    #Example of return: {"User with the most played hours for Genre X" : us213ndjss09sdf, 
    #                     "Playtime":[{Year: 2013, Hours: 203}, {Year: 2012, Hours: 100}, {Year: 2011, Hours: 23}]}

    # We only use the variables that we need from the DataFrame df_F12
    df_f2 = df_F12[['user_id','genres','release_year','sum_playtime_forever']]

    # Normalize unique_genres_list
    normalized_genres = normalize_list_of_words(unique_genres_list)

    #We need to mke sure that the genre inserted by the user is a string
    if isinstance(genre, str):
    
        # The first thing that we need to do is to validate that the genre entered by the user is in the list of genres.
        # We need to normalize the genre entered by the user.
        norm_genre = capitalize_first_words_in_sentence(genre)

        #Now, let's look for that genre in the normalized list of genres:
        if norm_genre in normalized_genres:

            # Let's suppose that we are going to look for the genre 'norm_genre' as an input from the user.
            genre_to_find = norm_genre

            # There are two genres that need special attention: "Free to Play","RPG". Because they are not normalized in the
            # original dataframe where we are going to search later. So let's transform them:
            if norm_genre == 'Free To Play':
                genre_to_find = 'Free to Play'
            elif norm_genre == 'Rpg':
                genre_to_find = 'RPG'

            # We create a mask to select the rows that contain the desired genre and we take care of the NaN values
            mask = df_f2['genres'].apply(lambda x: genre_to_find in x)

            # We filtrate the DataFrame with the mask
            df_f2_by_genre = df_f2[mask]

            # We need to do a group by 'user_id' and 'release_year' summing 'sum_playtime_forever'
            grouped_df_f2_by_user_year = df_f2_by_genre.groupby(['user_id','release_year'])['sum_playtime_forever'].sum().reset_index()

            # Also we are going to group by 'user_id' summing 'sum_playtime_forever' in order to find the user that played the most.
            grouped_df_f2_by_user = df_f2_by_genre.groupby(['user_id'])['sum_playtime_forever'].sum().reset_index()

            # Now we sort by the 'sum_playtime_forever' to have the year with most played hours:
            grouped_df_f2_by_user.sort_values(by='sum_playtime_forever', ascending=False, inplace=True)

            # We reset the index
            grouped_df_f2_by_user.reset_index(drop=True, inplace=True)

            #User with more hours played
            user_most_hours_played = grouped_df_f2_by_user.iloc[0,0]

            # Now we are going to make a mask in the dataframe 'grouped_df_f2_by_user_year' just to have the data of that user
            # that had the maximum hours played
            mask = grouped_df_f2_by_user_year['user_id'] == user_most_hours_played
            resultF2 = grouped_df_f2_by_user_year[mask]

            #Now, we create a list of dictionaries for the year and sum_playtime_forever
            playtime_list = resultF2.rename(columns={'release_year': 'Release Year', 'sum_playtime_forever': 'Hours'})[['Release Year', 'Hours']].to_dict(orient='records')

            # Response
            return {
                f"User with the most played hours for Genre {genre_to_find}": user_most_hours_played,
                "Playtime": playtime_list
            }
        else:
            return "The genre entered is not valid. Please try again."
    else:
        return "Plese insert a genre as a string value (Text)."

# Endpoint 3
@app.get("/UsersRecommend/{year}")
def UsersRecommend(year:int):

    """
    Returns the top 3 games MOST recommended by users for the given year (reviews.recommend = True and positive/neutral comments).

    Parameters:
        year (int): The year for which the top 3 recommended games are requested.

    Returns:
        list of dict: A list containing the top 3 recommended games with their ranks.
                      Example: [{"Rank 1": item1}, {"Rank 2": item2}, {"Rank 3": item3}]
    """

    # This function returns the top 3 games MOST recommended by users for the given `year` (reviews.recommend = True and positive/neutral comments).
    # Example of return: [{"Rank 1" : item1}, {"Rank 2" : item2},{"Rank 3" : item3}]

    # We only use the variables that we need from the DataFrame df_F345
    df_f3 = df_F345[['item_id','title','recommend','sentiment_analysis','review_year']]

    # Eliminate all the rows that have 'recommend = False' and have sentiment_analysis = 0
    mask = (df_f3['recommend'] == True) & (df_f3['sentiment_analysis'] != 0)
    df_f3f = df_f3[mask]
    df_f3 = df_f3f.reset_index(drop=True)

    # First, we need to make sure that the year inserted by the user is a number
    if type(year) == int: 
        # Now, let's select only the data for the year given by the user.
        mask = df_f3['review_year'] == year
        df_f3_review_year = df_f3[mask].reset_index(drop=True)

        # Group by 'title' summing all the sentiment_analysis and sort the dataframe by sum of sentiment_analysis.
        grouped_df_f3_review_year = df_f3_review_year.groupby(['title'])['sentiment_analysis'].sum().reset_index()

        # Now we sort by the 'sentiment_analysis' to have the items with the highest sentiment_analysis:
        grouped_df_f3_review_year.sort_values(by='sentiment_analysis', ascending=False, inplace=True)

        # We need to verify if the Dataframe is not empty, which means that there are no reviews for that year:
        if not grouped_df_f3_review_year.empty:
            #Rank 1
            item_Rank_1 = grouped_df_f3_review_year.iloc[0,0]
            #Rank 2
            item_Rank_2 = grouped_df_f3_review_year.iloc[1,0]
            #Rank 3
            item_Rank_3 = grouped_df_f3_review_year.iloc[2,0]

            #Now, we create the dataframe that we are going to convert into JSON Format
            dataF3 = {
                'Rank': ['Position 1', 'Position 2', 'Position 3'],
                'title': [item_Rank_1, item_Rank_2, item_Rank_3]
            }

            # Response
            return [{"Rank " + str(index + 1): item} for index, item in enumerate(dataF3['title'])]

        else:
            return "The year inserted has not reviews to calculate the ranking of the most recommended items. Please try with another year."
    else:
        return "Please insert a valid year as an integer number."

# Endpoint 4
@app.get("/UsersNotRecommend/{year}")
def UsersNotRecommend(year:int):

    """
    Returns the top 3 games LEAST recommended by users for the given year (reviews.recommend = False and negative comments).

    Parameters:
        year (int): The year for which the top 3 least recommended games are requested.

    Returns:
        list of dict: A list containing the top 3 least recommended games with their ranks.
                      Example: [{"Rank 1": X}, {"Rank 2": Y}, {"Rank 3": Z}]
    """

    # This function returns the top 3 games LEAST recommended by users for the given `year` (reviews.recommend = False and negative comments).
    # Example of return: [{"Rank 1" : X}, {"Rank 2" : Y},{"Rank 3" : Z}]

    # We only use the variables that we need from the DataFrame df_F345
    df_f4 = df_F345[['item_id','title','recommend','sentiment_analysis','review_year']]

    # Eliminate all the rows that have 'recommend = True'.
    mask = (df_f4['recommend'] == False) & (df_f4['sentiment_analysis'] == 0)
    df_f4f = df_f4[mask]
    df_f4 = df_f4f.reset_index(drop=True)

    # First, we need to make sure that the year inserted by the user is a number
    if type(year) == int: 
        # Now, let's select only the data for the year given by the user.
        mask = df_f4['review_year'] == year
        df_f4_review_year = df_f4[mask].reset_index(drop=True)

        # We need to verify if the Dataframe is not empty, which means that there are no reviews for that year:
        if not df_f4_review_year.empty:

            # We count the values considering the 'item_id'
            counter = df_f4_review_year['title'].value_counts()

            # Three east recommended games
            top_3_least_recommended = counter.head(3)

            #Response
            return [{"Rank " + str(index + 1): item} for index, item in enumerate(top_3_least_recommended.index)]
            
        else:
            return "The year inserted has not reviews to calculate the ranking of the most recommended items. Please try with another year."
    else:
        return "Please insert a valid year as an integer number."


# Endpoint 5
@app.get("/Sentiment_analysis/{year}")
def Sentiment_analysis(year:int):

    """
    Based on the release year, returns a dictionary with the count of user review records categorized with sentiment analysis.

    Parameters:
        year (int): The release year for which sentiment analysis statistics are requested.

    Returns:
        dict: A dictionary containing the count of reviews categorized as "Negative," "Neutral," and "Positive."
              Example: {"Negative": 182, "Neutral": 120, "Positive": 278}
    """

    # Based on the release `year`, it returns a list with the count of user review records categorized with sentiment analysis.
    # Example of return: {"Negative": 182, "Neutral": 120, "Positive": 278}

    # We only use the variables that we need from the DataFrame df_F345
    df_f5 = df_F345[['sentiment_analysis','release_year']]

    # First, we need to make sure that the year inserted by the user is a number
    if type(year) == int: 
        # Now, let's select only the data for the year given by the user.
        mask = df_f5['release_year'] == year
        df_f5_review_year = df_f5[mask].reset_index(drop=True)

        # We need to verify if the Dataframe is not empty, which means that there are no reviews for that year:
        if not df_f5_review_year.empty:
            # We count the values considering the 'sentiment_analysis'
            counter = df_f5_review_year['sentiment_analysis'].value_counts().sort_index()

            # Response
            return {
                "Negative": int(counter.get(0, 0)),
                "Neutral": int(counter.get(1, 0)),
                "Positive": int(counter.get(2, 0))
            }

        else:
            return "The year inserted has not reviews to calculate the categories of sentiment analysis. Please try with another year."
    else:
        return "Please insert a valid year as an integer number."

# Endpoint 6
@app.get("/Game_Recommendation/{item_id}")
def Game_Recommendation(item_id:int):
    """
    Recommends similar games using Locality-Sensitive Hashing (LSH).

    Parameters:
        item_id (int): The ID of the game for which recommendations are requested.
        engine (LSH Engine): The LSH engine used for similarity search.

    Returns:
        list: A list of recommended games in JSON format.
    """
    # Combine text columns into a single column for vectorization
    df_FML1_for_Model['combined_features'] = df_FML1_for_Model['tags']

    # We need to use a vectorizer and create the matrix filling the NaN with ''
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_FML1_for_Model['combined_features'].fillna(''))

    # We set the number of dimensions in the TF-IDF matrix
    num_dimensions = tfidf_matrix.shape[1]

    # Also, we set the number of hash functions (random binary projections)
    num_hash_functions = 4  # We can adjust this value according to our needs

    # Create the LSH engine
    engine = Engine(num_dimensions, lshashes=[RandomBinaryProjections('rbp', num_hash_functions)])

    # Loop through each row and its index in the TF-IDF matrix
    for i, row in enumerate(tfidf_matrix):

        # Extract the item_id from the corresponding row in the DataFrame 'df_FML1_for_Model'
        game_id = df_FML1_for_Model.iloc[i]['item_id']

        # Store the TF-IDF vector as a flattened array in the LSH engine, associated with the game ID
        engine.store_vector(row.toarray().flatten(), data=game_id)

    # By entering the `item_id`, we should receive a list with 5 recommended games similar to the one entered.
    # We review the user enter a number
    if (type(item_id) == float) | (type(item_id) == int):
        # Now, let's review the item_id exists
        if item_id in df_FML1_for_Model['item_id'].values:
                # We Get the LSH index of the input game

                # Query the TF-IDF matrix to get the TF-IDF vector for the input game
                query = tfidf_matrix[df_FML1_for_Model['item_id'] == item_id].toarray().flatten()

                # Use LSH to find similar games (neighbors) to the input game
                neighbors = engine.neighbours(query)

                # Recommendations based on LSH

                # Extract the game IDs of the recommended games, excluding the input game, and limit to the top 5
                recommended_game_ids = [neighbor[1] for neighbor in neighbors if neighbor[1] != item_id][:5]

                # Filter the DataFrame to get details of the recommended games (titles)
                recommended_games = df_FML1_for_Model[df_FML1_for_Model['item_id'].isin(recommended_game_ids)][['title']]

                # Construct the list of recommendations in JSON format
                result = [{'Rec {}'.format(i + 1): game} for i, game in enumerate(recommended_games['title'])]

                return result
        else:
            return 'The item_id does not exists. Please try again.'
    else:
        return "The item_id must be a number, please enter a new item_id"