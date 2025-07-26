import pandas as pd 
import os

# importing all the models needed

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb

# importing the required metrics for testing

from sklearn.metrics import r2_score, root_mean_squared_error

# importing joblib to save our model in a pkl file

import joblib as jb

if not os.path.exists("cleaned_data.pkl"):
    # Cleaning the dataset
    movies = pd.read_csv("movies_metadata.csv", low_memory=False)
    ratings = pd.read_csv("ratings.csv")

    movies.drop(columns = ['adult', 'belongs_to_collection', 'budget','homepage', 'original_language','overview','release_date', 'revenue', 'runtime',
        'spoken_languages', 'status', 'tagline', 'original_title', 'video', 'vote_count',"poster_path","production_companies","production_countries","imdb_id","popularity"], inplace = True)
    ratings.drop(columns="timestamp",inplace=True)
    movies.drop_duplicates(inplace=True)
    movies.dropna(inplace = True)

    # Changing the datatype of the features and sorting them

    movies["id"] = movies["id"].astype("int32")
    movies["vote_average"] = movies["vote_average"].astype("float32")
    movies["genres"] = movies["genres"].apply(eval)
    movies["genres"] = movies["genres"].apply(lambda x: [y["name"] for y in x])
    movies.sort_values("id", inplace = True)
    movies.reset_index(drop=True,inplace=True)

    # Saving the title of movies in a dict with id and title as key value pair

    title_dict = dict(zip(movies["id"],movies['title']))

    movies.drop(columns = "title", inplace = True)
    ratings.rename(columns = {"movieId": "id"}, inplace = True)

    #  One hot encoding the genre feature

    enc = MultiLabelBinarizer()
    new_data = pd.DataFrame(enc.fit_transform(movies["genres"]), columns = enc.classes_ , index = movies.index)
    movies = pd.concat([movies,new_data],axis=1)
    movies.drop("genres",axis=1,inplace=True)

    # Merging the datasets and making it as for our training and testing

    x = pd.merge(movies, ratings, on = "id", how = "inner")
    column_order = ['userId', 'id','Action', 'Adventure', 'Animation', 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Foreign',
        'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
        'TV Movie', 'Thriller', 'War', 'Western','vote_average','rating']
    x = x[column_order]

    # Feature engineering new features from userId and id

    x["avg_rating_to_movie"] = x.groupby("id")["rating"].transform("mean")
    x["avg_rating_by_user"] = x.groupby("userId")["rating"].transform("mean")
    x["total_rating_by_user"] = x.groupby("userId")["rating"].transform("count")
    x["total_rating_to_movie"] = x.groupby("id")["rating"].transform("count")
    x['rating_deviation'] = x['avg_rating_to_movie'] - x['avg_rating_by_user']

    # Comment this if you don't want cleaned_data.pkl file 
    # Note -> cleaned_data.pkl file is of 2GB approx

    jb.dump(x,"cleaned_data.pkl")
    jb.dump(title_dict,"title.pkl")

    # Seperating our label and features
    y = x["rating"].values
    x.drop(columns= ["rating","userId","id"], inplace=True)
    x = x.values

    # Making train test dataset from train test split
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.4, random_state=1)

    # Checking if model pkl file doesn't exist then train the model and create the pkl file
    
    ''' This is done in a nested if statement because this model trains in approx 1 hr on Amd Ryzen 7 7840hs, 16Gb ram
    so if you have downloaded the model file then it is easier to load the model than to train the model '''

    if not (os.path.exists("model.pkl")):
        # Defining our base estimators for our stacking model
        estimators = [
            ("rf", RandomForestRegressor(n_estimators=80, max_depth=25, n_jobs = -1,random_state=24)),
            ("ridge", Ridge()),
            ("xgbr", xgb.XGBRegressor(
                tree_method= 'approx',
                n_estimators= 750,
                max_depth= 8,
                subsample= 0.7,
                colsample_bytree= 0.8,
                learning_rate= 0.3,
                n_jobs=-1, 
                random_state=42,
                device = "cuda"
                )
            )
        ]

        # Creating the Stacking regressor
        # Note -> I have used Ridge() as meta model you can also try other models 
        model = StackingRegressor(
            estimators=estimators,
            final_estimator= Ridge(),
            cv= 5,
            verbose=3
        )

        # Fitting our model 
        model.fit(x_train, y_train)

        # Dumping it on a pkl file
        # Note -> this is approx 7GB fiel so comment it out if you haven't enough memory
        jb.dump(model,"model.pkl")

    else:
        # Loading the model pkl files
        model = jb.load("model.pkl")

        # Inferencing on the test dataset
        pred = model.predict(x_test)

        # This line prints the r2 score and rmse on test dataset
        print(f"The r2 score is {r2_score(y_test,pred)} and the rmse is {root_mean_squared_error(y_test,pred)}")


else:
    x = jb.load("cleaned_data.pkl")

    # Seperating our label and features
    y = x["rating"].values
    x.drop(columns= ["rating","userId","id"], inplace=True)
    x = x.values

    # Making train test dataset from train test split
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.4, random_state=1)

    # Checking if model pkl file doesn't exist then train the model and create the pkl file
    ''' This is done in a nested if statement because this model trains in approx 1 hr on Amd Ryzen 7 7840hs, 16Gb ram
    so if you have downloaded the model file then it is easier to lead the model than to train the model '''

    if not (os.path.exists("model.pkl")):
        # Defining our base estimators for our stacking model
        estimators = [
            ("rf", RandomForestRegressor(n_estimators=80, max_depth=25, n_jobs = -1,random_state=24)),
            ("ridge", Ridge()),
            ("xgbr", xgb.XGBRegressor(
                tree_method= 'approx',
                n_estimators= 750,
                max_depth= 8,
                subsample= 0.7,
                colsample_bytree= 0.8,
                learning_rate= 0.3,
                n_jobs=-1, 
                random_state=42,
                device = "cuda"
                )
            )
        ]

        # Creating the Stacking regressor
        # Note -> I have used Ridge() as meta model you can also try other models 
        model = StackingRegressor(
            estimators=estimators,
            final_estimator= Ridge(),
            cv= 5,
            verbose=3
        )

        # Fitting our model 
        model.fit(x_train, y_train)

        # Dumping it on a pkl file
        # Note -> this is approx 7GB file so comment it out if you haven't enough memory
        jb.dump(model,"model.pkl")

    else:
        # Loading the model pkl files
        model = jb.load("model.pkl")

        # Inferencing on the test dataset
        pred = model.predict(x_test)

        # This line prints the r2 score and rmse on test dataset
        print(f"The r2 score is {r2_score(y_test,pred)} and the rmse is {root_mean_squared_error(y_test,pred)}")