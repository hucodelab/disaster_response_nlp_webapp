import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    input - messages_filepath: X data directory
    categories_filepath: Y data directory
    
    output - a merged dataframe with each message and the
    categories of each message (df)
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath).drop(columns="original")
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, how='left', on='id')
    
    return df

def clean_data(df):
    """
    input - merged df with messages and categories
    
    output - dataframe with messages and the categories in each
    column with the categories in binary data type
    """
    
    # we create a categories' df to manipulate the data
    categories = df.pop('categories')
    
    # create a dataframe of the 36 individual category columns
    categories = categories.str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories[0:1].T
    
    # use this row to extract a list of new column names for categories.
    # we apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    # applymap works well with strings, .apply doesn't
    category_colnames = list(row.applymap(lambda x: x[:-2])[0])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    try:
        categories = categories.applymap(lambda x: int(x[-1]))
    except:
        pass
    
    categories['id'] = df['id']
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.merge(categories, how='left', on='id')
    
    # there are some values in the df with the number 2.
    # Let's drop the 2 values to get binary data
    df[df.columns[4:]] = df[df.columns[4:]].replace(2, np.NaN)
    df.dropna(inplace=True)
    
    # drop duplicates in the dataframe
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    '''
    This function saves the clean dataset into an sqlite database
    '''
    # Save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///{}'.format(database_filename))    
    df.to_sql('YourTableName', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()