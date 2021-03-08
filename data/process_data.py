import sys
import pandas as pd
import sqlite3

def load_data(messages_filepath, categories_filepath):
    '''
    Load Data From CSV Files
    
    Args:
        messages_filepath : Path to messages file
        categories_filepath : Path to categories file

    Returns:
        df : DataFrame after merging both messages ans categories file
    '''
    # Loading Messages file as pandas DataFrame
    messages = pd.read_csv(messages_filepath)

    # Loading Categories file as pandas DataFrame
    categories = pd.read_csv(categories_filepath)

    # Merging messages and categories DateFrame
    df = pd.merge(categories, messages,on="id")

    return df

def clean_data(df):
    '''
    Cleaning the DataFrame

    Args:
        df : merged DataFrame returned by load_data function
    Returns:
        df : 
    '''
    # Creating Different column for different category
    categories = df['categories'].str.split(';',expand=True)
    # Changing the names of the newly created columns
    row = categories.loc[0]
    category_colnames = ['category_'+col.split('-')[0] for col in row]
    categories.columns = category_colnames
    
    # Changing the values to 0 or 1 in each category column and also changing datatype to int
    for column in categories.columns:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1] )
        # convert column from string to numeric
        categories[column] = categories[column].astype('int64')

    # Droping old category column of dataframe
    df.drop('categories', axis=1, inplace=True)

    # concatenating new category dataframe to df
    df = pd.concat([df, categories] , axis=1)
    
    # Droping Rows with Duplicates Data
    df.drop_duplicates(inplace=True)
    
    # As there are some rows with different lebal for same feature, 
    # It is not good for our modal So removing these rows will be a better choice
    # as no of rows to be romove is 72 which is very less as comp to 26216 rows.
    r_rows = list(df['id'].value_counts()[(df['id'].value_counts() >1)].index)
    
    for row in r_rows:
        df = df[(df['id'] != row)]
    
    #Removing rows with category_related = 2
    df = df[df['category_related']!=2]
    
    return df



def save_data(df, database_filename):
    '''
    Save Cleaned Data to a SQLite DataBase

    Args:
        df : DataFrame which is to be Saved
        database_filename : Path to the DataBase File
    '''
    # Creating sqlite DataBase Connection Object
    conn = sqlite3.connect(database_filename)
    
    # Dumping all cleaned data to sqlite database
    df.to_sql('Messages', conn, index=False)


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