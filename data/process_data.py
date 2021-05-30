import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)    
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    categories = df['categories'].str.split(";", n = 36, expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2]).unique()
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: int(x))
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    return df

def clean_data(df):
    """
    Return Filtered, a column dropped, and duplicates dropped dataframe
    :param df: dataframe
    :return: df: dataframe
    """
    # only 0 or 1 is allowed as this value should be binary.
    df = df[df['related'] != 2]
    # child_alone is just one value, so does not affect any performance
    df = df.drop('child_alone', axis=1)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    '''
    Save input dataframe to sqlite.
    :param df: input dataframe
    :param database_filename: databased file name to be written
    :return: nothing
    '''
    # database_filename = DisasterResponse.db
    engine = create_engine('sqlite:///{}'.format(database_filename))
    table = database_filename.replace('.db', '_table').replace('data/', '')
    # if exits, overwrite table
    df.to_sql(table, con = engine, index=False, if_exists = 'replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        print(df.head())
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