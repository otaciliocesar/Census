

# Import pandas with alias
import pandas as pd

# Read in the census dataframe
census = pd.read_csv('census_data.csv', index_col=0)

"""1. The census dataframe is composed of simulated census data to represent demographics of a small community in the U.S. Call the .head() method on the census dataframe and print the output to view the first five rows."""
"""2. Review the dataframe description and values returned by .head() to assess the variable types of each of the variables. This is an important step to understand what preprocessing will be necessary to work with the data."""
#print(census.head())

"""3. Compare the values returned from the .head() method with the data types of each variable by calling .dtypes on the census dataframe and print the result."""
#print(census.dtypes)

"""4. The manager of the census would like to know the average birth year of the respondents. We were able to see from .dtypes that birth_year has been assigned the str datatype whereas it should be expressed in int.
Print the unique values of the variable using the .unique() method."""
birth_year = census['birth_year'].unique()
#print(birth_year)

"""5. There appears to be a missing value in the birth_year column. With some research you find that the respondent’s birth year is 1967.
Use the .replace() method to replace the missing value with 1967, so that the data type can be changed to int. Then recheck the values in birth_year by calling the .unique() method and printing the results."""

census['birth_year'] = census['birth_year'].replace('missing', 1967)

"""6. Now that we have adjusted the values in the birth_year variable, change the datatype from str to int and print the datatypes of the census dataframe with .dtypes. """
census['birth_year'] = pd.to_numeric(census['birth_year'])
#print(census['birth_year'].unique())
#print(census.dtypes)

"""7. Having assigned birth_year to the appropriate data type, print the average birth year of the respondents to the census using the pandas .mean() method."""
birth_mean = census['birth_year'].mean()
#print(birth_mean)


"""8. Your manager would like to set an order to the higher_tax variable so that: strongly disagree < disagree < neutral < agree < strongly agree.
Convert the higher_tax variable to the category data type with the appropriate order, then print the new order using the .unique() method."""
census['higher_tax'] = pd.Categorical(
  census['higher_tax'],
  categories=['strongly disagree', 'disagree', 'neutral', 'agree', 'strongly agree'], ordered=True
)
higher_tax = census['higher_tax'] = census['higher_tax'].astype('category')
print(census['higher_tax'])
print(census.dtypes)
print(census['higher_tax'].unique())


"""9. Your manager would also like to know the median sentiment of the respondents on the issue of higher taxes for the wealthy. Label encode the higher_tax variable and print the median using the pandas .median() method."""

numeric_values = census['higher_tax'].cat.codes
median_tax = numeric_values.median()
print("Median: ", median_tax)
median_category = census['higher_tax'].cat.categories[int(median_tax)]
print("Mediana (categórica):", median_category)

"""Your manager is interested in using machine learning models on the census data in the future. To help, let’s One-Hot Encode marital_status to create binary variables of each category. Use the pandas get_dummies() method to One-Hot Encode the marital_status variable.

Print the first five rows of the new dataframe with the .head() method. Note that you’ll have to scroll to the right or expand the web-browser to see the dummy variables."""
census_encoded = pd.get_dummies(census, columns=[
  'marital_status'], prefix='status')
print(census_encoded.head())

"""11. a. Create a new variable called marital_codes by Label Encoding the marital_status variable. This could help the Census team use machine learning to predict if a respondent thinks the wealthy should pay higher taxes based on their marital status."""
census['marital_codes'] = census['marital_status'].astype('category').cat.codes
print(census['marital_codes'])

"""11. b. Create a new variable called age_group, which groups respondents based on their birth year. The groups should be in five-year increments, e.g., 25-30, 31-35, etc. Then label encode the age_group variable to assist the Census team in the event they would like to use machine learning to predict if a respondent thinks the wealthy should pay higher taxes based on their age group."""
current_year = 2024
census['age'] = current_year - census['birth_year']


census['marital_codes'] = census['marital_status'].astype('category').cat.codes


census['age_group'] = pd.cut(
    census['age'],
    bins=range(17, 83, 5),  
    right=False,            
    labels=[
        '18-23', '23-28', '28-33', '33-38', '38-43', '43-48', 
        '48-53', '53-58', '58-63', '63-68', '68-73', '73-78', '78-83'
    ]
)


census['age_group_codes'] = census['age_group'].astype('category').cat.codes


print(census)
