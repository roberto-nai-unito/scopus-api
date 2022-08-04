from datetime import datetime

from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt

import pybliometrics
from pybliometrics.scopus.utils import config
from pybliometrics.scopus import ScopusSearch

import pandas as pd

import os
import csv
import numpy as np

import re

separator = ";"
result_csv_dir = "result"

dictionary = {}

# Obtain Scimago Journal Rating DF from CSVs (1999-2020)
def scimagojr_to_df(year):
    """Obtain Scimago Journal Rating DF from CSV"""
    scimagojr_csv_dir = "sjr"
    global separator
    scimagojr_file_name_prefix = "scimagojr_"
    scimagojr_file_ext = ".csv"
    scimagojr_file_name = scimagojr_file_name_prefix + str(year) + scimagojr_file_ext
    csv_path = scimagojr_csv_dir + os.sep + scimagojr_file_name
    check = os.path.isfile(csv_path) # check if file exists
    if (check == True):
      # silent warning with low_memory=False)
      scimagojr_df = pd.read_csv(csv_path, delimiter=separator, header=0, low_memory=False, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
      # print(list(scimagojr_df))    # debug check
      # scimagojr_df.info()          # debug check
      # print(scimagojr_df.head())   # debug check
      return scimagojr_df
    else:
      print("Data not found")
      return pd.DataFrame()

# Craeate network of authors
def authors_explore(scopus_result):
    """Craeate network of authors"""
    global result_csv_dir
    authors = []
    for i in scopus_result.results:
        if i.author_ids != None:
            authors.append(i.author_ids.split(';'))
            # scopus_authors = [i.author_ids.split(';') for i in scopus_result.results]
    # print(scopus_authors)

    # create a list of pairwise combinations
    combs = [list(combinations(i,2)) for i in authors]
    edges = [i for j in combs for i in j]

    # create a network
    G = nx.Graph()
    G.add_edges_from(edges)
    print(nx.info(G)) # info about the graph

    # nx.draw(G, node_size=2)
    nx.draw(G, node_size=2)
    # plt.savefig('network.pdf', bbox_inches='tight', figsize=(50,50))
    path_result = result_csv_dir + '/' + 'network.pdf'
    plt.savefig(path_result, dpi='figure', format='pdf')
    print("Done in:",path_result)

# Join Articles DF with Scimago Journal Rating DF (key=publicationName)
def scopus_join_scimagojr(year):
    """Join Articles DF with Scimago Journal Rating DF (key=publicationName)"""
    global result_csv_dir
    result_scopus_scimago_all_csv_file = "result_scopus_join_scimago_large.csv"
    global separator
    print()
    print("Join Articles with ScimagoJR, year " + str(year) + "...")
    scimagojr_df = scimagojr_to_df(year)
    if len(scimagojr_df) > 0:
      articles_df = pd.merge(scopus_df, scimagojr_df, how='left', left_on='publicationName', right_on='Title')
      articles_df.insert(loc=0, column='Num', value=np.arange(len(articles_df))) # add a first column with rownumber
    result_path = result_csv_dir + os.sep + result_scopus_scimago_all_csv_file
    # save pandas.DataFrame to csv
    articles_df.to_csv(result_path)
    print("Done in:", result_path)
    return articles_df

# From Scopus join Scimago joint, extract the Scimago Categories
def scopus_to_categories(articles_df):
  """Extract the Scimago Categories"""
  global dictionary
  global result_csv_dir
  result_scopus_categories = "result_categories.csv"
  categories_text = ""
  if len(articles_df) > 0:
    categories_df = articles_df[["Categories"]] # Categories is a Series
    # print(type(abstract_df)) # debug check
    # get the abstract
    for index, row in categories_df.iterrows():
      # print(row) # debug check
      categories_text=str(row[0]).replace(' (Q1)', '').replace(' (Q2)', '').replace(' (Q3)', '').replace(' (Q4)', '')
      categories_text = categories_text.replace('; ', ';')
      categories_text = categories_text.rstrip(' ')
      categories_text = re.sub(r"^\W+", "", categories_text.lstrip())
      # print(categories_text) # debug check
      list_categories = categories_text.split(";")
      # print(type(list_categories)) # debug check
      categories_to_frequency(list_categories)

    # print(dictionary) #debug check
    result_path = result_csv_dir + os.sep + result_scopus_categories
    with open(result_path, 'w') as csv_file:
        writer = csv.writer(csv_file,delimiter = ";")
        for key, value in dictionary.items():
            writer.writerow([key, value])
    print("Done in:",result_path)

def categories_to_frequency(list_categories):
    """Exctract Scimago Categories frequency"""
    global dictionary
    if (len(list_categories)>0):
        for elements in list_categories:
            if elements in dictionary:
              dictionary[elements] += 1
            else: # not in dictionary, starts from 1
              dictionary.update({elements: 1})

# MAIN
print()
print("*** Scopus API Search ***")
print()

# same keyowrds of scopus.com except "LIMIT-TO()" e "INDEXTERMS()"

# sample query 1
query = 'TITLE-ABS-KEY ( ( ( "PUBLIC TENDERS"  OR  "PUBLIC PROCUREMENTS" )  AND  ( "DETECTION"  OR  fraud  OR  corruption ) ) )'

# sample query 3
# query = 'TITLE-ABS-KEY ( ( ( "PUBLIC TENDER" OR "PUBLIC PROCUREMENT" OR "E-PROCUREMENT" OR "public competitions" OR "public regulations" OR "state laws" ) AND ( "DETECTION" OR fraud OR corruption OR crime OR criminal ) AND ( prediction OR predictive OR "machine learning" OR "deep learning" OR "neural networks" OR "modeling" OR "artificial intelligence" ) ) )'

start = datetime.now().replace(microsecond=0)

print("Query execution: ", query)

scopus_result = ScopusSearch(query)

end = datetime.now().replace(microsecond=0)

print()
print("Results found:", scopus_result.get_results_size()) # number of results found
print("Timing:", end - start)

# results of the query (scopus_result.results) saved inside a pandas.DataFrame (DF)
scopus_df = pd.DataFrame(pd.DataFrame(scopus_result.results)) # Articles DF

articles_df = scopus_join_scimagojr(2020) # join Articles DF with Scimago Journal Rating DF from a specific year

print()
print("Categories CSV...")
list_categories = scopus_to_categories(articles_df)

# authors_explore(scopus_result) # get network of authors
