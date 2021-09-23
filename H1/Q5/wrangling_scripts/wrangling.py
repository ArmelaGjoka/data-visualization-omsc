"""
cse6242 s21
wrangling.py - utilities to supply data to the templates.

This file contains a pair of functions for retrieving and manipulating data
that will be supplied to the template for generating the table. """
import csv

def username():
    return 'agjoka3'

def data_wrangling():
    with open('data/movies.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        table = list()
        # Feel free to add any additional variables
        
        # Read in the header
        for header in reader:
            break
        counter = 0
        # Read in each row
        for row in reader:
            if counter == 100:
                break
            table.append(row)
            counter += 1
        
        # Order table by the last column - [3 points] Q5.b
        table.sort(key = get_average, reverse=True)
    
    return header, table


def get_average(movie):
    return float(movie[2])

