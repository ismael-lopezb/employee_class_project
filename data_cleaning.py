#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 23:20:36 2021

@author: ismaellopezbahena
"""
#import usefull libraries
import pandas as pd
import numpy as np

#read csv file and get the info 
df = pd.read_csv('aug_train.csv')
df.info()
#let's replace the gender missing data with the mode
gmode = df.gender.mode()[0]
df['gender'].fillna(gmode, inplace=True)
#fill missing data in enrolled university with the mode
df['enrolled_university'].fillna(df.enrolled_university.mode()[0], inplace=True)
#fill eduaction level missing values with the mode
df['education_level'].fillna(df.education_level.mode()[0], inplace=True)

#let's do the same with major discipline, experience, company size, company type, last new job
for column in df.columns:
    df[column].fillna(df[column].mode()[0], inplace=True)
#make sure we don't have more missing data    
df.info()
#let's see relevent experience values
df['relevent_experience'].value_counts()
#we have two values so let's make 'yes'=relevent experience and 'no'= no relevet experince 
df['relevent_experience'] = df['relevent_experience'].apply(lambda x: 'Yes' if 'Has relevent experience' in x else 'No')
#let's see the unique values of enrolled_university
df['enrolled_university'].value_counts()
#we have 3 categories so leave it like that. Education level and major values
df['education_level'].value_counts()
df['major_discipline'].value_counts()
#we have 4 categories leave it as well with major discipline. We want experince to be int
#remove the < and > from experince
df['experience'] = df['experience'].apply(lambda x: x.replace('<', ''))
df['experience'] = df['experience'].apply(lambda x: x.replace('>', ''))
df['experience'] = df['experience'].astype(int)
#lets put compani size in the same format n-m
df['company_size'] = df['company_size'].apply(lambda x: x.replace('/', '-'))
#we want last-new job be an int 
df['last_new_job']=df['last_new_job'].apply(lambda x: x.replace('+',''))
df['last_new_job']=df['last_new_job'].apply(lambda x: x.replace('>',''))
df['last_new_job']=df['last_new_job'].apply(lambda x: x.replace('never','0'))
df['last_new_job'] = df['last_new_job'].astype(int)
df.info()
#we don't have missing values and we get numerical and categorical data. Save it
df.to_csv('data_cleaned.csv', index=False)
