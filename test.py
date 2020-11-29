import pandas as pd 
from gatorvader import *

df = pd.read_csv('./rawdata/media_20201128.csv')

print(df['news_source'].value_counts())



df[['score_compound','score_pos','score_neu','score_neg','pos_words','neg_words']] = df['headline_descriptions'].apply(sentiment_scores)
df[['coordinating_conjunction','cardinal_digit','determiner','existential','foreign','preposition','adjective','numbering','modal','noun','possessive','pronoun','adverb','giveup','to_go','interjection','verb']] = df['headline_descriptions'].apply(part_of_speech)

print(df.head(5))