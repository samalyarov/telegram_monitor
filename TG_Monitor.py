# General libraries for work
import configparser
import asyncio
import pandas as pd
import pytz
import pickle
from datetime import datetime

# Interaction with Telegram API is done via the Telethon library
import telethon
from telethon.tl.types import PeerChannel
from telethon.tl.functions.messages import ForwardMessagesRequest

# Text preparation
import re
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords

# Logistic regresssion model operation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

# Importing neccessary models 
vectorizer = pickle.load(open('vectorizer.pkl', 'rb')) # Vectorizer model
saved_vocabulary = pickle.load(open("feature.pkl", 'rb')) # Vocabulary for the vectorizer model
logit_model = pickle.load(open('logit_model.pkl', 'rb')) # Logistic regression model

# Basic parameters and functions for text transformation
patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
morph = MorphAnalyzer()
stopwords_ru = stopwords.words("russian")

def lemmatize(doc:str):
    '''
    Function for lemmatizing strings

    Args:
    doc(str): a string to be lemmatized 

    Returns:
    tokens(str): a lemmatized string
    '''
    # Only keep the relevant symbols and words
    patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
    morph = MorphAnalyzer()
    stopwords_ru = stopwords.words("russian")
    
    # Conduct the actual procedure
    doc = re.sub(patterns, ' ', doc)
    tokens = []
    for token in doc.split():
        if token and token not in stopwords_ru:
            token = token.strip()
            token = morph.normal_forms(token)[0]
            tokens.append(token)
            ' '.join(tokens)
    
    # Return only data with more then 2 words in it
    if len(tokens) > 2:
        return tokens
    return None

# Read the config file
config = configparser.ConfigParser()
config.read("config.ini")

# Acquire data for Telegram connection
api_id = config['Telegram']['api_id']
api_hash = config['Telegram']['api_hash']
api_hash = str(api_hash)
username = config['Telegram']['username']

# Reading the channels excel for list of channels to parse:
channels_excel = pd.read_excel('t_channels.xlsx')
channels_links = channels_excel['Channel Link'].tolist()
channels_short_names = channels_excel['Short Name'].tolist()
channels_names = channels_excel['Channel Name'].tolist()
print('Loading Channels: ' + str(channels_short_names))

# Working with script parameters:
local_tz = pytz.timezone('Asia/Tbilisi') # Setting the local timezone (dependant on your location)

# Input the relevant data:
from_date = input('Up to which date and time should the messages in channels be parsed? Input the answer in "2023-12-01 00:00:00" format: ')
threshold = float(input('What model confidence level should the script use? Enter a number between 0 and 1 (0.2 is the suggested value): '))

# Changing the input date into datetime format and adding the timezone info
from_date = datetime.strptime(from_date, '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.utc).astimezone(local_tz)

# Creating a telethon client
client = telethon.TelegramClient(username, api_id, api_hash)

# Launching the client and informing user that it is operational
client.start()
print('Client Operational')

async def monitor_channels():
    # Create a dictionary with already forwarded messages (get rid of duplicates due to telegram errors and such)
    forwarded_messages = {}

    # Insert the link to a channel into which the script will forward the relevant messages
    forward_chat_id = await client.get_entity('https://t.me/######')


        # Iterate over channels
    for i, channel in enumerate(channels_links):
            print(f'Парсим паблик {channels_short_names[i]}')
            # Acquiring entity for each channel
            user_input_channel = channel 
            if user_input_channel.isdigit():
                entity = PeerChannel(int(user_input_channel))
            else:
                entity = user_input_channel


            entity = await client.get_entity(entity)

            await client.get_participants(entity)
            
            forwarded_messages = []
            # Iterate over all the messages on that channel
            async for message in client.iter_messages(entity):
                try:
                    current_text = message.message.lower().replace("ё","е")
                    current_text = lemmatize(current_text)
                    current_text = ' '.join(current_text)
                    
                # Checking whether the message text is appropriate and not already forwarded
                    if message.message not in forwarded_messages:
                        text_transformed = vectorizer.transform([message.message])
                        certainty = logit_model.predict_proba(text_transformed)
                        if (certainty[:, 1] > threshold).astype(int):
                            forwarded_messages.append(message.message)
                            print(f'Adding message with certainty = {certainty[:, 1]}')
                            await client(ForwardMessagesRequest(from_peer=entity, id=[message.id], to_peer=forward_chat_id))
                except:
                    print('Skipping Empty Message')
                if message.date <= from_date:
                    print('Checked all channel messages')
                    break

# Loop over all channels after which - shut down the client
client.loop.run_until_complete(monitor_channels())
client.disconnect()