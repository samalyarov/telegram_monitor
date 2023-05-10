# Telegram monitoring script

A script is based on telethon library and utilizes a personal Telegram account for acquiring messages from open channels or channel to which the user has access. Data can be collected from multiple channels (from a list in separate files). That particular version of the scripts then utilizes text pre-processing and a trained logistic regression model to determine if the message is "of interest" and if so - forwards it to a specified channel.
Whether the message is "of interest" is arbitrary and based purely on the samples and interests of one using the script. 

Files included in repository:

- `t_channels.xlsx` is a list of channels (with names and links) from which the posts are to be loaded. The list is done in a .xlsx file for ease of communicating with colleagues (especially those with no coding experience) and leaving comments and remarks.
- `config.ini` is a config file listing username, password, phone number of a user as well as bot API settings for connection. 
- `TG_Monitor.py` is a script itself
- `Posts_prediction.ipynb` is an Jupyter Notebook with preliminary data analysis and model training, as well as prospects for future development and overall analysis of the model.
- In order to be operational the folder should also include 3 .pkl files - a trained logistic regression model, a trained vectorizer model and vocabulary for said vectorizer model. All three can be created through the use of the "Posts_prediction" notebook.

The folder is designed to later on be compressed into a single .exe file (via the use of pyinstaller) for easy use.

*Libraries used: telethon, configparser, datetime, numpy, pandas, pytz, matplotlib, seaborn, pickle, re, pymorphy2, nltk, scikit-learn(sklearn), eli5, tqdm (tqdm_notebook)*
