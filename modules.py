
#fUNCTION call
import re
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

def text_cleaning(text):
    """This function removes texts with anomalies such as URLS,@NAME, WASHINGTON (Reuters) and also to convert text into lowercase

    Args:
        text (_type_): _description_

    Returns:
        _type_: _description_
    """

    # have URL (bit.ly/djwidiwjdjawd)
    text = re.sub('bit.ly/\d\w{1,10}','',text)

    # have @realDonaldTrump
    text = re.sub('@[^\s]+', '', text)

    # WASHINGTON (Reuters) : New Header
    text = re.sub('^.*?\)\s*-', '', text)

    # [1901 EST] 
    text = re.sub('\[.*?EST\]', '', text)

    # numbers and special characters and punctuations -->must last
    text = re.sub('[^a-zA-Z]', '', text).lower()
    
    return text

#%%

def lstm_model(num_words, nb_classes, embedding_layer=64, dropout=0.3, num_neurons= 64, ):
    """This function creates LSTM model with embedding layer, 2 LSTM layers, with dropout and summary_

    Args:
        num_words (int): number of vocabulary
        nb_classes (int): number of class
        embedding_layer (int, optional): the number of output embedding llayer. Defaults to 64.
        dropout (float, optional): the rate of dropout. Defaults to 0.3.
        num_neurons (int, optional): number of rbain cells. Defaults to 64.

    Returns:
        model: returns the model created using sequential API.
    """
    model = Sequential()
    model.add(Embedding(num_words, embedding_layer))
    model.add(LSTM(embedding_layer, return_sequences= True))
    model.add(Dropout(dropout))
    model.add(LSTM(num_neurons))
    model.add(Dropout(dropout))
    model.add(Dense(nb_classes, activation = 'softmax'))
    model.summary()

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['acc'])
    plot_model(model)
    return model