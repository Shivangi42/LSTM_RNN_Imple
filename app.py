import streamlit as slt
import numpy as np 
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load the LSTM model
model=load_model('next_word_lstm.h5')

#Load the Tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

#Function to predict the next word
def predict_next_word(model,tokenizer,text,max_sentence_length):
  token_list=tokenizer.texts_to_sequences([text])[0]
  if(len(token_list))>=max_sentence_length:
    token_list=token_list[-(max_sentence_length-1):]
  token_list=pad_sequences([token_list],maxlen=max_sentence_length-1,padding='pre')
  predicted=model.predict(token_list,verbose=0)
  predicted_word_index=np.argmax(predicted,axis=1)
  for word,index in tokenizer.word_index.items():
    if index==predicted_word_index:
      return word
  return None
 
 #streamlit app
 slt.title("Next Word Prediction using LSTM and EarlyStopping")
input_text=slt.text_input("Enter the sequence of words","To be or not to")
if slt.button("Predict Next Word"):
  max_sentence_length=model.input_shape[1]+1
  next_word=predict_next_word(model,tokenizer,input_text,max_sentence_length)
  slt.write(f'Next Word: {next_word}')