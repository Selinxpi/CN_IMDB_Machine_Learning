from imp import new_module
from sre_parse import State
import tensorflow as tf

new_model = tf.keras.models.load_model('imdb_classifier.model')

def predictor(message):
    prediction = list(new_model.predict(message))
    prediction[0][0] *= 100
    prediction = round(prediction[0][0], 2)
    if prediction < 40:
        state = "Negative "
    elif prediction >= 40 and prediction <60:
        state = "Neutral "
    else:
        state = "Positive "
    prediction = str(prediction) + "%   ~>  "+ state + "review"
    return prediction