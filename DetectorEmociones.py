import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def detectar_emociones_habla(texto):
    sid = SentimentIntensityAnalyzer()
    emociones = sid.polarity_scores(texto)
    return emociones

def detectar_emociones_mensaje(texto):
    sid = SentimentIntensityAnalyzer()
    emociones = sid.polarity_scores(texto)
    return emociones

# Ejemplo de uso para el reconocimiento de emociones en el habla
texto_habla = "Estoy muy emocionado por el próximo concierto. Será increíble."
emociones_habla = detectar_emociones_habla(texto_habla)
print("Emociones en el habla:")
for emocion, valor in emociones_habla.items():
    print(emocion + ": " + str(valor))

# Ejemplo de uso para el reconocimiento de emociones en los mensajes
texto_mensaje = "Estoy muy feliz por tu éxito. ¡Felicidades!"
emociones_mensaje = detectar_emociones_mensaje(texto_mensaje)
print("\nEmociones en el mensaje:")
for emocion, valor in emociones_mensaje.items():
    print(emocion + ": " + str(valor))
