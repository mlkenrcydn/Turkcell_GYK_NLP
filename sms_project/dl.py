import pandas as pd
import re, string

df = pd.read_csv('spam.csv', encoding="latin-1")
df = df[['v1','v2']]
df.columns = ['label','text']

df['label'] = df['label'].map({'ham':0,'spam':1})
#print(df.head())

# Bütün verimi normalize etmek için. (Hepsi küçük harf, sayıları temizlenmiş, noktalama işaretleri temizlenmiş)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text) # sayıları temizle
    text = text.translate(str.maketrans('','', string.punctuation)) # noktalama işaretlerini temizle
    return text

df['clean_text'] = df['text'].apply(clean_text)
#print(df.head())

# Makine Öğrenmesi -> train_test_split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)

# Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer

# num_words => Kelime sayısı
# out-of-vocabulary token => <OOV>

# num_words => Toplam kelime sayısı -> verilmezse -> tüm kelimeleri alır.
tokenizer = Tokenizer(num_words=5000,oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

#print(tokenizer.word_index.items())
vocab_size = len(tokenizer.word_index) + 1 # 1 ekleniyor çünkü 0 index kullanılmaz. (Padding için)


# Metinleri sayıya çevir (sequence)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# # İlk 3 orijinal mesaj
# for i in range(3):
#     print(f"{i+1}. Metin: {X_train.iloc[i]}")
# # İlk 3 sayısal mesaj
# for i in range(3):
#     print(f"{i+1}. Metin: {X_train_seq[i]}")


# Padding
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Bütün cümleleri belirlenen uzunluğa getir.
max_length = 100

# truncating => Eğer cümle uzun ise kısaltır.
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

for i in range(3):
    print(f"{i+1}. Metin: {X_train_pad[i]}")


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout

# Training verisinde kısa spam örnekleri yok, eklenebilir.

# Embedding Layer Detayları

# Model spam patternlarını cümle yapısından ayırt edemiyor?

# n-gram awareness => sağlayacak bir şeyler eklemek?

# "win" + "prize" + "click" => spam
# "win","prize"

model = Sequential()
# input_dim => Toplam kelime sayısı
# output_dim => Vektör boyutu (Her idyi kaç boyutlu vektörle temsil edeceğiz)
# input_length => Her mesajın kaç kelimelik olacağı

model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length, mask_zero=True)) # Kelimeleri sayılardan oluşan anlamlı vektörlere çevir.

# 42 -> [0.25,0.13,-0.27, 0.5]
# Kelimelere id atamak (Tokenizer)
# 42 gibi sayılar model için anlamlı değil.. 
# ID => Anlamlı bir uzayda vektörel temsil edeyim.

# Long Short Term Memory (LSTM) Layer
# RNN => Recurrent Neural Network
# Bidirectional => LSTM'in hem ileri hem geri yönde çalışmasını sağlar.

model.add(Bidirectional(LSTM(64))) # 64 => Nöron sayısı

# Dropout katmanı?
model.add(Dropout(0.5)) # 0.5 => %50 dropout => Nöronların %50'sini kapatır.
#
# Bağlamı hatırlamak için.
# Ali sabah okula gitti. Akşam eve geldi.
# Forget Gate, Input Gate, Output Gate (ARAŞTIRMA)
# LSTM sırayla gelen verileri işlerken geçmişte gördüğü önemli bilgileri hatırlayan bir yapay zeka katmanı.
# Activation Function nedir?
# Dense -> Fully Connected Layer
# 1 => 1 tane nöron çünkü binary classification -> spam değil mi?
model.add(Dense(1, activation="sigmoid")) # sigmoid => 0-1 arası değerler verir. %75 spam %25 ham 0.75

# loss => Her tahminde ne kadar hata yapıldığını ölçer.
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

# Early Stopping
from tensorflow.keras.callbacks import EarlyStopping

# Patience = 3 => 3 epoch sonra kontrol et. => val_loss N adet epoch boyunca iyileşmezse dur.
# Restore Best Weights => En iyi epoch'i kullan.
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train_pad, y_train, epochs=10, validation_data=(X_test_pad, y_test), callbacks=[early_stopping])
# Epoch => veriyi baştan sona kaç kere göreyim?

# Eğitim sonunda modeli kaydet.
model.save("sms_model.h5")
#model.save("sms_model2.keras") 