##############################################################
# Kütüphanelerin Import Edilmesi
##############################################################
import yfinance as yf
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from PIL import Image
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


st.write(""" 
#       **Welcome**     """)

image = Image.open('titanic.jpg')
st.image(image, caption='Titanic Ship')



st.write(""" 
## **Artificial Intelligence Prediction Automation!**""")

st.write("""
Değerli kullanıcılarım uygulamaya hoşgeldiniz. :sunglasses:

Size kısaca Titanic gemisinden bahsedip ardından uygulamayı nasıl kullanabileceğinizi anlatacağım.
Titanic gemisi Harland and Wolff (Belfast, Kuzey İrlanda) tersanelerinde üretilmiştir. 
15 Nisan 1912 gecesi daha ilk seferinde bir buz dağına çarpmış ve yaklaşık iki saat kırk dakika içinde Kuzey 
Atlantik'in buzlu sularına gömülmüştür. 
1912'de yapımı tamamlandığında dünyanın en büyük buharlı yolcu gemisiydi. (Wikipedia)

Bu veri seti Titanik gemisine ait o kişilerin bilgilerini içeren bir veri setidir.
Bu veri seti ile gemi batmadan önce binecek olan bir kişinin hayatta kalıp kalmaması durumu tahmin edeceğiz.

Alınan bilgiler ise kısaca şunlar:

    * Pclass: Yolculuk Sınıfı,
    * Sex: Cinsiyet, 
    * Age: Yaş, 
    * SibSp: Yolcunun gemide bulunan kardeş sayısı, 
    * Parch: Yolcunun gemide bulunan akraba sayısı, 
    * Fare Yolcunun bilet için ödediği tutar,
    * Cabin: Seyahat eden yolcu ve personellere verilen kabin numaralarıdır
    * Embarked: Yolcunun gemiye bindiği kapı (S (En iyisi), C (Orta), Q (En Kötüsü).
    
Bu bilgiler doğrultusunda bu kişinin gemiden sağ çıkıp çıkmama durumunu  tamin eden bir model geliştirdim. 
Uygulama açık değilse sol üstteki ">" işareti ile gösterilen barı açarak tahmin için değerleri seçebilirsiniz.
Ardından özeelikleri değiştirerek sonucunuzu "Prediction" başlığı altında görebilirsiniz.
""")

st.write("""Uygulamanın tahmin başarsı (**Accuracy Score**): %82""")


st.sidebar.header('User Input Parameters for Titanic')
df1 = pd.read_csv("titanic.csv")
df1 = df1.drop(["PassengerId", "Name", "Ticket"], axis = 1)
df1.SibSp.unique()
def user_input_features():
    #Pclass
    pclass = st.sidebar.radio("Hangi sınıfta yolculuk ediyorsunuz?", ('En iyi', 'Orta', 'En kötü'))

    if pclass == 'En iyi':
        pclass = 1
    elif pclass == "pclass_input":
        pclass = 2
    else:
        pclass = 2

    # SEX
    sex = st.sidebar.radio("Cinsiyetinizi seçiniz:", ('Erkek', 'Kadın'))

    if sex == 'Erkek':
        sex = "male"
    else:
        sex = "female"


    # AGE
    age = st.sidebar.number_input('Yaşınızı Giriniz: ')


    # SIBSP
    sibsp = st.sidebar.slider('Gemi de Kaç Kardeşiniz Var?', 0, 4, 0)



    # Parch
    parch = st.sidebar.slider('Gemide Kaç Akrabanız Var?', 0, 3, 0)



    # FARE
    fare = st.sidebar.slider('Bilet Ücreti:',  0.00, 512.3292, 0.00)


    # CABIN
    cabin_ = st.sidebar.number_input('Kabin Numaranızı Giriniz: ')
    cabin_ = int(cabin_)
    cabin = str(cabin_)
    cabin = "C" + cabin

    #EMBARKED
    embarked = st.sidebar.radio("Gemiye Hangi Kapıdan Bindniniz: S(En İyi) - C(Orta) - Q(En Kötü)", ('S', 'C', "Q"))




    data = {'Pclass': pclass,
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,
            "Parch": parch,
            "Fare" : fare,
            "Cabin": cabin,
            "Embarked": embarked
            }
    features = pd.DataFrame(data, index = [df1.index[-1] + 1])
    return features

df2 =  user_input_features()

st.write("**Values**")
df2


dff = pd.concat([df1, df2]).reset_index()
dff = dff.drop("index", axis = 1)
#dff


from titanic_pipeline import diabetes_data_prep
X, y = diabetes_data_prep(dff)
X_sample = X[-1:]

new_model = joblib.load("voting_clf.pkl")
new_model.predict(X_sample)



st.subheader('Prediction')


if new_model.predict(X_sample)[0] == 0:
    st.write("Didn't Survive")
else:
    st.write("Survived")



st.write("""
### Bana Ulaşmak İsterseniz:""")

st.markdown("""**[Linkedin](https://www.linkedin.com/in/muratcelebi3455)**""")
st.markdown("""**[Medium](https://medium.com/@celebim.murat)**""")
st.markdown("""**[Github](https://github.com/muratcelebim)**""")
st.markdown("""**[Kaggle](https://www.kaggle.com/clbmurat)**""")

