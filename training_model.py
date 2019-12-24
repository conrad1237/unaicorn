{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# A)Trabalhando Dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Impontando pacotes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Definindo Base"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('train.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Retirando outliers"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Temos 3 registros com Fare >= 500, agora vamos substituilos pela média"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.loc[df[\"Fare\"]>400, \"Fare\"] = df[\"Fare\"].median()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Verificamos novamente o histograma"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Temos 3 registros com Age > 70, agora vamos substituilos pela média"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.loc[df[\"Age\"]>70, \"Age\"] = df[\"Age\"].median()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Tratando valores nulos"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Verificando quais colunas tem valores nulos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"Age\"].fillna(df[\"Age\"].median(), inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Agora vamos corrigir a de embarques"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Como este é um campo de descrição vamos verificar quantas ocorrencias temos em cada descrição"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "A mais recorrente é a \"S\", logo vamos substituir os nulos por \"S\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"Embarked\"].fillna(\"S\", inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Agora vamos corrigir a de cabines"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Como temos 889 registros e a coluna Cabin tem 687 nulos, vamos optar por deletar esta coluna"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "del df[\"Cabin\"]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Agora vamos olhar no dataset se temos mais alguma informação útil para usarmos no machine learning"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Vemos que tem Miss, Mrs entre outros, vamos separa isso"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Vamos criar uma função chamada get_title"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_title(name):\n",
    "    if '.' in name:\n",
    "        return name.split(',')[1].split('.')[0].strip()\n",
    "    else:\n",
    "        return \"No title in name\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Aplicando a função"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "titles = set([x for x in df.Name.map(lambda x: get_title(x))])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Agora vamos criar uma função que retorne um sub-grupo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "def shorter_titles(x):\n",
    "    title= x[\"Title\"]\n",
    "    if title in ['Capt','Col','Major']:\n",
    "        return 'Officer'\n",
    "    elif title in ['Jonkheer','Don','the Countess','Don','Lady','Sir']:\n",
    "        return 'Royalty'\n",
    "    elif title== \"Mme\":\n",
    "        return 'Mrs'\n",
    "    elif title in ['Mlle','Ms']:\n",
    "        return 'Miss'\n",
    "    else:\n",
    "        return title"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Agora vamos criar ua coluna no dataset chamada title aplicando a função get_title"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"Title\"] = df[\"Name\"].map(lambda x: get_title(x))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Vamos aplicar a função shorter_titles na coluna que acabamos de criar \"Title\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"Title\"] = df.apply(shorter_titles,axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###### Vamos deletar a coluna name, já que temos a title, porém de uma forma diferente del df[\"Name\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop(\"Name\",axis=1,inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Deletamos támbem a coluna ticket"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop(\"Ticket\",axis=1,inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Nosso dataset parece estar completo"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Converter a colunas descritivas em números"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Convertendo sexo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.Sex.replace(('male','female'),(0,1),inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Convertendo Embarked"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.Embarked.replace(('S','C','Q'),(0,1,2),inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Convertendo Title"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.Title.replace(('Mr', 'Miss', 'Mrs','Master','Dr','Rev', 'Officer', 'Royalty'),(0,1,2,3,4,5,6,7),inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Nosso dataset parece estar completo e convertido"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# B) Gerando Modelo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Importando método de separar as bases de treino e teste\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Variáveis explicativas\n",
    "x = df.drop(['Survived', 'PassengerId'], axis=1) #retirando Survived pois é nossa variável de resposta e PassengerId que é o ID"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Variáve de resposta\n",
    "y = df[\"Survived\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Separando base de treino e teste\n",
    "x_train, x_val, y_train, y_val = train_test_split(x,y,test_size = 0.1) # tamanho da base de teste é de 10%\n",
    "\n",
    "# Com isso ele criara 2 csv dentro da pasta que estamos trabalhando\n",
    "# 1 que será o teste, logo, test.csv\n",
    "# 2 que sera o de treino, logo, train.csv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy:85.56\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Conrad\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n"
     ]
    }
   ],
   "source": [
    "#executando o modelo\n",
    "import pickle #este é importante para salvar o modelo\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "randomforest = RandomForestClassifier()\n",
    "randomforest.fit(x_train, y_train)\n",
    "y_pred = randomforest.predict(x_val)\n",
    "acc_randomforest = round(accuracy_score(y_pred,y_val) * 100, 2)\n",
    "#print(acc_randomforest)\n",
    "print(\"Accuracy:{}\".format(acc_randomforest))\n",
    "\n",
    "#Aqui ele ira salvar o modelo dentro de sua pasta de trabalho\n",
    "pickle.dump(randomforest,open('titanic_model.sav','wb'))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### Agora vamos fazer uma função utilizando o modelo que salvamos, titanic_model.sav, aplicar parâmetros fake, e observar o resultado"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "def prediction_model(pclass,sex,age,sibsp,parch,fare,embarked,title):\n",
    "    import pickle\n",
    "    x = [[pclass,sex,age,sibsp,parch,fare,embarked,title]]\n",
    "    randomforest = pickle.load(open('titanic_model.sav','rb'))\n",
    "    predictions = randomforest.predict(x)\n",
    "    print(predictions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1]\n"
     ]
    }
   ],
   "source": [
    "prediction_model(1,1,11,1,1,19,1,1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### O modelo acima está funcionando e pronto para aplicação"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
