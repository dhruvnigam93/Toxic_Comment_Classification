{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Toxic Comment challenge\n",
    "This is an NLP based binary classification problem. I plan to use a variety of approaches to solve the problem ranging from classical methods to more advance deep learning based methods. My primary objective is to learn how to efficiently represent text in a form amenable to application of ML techniques. Some of the methods I explore are - \n",
    "\n",
    "1. Bag of words\n",
    "2. TF-IDF vectors\n",
    "3. Using pre-Learned word Embeddings - GLOVE\n",
    "4. Effectively using GLOVE for paragraph representations\n",
    "\n",
    "The data set used is from a challenge hostedon kaggle by google. The data set contains a large number of user comments that have been labeled(binary) by humans for toxic behaviour. The objective is to learn a model from this labelled data that can identify toxic behaviour in commnet text.\n",
    "\n",
    "## EDA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 298,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>comment_text</th>\n",
       "      <th>toxic</th>\n",
       "      <th>severe_toxic</th>\n",
       "      <th>obscene</th>\n",
       "      <th>threat</th>\n",
       "      <th>insult</th>\n",
       "      <th>identity_hate</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0000997932d777bf</td>\n",
       "      <td>Explanation\\nWhy the edits made under my usern...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>000103f0d9cfb60f</td>\n",
       "      <td>D'aww! He matches this background colour I'm s...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>000113f07ec002fd</td>\n",
       "      <td>Hey man, I'm really not trying to edit war. It...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0001b41b1c6bb37e</td>\n",
       "      <td>\"\\nMore\\nI can't make any real suggestions on ...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0001d958c54c6e35</td>\n",
       "      <td>You, sir, are my hero. Any chance you remember...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                 id                                       comment_text  toxic  \\\n",
       "0  0000997932d777bf  Explanation\\nWhy the edits made under my usern...      0   \n",
       "1  000103f0d9cfb60f  D'aww! He matches this background colour I'm s...      0   \n",
       "2  000113f07ec002fd  Hey man, I'm really not trying to edit war. It...      0   \n",
       "3  0001b41b1c6bb37e  \"\\nMore\\nI can't make any real suggestions on ...      0   \n",
       "4  0001d958c54c6e35  You, sir, are my hero. Any chance you remember...      0   \n",
       "\n",
       "   severe_toxic  obscene  threat  insult  identity_hate  \n",
       "0             0        0       0       0              0  \n",
       "1             0        0       0       0              0  \n",
       "2             0        0       0       0              0  \n",
       "3             0        0       0       0              0  \n",
       "4             0        0       0       0              0  "
      ]
     },
     "execution_count": 298,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "from sklearn.metrics import roc_auc_score \n",
    "\n",
    "from sklearn.feature_selection import SelectKBest\n",
    "from sklearn.feature_selection import f_classif\n",
    "\n",
    "all_data = pd.read_csv(\"train.csv\")\n",
    "\n",
    "all_data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- The 'id' column is a unique id identifying each row - this will not contribute to our model.\n",
    "- 'comment_text'is the actual text of the comment - the meat of the problem.\n",
    "- The rest of the the 6 columns are the tags that each identify wheater the 'comment_text' beongs to that category. For now, the only column of concern is the 'toxic' column\n",
    "\n",
    "The task is to build a model that given a comment can identify if it falls into the category 'toxic' or not."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 299,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "toxic           0.095844\n",
       "severe_toxic    0.009996\n",
       "obscene         0.052948\n",
       "threat          0.002996\n",
       "insult          0.049364\n",
       "dtype: float64"
      ]
     },
     "execution_count": 299,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "all_data.iloc[:,2:7].sum(axis = 0)/all_data.shape[0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Another interesting thing to note is that there are relatively few instances of toxic comments under each category. This indicates a class imblalance -something that needs to be kept in mind while building and evaluating the classification model.\n",
    "\n",
    "There may be interesting relations between the multiple independednt binary variables. Below I explore the similarity of the catogories in the train data using Jaccards coefficient - a simple metric to calculated similarity between binary varibles."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 300,
   "metadata": {},
   "outputs": [],
   "source": [
    "def cumpute_jaccard(x,y):\n",
    "    count_df= pd.crosstab(x, y)\n",
    "    jaccard = count_df.iloc[1,1] / ( count_df.iloc[0,1] +  count_df.iloc[1,1] + count_df.iloc[1,0])\n",
    "    return jaccard"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 301,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>toxic</th>\n",
       "      <th>severe_toxic</th>\n",
       "      <th>obscene</th>\n",
       "      <th>threat</th>\n",
       "      <th>insult</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>toxic</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.104289</td>\n",
       "      <td>0.501106</td>\n",
       "      <td>0.029302</td>\n",
       "      <td>0.464017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>severe_toxic</th>\n",
       "      <td>0.104289</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.177905</td>\n",
       "      <td>0.057114</td>\n",
       "      <td>0.169238</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>obscene</th>\n",
       "      <td>0.501106</td>\n",
       "      <td>0.177905</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.034895</td>\n",
       "      <td>0.605152</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>threat</th>\n",
       "      <td>0.029302</td>\n",
       "      <td>0.057114</td>\n",
       "      <td>0.034895</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.038146</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>insult</th>\n",
       "      <td>0.464017</td>\n",
       "      <td>0.169238</td>\n",
       "      <td>0.605152</td>\n",
       "      <td>0.038146</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                 toxic  severe_toxic   obscene    threat    insult\n",
       "toxic         1.000000      0.104289  0.501106  0.029302  0.464017\n",
       "severe_toxic  0.104289      1.000000  0.177905  0.057114  0.169238\n",
       "obscene       0.501106      0.177905  1.000000  0.034895  0.605152\n",
       "threat        0.029302      0.057114  0.034895  1.000000  0.038146\n",
       "insult        0.464017      0.169238  0.605152  0.038146  1.000000"
      ]
     },
     "execution_count": 301,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "all_data.iloc[:,2:7].corr(method = cumpute_jaccard)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There are some significant overlaps between the categories for sure. We may be able to build a better model if we take these overlaps into consideration. However, for now, I will ignore these and focus on building a model to identify 'toxic' comments.\n",
    "\n",
    "## Linear model to predict 'toxic' variable\n",
    "\n",
    "Starting simple. Fit a linear model to the comment text to predict the target binary variable 'toxic'.\n",
    "\n",
    "### Cleaning the text data\n",
    "\n",
    "1. Remove all punctuations (may be revisited because certain puctuations !,# might be helful to identify toxic behavior)\n",
    "2. Remove all non-alphabet info - likely will not be useful. For typos and non-english text - i will remove rarely occuring words in corpus during training.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 302,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'D\\'aww! He matches this background colour I\\'m seemingly stuck with. Thanks.  (talk) 21:51, January 11, 2016 (UTC)---Hey man, I\\'m really not trying to edit war. It\\'s just that this guy is constantly removing relevant information and talking to me through edits instead of my talk page. He seems to care more about the formatting than the actual info.---\"\\nMore\\nI can\\'t make any real suggestions on improvement - I wondered if the section statistics should be later on, or a subsection of \"\"types of accidents\"\"  -I think the references may need tidying so that they are all in the exact same format ie date format etc. I can do that later on, if no-one else does first - if you have any preferences for formatting style on references or want to do it yourself please let me know.\\n\\nThere appears to be a backlog on articles for review so I guess there may be a delay until a reviewer turns up. It\\'s listed in the relevant form eg Wikipedia:Good_article_nominations#Transport  \"---You, sir, are my hero. Any chance you remember what page that\\'s on?'"
      ]
     },
     "execution_count": 302,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\"---\".join(all_data['comment_text'][1:5])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 303,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cleaning the data\n",
    "import re\n",
    "\n",
    "def clean_text(text):\n",
    "    text_clean = re.sub( \"[/(){}\\[\\]\\|@,;:\\n]\" , \" \" ,text.lower()) ## replace punctuation by space\n",
    "    text_clean = re.sub( \"[^a-zA-Z\\s:]\" , \"\" ,text_clean) ## retain only text - remove all other charecters\n",
    "    return text_clean\n",
    "\n",
    "all_data['clean_comment_text'] = [clean_text(x) for x in all_data['comment_text']]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 304,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Splitting training and testing data at this point, 80:20\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "train_data, test_data = train_test_split(all_data, test_size=0.2, random_state = 1)\n",
    "\n",
    "y_train = train_data['toxic']\n",
    "y_test = test_data['toxic']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Word Count analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 305,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import nltk, itertools\n",
    "tokenizer = nltk.tokenize.WhitespaceTokenizer()\n",
    "fdist = nltk.FreqDist(list(itertools.chain.from_iterable([tokenizer.tokenize(x) for x in train_data['clean_comment_text']])))\n",
    "fdist.plot(30 , cumulative=False , title = \"Word count - full corpus - most common\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 306,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAZUAAAEtCAYAAADAwv0jAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzt3Xl8VNXZwPHfk8kOgYQ97CCI4m5QEHdt3arVtlZrW8VqpXVpbe1bra1W6/KqfetStbVu1K3udUNRRFQQRSAosiibbGGRPRAIW5Ln/eOcgZthJplJZjKBPN/P534yc+655557MzPPveece6+oKsYYY0wyZKS7AsYYY/YeFlSMMcYkjQUVY4wxSWNBxRhjTNJYUDHGGJM0FlSMMcYkjQUVA4CI3Cwiz6S7Hs2ViBwrInPSXQ9jmjsLKs2QiFwvIqMi0ubFSPtR09auaYjICSKytJFlXCwiE5JRH1X9SFUHJKMsUz8ReUJEbkt3PUziLKg0T+OBo0UkBCAiXYAs4PCItH4+b9zEsf/7HsT+Z2ZPYh/U5mkKLogc6t8fB3wAzIlI+1pVlwOIyFARmSIiG/zfoeHCRORDEbldRD4GKoG+ItJHRMaJSIWIjAE61FUhETlbRKaJyEYR+VpETvPpXUXkDRFZJyLzReSywDK1jjYjzz5EZJGI/I+ITPf1fkFEckWkFfA20FVENvmpayI7UET2B/4FHOWXL/fpbUXkKRFZLSKLReSG8A+2iDwkIi8HyrhLRMb6H/XIuvcQkVd8OWtF5MEY9QiJyB/9PqsQkaki0sPPS/R/9qGI3CEik/0yr4tIu2j7NrB/v+VfHykipf7/t1JE7klkf0aUqyJyhT9TrhCRW0VkHxGZ6Mt/UUSyA/kv85+Ndf6z0tWni4jcKyKr/PZMF5EDRWQ48BPgWv+/GxmjHgeIyBhf7koR+aNPzxGR+0RkuZ/uE5Gc4H4SkWv9eleIyDkicoaIzPVl/TGwjptF5CURecZv6wwR2Vdca8IqESkTkVMC+ev6Ptzs981TvqxZIjKoof+HZktVbWqGEy6I/Na/fhC4BLg9Im2Ef90OWA9cCGQCF/j37f38D4ElwAF+fhYwEbgHyMEFqArgmRh1ORLYAHwbdyDSDdjPzxsH/BPIxQW81cDJft4TwG2Bck4AlgbeLwImA139NnwF/DJa3gbuw4uBCRFpTwGvAwVAb2AucKmfl+/fXwwcC6wBukfWBwgBXwD3Aq38th8Tow6/B2YAAwABDgHaN/B/9iGwDDjQr/e/4f9ZtP3l9++3/OuJwIX+dWtgSCP2qwJvAG18/bYBY4G+QFvgS2CYz3uS34+H+8/aA8B4P+9UYCpQ6PfN/kBxtM9OlDoUACuA3/n9XwAM9vNuAT4FOgEdgU+AWwP7qQr4s9+nl+E+s8/6Mg4AtgJ9ff6b/ftT/f/hKWAh8KfA8gsD9arr+xAu6wz/GboD+DTdvzXJntJeAZti/GPcB/BV//oLoD9wWkTaMP/6QmByxPITgYv96w+BWwLzevovVqtA2rPEDioPA/dGSe8BVAMFgbQ7gCf861o/DEQPKj8NvP8r8K9oeRu4Dy8mEFT8F3kbMDCQ9gvgw8D7I4F1wGLggmh1B47yPxaZcdRhDnB2lPSE/meBtDsD7wcC2/127ba/qB1UxgN/ATok4bOpwNGB91OB6wLv7wbu868fB/4amNca2IEL6CfhgvgQICNiHbU+O1HqcAHweYx5XwNnBN6fCiwK/B+3ACH/vsBvz+CI7Tkn8D0cE5h3FrApyvKF1P99uBl4L+L/t6Wx/4/mNlnzV/M1HjhGRIqAjqo6D3fENdSnHciu/pSuuB/BoMW4M4qwssDrrsB6Vd0ckT+WHrgvaqSuwDpVrahjvfX5JvC6EvejUy8R6Sm7msY2xbmuDkA2tbe1Vn1VdTKwAHfk/GKMcnoAi1W1Ko511rXvEvmfRUtbjDtarrPp0rsU2BeY7ZvazoyWSUTeDuzXn9RR3srA6y1R3of/j7W2U1U3AWuBbqr6Pu6M+x/AShF5RETaxLEtEHu/7rZO/zrYfLpWVasDdY22PcHPYeS8NVGWb01834fIz3uuiGTG2I49kgWV5msirilhOPAxgKpuBJb7tOWqutDnXQ70ili+J66pJCx4O+oVQJG4votg/ljKgH2ipC8H2olIQYz1bsY1KYV1qWMdkeq8fbaqLlHV1uEpzjLW4I6Sg/uq1n4SkStxzTTLgWtjlFsG9Izzx6CufZfI/yysR0T+HbjtqrWvxQ3o6LizINV5qnoBrknoLuDliP9/ON/pgf36n7o2LE61ttOvsz1+O1X1flUtwTU77YtrLoR6/v/E3q+7rRO3n5YnXPPE1fd9aBEsqDRTqroFKAWuAT4KzJrg04KjvkYB+4rIj0UkU0TOx51avxmj7MW+7L+ISLaIHIM7rY/lceBnInKyiGSISDcR2U9Vy3BnT3eI62A/GHdEHP4xmgacISLtxI1W+00Cu2Al0F5E2iawTLQyuoc7jf3R5YvA7SJSICK9cPvyGQAR2Re4DfgprnnqWhE5NEq5k3GB+U4RaeW3/egYdXgMuFVE+vuO6YNFpD0J/s8CfioiA0UkH9d38LLfrrm4o97viEgWcAMuOOK37aci0lFVa4Byn1wdWXgKPIv77BzqO8v/F5ikqotE5AgRGezruxnX3xCu00pcH00sbwJdROQ3vmO+QEQG+3nPATeISEcR6YDrP0n5NVhxfB9aBAsqzds43JFl8FqLj3zazqCiqmuBM3GdlmtxR9hnquqaOsr+MTAY139wE64DMirfJPQzXMf0Bl+v8JHgBbj28eXAq8BNqjrGz3sa1/ezCHgXeKHuza21ztm4H4cFIlIuCY7+8t4HZgHfiEh4X/wK9wO2ALdfnwVG+LOOZ4C7VPUL39z4R+Dp8MihQN2qcUG4H64zfSlwfow63IMLZO8CG3EBOq+B/zNw+/QJXDNKLvBrX6cNwBW4ILbMb2NwNNhpwCzfVPh34EequrWedTWaqo4FbsQNKliBO7sIX1vVBngUN0BhMW4//M3PexwY6P/3r0UptwI3cOQs3L6YB5zoZ9+GO2iajhsk8ZlPawp1fR9aBPEdRsaYZk5EPsQNpngs3XUxJhY7UzHGGJM0FlSMMcYkjTV/GWOMSRo7UzHGGJM0FlSMMcYkzV51JWc8OnTooL17927Qslu2bCEvLy9p+axMK9PKtDKbY5nRTJ06dY2qdqw3Y7rvE9PUU0lJiTZUaWlpUvNZmVamlWllNscyowFK1e79ZYwxpilZUDHGGJM0FlSMMcYkjQUVY4wxSWNBxRhjTNJYUDHGGJM0FlTipKps3l6T7moYY0yz1uIufmyIJWsr+fa94yjMESYdle7aGGNM82VBJQ6d2uSwraqGNdVQXaOEMiTdVTLGmGbJmr/ikJsVomNBDtUKKzem/GF5xhizx7KgEqfuRe6eOUvXb0lzTYwxpvmyoBKn7kX5ACxdX5nmmhhjTPNlQSVOdqZijDH1s6ASp3BQKVtnZyrGGBOLBZU49djZ/GVnKsYYE4sFlTjtbP4qtzMVY4yJxYJKnLoWuqCyonwrVdV2Zb0xxkRjQSVOuVkhinIzqKpRVlZsS3d1jDGmWbKgkoBOrUIALLXOemOMicqCSgI65rugUmad9cYYE5UFlQTsPFOxCyCNMSaqlAUVEekhIh+IyFciMktErvbpN4vIMhGZ5qczAstcLyLzRWSOiJwaSD/Np80XkT8E0vuIyCQRmSciL4hIdqq2B4JBxc5UjDEmmlSeqVQBv1PV/YEhwJUiMtDPu1dVD/XTKAA/70fAAcBpwD9FJCQiIeAfwOnAQOCCQDl3+bL6A+uBS1O4PXamYowx9UhZUFHVFar6mX9dAXwFdKtjkbOB51V1m6ouBOYDR/ppvqouUNXtwPPA2SIiwEnAy375J4FzUrM1TrhPxc5UjDEmuibpUxGR3sBhwCSfdJWITBeRESJS5NO6AWWBxZb6tFjp7YFyVa2KSE+ZcFBZscGuVTHGmGhEVVO7ApHWwDjgdlV9RUQ6A2sABW4FilX1EhH5BzBRVZ/xyz0OjMIFvlNV9ec+/ULc2cstPn8/n94DGKWqB0Wpw3BgOEBxcXHJyJEjG7QtlZWVXD12E+u21vDQGR13NodFy5efnx93mfHktTKtTCvTymyqMqMZNGjQVFUdVG9GVU3ZBGQBo4FrYszvDcz0r68Hrg/MGw0c5afRgfTr/SS44JTp02vlizWVlJRoQ5WWlur3//mx9rruTf1k/po68yVSZjLzWZlWppVpZTa2zGiAUo3jdz+Vo78EeBz4SlXvCaQXB7J9D5jpX78B/EhEckSkD9AfmAxMAfr7kV7ZuM78N/xGfgCc65cfBryequ0J67HzFvjWWW+MMZFS+Yz6o4ELgRkiMs2n/RE3eutQXPPXIuAXAKo6S0ReBL7EjRy7UlWrAUTkKtyZSwgYoaqzfHnXAc+LyG3A57ggllLd7W7FxhgTU8qCiqpOwDVRRRpVxzK3A7dHSR8VbTlVXYDrX2ky9rAuY4yJza6oT5A9VtgYY2KzoJIgO1MxxpjYLKgkqLgwFxFYsWELO+xaFWOMqcWCSoJyMkN0aZNLjcI3G7amuzrGGNOsWFBpgHATWJn1qxhjTC0WVBrAhhUbY0x0FlQawDrrjTEmOgsqDdDdrqo3xpioLKg0wM7mr3V2pmKMMUEWVBrAzlSMMSY6CyoNUNw2jwyBbzZuZXuVXatijDFhFlQaIDszw65VMcaYKCyoNJDdA8wYY3ZnQaWBbFixMcbszoJKA1lnvTHG7M6CSgOFm7/K7EzFGGN2sqDSQN3b2ZmKMcZEsqDSQD3s/l/GGLMbCyoN1KVtrl2rYowxESyoNFBWKIPitnmougd2GWOMsaDSKN1sWLExxtRiQaURdj6sa5111htjDFhQaRTrrDfGmNosqDSCXQBpjDG1WVBpBHussDHG1GZBpRHs/l/GGFObBZVGKG6bSyhDWFmxlW1V1emujjHGpJ0FlUbIDLnnqqjCinJ7rooxxlhQaaSdw4qts94YYyyoNFaPdtZZb4wxYRZUGsmGFRtjzC4pCyoi0kNEPhCRr0Rklohc7dPbicgYEZnn/xb5dBGR+0VkvohMF5HDA2UN8/nniciwQHqJiMzwy9wvIpKq7YnFhhUbY8wuqTxTqQJ+p6r7A0OAK0VkIPAHYKyq9gfG+vcApwP9/TQceAhcEAJuAgYDRwI3hQORzzM8sNxpKdyeqGxYsTHG7JKyoKKqK1T1M/+6AvgK6AacDTzpsz0JnONfnw08pc6nQKGIFAOnAmNUdZ2qrgfGAKf5eW1UdaKqKvBUoKwmY81fxhizS5P0qYhIb+AwYBLQWVVXgAs8QCefrRtQFlhsqU+rK31plPQm1aWNv1Zl4za27rBrVYwxLZu4g/wUrkCkNTAOuF1VXxGRclUtDMxfr6pFIvIWcIeqTvDpY4FrgZOAHFW9zaffCFQC433+b/n0Y4FrVfWsKHUYjmsmo7i4uGTkyJEN2pbKykry8/N3S7981GpWba7mgdM60LUgM2a+RMpsaD4r08q0Mq3MxpYZzaBBg6aq6qB6M6pqyiYgCxgNXBNImwMU+9fFwBz/+mHggsh8wAXAw4H0h31aMTA7kF4rX6yppKREG6q0tDRq+o8enqi9rntTx81ZVWe+RMpsaD4r08q0Mq3MxpYZDVCqcfzup3L0lwCPA1+p6j2BWW8A4RFcw4DXA+kX+VFgQ4AN6prHRgOniEiR76A/BRjt51WIyBC/rosCZTUp66w3xhgnM4VlHw1cCMwQkWk+7Y/AncCLInIpsAT4oZ83CjgDmI9r3voZgKquE5FbgSk+3y2qus6/vhx4AsgD3vZTk9s1rNg6640xLVvKgoq6vpFY142cHCW/AlfGKGsEMCJKeilwYCOqmRR2pmKMMY5dUZ8Edv8vY4xxLKgkgd3/yxhjHAsqSdC5TS6ZGcLqCrtWxRjTsllQSYJQhtC10DWBLSu3sxVjTMtlQSVJrLPeGGMsqCSN3QPMGGMsqCSN3QLfGGMsqCTNzmHF6+xMxRjTcllQSRIbVmyMMRZUksY66o0xxoJK0nQqyCUrJKzZtI1t1al9nIAxxjRXFlSSJHityurNdgGkMaZlsqCSROEmsNWVFlSMMS2TBZUk6l7oOutX2pmKMaaFsqCSRD3aWfOXMaZls6CSROELIFdZUDHGtFAWVJLI+lSMMS2dBZUksjMVY0xLZ0EliToV5JAVEjZsq2HLdgssxpiWx4JKEmVkCN0K7W7FxpiWy4JKkoWbwKaVlaNqV9YbY1qWhIOKiBSJyMGpqMzeoHcHF1R+//J0ht75Pn/473TemfkNFVt3pLlmxhiTepnxZBKRD4Hv+vzTgNUiMk5Vr0lh3fZIvzx+H5Z9s5oZa2pYsWErz08p4/kpZWRmCIN6F3HCgE6cMKAjAzoXICLprq4xxiRVXEEFaKuqG0Xk58C/VfUmEZmeyortqboX5XPVEW057LDD+XLFRj6cs4oP56zmsyXr+XTBOj5dsI47355Ncdtcjt+3I91CW+k1YBsdWueku+rGGNNo8QaVTBEpBs4D/pTC+uw1MjKEA7u15cBubbnqpP6UV27no3lr+HDOasbNXb3zLAbg7k/fo1+n1gzu044hfdszuG87OhXkpnkLjDEmcfEGlb8Ao4EJqjpFRPoC81JXrb1PYX42Zx3SlbMO6UpNje48ixk9bRHz1lcxf9Um5q/axH8mLQGgb8dWDO7TniF92zG4T/s0194YY+ITb1BZoao7O+dVdYGI3JOiOu31gmcxR7XdyEGHHMaMZeW+eWwtpYvWs2D1Zhas3sxzk12Q6V4Q4okeFfTrVJDm2htjTGzxBpUHgMPjSDMNkJ2ZQUmvdpT0aseVJ/ZjR3UNM5ZtYNLOILOOpRXVPPbRQu78gQ28M8Y0X3UGFRE5ChgKdBSR4EivNkAolRVrybJCGRzes4jDexZx+Qn7MH/VJr51zzhGfrGcP581kPzseI8FjDGmadV3nUo20BoXfAoC00bg3NRWzYT169SaAe2z2Ly9mlEzvkl3dYwxJqY6D3lVdRwwTkSeUNXFTVQnE8VJvfOYs3YHL5aWcW5J93RXxxhjoor3ivocEXlERN4VkffDU0prZmoZ2iOXvKwQkxeuY9GazemujjHGRBVvUHkJ+By4Afh9YIpJREaIyCoRmRlIu1lElonIND+dEZh3vYjMF5E5InJqIP00nzZfRP4QSO8jIpNEZJ6IvCAi2XFuyx4pPyuD0w/qAsDLU5emuTbGGBNdvEGlSlUfUtXJqjo1PNWzzBPAaVHS71XVQ/00CkBEBgI/Ag7wy/xTREIiEgL+AZwODAQu8HkB7vJl9QfWA5fGuS17rPMG9QBcUKmusZtVGmOan3iDykgRuUJEikWkXXiqawFVHQ+si7P8s4HnVXWbqi4E5gNH+mm+qi5Q1e3A88DZ4m6adRLwsl/+SeCcONe1xxrcpx292ufzzcatfDRvdbqrY4wxu4k3qAzDNXd9Akz1U2kD13mViEz3zWNFPq0bUBbIs9SnxUpvD5SralVE+l5NRPih76R/yZrAjDHNkKTymR8i0ht4U1UP9O87A2sABW4FilX1EhH5BzBRVZ/x+R4HRuGC3qmq+nOffiHu7OUWn7+fT+8BjFLVg2LUYzgwHKC4uLhk5MiRDdqeyspK8vPzk5avIWWuqazml2+tJpQBj53ZiYKc3Y8LmkM9rUwr08rcc8uMZtCgQVNVdVC9GVW13gm4KNoUx3K9gZn1zQOuB64PzBsNHOWn0YH06/0kuOCU6dNr5atrKikp0YYqLS1Nar6Glnnh45O013Vv6r8nLEhamcnKa2VamVbmnl9mNECpxvEbG2/z1xGB6VjgZtzzVRLi73Qc9j0gPDLsDeBHIpIjIn2A/sBkYArQ34/0ysZ15r/hN/ADdl2AOQx4PdH67KnOG+SawF4stSYwY0zzEtf9PlT1V8H3ItIWeLquZUTkOeAEoIOILAVuAk4QkUNxzV+LgF/48meJyIvAl0AVcKWqVvtyrsKduYSAEao6y6/iOuB5EbkNN9z58Xi2ZW/w7YGdKczP4ssVG5m5bAMHdmub7ioZYwwQ/w0lI1XiziZiUtULoiTH/OFX1duB26Okj8L1r0SmL8D1r7Q4OZkhzj6kK09OXMzLU5daUDHGNBtxNX+JyEgRecNPbwFzaEHNTc3RD/01K69+voytO6rTXBtjjHHiPVP5W+B1FbBYVa1BP40O7NaWgcVt+HLFRt77aiVnHtw13VUyxpj4zlTU3VhyNu4OxUXA9lRWysTHOuyNMc1NvM1f5+FGY/0Q95z6SSJit75Ps7MP7UZ2KIOP5q1mefmWdFfHGGPiHlL8J+AIVR2mqhfhOshvTF21TDyKWmXz7YGdUYVXPrOzFWNM+sUbVDJUdVXg/doEljUp9MNAE1iN3WTSGJNm8QaGd0RktIhcLCIXA28RZZivaXrH9u9Ilza5LFlXyeRF8d6/0xhjUqPOoCIi/UTkaFX9PfAwcDBwCDAReKQJ6mfqEcqQnU+CfMk67I0xaVbfmcp9QAWAqr6iqteo6m9xZyn3pbpyJj7hoDJqxgoqtu5Ic22MMS1ZfUGlt6pOj0xU1VLcDSFNM9C7QyuO7NOOLTuqeWv6inRXxxjTgtUXVHLrmJeXzIqYxgk/FfLF0rJ6chpjTOrUF1SmiMhlkYkicinuQV2mmTjjoC60yg7x2ZJylm6sqn8BY4xJgfpu0/Ib4FUR+Qm7gsggIBt363rTTORnZ3LmwV15obSMl7/aRJviVeRnh2iVk0nrnEzyc0K0zskkLyuEexqzMcYkX51BRVVXAkNF5ETgQJ/8lqq+n/KamYSdd0R3Xigt46MlW/noiSlR84hAq+xMWuWEaJVRzSM9NtGvU+smrqkxZm8V7/NUPsA9FMs0Y4f3LOIPp+/HuBmLyM4vYPO2KjZvr3Z/t1WxeXsVW3fUsGlbFZu2uSayu96ZzaMX1f+EUGOMiUdDn6dimiER4ZfH78MRrcspKSmJmqequobN26tZtXEr37l/PGO+XMmMpRs4qLs9k8UY03h2q5UWJjOUQdu8LPp3LuD0fq0AuO+9uWmulTFmb2FBpQU7e0Ar8rNDjJ29imll5emujjFmL2BBpQVrm5PBsKG9Abh3jJ2tGGMaz4JKCzf82L60yg4xbu5qpi5en+7qGGP2cBZUWriiVtn87Og+gPWtGGMaz4KK4efH9qEgJ5OP5q1h8kK7fb4xpuEsqBgK87O55Bh3tmJ9K8aYxrCgYgC45Jg+FORmMnHBWiZ+vTbd1THG7KEsqBgA2uZlcdmxfQG49725qNqjiY0xibOgYnb62dG9aZuXxeSF6/jEzlaMMQ1gQcXsVJCbxfDj3NnKPWPsbMUYkzgLKqaWYUN7U5SfxdTF6/lo3pp0V8cYs4exoGJqaZ2TyS+O3wewsxVjTOIsqJjdXHRUL9q3ymZaWTkfzlmd7uoYY/YgFlTMbvKzM/mlP1uxkWDGmERYUDFR/XRILzq0zmH60g2M/WpVuqtjjNlDpCyoiMgIEVklIjMDae1EZIyIzPN/i3y6iMj9IjJfRKaLyOGBZYb5/PNEZFggvUREZvhl7hd78HpS5WWHuOIEO1sxxiQmlWcqTwCnRaT9ARirqv2Bsf49wOlAfz8NBx4CF4SAm4DBwJHATeFA5PMMDywXuS7TSD8e3JNOBTnMWr6Rycu3pbs6xpg9QMqCiqqOByLvTng28KR//SRwTiD9KXU+BQpFpBg4FRijqutUdT0wBjjNz2ujqhPVHUI/FSjLJEluVogrT+wHwOOfb6R0kd1s0hhTt6buU+msqisA/N9OPr0bUBbIt9Sn1ZW+NEq6SbLzj+jBYT0LWbulhvMensjd785hR3VNuqtljGmmJJVt5SLSG3hTVQ/078tVtTAwf72qFonIW8AdqjrBp48FrgVOAnJU9TaffiNQCYz3+b/l048FrlXVs2LUYziuqYzi4uKSkSNHNmh7Kisryc/PT1q+PaXMHTXKM9PW89bX21GgX1EWVw9uS9eCzGZVTyvTyrQyk1NmNIMGDZqqqoPqzaiqKZuA3sDMwPs5QLF/XQzM8a8fBi6IzAdcADwcSH/YpxUDswPptfLVNZWUlGhDlZaWJjXfnlbmxK/X6NA7xmqv697U/W54W5+dtFhramqaXT2tTCvTymxcmdEApRrHb2xTN3+9AYRHcA0DXg+kX+RHgQ0BNqhrHhsNnCIiRb6D/hRgtJ9XISJD/KiviwJlmRQZ0rc9o64+lrMP7cqWHdVc/8oMhj89lXWbt6e7asaYZiKVQ4qfAyYCA0RkqYhcCtwJfFtE5gHf9u8BRgELgPnAo8AVAKq6DrgVmOKnW3wawOXAY36Zr4G3U7UtZpe2eVn8/UeHcd/5h1KQk8mYL1dy6n3j+XCOXctijIHdG8WTRFUviDHr5Ch5FbgyRjkjgBFR0kuBAxtTR9Nw5xzWjUG9i7jmhS+YvGgdF/97ChcP7c0pXex6FmNaMrui3jRY96J8nhs+hN+fOoDMDOGJTxZx3XtrKVtXme6qGWPSxIKKaZRQhnDlif145Yqh9O3QirKNVQwbMdn6WYxpoSyomKQ4uHshr111NL3aZrJgzWYueWIKldur0l0tY0wTs6BikqZNbhY3HFtEt8I8ppWVc+V/PrMLJY1pYSyomKRqlxfiqUuPpCg/iw/mrOaPr8ywm1Ea04JYUDFJt0/H1oy4+AjyskK8NHUpd787N91VMsY0EQsqJiUO61nEP35yGKEM4cEP5vPUxEXprpIxpglYUDEpc9J+nbnj+wcBcNMbsxg1Y0Waa2SMSTULKialzhvUg9+fOgBV+M3z0/h0wdp0V8kYk0IWVEzKXXHCPlx0VC+2V9dw2VOlzP5mY7qrZIxJEQsqJuVEhJvOOoDTD+xCxVZ3ceSy8i3prpYxJgUsqJgmEcoQ7j3/UI7s046VG7dx0eOTqNhm17AYs7exoGKaTG5WiEcvGsSAzgV8vXoz141dy1MTF9mV98bsRSyomCbVNi+LJy85kr4dW7FyczV/fn0WQ/53LHe8/RXLrUmOVSOvAAAgAElEQVTMmD2eBRXT5Lq0zeXd3xzH74YUcnjPQjZureLhcQs49q8f8KvnPmdaWXm6q2iMaaCUPU/FmLpkhjIY2iOXX51TwudL1vP4hIW8PfMbRn6xnJFfLKekVxGXHtOHUwZ2TndVjTEJsKBi0u6wnkU8+OMilpVv4alPFvHs5CVMXbyeqYvX060wj2/1yuSAg6vJzQqlu6rGmHpY85dpNroV5nH9Gfvz6fUn85fvHkDv9vksK9/Ck19UcOp94xk3d3W6q2iMqYcFFdPstMrJZNjQ3rz/uxN49KJB9GiTyeK1lQwbMZkr//MZ32zYmu4qGmNisOYv02xlZAjfHtiZNpvbM62yiPvem8dbM1bw4ZxVXHPKAIYd1YvMkB0XGdOc2DfSNHuZGcIvjt+H9353PKcM7Mzm7dXc+uaXnPXgx3y2ZH26q2eMCbCgYvYY3QrzeOSiQTw+bBDdi/L4asVGfvDQJ1z/ygzKK7enu3rGGCyomD3Qyft3Zsxvj+eKE/YhM0N4bvISTrp7HC+VltlTJo1JM+tTMXukvOwQ1562H98/vBs3vDaTTxes4/cvT6d1ttD1o3F0aJ1Dx4KcKH+z6ViQQ7UFH2NSwoKK2aP161TAc5cN4bVpy/jfUbNZXbGNuSs3MXflpjqXy80UfrP5a35+TB/r7DcmiSyomD2eiPC9w7rz3UO68eHEKXTtux+rK7axZtO2iL/bd75fu3k7d749m7dnrOCv5x7CgC4F6d4MY/YKFlTMXiOUIRTmhti/uA37F9ed97G3PmHE9C18sXQDZz7wEb8+qT+/PGEfsuysxZhGsW+QaZEO65LD6N8ex08G92RHtXL3mLmc/eDHzFy2Id1VM2aPZkHFtFgFuVnc/r2DePbng+nRLo8vV2zknH98zN3vzmFbVXW6q2fMHsmCimnxhvbrwOjfHMfFQ3tTrcoD78/nzPsn2C34jWkA61MxBsjPzuTm7x7Adw4u5rqXpzNv1Sa+/8+PuezYvpQUVLO6YhsZ4vptRIRQhpAhkCFChn9v18gYk6agIiKLgAqgGqhS1UEi0g54AegNLALOU9X1IiLA34EzgErgYlX9zJczDLjBF3ubqj7ZlNth9j5H9G7HqKuP5d4xc3n0owU8PH6Bm/HWe/Uum58p/HLDPIYf19du029arHQ2f52oqoeq6iD//g/AWFXtD4z17wFOB/r7aTjwEIAPQjcBg4EjgZtEpKgJ62/2UrlZIa4/Y39eueJoSnoVUZibQYfW2RTlZ9E2L4uCnEzys0PkZmWQHcrwZy9QWaXcM2Yu3753HO/O+sbOXEyL1Jyav84GTvCvnwQ+BK7z6U+p+4Z+KiKFIlLs845R1XUAIjIGOA14rmmrbfZWh/Yo5L+XD2Xq1KmUlJTUm/+Jtz/hudlVzFlZwfCnp3Lcvh256ayB7NOxdRPU1pjmIV1nKgq8KyJTRWS4T+usqisA/N9OPr0bUBZYdqlPi5VuTFoc1CmHt359DDedNZCC3EzGz13NafeN545RX7FpW1W6q2dMk5B0nKKLSFdVXS4inYAxwK+AN1S1MJBnvaoWichbwB2qOsGnjwWuBU4CclT1Np9+I1CpqndHWd9wXNMZxcXFJSNHjmxQvSsrK8nPz09aPitz7y1zw9Zq/jNzE+8v3IICRbkZXHhwAcf1zEVEmk09rUwrM16DBg2aGuiuiE1V0zoBNwP/A8wBin1aMTDHv34YuCCQf46ffwHwcCC9Vr5YU0lJiTZUaWlpUvNZmXt/mdOWrNezH5ygva57U3td96b+4J8f64yl5c2unlamlVkfoFTj+E1v8uYvEWklIgXh18ApwEzgDWCYzzYMeN2/fgO4SJwhwAZ1zWOjgVNEpMh30J/i04xpNg7pUcgrlw/l/849mA6tsyldvJ7vPjiBm8et48bXZjJiwkI+mLOKRWs2U1Vdk+7qGtNo6eio7wy86kYKkwk8q6rviMgU4EURuRRYAvzQ5x+FG048Hzek+GcAqrpORG4Fpvh8t6jvtDemOcnIEH44qAenHtiFv783jyc+WcSMVduZsWpxrXyZGULPdvn06dCK3h1a0adDK7at3UZmWTlt8rIoyM2kIDeTnEwbrmyaryYPKqq6ADgkSvpa4OQo6QpcGaOsEcCIZNfRmFRok5vFjWcO5LJj+/L6+KmECotZuGYzi9ZuZuHqzSzfsJUFazazYM3m2gt+9HGtt9mZGbTJzaKNDzLhgFPEJor32ULXwrwm3CpjamtOQ4qNaRG6tM1lUNdcSkr61krfsr2axetcgFnoA83sslVoVi4VW6uo2FrFxi072F5Vw5pN7hb+kZ6d+T5D+rTn+4d34/SDimmdY19x07TsE2dMM5GXHWK/Lm3Yr0ubnWmR18ioKlt31FCxdQcbt1bt/Lt+83Ze+mQ2U1ZsZ+KCtUxcsJYbX5/JqQd04XuHdeOYfh3sYWSmSVhQMWYPIiLkZYfIyw7RqU3teT1qvqH/AQfz9owV/PezZUxeuI7Xpy3n9WnL6ViQw9mHdOV7h3ezK/1NSllQMWYv0iY3i/OP6Mn5R/SkbF0lr32+jFc/X8aCNZt5bMJCHpuwkM6tQnT45CN/M0wXqII3xxR/48wMEXZsqWDfspl0aJ1Dh4Js97d1Dh39+/xs+wkxtdknwpi9VI92+fzq5P5cdVI/ppWV8+rny3jji+Ws3LyDlZs3xl3OpGWLY85rlR2iQ0EO2bqDwimfkCFCZkgIZWQQEghlZJCZIYRCQsjP60AlAw+qJi/bRrHtjSyoGLOXExEO61nEYT2LuOE7A3lz/GT2HbA/qlCtSo2/aK1GoaZGqVZ182qUaV/OoaBjNzcwoGL7zgECqyu2sWbTdjZvr2bz2kq3onXr467Ty3Pe58IhvbjoqF60b52Toi036WBBxZgWJDszg15tsziwW9u48reqWEJJSZ+o81SVjVurWLNpG5M+n0G//gOoqqmhukapqlFq/N/g+4ptVTw5bg7z12/n72Pn8a9xX3NuSXcuO7YvvTu0SuammjSxoGKMaRARoW2eexxAeftsSvq0i2u5/TNXU92uD4+MX8DY2av4z6QlPDt5CacO7MLw4/tyeE97gsWezIKKMaZJiQiD+7ZncN/2zFtZwaMfLeC1z5fzzqxveGfWNxzRu4jhx+1DoY1S2yNZUDHGpE3/zgX89dxD+J9TBvDvTxbxzKeLmbJoPVMWldI+L4NOH39EVsg9rjkzQ8jMyPADAXa9D4WEjeXldPr6C7JCbjBAph8gkBnK2Ll8ViiDNd9sZlX2CroV5dGtMI92rbLxt4wySWJBxRiTdp3a5HLdaftx5Yn9eGFKGSMmLGRZ+RbWbol/lBpLlsaV7d9ffLbzdV5WaGeACf/t7v8u21hFl/It5Ge564JyMjMsAMXBgooxptlonZPJpcf0YdhRvXhz/BT67bvfzo7+qupdgwCqamqoqnaDAHbUKPO/XkD3nr2oqt41r6qmhh3VuiutRllYtoKq7NYsXb+FZeVbqNhaxfxVm5i/alP0Co1+f+fLDHFBKHzxaX5WJrnZIaq2VtJu2iSy/FlRZsg9ZjozQ8jKzCDLnyVlhjJYs6qCj8vnkRkSsjJ25c/yZ1fh/GUrtsLi9RTmZ1Ho+632lDsiWFAxxjQ7maEMerTJjHuU2tSqFZSU9Kg/39TNtW57s2HLDpb5ALNsfaX7W76FZeVbWb2+gpqMLLbsqGbL9mq2V9e4IdTbq3cvePWauLeNr+bGl2/CJ7XeFuRkUtgqi8K8bArzXaApzM9i47qNvLd6tgtqPpBlZgjZmRkuUIVcUCsr20qP/lvp1CY3/ro2gAUVY0yLFR69NrBrm93mRd53raq6ZmeA2bKjmsrtbpo+6yv67NOPqmplR3UNO2qUHVU1VNXUsL3anWHtqHZnTUvKltGxcxd2hM+mauV3y1dVKyvXroPsfMord1BeuZ0NW3ZQsa2Kim1VlLFl9w2Z+3Vc23vg/hstqBhjTHOQGcqgIJRBQW5W7RlrcigZ0CmuMqZO3UBJyYA48tUOaOFrfDZU7mB95XbKt+wKNvMXLqFzcVcfuFxQ2u7/hoPZjuoaVq9dS+eC1AYUsKBijDHNXkbGrmuCerav/Zz5qdlrKSnpV28ZU6dOjXpGlmx7Rs+PMcaYPYIFFWOMMUljQcUYY0zSWFAxxhiTNBZUjDHGJI0FFWOMMUljQcUYY0zSWFAxxhiTNKIt7JkFIrIaiP3Q7bp1AOK5yU+8+axMK9PKtDKbY5nR9FLVjvXmUv98apvqn4DSZOazMq1MK9PKbI5lNmay5i9jjDFJY0HFGGNM0lhQScwjSc5nZVqZVqaV2RzLbLAW11FvjDEmdexMxRhjTNJYUDHGGJM0FlSaOREpEpEjReS48JTuOoWJyNP+79Vx5M0QkfMaub6cZC8fq8xo2xTPdtaz/nYJ5M0TkfofEZgCieynOMoKicgzja/VbuX2iSctVUTkh/GkJVBehogMbcByrRq6zpRpinHLe+oEXBRtamSZnYHHgbf9+4HApTHy/hyYAawHPgC2AO8nYbs6A2f6qVMd+Y4GWvnXPwXuwV0AFZ7/JdAL+AIoAtoFpyjljU+gjiMi3rcGxjZyf34WT1odeT+PkXco8OP6PiPAPOAl4Ax8f2aMfGcBc4CF/v2hwBtR8l0NtAHE74PPgFNilJnj6/hH4M/hqTH7CdgXGAvM9O8PBm6Ikm80kJ3A/77e/RmjPlNjlNcWuBco9dPdQNt465OEz1Jcn1FgYoL76EtgiX9/CPDPKPk6+v/5I8CI8NSYba9vsscJ1+2IwOtc4GTcF/epcKKITFDVY0SkAgiOehBAVTXy+Z1PAP8G/uTfzwVewH3oIl3t6/Cpqp4oIvsBfwlmiLLeWiLX788W/g/40NfxARH5vaq+HGXxh4BDROQQ4Fpfx6eA4/38fwHvAH2BqcHV+Dr1jShvjIj8j9/ezYE6rouy7mUi8pCqXi4iRcBbwKNR8j1BPftTRLoA3YA8ETnM1w/cD3KtZ7OKyAW4H7Q+IvJGYFYBsDZy5f5sbR9gGlAd3iQCn5GAfYFvAZfg9vsLwBOqOjci383Akbj/Eao6TUR6RynvElX9u4icivvx+BluX7wbJe/rwAbc/2lblPkJ7SfvUeD3wMO+ntNF5Fngtoh8i4CP/f4M/t/viVKHOven/w4cALQVke8HFm2D+45GMwKYCYTPlC/E7afv+zJjfYd2+w6LyOm4g4JuInJ/xPqrYqz/CeL7zr8rIj8AXlEfEepwL3Aq8Aaukl/EaMV4HfgIeI9d+zOlLKjUQVV/FXwvIm2BpyPyHOP/FsRZbAdVfVFErvfLVYlIrH/2VlXdKiKISI6qzo5sEgmvV0RuAb7x9RPgJ7gfwkh/Ao5Q1VV+uY64D1y0oFKlqioiZwN/V9XHRWRYYN33A/eLyEO4ABP+UI9X1S+ilHeJ/3tlcBPYPfigqjeKyF0i8i+gBLhTVf8bpcx49uepwMVAd9zZVlgF7igu6BNgBe6WFndH5J0eZf2DgIFx/Ajg84zBBdcTgWeAK0TkC+APqjrRZ61S1Q0iEquosHCGM4B/+x+WWAt1V9XT6ikvkf0EkK+qkyNWGe2HdbmfMoj+mQyqb38OwJ1hF+LO6IJ1vCzGMvuo6g8C7/8iItPCbxL47oLbjlLgu9Q+kKoAfhtjmXi/89cArYBqEdlC7APTcL3LIvZ9tDLzVfW6ujYo2SyoJKYS6N/IMjaLSHv8kZGIDMEdQUazVEQKgddwP0TrcR/qaE5V1cGB9w+JyCTgrxH5MsIBxVtL7L61Cv9F+ClwnIiEgKwo+WbjfiBfwX0RnhaRR1X1gWAmVa23zTvi6HMycKP/qyLyfVV9JWKRevenqj4JPCkiP4gRmIJ5F+PuDXdUfXX1ZgJdcIGoTr6eP8U16XwD/Ap3pHkorlksvH9misiPgZCI9Ad+jQt2kaaKyLt+uetFpACoibH6T0TkIFWdEat+iewnb42I7MOufX8uUfaDqv7Fzy9wb3VTHWXWuT9V9XXgdRE5KhCE67NFRI5R1Qm+HkfjmpLx7+vs6wqeSfuDpS9E5FlV3RHn+uP6zicY3Mp8H4yKSDbuM/JVlHxvisgZqjoqgbIbxa5TqYOIjGTXaXEI2B94UVX/0IgyDwceAA7EfYE6AueqarSj4OByx+Paht9R1e1R5n8C/AN43tf5AuBKVR0ake+vuPbX53zS+cD0aEczvjnkx8AUVf1IRHoCJ6jqUxH5pgNHqepm/74Vrn344Ih8F0XbtmB5IvLv2HsBVdVLggmJ7k8R+Q6u+WRnU4mq3hKYn1Bzpoh8gAsKkwk0K6nqd6Osey7uTHKEqi6LmHedqt7lX+fjzihP8bNHA7eq6raIZTL8uheoarn/4eoWbdtF5EvcAdECX8/w9hwcJW8hrs8lfOY5DrhFVTdE5OuLa6sfiuv3Wwj8xAfmYL4D/XaHf7zX4PpJZgXyhL9rBdSxP0XkWlX9q4g8QJQmK1X9dZTtOQTXfNbWJ60HhoX3k4gs9GUFD/vD71VVdzuT9oHpZlyfYmY9ecOf0QOAWcT4jPqzzJ8AfVT1VhHpARSr6uQoZXYA/o5rThVck+fVqro2Il8F7uxnG7CDGJ/jZLKgUgf/Qx5WBSxW1aVJKDcTdxovwJwEjnjqKrM37kN2NO4L8THwG1VdFJHvLmAScIxf/3hgSGNOkUVkBq5Jbat/n4sLRAdF5Aueuezso1LVcxu43gxgCO4HqN796ZvS8oETgceAc4HJqnppQ9bvyzw+WrqqjouS9whcM1L4hyicNzL4DsIFld6BfDsDgIjs55tCD4+x7s+irLsXbjDFsT5pPFAeGQB83v/iAvSTPulC4BBV/X5EvpCqVvuDiAxVrYhWH3/A8ydV/cC/PwH43+ABT6z9GNimcT7fWao6MtgMG5EvXGdE5JpgNXA/ruD6dVSj9+m0wwXf4EFHtP/lbFxz11QCzU6RP+o+by5wFa55sQKYCDwQ/r4E8j2EO9M8SVX3F9eX+K6qHhGlzHYa0RcpIn1UdWFDtylpNIWjAPaGiThHSiVYZlyjhVK0PdFGrUyPeD/B/60ANgamCmBjlOWvwY0Au9lP03ABrb66tCXKqCY/70mgMPC+iCijVkhsxMz0iL+tcV/aptr3c3D9AH1wgaUXgdF08eYDHvF/P4gyRR0diBv0MQM30OMWXP/Qr2LknRZn2hLcmcrJ1D2a7Yt40nz6XfGkxbm/b/LTs7iRd3/D9ZPNBR6Lkj/aaMvdRhz6vJMSqMeLuIOYE/30CPBSlHyf+b+fB9Ji7aePgTaB9/vjR+E1dJuS9jlPZeF7+oQbLbLY/8A9hTu9P7eRZT6Nax//J+6U+AHg/iTUtc6hg8Dl/sO12f+ghKeFwDNJWP/huHbdq4HD4lwmC/gqxrzdhu/GSPsL8IO6ftQCeSf5v58CXXHDbOc1cHsTCrzBZeItO8mf5en44eH+fSsiDiYC8yYCxwTeH02U4A3k+e/IK7gRXg8GlwvkexXXN9bbTzcAr8VYd70HPT5tX/9Zfxd4PzzFKPNdoCDwvgDXjByZbwbuaH6af78f8EKMMu/EjaI8yn/2DwcOj5E3rqCKa0EIsSu4dIz2mffzvoNrlmyNG8gyCzi0MduUrMk66uuWyEipeMU9WihB9Q0dfBZ4G7gDCPYJVWj0Ib0JUdfksluzS1CsPqoY2TNEpEhV1/tl2xF9YEl4xEyViGyl7jbjN31/wf/5uiruCDJhmvioP4CbROQx3LUdwf6CyMEH8ebDd9b2pnZzWrThzELtz0U1tfsQgi7HddjX6oOIzKSqW3D/vxd9U83fcT90IV+3p1X1Qtznsje7BnKMww1/Dm7H5cAVQF/fRxdWgDsqj/QSbsThY9Q/VLYnEOyH3O7rE6ne0ZYB4UExgwJpCpwUJe/nIjJEVT8FEJHBRN+m+3EBuJOI3I5rnr0h2spV9S0RycIHTOAcVZ3XyG1KCgsqdUtkpFS84h4tlKA6hw6q62TdgOvAT5e/BV7X10d1N27EUjiA/xC4PTKTqhZEazOORlVv9S//KyJvArka0fmcYj/DHSlmsWuUluJ+bBPOJ4ldI/NvYJKIvOrfn0P0a6PAjSL6qy+7EPe5OYcoQ6p9X8j5wOnAFHZdCwJQ4vtyhuGafcLXL8HuAS3Rg54qVX0oRv0jPQ1M9tuuwPfY1V8UFPdoS1U9sb6V+r5Gxf0fLxKRJf59L9yFi5Fl/kdEpuKbE3GBotaIrigDFNrgBl/8SkTQ3QcqJDKCNCmso74OiYyUiqOsuEa3NKKutwGfaBMOHWwIEenMrotKJ0cE7ci8A3FHfoJrB97tiygiP8c1uXXH/bgOwe2Hk2OUGe+RfdKJyAyNGLzQyHxfkcBZr+/Y3zlAQ1U/j5HvHaAcdzYX7IS+OyLfQtw+fxHXN7Y5Yv6vcWc9fYHgaLfdRkqJSBtV3SgxhveGA0tg/q+BVbgj+22R+WJs+85BCrG2PZD/eOoebdkW118Tc4ScD6gx6e6j5KJte4UGBp7EGqAQKDNasAwvW+c2JYsFlTr4L0UZ7sMY/iK+WvdSMcs63pdxF+7q9J2zcB2Rg6MuGH/5TT50MFGy+9X8xwJRr+YXN3x5N6q6JCLfDHbddeBQ8XcdUNXzo5QZ9cg+ytFdSojIo8C90YJjA/O9BPxaVZN61isiM1X1wDjytVHVjXHke0hVL68nz5uqemas4b3hAFTH8N9wxt2G9KZCvCPkEixzEdAD19wouLPEFbjgeZmqTo29dPNhzV9164Q7IvoM1/E9uqEF6a4hkVkaMZxPRPIaU0lfftzNQGmUSB/VW+z6scjDjYSagxvrH5RIm3Gq+rPidQwwzP8w1nWtSJ35Is56vxSRpJ71EseFkt52EbmS3a/7qXUtUX0Bxec507+cgBvu/JGqzo6Srw/sPEB5x5/d3IjrKL81Mn8K1XmVfgO9A7yqqqMBROQU4DTcmeA/gcEi8qKqnhdoWqslymepyVlQqYOq3uA/sKfg2rkfFJEXgcdV9etEympAR2RCYjUD4dpnm4u4+6gim39888UvomRNpM04Vf1Z8arvNinx5vsbu856zwmkh9Ma6xjg4jiC39O4uymcihum/BOiX9WdiH/79T8g7uLKz3EB5u8R+W5Qd+uTY4Bv4/rgHmJXB3qq1XmVfgMNUtVfht+o6rsi8r+qeo3sukt0+E7ZZ+6+ePNgzV9xEHdF7s9wX/YPcD/YY1T12joXrF1GW9y1FikZfZVIM1C6NLaPSkQ+U9WoF/z5+VHbjFPdn5Uu0faHiExv7NFqrL6AKH0An6vqYeF1+tFIo1U12gioRNYfwn2WTwR+CWxR1f1irPsOYIaqPhtOa8y6E6hjnVfpN7DMd3Ej/p73SefjAuZpuIuJY372mxM7U6mD71MZhrutxGO49v8d4q7knkftvpE6NcHoqyYfOtgAS3HXQIT7qB6J1Uclta+GzsCNxV9dV+GRzYoBqT6yb1KpPuuNDB51CHcgl4u7Fcs3RB+qGzcRGYvrG5yIG4q8s7k0wjIReRh3m5K7/JF8yp8PFfG5fIraV+l/i+g3HY3Xj3Gd/6/hPpsTfFoIP6pOErijcrpYUKlbB+D7kV8yVa0RkeZ2+tnkQwcboN4+Ktl1bcOfcbf3Bjf8+E0gnpsc7ibV/VlpkNJrjhLwiLjrU27A3RizNe4ix8aYjjuAOBB3EFYuIhPVXRMTdB7uCP5v6u57Voy7DX+qha9LGoA7m3od94P+U1xfUIOp6hrcTUajme/zJHJdVFpY89deqKmGDjaEiAi7+qgG4Tohd/ZRibvx4enASOCEyOUb8qMZPLIHgn1hBcDHqvrTRMs04M8OfoA7OwnfvVo1cIPORpTdGvcZ+R+gi6o26qmfyeabqn6g/n5n4u7A/JLW/3iBusoMXhwctgF3q/2HNeJeYc2VnanshepoBko7VVUR+QbXVFKF62d6WUTCfVThB3/1wX2ZwmI9+CsezeXIfm9T74O/EiUiV+GaR0twt0gagWsGa27ivUo/EQtwt2YJ9jmuxN2S5lHcsOVmz85UTJOJ0kf1WrCPSlX3CeSt99oGk17xXs+SYJm/xzUjTVXVWE9STDsR+ROuCS54lf4LqnpHI8ocr6rHRUsTkVmqGjmcvlmyMxXTlOLuo7KAskeI93qWuKnq/yWrrFRS1dtF5G12XaX/M63nKv04dBSRnuELfP0FwB38vGbVjF0XO1MxxiQkcOFdJnE++MvUT0TOwDX/fo3bl31wfYEf4q6ovy99tYufBRVjTEJiXccSlsCQZBPBD37YDxdUZu8pnfNBFlSMMSaNROQkVX1fRKLeN0yjPPKgObM+FWOMSa/jcQ8ZO8u/Dz4eINqjEZo1O1MxxphmQNyz7MPX/YQP+JNy3U9TsjMVY4xpHl5j13Nswn0pe9xRv52pGGNMM5CK637SIeU3YDPGGBOXT0Sk3id+Nnd2pmKMMc2Av+9dP6C+59g0axZUjDGmGYj3OTbNnQUVY4wxSWN9KsYYY5LGgooxxpiksaBiTAOJyJ9EZJaITBeRaSIyOIXr+lBEBqWqfGOSxS5+NKYBROQo4EzgcFXdJiIdgOw0V8uYtLMzFWMaphhYo6rbwD1fXFWXi8ifRWSKiMwUkUf845PDZxr3ish4EflKRI4QkVdEZJ6I3Obz9BaR2SLypD/7eVlE8iNXLCKniMhEEflMRF7yj95FRO4UkS/9sn9rwn1hzE4WVIxpmHeBHiIyV0T+KSLH+/QHVfUIf2V0Hu5sJmy7f7Lfv3CP4r0SOBC4WETa+zwDgEf8tQkbcc/T2MmfEd0AfEtVD8c9cvkaEWmHe/rgAX7Z21KwzcbUy4KKMQ2gqptwz1EfDqwGXhCRi4ETRWSSf5DVSUDwEbBv+L8zgFmqusKf6SwAevh5Zar6sX/9DHBMxKqHAAOBj0VkGu7xzAZ3JqQAAAEeSURBVL1wAWgr8Ji/hXpl0jbWmARYn4oxDaSq1bin8n3og8gvgIOBQapaJiI3A7mBRbb5vzWB1+H3O+9KG7maiPcCjFHVCyLrIyJHAicDPwKuwgU1Y5qUnakY0wAiMkBE+geSDgXm+NdrfD/HuQ0ouqcfBABwATAhYv6nwNEi0s/XI19E9vXra6uqo4Df+PoY0+TsTMWYhmkNPCAihUAVMB/XFFaOa95aBExpQLlfAcNE5GFgHvBQcKaqrvbNbM/5R8+C62OpAF73z+QQ4LcNWLcxjWa3aTGmmRCR3sCbe8Ptz03LZc1fxhhjksbOVIwxxiSNnakYY4xJGgsqxhhjksaCijHGmKSxoGKMMSZpLKgYY4xJGgsqxhhjkub/AUqU/Ada45uIAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fdist = nltk.FreqDist(list(itertools.chain.from_iterable([tokenizer.tokenize(x) for x in train_data['clean_comment_text'][train_data['toxic']==1]])))\n",
    "fdist.plot(30 , cumulative=False , title = \"Word count - toxic corpus - most common\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 307,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('elyrian', 1),\n",
       " ('unhurt', 1),\n",
       " ('meggie', 1),\n",
       " ('everett', 1),\n",
       " ('engineer', 1),\n",
       " ('subjected', 1),\n",
       " ('dealwrongdoing', 1),\n",
       " ('ponpon', 1),\n",
       " ('wrecking', 1),\n",
       " ('tanthony', 1),\n",
       " ('sandchigger', 1),\n",
       " ('juppiter', 1),\n",
       " ('escorrt', 1),\n",
       " ('caty', 1),\n",
       " ('yrgh', 1),\n",
       " ('snowfire', 1),\n",
       " ('vandalizer', 1),\n",
       " ('pleasedontdeletemyedits', 1),\n",
       " ('geographical', 1),\n",
       " ('hounded', 1),\n",
       " ('passiveaggressive', 1),\n",
       " ('peons', 1),\n",
       " ('wwthevergecom', 1),\n",
       " ('whyareeditorsleavingtheworldsbiggestencyclopedia', 1),\n",
       " ('parks', 1),\n",
       " ('warts', 1),\n",
       " ('fuku', 1),\n",
       " ('perpetual', 1),\n",
       " ('fountains', 1),\n",
       " ('edisontechcentercom', 1),\n",
       " ('xck', 1),\n",
       " ('spk', 1),\n",
       " ('changer', 1),\n",
       " ('meaningfulmeaning', 1),\n",
       " ('iti', 1),\n",
       " ('reddish', 1),\n",
       " ('aryour', 1),\n",
       " ('ticle', 1),\n",
       " ('redrose', 1),\n",
       " ('satin', 1),\n",
       " ('trevor', 1),\n",
       " ('tred', 1),\n",
       " ('oooooooooooooooooooooooooooo', 1),\n",
       " ('ooooooooooooooooooooooooooooooooooooooooooooooooooooo', 1),\n",
       " ('scoobydoo', 1),\n",
       " ('greekwarrior', 1),\n",
       " ('mysterion', 1),\n",
       " ('clyde', 1),\n",
       " ('concise', 1),\n",
       " ('pseudointellect', 1)]"
      ]
     },
     "execution_count": 307,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Least common words\n",
    "fdist.most_common()[-50:]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "As expected stop words are the most common in the whole corpus and even in the 'toxic' corpus. Also, typos and namedreferences are among the most infrequently accuring words. While training model, we will take care to exclude both these classes of words.\n",
    "\n",
    "### Bag of words"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 308,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "vectorizerBow = CountVectorizer(ngram_range=(1,2), stop_words = 'english',max_features =5000, min_df = 50 , max_df = 0.99)\n",
    "\n",
    "X_train_bow = vectorizerBow.fit_transform(train_data['clean_comment_text'])\n",
    "X_test_bow = vectorizerBow.transform(test_data['clean_comment_text'])\n",
    "\n",
    "feature_names_bow = vectorizerBow.get_feature_names()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 309,
   "metadata": {},
   "outputs": [],
   "source": [
    "def fitAndTestmodel(X_train , X_test, y_train, y_test):\n",
    "    logregModel = LogisticRegression(penalty = 'l2' , C = 1)\n",
    "    logregModel.fit(X_train,y_train)\n",
    "    predicted_y_test = logregModel.predict_proba(X_test)\n",
    "    rocauc = roc_auc_score(y_test ,predicted_y_test[:,1] )\n",
    "    print(\"ROC AUC Score -\" ,rocauc)\n",
    "    return "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 310,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC AUC Score - 0.9448415722152508\n"
     ]
    }
   ],
   "source": [
    "fitAndTestmodel(X_train_bow , X_test_bow , y_train, y_test)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 311,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Most important features : article,cock,crap,fucked,fucking,idiots,motherfucker,prick,retard,ur\n"
     ]
    }
   ],
   "source": [
    "# Get most important features\n",
    "\n",
    "def printImportantFeatures(X_train,y_train,feature_names):\n",
    "    X_kbest = fvalue_selector.fit_transform(X_train, y_train)\n",
    "    feature_names.__len__()\n",
    "    y = fvalue_selector.get_support()\n",
    "    important_features = [x for x,i in zip(feature_names,y) if i]\n",
    "    print(\"Most important features :\" , \",\".join(important_features))\n",
    "    return\n",
    "\n",
    "printImportantFeatures(X_train_bow,y_train , feature_names_bow)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### TF-IDF vectors"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 312,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "\n",
    "# using 1 and 2 grams of words\n",
    "# uning l2 normalisation for feature vector of each document\n",
    "# removing stopwords\n",
    "# setting a cap on max features to speed up process and avoid over - fitting\n",
    "# removing too rarely occuring words which are probably typos or named references\n",
    "\n",
    "tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), norm='l2', stop_words = 'english',max_features =5000, min_df = 50 , max_df = 0.99 ) \n",
    "\n",
    "X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['clean_comment_text'])\n",
    "X_test_tfidf = tfidf_vectorizer.transform(test_data['clean_comment_text'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 313,
   "metadata": {},
   "outputs": [],
   "source": [
    "feature_names_tfidf = tfidf_vectorizer.get_feature_names()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Fit logistic model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 314,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC AUC Score - 0.9614729260004594\n",
      "Most important features : ass,asshole,bitch,dick,fuck,fucking,idiot,shit,stupid,suck\n"
     ]
    }
   ],
   "source": [
    "fitAndTestmodel(X_train_tfidf , X_test_tfidf , y_train, y_test)\n",
    "\n",
    "printImportantFeatures(X_train_tfidf,y_train , feature_names_tfidf)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Our relatively simple model using basic tf-idf features does well. Also note that among the 10 most important features - all are words that are umm... toxic. This demostrates why it is a better approach that bow.\n",
    "\n",
    "The model does well also perhaps because the mere presence of certain unigrams correlates very well with the cooment being toxic - something that might not work if the information we needed was much more subtle. It would be interesting to see if learning dense vector representations ccould improve this performance.\n",
    "\n",
    "### GLOVE features\n",
    "\n",
    "Now let's try using the hard work of the good people at Stanford. They have made available vector reprenstations of words - which have been learned in an unsupervised way over a large corpus. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 315,
   "metadata": {},
   "outputs": [],
   "source": [
    "## Tokenizing uning previously used whitespace tokenizer from nltk\n",
    "train_data['clean_comment_text_tokenized'] = [tokenizer.tokenize(x) for x in train_data['clean_comment_text']]\n",
    "test_data['clean_comment_text_tokenized'] = [tokenizer.tokenize(x) for x in test_data['clean_comment_text']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 316,
   "metadata": {},
   "outputs": [],
   "source": [
    "import gensim"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 317,
   "metadata": {},
   "outputs": [],
   "source": [
    "#from gensim.scripts.glove2word2vec import glove2word2vec\n",
    "#glove_input_file = 'glove.6B.100d.txt'\n",
    "#word2vec_output_file = 'word2vec.txt'\n",
    "#glove2word2vec(glove_input_file, word2vec_output_file)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 318,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Loading the glove model from disk\n",
    "from gensim.models import KeyedVectors\n",
    "filename = 'word2vec.txt' #dowloaded model file - converted into word2vec format\n",
    "glove_model = KeyedVectors.load_word2vec_format(filename, binary=False) #load glove model into RAM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 319,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The vector representation for 'the cat sat on the mat' is - \n",
      " [-7.7008998e-01  1.2073100e+00  8.5778999e-01 -1.6087000e+00\n",
      " -1.5735201e+00  2.0247800e+00 -1.9553199e-01  1.2699299e+00\n",
      " -7.3517299e-01 -1.5889711e+00  8.3257002e-01  6.3399598e-04\n",
      "  1.1765200e+00  2.2569002e-01  1.6922300e+00  4.9297997e-01\n",
      "  4.3902999e-01  8.0958605e-01 -1.8552998e-01 -1.7070899e+00\n",
      "  1.3461000e-01  1.2771699e+00  6.0156101e-01 -7.5835001e-01\n",
      "  1.9741200e+00  1.5423601e+00 -1.0521600e+00 -1.1438490e+00\n",
      " -4.0687019e-01  2.0985001e-01 -1.7286998e-01  6.6258156e-01\n",
      "  7.0366597e-01  3.7584999e-01  5.8648002e-01  1.2335500e+00\n",
      " -2.1295300e-01  1.9678899e+00  1.0941401e+00 -5.8835000e-01\n",
      " -6.0928005e-01  8.7169021e-02  9.8383999e-01 -1.5329900e+00\n",
      " -5.7173002e-01  1.5954000e-01 -2.5162599e+00  1.5406300e+00\n",
      "  8.1229001e-01 -6.8897325e-01 -1.1576816e+00 -4.9469000e-01\n",
      "  3.9929998e-01  2.5583701e+00 -1.0256200e+00 -3.0744500e+00\n",
      " -4.1704601e-01  7.0223999e-01  1.7904699e+00  5.0038999e-01\n",
      "  1.0677600e+00  1.6058490e+00 -3.5026002e-01  7.6813006e-01\n",
      "  7.2922999e-01  8.0419654e-01  1.3725700e+00 -4.0538001e-01\n",
      " -4.8859000e-01  9.4517505e-01 -1.3529700e+00  9.7538006e-01\n",
      "  7.9514360e-01 -6.9393003e-01 -9.6833003e-01  4.1580904e-01\n",
      " -8.4608197e-01 -3.1318900e-01  4.5210004e-01 -1.1456007e-01\n",
      "  2.1769002e-01 -1.6674401e+00  4.9176502e-01 -1.1013219e+00\n",
      " -1.1305650e+00 -1.3763500e+00 -5.3480044e-03 -2.8776300e-01\n",
      "  4.5715779e-01  1.2647200e+00  1.0083300e+00 -1.0072800e+00\n",
      "  6.0556006e-01  1.2257880e+00  3.2912999e-01  1.3447002e-01\n",
      "  2.0579994e-02 -1.4043000e-01  1.4917300e+00 -9.8721701e-01]\n"
     ]
    }
   ],
   "source": [
    "glove_vocab = glove_model.vocab\n",
    "from nltk.corpus import stopwords\n",
    "stop_words = set(stopwords.words('english'))\n",
    "\n",
    "def getGloveVector(sentence , model , vocab):\n",
    "    sentence = [x for x in sentence if x in vocab and x not in stop_words ] ## remove words not in dictionary and stop words\n",
    "    gv = np.zeros((1,100))\n",
    "    if(sentence.__len__() > 0):\n",
    "        gv = model.wv[sentence].sum(axis = 0)\n",
    "    return gv\n",
    "\n",
    "print(\"The vector representation for 'the cat sat on the mat' is - \\n\",getGloveVector( ['the', 'cat','sat', 'on','the', 'mat', ] , glove_model , glove_vocab )) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 320,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "def getGloveMatrix( corpus ,glove_model ,glove_vocab  ):\n",
    "    y = [getGloveVector(x , glove_model ,glove_vocab ) for x in corpus] # get vec representation for each sentence\n",
    "    X_glove = np.zeros(( corpus.shape[0] ,100))\n",
    "    for i in range(corpus.shape[0]):\n",
    "        X_glove[i,:] = y[i]\n",
    "        X_glove[i,:] = X_glove[i,:]/(np.linalg.norm(X_glove[i,:] , ord =2) + 1e-4) # normalize each text vector\n",
    "    return X_glove"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 321,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_glove = getGloveMatrix(train_data['clean_comment_text_tokenized'] , glove_model , glove_vocab)\n",
    "X_test_glove = getGloveMatrix(test_data['clean_comment_text_tokenized'] , glove_model , glove_vocab)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 322,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC AUC Score - 0.9336660927302777\n"
     ]
    }
   ],
   "source": [
    "fitAndTestmodel(X_train_glove , X_test_glove , y_train, y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It seems like the performance is subpar compared to out tf-idf vectors. However, we use significantly less number of features - only 100. To get performance that is comparable - we have to use 1000 tf-idf features.\n",
    "Using higher dimentional word embeddings will probably make a better model."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Hybrid approach - word vectors weighted by tifidf vectors\n",
    "So far - to get the glove embedding of a certain chunk of text - I have been adding the vector represntations of all words(which are also in the GOLVE vocab) in the text - . There are a couple of ways I think this could be made more efficient - \n",
    "\n",
    "1. Removing stop words and rare or infrequent words\n",
    "2. instead of adding word representatios of all words - use the TF-idf vector to weight the contribution of each word. This way - words with high tf-idf values for a certain text - have higher weight in the dense vector representation.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 323,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "4023 out of 5000 most common corpus words(excluding stop words and too frequent words) in glove vocab\n"
     ]
    }
   ],
   "source": [
    "y = [i for i in range(feature_names.__len__()) if feature_names[i] in glove_vocab]\n",
    "print( y.__len__() ,\"out of 5000 most common corpus words(excluding stop words and too frequent words) in glove vocab\" )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 324,
   "metadata": {},
   "outputs": [],
   "source": [
    "feature_names_gloveP = [feature_names[x] for x in y]\n",
    "word_vec_features = glove_model.wv[feature_names_gloveP]\n",
    "text_word_tfidf = X_train_tfidf[:,y]\n",
    "\n",
    "X_train_tfidf_glove = text_word_tfidf*word_vec_features\n",
    "X_test_tfidf_glove = X_test_tfidf[:,y]*word_vec_features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 325,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ROC AUC Score - 0.9279955739160318\n"
     ]
    }
   ],
   "source": [
    "fitAndTestmodel(X_train_tfidf_glove , X_test_tfidf_glove , y_train, y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The Tfidf weighted word vectors were not as effective as weights for glove word vectors. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### A summary of results so far\n",
    "| Features | AUC ROC | feature dimentionality |\n",
    "| --- | --- | --- |\n",
    "| Bag of words |  0.944 | 5000 |\n",
    "| Tfidf vectors |  0.961 | 5000 |\n",
    "| Glove  |  0.933  | 100 |\n",
    "| Tfidf weighted Glove |  0.927 | 100 |"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
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
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
