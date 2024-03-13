import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt
import wordcloud
from wordcloud import WordCloud


def plot_word_cloud(data, type):
    email_corpus = "".join(data['v2'])
    plt.figure(figsize=(7, 7))
    wc = WordCloud(background_color='black',
                   max_words=100,
                   width=800,
                   height=400,
                   collocations=False).generate(email_corpus)
    plt.imshow(wc, interpolation='bilinear')
    plt.title(f'WordCloud for {type} emails', fontsize=15)
    plt.axis('off')
    plt.show()


def main():
    spam = pd.read_csv('spam.csv')
    print(spam.shape)
    sns.countplot(x='v1', data=spam)  # plots the amount of spam vs non spam emails
    plt.show()

    spam_msg = spam[spam.v1 == 'spam']
    ham_msg = spam[spam.v1 == 'ham']
    # Downsampling to balance data
    ham_msg = ham_msg.sample(n=len(spam_msg), random_state=42)
    balanced_data = ham_msg.append(spam_msg) \
        .reset_index(drop=True)
    print(balanced_data)
    # Plots the word cloud of most used words from spam and non spam emails
    plot_word_cloud(balanced_data[balanced_data['v1'] == 'ham'], type='Non-Spam')
    plot_word_cloud(balanced_data[balanced_data['v1'] == 'spam'], type='Spam')

    z = spam['v2']
    y = spam["v1"]
    # z_train trains the input
    # z_test tests the input
    # y train trains the labels and y_test tests the labels
    # test size is set to 20% of z and y
    z_train, z_test, y_train, y_test = train_test_split(z, y, test_size=0.2)

    cv = CountVectorizer()  # randomly assigns a number to each word (tokenization)
    features = cv.fit_transform(z_train)  # counts occurrences of each word

    # Support Vector machine algo
    # Linear model for classification and regression
    model = svm.SVC()
    model.fit(features, y_train)

    # Makes predictions from z_test
    features_test = cv.transform(z_test)
    print('Accuracy: {}'.format(model.score(features_test, y_test)))


if __name__ == "__main__":
    main()
