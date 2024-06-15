import nltk
import re
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from django.core.paginator import Paginator
from django.db.models import Q
from django.shortcuts import render, redirect
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz, process

from .models import Smartphones

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

nltk.download('punkt')

stop_factory = StopWordRemoverFactory().get_stop_words()

def stemmed_words(doc):
    return ' '.join([stemmer.stem(word) for word in nltk.word_tokenize(doc) if word.lower() not in stop_factory])

def get_fuzzy_matches(query, choices, limit=5):
    # Menggunakan fuzzywuzzy untuk mencocokkan query dengan data
    results = process.extract(query, choices, limit=limit)
    return results

def preprocess_query(query):
    # Menghapus spasi tambahan
    query = re.sub(r'\s+', ' ', query).strip()
    # Menghapus karakter yang berulang
    query = re.sub(r'(.)\1+', r'\1', query)
    return query

# Create your views here.

def index(request):
    query = request.GET.get('q', '')
    if query:
        query = preprocess_query(query)
        queryset = Smartphones.objects.all()
        data = list(queryset.values())

        df = pd.DataFrame(data)
        df['stemmed_model'] = df['model'].apply(stemmed_words)
        df['stemmed_processor'] = df['processor'].apply(stemmed_words)
        df['stemmed_ram'] = df['ram'].apply(stemmed_words)
        df['stemmed_battery'] = df['battery'].apply(stemmed_words)
        df['stemmed_camera'] = df['camera'].apply(stemmed_words)

        df['combined_text'] = df['stemmed_model'] + ' ' + df['stemmed_processor'] + ' ' + df['stemmed_ram'] + ' ' + df['stemmed_camera'] + ' ' + df['stemmed_battery']

        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(df['combined_text'])

        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        smartphone = []
        suggestion = None
        stemmed_query = [stemmed_words(query)]
        X_query_counts = count_vect.transform(stemmed_query)
        X_query_tfidf = tfidf_transformer.transform(X_query_counts)

        cosine_similarities = cosine_similarity(X_query_tfidf, X_train_tfidf).flatten()
        df['weight'] = cosine_similarities
        df = df.sort_values(by='weight', ascending=False)

        smartphone = df[df['weight'] > 0].to_dict(orient='records')

        if len(smartphone) == 0:
            choices = df['combined_text'].tolist()
            fuzzy_matches = get_fuzzy_matches(query, choices)
            if fuzzy_matches:
                suggestion = fuzzy_matches[0][0]
    else:
        smartphone = Smartphones.objects.all().order_by('id')
        paginator = Paginator(smartphone, 20)
        page_number = request.GET.get('page')
        smartphone = paginator.get_page(page_number)

    context = {
        'smartphone': smartphone,
        'query': query,
    }

    return render(request, "index.html", context)

# def search(request):
#     query = request.GET.get('q', '')

#     if query:
#         query = preprocess_query(query)

#     queryset = Smartphones.objects.all()
#     data = list(queryset.values())

#     df = pd.DataFrame(data)
#     df['stemmed_model'] = df['model'].apply(stemmed_words)
#     df['stemmed_processor'] = df['processor'].apply(stemmed_words)
#     df['stemmed_ram'] = df['ram'].apply(stemmed_words)
#     df['stemmed_battery'] = df['battery'].apply(stemmed_words)
#     df['stemmed_camera'] = df['camera'].apply(stemmed_words)

#     df['combined_text'] = df['stemmed_model'] + ' ' + df['stemmed_processor'] + ' ' + df['stemmed_ram'] + ' ' + df['stemmed_camera'] + ' ' + df['stemmed_battery']

#     count_vect = CountVectorizer()
#     X_train_counts = count_vect.fit_transform(df['combined_text'])

#     tfidf_transformer = TfidfTransformer()
#     X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#     results = []
#     suggestion = None
#     if not query:
#         stemmed_query = [stemmed_words(query)]
#         X_query_counts = count_vect.transform(stemmed_query)
#         X_query_tfidf = tfidf_transformer.transform(X_query_counts)

#         cosine_similarities = cosine_similarity(X_query_tfidf, X_train_tfidf).flatten()
#         df['weight'] = cosine_similarities
#         df = df.sort_values(by='weight', ascending=False)

#         results = df[df['weight'] > 0].to_dict(orient='records')

#         if len(results) == 0:
#             choices = df['nama'].tolist()
#             fuzzy_matches = get_fuzzy_matches(query, choices)
#             if fuzzy_matches:
#                 suggestion = fuzzy_matches[0][0]

#     return render(request, 'index.html', {'results': results, 'query': query, 'suggestion': suggestion})

def deleteSmartphone(request, id):
    mkn = Smartphones.objects.get(id=id)
    mkn.delete()
    return redirect("/alldata")


def test(request):
    query = request.POST.get('q', '')

    queryset = Smartphones.objects.all()
    data = list(queryset.values())

    # Convert the food data into a DataFrame
    df = pd.DataFrame(data)
    df['stemmed_deskripsi'] = df['deskripsi'].apply(stemmed_words)
    df['stemmed_bahan'] = df['bahan'].apply(stemmed_words)
    df['stemmed_nama'] = df['nama'].apply(stemmed_words)
    df['combined_text'] = df['stemmed_deskripsi'] + ' ' + df['stemmed_bahan'] + ' ' + df['stemmed_nama']


    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(df['combined_text'])

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    if query:
        stemmed_query = [stemmed_words(query)]
        X_query_counts = count_vect.transform(stemmed_query)
        X_query_tfidf = tfidf_transformer.transform(X_query_counts)
        cosine_similarities = cosine_similarity(X_query_tfidf, X_train_tfidf).flatten()
        df['weight'] = cosine_similarities
        df.sort_values(by='weight', ascending=False, inplace=True)

    # Print the DataFrame
    # print(df[['name', 'deskripsi', 'stemmed_deskripsi', 'weight']])
    return render(request, 'search_test.html', {'data': df.to_html()})


# def search():
#     queryset = Smartphones.objects.all()
#
#     data = list(queryset.values())
#
#     # Convert the food data into a DataFrame
#     df = pd.DataFrame(data)
#     dd(data)
