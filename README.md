### Założenia i cele projektu

Celem projektu jest zbadanie możliwości ekstrakcji, strukturyzacji i analizy danych tekstowych z portalu społecznościowego Reddit. Badanie polegać będzie na eksploracji metod grupowania komentarzy zaciągniętych z portalu.

Jednym z głównych wyzwań, pojawiających się w analizach danych z mediów społecznościowych, jest problem inkorporacji danych wizualnych, takich jak zdjęcia, grafiki i nagrania. W przeciwieństwie do danych uzyskiwanych z popularniejszych mediów społecznościowych, np. Twittera czy Facebooka, r/AskReddit zapewnia gwarancję czysto tekstowego charakteru wypowiedzi.

Do niniejszej analizy został wybrany wątek - **Older people of Reddit. How do you carry on with life when the weight of mistakes and regrets only seems to grow larger as you age?** *(tłum. Starsi ludzie z reddita. Jak żyć dalej, gdy ciężar błędów i żalu wydaje się tylko rosnąć wraz z wiekiem?)*

Źródło posta: 

[Older people of reddit. How do you carry on with life when the weight of mistakes and regrets only seems to grow larger as you age?](https://www.reddit.com/r/AskReddit/comments/14oszge/older_people_of_reddit_how_do_you_carry_on_with/)

W ramach projektu zostaną **zidentyfikowane grupy podobnych do siebie komentarzy**, będące odpowiedziami na pytanie zadane w wątku.

Podczas projektu przeprowadzone zostaną:

- Ekstrakcja komentarzy z wątku
- Przetworzenie (czyszczenie) tekstów
- Grupowanie tekstów wg podobieństwa
- Identyfikacja cech definiujących każdą z grup
- Wizualizacja

Cały proces odbywa się w ramach środowiska języka programowania Python, opisany funkcja po funkcji. Pełen kod zostanie dołączony do opracowania.

Stawiane w badaniu **hipotezy** dotyczą identyfikacji konkretnych grup komentarzy:

- Komentarze **negujące tezę** zadaną w pytaniu - *z wiekiem jest coraz łatwiej radzić sobie z problemami*
- Komentarze cechujące się **obojętnością**, nawiązujące do rutyny życia - *jakoś to będzie*
- Komentarze **potwierdzające** tezę z pytania - *jest źle, nie daję sobie rady* etc.

### Ekstrakcja danych z serwisu

Wydobycie danych z serwisu Reddit nie stanowi znacznego wyzwania technicznego. Portal udostępnia dobrze udokumentowany interfejs aplikacji (API). Do zebrania komentarzy z wątku wykorzystano Praw - bibliotekę dla języka programowania Python.

```python
import praw
import csv

reddit = praw.Reddit(client_id='ID', 
                     client_secret='SECRET', 
                     user_agent='AGENT')

post_id = '140xj5s'
post = reddit.submission(id=post_id)
post.comments.replace_more(limit=0)
comments_list = post.comments.list()
```

Najpierw należy zdefiniować poświadczenia API Reddita, tj. identyfikator klienta (client_id), sekret klienta (client_secret) oraz identyfikator użytkownika (user_agent). Następnie należy podać identyfikator posta, z którego chcemy pobrać komentarze. Kolejnym krokiem jest pobranie posta z użyciem funkcji `reddit.submission(id=post_id)`.

Aby pobrać wszystkie komentarze i odpowiedzi, należy użyć metody `replace_more` na obiekcie `comments` oraz `list()` na obiekcie `comments` zwracanym przez funkcję `post.comments`. Następnie, za pomocą pętli `for`, zapisywane są teksty komentarzy oraz liczba ich polubień (upvotes) w pliku CSV.

```python
with open(f'comments_{post.name}.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['author', 'comment', 'upvotes']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for comment in comments_list:
        writer.writerow({'author': comment.author, 
				'comment': comment.body, 'upvotes': comment.score})
```

Wynikiem wywołania funkcji jest tabela w formacie CSV, zawierająca komentarze wraz z przypisaną liczbą polubień i pseudonimem autora.

### Przygotowanie komentarzy

Przed przystąpieniem do analizy, komentarze zostaną starannie przetworzone, aby maksymalnie zredukować ich objętość. 

Fragment kodu wczytuje listę słów kluczowych, wczytuje gotowy model `spaCy` dla języka angielskiego i odczytuje plik CSV z komentarzami. 

```python
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")
df = pd.read_csv('comments.csv', sep=',')
```

Klasyczną stoplistę dla języka angielskiego uzupełniłem o słowa charakterystyczne dla badanego tematu, spodziewając się ich znacznego udziału w dyskusji. Aby dokonać rozszerzenia stoplisty wykorzystuję metodę `extend`.

```python
with open('stopwords_en.txt', 'r', encoding='utf-8') as f:
    unique_stopwords = ['life', 'people', 'age', 'young', 'old']
    STOP_WORDS = f.read().splitlines()
    STOP_WORDS.extend(unique_stopwords)
```

Definiowanie funkcji oczyszczającej zaczynam od usunięcia znaków niealfanumerycznych, za pomocą wyrażenia regularnego. Wykorzystując zasoby biblioteki  przeprowadzam tokenizację tekstu, czyli dzielę go na mniejsze jednostki zwane tokenami. W tym przypadku, tokeny to po prostu pojedyncze słowa. Następnie w pętli `for` porównuję lemat tokenu (uzyskany dzięki `spaCy`) ze stoplistą. Jeśli lemat nie znajduje się wśród słów wykluczonych, zostaje dodany do listy `clean_tokens`. Na końcu tokeny z listy łączone są w tekst komentarza.

```python
import re

def clean_text(text):
    text_cleaned = ''
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = text.split()
    tokens = " ".join([i for i in text.lower().split()])
    tokens = nlp(tokens)
    clean_tokens = []

    for word in tokens:
        if word.lemma_ not in STOP_WORDS:
            clean_tokens.append(word.lemma_)

    text_cleaned = ' '.join(clean_tokens)
    text_cleaned = str(text_cleaned)

    return text_cleaned
```

Powyższa funkcja wywoływana jest na każdym z komentarzy, za pomocą metody `.apply()` z biblioteki `pandas`.

### Grupowanie (clustering) komentarzy

**Wektoryzacja**

Pierwszym krokiem do pogrupowania komentarzy jest przetworzenie ich zawartości na wektory liczbowe. 

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

def df_to_list(df=df, content_col='comment', min_length=10):
    comments = []
    ids = []

    for index, row in df.iterrows():
        comment = row[content_col]
        if len(comment) >= min_length:
            comments.append(comment)
            ids.append(index)
    
    return [comments, ids]

def vectorize_comments(document_list):
    vectorizer = TfidfVectorizer()
    vectorized_docs = vectorizer.fit_transform(document_list)

    pca = PCA(n_components=2)
    reduced_docs = pca.fit_transform(vectorized_docs.toarray())

    return reduced_docs
```

Funkcja `df_to_list` zamienia ramkę danych na listę zawierającą tylko komentarze. W pętli iteruję po ramce danych i wyciągam wartość kolumny `'comment'`. Dodaję je do listy, jeśli spełnią warunek minimalnej liczby znaków. Zdecydowałem się na minimum 20 znaków, polegając na badaniu wg którego wyrazy w języku angielskim mają zazwyczaj ok. 5 liter[1]. 20 znaków daje więc teoretycznie możliwość złożenia prostego zdania.

Funkcja `vectorize_comments` używa algorytmu `TF-IDF Vectorization` do zamiany listy tekstów na macierz wektorów TF-IDF. Następnie, przy pomocy algorytmu `PCA` następuje redukcja wymiarów, przy próbie jak najlepszego zachowania informacji o zawartości dokumentów. Wynik tych operacji to macierz złożona z dwóch wymiarów, która nadaje się do wizualizacji **wykresem punktowym**.

Do wizualizacji wykorzystuję funkcję `draw_viz`, stworzoną w oparciu o pakiet `matplotlib:`

```python
def draw_viz_raw(reduced_docs, title='Rozmieszczenie komentarzy na płaszczyźnie'):

    fig, ax = plt.subplots(figsize=(12, 12))
    scatter = ax.scatter(reduced_docs[:, 0], reduced_docs[:, 1])
    
		plt.style.use('default')
    plt.title(title)
    plt.show()
```

Wywołanie funkcji skutkuje poniższą wizualizacją. Osie X i Y **nie zostały oznaczone celowo**, ponieważ reprezentowane przez nie wymiary są umowne, stanowiąc kartezjański układ współrzędnych.

![Figure_1.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/38cac06f-29d1-48b2-a034-5d5fafceb1ca/Figure_1.png)

Powyższa wizualizacja pokazuje, że istnieje wyraźne skupisko, bardzo podobnych do siebie komentarzy oraz niewielkie wysepki komentarzy zupełnie niezwiązanych z innymi. Zadaniem algorytmów grupujących będzie zdefiniowanie występujących grup.

**Grupowanie DBSCAN**

Istnienie wyraźnego skupiska tekstów podpowiada, aby wypróbować algorytm grupujący DBSCAN. Algorytm DBSCAN (*Density-Based Spatial Clustering of Applications with Noise*) jest używany do grupowania danych w oparciu o gęstość przestrzenną. Działa na zasadzie znajdowania obszarów o wysokiej gęstości punktów, które są oddzielone obszarami o niższej gęstości. W Pythonie jego implementację oferuje biblioteka `sklearn`.

Poza wektorową reprezentacją komentarzy (`reduced_docs`), funkcja `DBSCAN_clustering` przyjmuje dwa argumenty - `epsilon` i `min`, co oznacza odpowiednio promień i minimalną liczbę punktów potrzebnych, aby utworzyć skupisko. Funkcja zwraca zredukowaną macierz wraz z etykietami utworzonych klastrów.

```python
import numpy as np
from sklearn.cluster import DBSCAN

def DBSCAN_clustering(reduced_docs, epsilon, min):
    dbscan = DBSCAN(eps=epsilon, min_samples=min)
    cluster_labels = dbscan.fit_predict(reduced_docs)

    return [reduced_docs, cluster_labels]
```

Do wizualizacji grupowania zmodyfikowałem funkcję `draw_viz`, aby przyjmowała etykiety utworzonych grup. Wykres przedstawia rozmieszczenie punktów w zredukowanej przestrzeni, gdzie **kolor każdego punktu oznacza przynależność do konkretnego klastra.**

Grupowanie algorytmiczne jest **procesem** iteracyjnym. Niemożliwym jest stwierdzenie *a priori* jakie parametry `epsilon` i `min` zwrócą skupiska odpowiadające oczekiwaniom.

Wybrane przykłady z procesu grupowania. Parametry epsilon i minimum odpowiednio: [0.05, 3]  oraz [0.04, 12]. Poniższe przykłady dobrze obrazują skrajności powstałe przy niewielkiej zmianie parametrów. Wykres po lewej cechuje duża liczba grup różnej wielkości. Ten po prawej pozwala na zidentyfikowanie głównego skupiska, nie będąc jednak w stanie pogrupować reszty komentarzy.

![Figure_2.3.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4c1ae191-14f8-43c2-89fc-32f8fbe31e33/Figure_2.3.png)

![Figure_2.2x.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a0aa3a57-1b22-46ba-971b-38489427f324/Figure_2.2x.png)

Eksperymentując z wartościami pośrednimi, można uzyskać ten wykres:

![Figure_7.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3fcf3d47-d62e-4c13-82aa-0d6416f017c0/Figure_7.png)

Algorytm DBSCAN dobrze radzi sobie z identyfikacją **głównego skupiska**. Pozwala zobrazować stopień podobieństwa - które komentarze są *samotnymi wyspami*, a które organizują się w małe *kontynenty*. 

Jednakże, z punktu widzenia grupowania komentarzy, bardziej niż na realizacji przez grupy pewnego podobieństwa, zależy mi na realizacji przez grupę pewnej cechy definiującej podobieństwo. 

Co rozumiem przez realizację cechy? W tym miejscu należy odwołać się do wiedzy na temat badanego zbioru, cofając się krok wstecz, do momentu wektoryzacji komentarzy. Wiadomo bowiem, że projekcja wielowymiarowych wektorów na dwuwymiarową płaszczyznę musi polegać na wyodrębnieniu jakiejś cechy X i Y. Układ przestrzenny odpowiada realizacji przez komentarz tych dwóch cech w jakimś stopniu. Tak też, np. komentarze najbardziej na prawo realizują cechę X w największym stopniu. Wynika z tego, że poszukując tej cechy jako definiującej grupę, bardziej niż odległość (gęstość) skupiska, interesuje mnie jego położenie przestrzenne. Algorytm DBSCAN nie realizuje tej funkcji w zadowalającym stopniu. 

**Grupowanie metodą k-średnich**

Alternatywnym, i zdecydowanie najpopularniejszym, sposobem grupowania dokumentów jest metoda k-średnich. Działa na zasadzie iteracyjnego przypisywania obiektów do grup w celu minimalizacji sumy kwadratów odległości między obiektami a centroidami. Centroid to pewien losowo wybrany punkt, stanowiący orientacyjne centrum grupy. Metoda k-średnich wymaga jednego parametru, jakim jest liczba skupisk, które algorytm ma utworzyć.

```python
from sklearn.cluster import KMeans

def kMeans_clustering(reduced_docs, n_clusters):
	kmeans = KMeans(n_clusters=n_clusters)
	cluster_labels = kmeans.fit_predict(reduced_docs)
	
	return [reduced_docs, cluster_labels]
```

Jak pokazują wizualizacje, uzyskiwane tą metodą grupy są mniej organiczne. Grupowanie przypomina bardziej tworzenie granic między dokumentami. 

![2.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f9e6fb46-5564-4130-a55b-ad24076ffc61/2.png)

![1.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2601766f-c82d-42bf-a15e-554180a510bf/1.png)

Można zauważyć, że na początku algorytm wydzielił dwie proste grupy - lewą i prawą. Zdając mu dodatkowe skupisko do wykonania, grupy przybierają bardziej złożone kształty. Podział zaczyna przebiegać na środku grafu. Co istotne, tworzone grupy mają charakter nie **tyle skupisk, co regionów.** W pewien sposób realizuje on założenie podziału przestrzeni, z którym DBSCAN miał problem. ****Żaden komentarz nie zostaje pominięty, nawet te bardzo odległe przypisane są do którejś z grup. Nie jest to pożądane zjawisko, niemniej wpływ tych pojedynczych dokumentów na zdefiniowanie cech grupy, z natury rzeczy będzie niewielki.

Najprawdopodobniej idealnym rozwiązaniem byłoby **połączenie wyników jednej i drugiej metody grupowania**. Tym sposobem możnaby odseparować luźno związane komentarze, a jednocześnie zachować przejrzyste grupy metody k-średnich. Wykracza ono jednak poza moje obecne umiejętności techniczne.

Ostatecznie, zdecydowałem się na grupowanie metodą k-średnich z czterema wyróżnionymi grupami.  W przyzwoity sposób wydzielone zostały tutaj silne skupiska po lewej **(fiolet)** i w centrum wizualizacji **(czerwień)**.

![Figure_8.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5bfdd7fc-4aac-43cf-a45d-8c73ad803059/Figure_8.png)

Ekstrakcja słów kluczowych przeprowadzona będzie na czterech grupach utworzonych metodą k-średnich. Zapewniają one odpowiednią ciągłość podobieństwa, zrównoważoną wielkość grup oraz separację przestrzenną.

### Identyfikacja cech grup

Po pogrupowaniu komentarzy, krokiem następnym jest identyfikacja cech grup, do których zostały zaliczone. Jest to próba odpowiedzi na pytanie - co wyróżnia każdą z grup? Jakie cechy komentarza decydują, że został zaliczony do danej grupy? Według jakich cech komentarze zostały opracowane przestrzennie?

**Agregacja słów kluczowych**

Jedną z najprostszych metod identyfikacji cech danej grupy, jest sprawdzenie, jakie słowa dominują w komentarzach w niej zebranych. Można to osiągnąć tworząc listę słów kluczowych dla każdej z grup.

Pierwszym krokiem do utworzenia listy słów kluczowych, jest przetworzenie komentarzy na luźną listę słów, tworząc swego rodzaju *worek słów*. W przypadku prowadzonej analizy, wystarczy podzielić tekst komentarza na słowa, korzystając z funkcji `split`. 

```python
comments = doc_list[0]
bag_of_words = [comment.split() for comment in comments]
```

Funkcja `aggregate_top_keywords` pobiera listę słów, etykiety klastrów oraz liczbę top słów kluczowych do wyodrębnienia. Dla każdego klastra wyodrębnia listę słów kluczowych i zlicza ile razy każde słowo kluczowe występuje we wszystkich klastrach. Następnie dla każdego klastra wybierane są top słowa kluczowe i zwracane jako słownik.

```python
import itertools
from collections import Counter

def aggregate_top_keywords(bag_of_words, cluster_labels, top_n=15):
    cluster_keywords = {}

    for cluster_label in set(cluster_labels):
        cluster_keywords_list = [kw for kw, lbl in zip(bag_of_words, cluster_labels) if lbl == cluster_label]

        flattened_keywords = list(itertools.chain.from_iterable(cluster_keywords_list))
        keyword_counts = Counter(flattened_keywords)
        top_keywords = keyword_counts.most_common(top_n)
        cluster_keywords[cluster_label] = top_keywords

    return cluster_keywords
```

Tak prezentuje się 15 słów kluczowych dla grup stworzonych metodą k-średnich:

| Grupa 1 | n słów | Grupa 2 | n słów | Grupa 3 | n słów | Grupa 4 | n słów |
| --- | --- | --- | --- | --- | --- | --- | --- |
| time | 47 | past | 51 | learn | 31 | mistake | 136 |
| regret | 43 | mistake | 25 | forgive | 17 | regret | 85 |
| make | 41 | change | 23 | mistake | 13 | make | 82 |
| think | 40 | make | 20 | let | 9 | learn | 57 |
| good | 39 | future | 18 | well | 9 | well | 35 |
| like | 39 | good | 14 | regret | 7 | know | 33 |
| well | 35 | regret | 12 | lesson | 7 | time | 25 |
| feel | 31 | live | 12 | way | 6 | think | 25 |
| give | 29 | focus | 10 | become | 6 | good | 23 |
| really | 28 | think | 9 | hard | 5 | like | 22 |
| try | 27 | move | 8 | grow | 4 | move | 18 |
| mistake | 26 | let | 6 | move | 4 | way | 17 |
| fuck | 26 | time | 6 | good | 4 | live | 17 |
| shit | 22 | happen | 5 | important | 4 | want | 16 |
| bad | 22 | important | 5 | weight | 3 | try | 15 |

Łatwo zauważyć, że wiele słów powtarza się między grupami. Można je poddać dalszej filtracji, eliminując wybrane wyrazy pospolite z *worka słów*. Słowa: mistake, make, think, good, well, regret, move, time - pojawiają się w więcej niż dwóch grupach. Zostaną usunięte z *worka słów*.

```python
common_words = ["mistake", "make", "think", "good", "well", "regret","move","time"]

def remove_words(bag_of_words, words_to_exclude):
    modified_bag_of_words = []

    for keywords in bag_of_words:
        modified_keywords = [word for word in keywords if word not in words_to_exclude]
        modified_bag_of_words.append(modified_keywords)

    return modified_bag_of_words
```

Proces ten również jest iteracyjny. Za pierwszym razem liczba słów powtarzających zmniejszyła się, ale wciąż trudno było zgromadzić słowa definiujące grupę. Ponownie zebrano słowa powtarzające się i dokonano filtracji.

Po zastosowaniu kilku filtracji, słowa kluczowe prezentują się następująco:

| Grupa 1 | n słów | Grupa 2 | n słów | Grupa 3 | n słów | Grupa 4 | n słów |
| --- | --- | --- | --- | --- | --- | --- | --- |
| feel | 31 | past | 51 | become | 6 | want | 16 |
| give | 29 | future | 18 | hard | 5 | past | 14 |
| really | 28 | focus | 10 | important | 4 | take | 11 |
| fuck | 26 | happen | 5 | weight | 3 | choice | 10 |
| shit | 22 | important | 5 | else | 3 | could | 10 |
| could | 22 | seem | 5 | many | 3 | experience | 10 |
| even | 21 | take | 5 | follow | 3 | remember | 10 |
| say | 20 | forward | 5 | repeat | 3 | say | 10 |
| start | 20 | worry | 4 | forward | 2 | feel | 10 |
| right | 19 | fuck | 4 | reason | 2 | thank | 10 |
| parent | 18 | realize | 4 | find | 2 | realize | 9 |
| back | 17 | opportunty | 3 | death | 2 | even | 9 |
| happen | 14 | weight | 3 | otherwise | 2 | keep | 9 |
| long | 13 | undrstand | 3 | true | 2 | forward | 8 |
| memory | 13 | end | 3 | failure | 2 | never | 8 |

We wszystkich grupach pojawiają się słowa związane z przyszłością: start, future, opportunity, become, forward, want, keep. Można zakładać, że w wypowiedziach dominują dobre rady - mimo wieku, nie przejmuj się bagażem przeszłości. Myśl o przyszłości. Tym co różnicuje wypowiedzi, jest ton i sposób przedstawienia rady. Od dosadnych i wulgarnych, poprzez optymistyczne, po bardziej wyszukane i filozoficzne.

### Analiza jakościowa

W interpretacji znaczenia zagregowanych słów kluczowych może pomóc analiza jakościowa wybranych komentarzy. Oczywiście, przeprowadzenie analizy jakościowej zaledwie kilku wypowiedzi nie może stanowić podstaw wnioskowania o całości grupy. Niemniej, losowo wybrane komentarze mogą rozjaśnić to, w jaki sposób słowa kluczowe używane są w danej grupie.

Wybrane komentarze z Grupy 1.

	As my dog would say, “kick some grass over that shit and move on.” Death will  come soon enough. Zero point in hastening it [...]
	[...] You can't continue to punish yourself for regrets and mistakes of the past. You have to let it go or it will suck all the joy out of your life and ruin opportunities in the future. As I age that shit I did is so miniscule. That broken heart at 16 was so long ago. The rent I couldn't pay at 20 [...].
 
Jak wskazują słowa kluczowe oraz powyższe przykłady, komentarze z Grupy 1. są najbardziej wulgarne i dosadne. Autorzy wypowiedzi zachęcają, aby skupić się na przyszłości i przestać przejmować się przeszłością.

Wybrane komentarze z grupy 2

	If you learn from your mistakes you have nothing to regret.
	They only grow if you don’t learn to let them go and treat them as lessons and not weight.
	What weight of mistakes and regret? You learn from it and move the fuck on
 
Przykłady komentarzy z tej grupy utrzymane są w podobnym tonie. Są jednak bardziej lakoniczne i dosadne. Wypowiedzi te, to krótkie i proste “dobre rady”.

Wybrane komentarze z grupy 3

	I was young and naive in 2005 when I contracted HIV from my first boyfriend.  My life would be very very different if that didn’t happen.  I had plans that I had been working on for years but I had to stop them because I needed a job with good insurance. It has been a major factor in limiting my romantic partners.
	I’m 43 and I feel this question so so much. I’m still luckier than many but I still feel like I was robbed of choices.

Grupa 3. jest jedną z najmniej licznych. Jest też jedyną grupą, w której znaleźć można pesymistyczne wypowiedzi. Zebrane komentarze wtórują tezom z zadanego pytania, nie mogąc pogodzić się z bagażem przeszłości.

Wybrane komentarze z grupy 4

	The opposite is sort of happening to me. The older I get, the less important all of the mistakes of the past seem to be. [...] You have to let it go or it will suck all the joy out of your life and ruin opportunities in the future.
	Spending all of your energy on regretting your decisions isn't going to make your life any better. You need to come to terms with your choices and put them in the past and move forward with your life.
	So you need to accept the fact that this is what your life is and the only way it's going to improve is by making better choices.

Komentarze z tej grupy cechuje optymistyczny ton. Są rozbudowane i pouczające. Autorzy dzielą się własnym doświadczeniem. Ich zdaniem z perspektywy lat, dawne błędy tracą znaczenie. Autorzy radzą, aby sprawy przeszłości zostawić w przeszłości - żyć tu i teraz - ucząc się na błędach.

### Wnioski

W toku badania zrealizowano jego podstawowe założenia. Zebrano i przetworzono komentarze z portalu Reddit. Dokonano ich grupowania i analizy. Hipotezą badawczą było wskazanie na możliwość wyraźnej identyfikacji poniższych grup w korpusie:
- Komentarze negujące tezę zadaną w pytaniu - z wiekiem jest coraz łatwiej radzić sobie z problemami
- Komentarze cechujące się obojętnością, nawiązujące do rutyny życia - jakoś to będzie
- Komentarze potwierdzające tezę z pytania - jest źle, nie daję sobie rady etc.
- 
Przeprowadzona analiza daje podstawy do stwierdzenia, że hipoteza o możliwości wyraźnej identyfikacji tychże grup powinna zostać odrzucona. Choć zidentyfikowano pojedyncze wypowiedzi cechujące się pesymizmem, stanowią one niewielki ułamek korpusu. W większości grup przeważają wypowiedzi optymistycznie nastawione do procesu starzenia się, negujące tezę, jakoby z wiekiem radzenie sobie z problemami stanowiło wyzwanie. Dominują rady, propozycje skupienia się na przyszłości i pozostawienia przeszłości w tyle. Głównym czynnikiem różnicującym grupy, okazał się charakter wypowiedzi (ton, dosadność, wulgarność, zwięzłość).
Dokładne określenie charakteru stworzonych grup mogłyby rozstrzygnąć dodatkowe, bardziej szczegółowe analizy. Wśród nich wymienić można: agregację zdań kluczowych, generację streszczeń dla grup, modelowanie tematyczne, czy filtrowanie wielokrotne. Zasobochłonność takich analiz wykracza jednak poza moje obecne możliwości.


### Źródła

[1] Bochkarev, Vladimir & Shevlyakova, Anna & Solovyev, Valery. (2012). *Average word length dynamics as indicator of cultural changes in society*. Social Evolution and History. 14. 153-175.

[2] by **[Jason Brownlee](https://machinelearningmastery.com/author/jasonb/)** on April 6, 2020, [10 Clustering Algorithms With Python - MachineLearningMastery.com](https://machinelearningmastery.com/clustering-algorithms-with-python/)
