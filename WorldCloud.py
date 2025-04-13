import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

df = pd.read_csv('TrashCanResponses.csv')

df = df.drop(columns=['Timestamp'])

df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
df['What do you think is the biggest issue with trash bins?'] = df['What do you think is the biggest issue with trash bins?'].replace(
    {'Imporper design': 'Improper design'}
)

print("Columns:", df.columns.tolist())
print(df.head(3))
label_freq = df['How often do you read what type of recycle bin you are using? '].value_counts()

print(label_freq)


categories = ["Always", "Quite often", "Sometimes do", "I prefer to look at the color", "I do not read at all"]
label_freq = label_freq.reindex(categories)
plt.figure(figsize=(6,4))
plt.bar(categories, label_freq.values, color='tab:red')
plt.title('Frequency of Reading Trash Bin Labels')
plt.ylabel('Number of Responses')
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(label_freq.values):
    plt.text(i, v+1, str(v), ha='center')
plt.tight_layout()
plt.show()
bin_quantity = df['Do you think there are enough trash bins in campus?'].value_counts()
print(bin_quantity)

plt.figure(figsize=(6,4))
plt.barh(bin_quantity.index, bin_quantity.values, color='skyblue')
plt.xlabel('Number of Responses')
plt.title('Are there enough trash bins on campus?')
# Annotate counts
for i, v in enumerate(bin_quantity.values):
    plt.text(v+1, i, str(v), va='center')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

struggled = df['Have you ever struggled to find a trash bin when you needed one?'].value_counts()
print(struggled)

labels = ['Yes', 'No']
sizes = [struggled.get('Yes, I had a problem.', 0), struggled.get('No, I never had a problem with it.', 0)]
plt.figure(figsize=(4,4))
plt.pie(sizes, labels=labels, autopct='%1.0f%%', startangle=90, colors=['#ff9999','#99ff99'], counterclock=False)
plt.title('Ever struggled to find a trash bin?')
plt.axis('equal')
plt.show()
locations = []
for resp in df['In which areas on campus would you like to see trash bins?'].dropna():
    for area in resp.split(','):
        locations.append(area.strip())
loc_counts = pd.Series(locations).value_counts()
print(loc_counts)

plt.figure(figsize=(6,4))
plt.bar(loc_counts.index, loc_counts.values, color='cornflowerblue')
plt.title('Most Desired Locations for Additional Trash Bins')
plt.ylabel('Number of Mentions')
plt.xticks(rotation=45, ha='right')
for i, v in enumerate(loc_counts.values):
    plt.text(i, v+1, str(v), ha='center')
plt.tight_layout()
plt.show()


issues_text = df['What do you think is the biggest issue with trash bins?'].dropna().astype(str)
stop_words = ['trash','bin','bins','trash bins','the','and','of','to','in','a','is','with','for','that']
vectorizer = CountVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(issues_text)
total_counts = X.sum(axis=0).A1
vocab = vectorizer.get_feature_names_out()
freq_dist = sorted(list(zip(vocab, total_counts)), key=lambda x: x[1], reverse=True)
print("Top 5 words:", freq_dist[:5])
issues_data = df['What do you think is the biggest issue with trash bins?'].dropna()
themes = {"Lack of Info/Clarity": 0, "Design": 0, "Quantity": 0, "Size": 0, "Location": 0,
          "User Behavior": 0, "Maintenance": 0}
for response in issues_data:
    resp = str(response).lower()
    if "lack of information" in resp or "нечёткие определения" in resp or "not sure if selecting correct bin" in resp:
        themes["Lack of Info/Clarity"] += 1
    if "design" in resp or "improper design" in resp:
        themes["Design"] += 1
    if "quantity" in resp or "not enough" in resp:
        themes["Quantity"] += 1
    if "size" in resp:
        themes["Size"] += 1
    if "inconvenient location" in resp or "location" in resp:
        themes["Location"] += 1
    if "people" in resp or "ignorant" in resp or "discipline" in resp:
        themes["User Behavior"] += 1
    if "not emptied on time" in resp or "maintenance" in resp:
        themes["Maintenance"] += 1

print("Issue themes counts:", themes)

plt.figure(figsize=(6,4))
issue_labels = list(themes.keys())
issue_counts = list(themes.values())
plt.barh(issue_labels, issue_counts, color='lightgreen')
plt.title('Most Cited Issues with Trash Bins')
plt.xlabel('Number of Mentions')
for i, v in enumerate(issue_counts):
    plt.text(v+0.5, i, str(v), va='center')
plt.tight_layout()
plt.show()
suggestions = df['How would you change the design of trash bins?'].dropna()

no_change_count = sum(1 for s in suggestions if str(s).strip().lower().startswith("no need"))
print(f"No-change suggestions: {no_change_count} respondents")

ideas = {"More eye-catching": 0, "Change lid opening": 0, "Standardize (recycling standards)": 0,
         "Bigger bins": 0, "Clearer labels": 0, "Education (not design issue)": 0}
for s in suggestions:
    text = str(s).lower()
    if "eye-catching" in text and "less eye-catching" not in text:
        ideas["More eye-catching"] += 1
    if "hole in the lid" in text or "lid" in text:
        ideas["Change lid opening"] += 1
    if "recycling standards" in text or "unification" in text:
        ideas["Standardize (recycling standards)"] += 1
    if "make them bigger" in text or "bigger bin" in text:
        ideas["Bigger bins"] += 1
    if "picture" in text or "label" in text:
        ideas["Clearer labels"] += 1
    if "not in the design" in text or "lack of education" in text:
        ideas["Education (not design issue)"] += 1

print("Common improvement suggestions:", ideas)


suggestion_text = " ".join(str(s) for s in suggestions if not str(s).lower().startswith("no need"))
stopwords = set(STOPWORDS)
stopwords.update(["trash", "bin", "bins", "make", "them", "trash bins", "would", "need", "change", "design"])
wc = WordCloud(width=600, height=400, background_color="white", stopwords=stopwords, collocations=False)
wc.generate(suggestion_text)
plt.figure(figsize=(6,4))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Suggested Design Improvements - Word Cloud')
plt.show()
