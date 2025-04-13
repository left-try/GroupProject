import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap

def custom_autopct(pct):
    return '{:.1f}%'.format(pct) if pct >= 1 else ''

df = pd.read_excel("Trash bins in AUCA (Responses).xlsx")

number_of_bins = df["Do you think there are enough trash bins in campus?"].value_counts()
reading_of_types = df["How often do you read what type of recycle bin you are using?"].value_counts()
struggling = df["Have you ever struggled to find a trash bin when you needed one?"].value_counts()
changing = df["How would you change the design of trash bins?"].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(number_of_bins, labels=number_of_bins.index, autopct=custom_autopct, startangle=90)
plt.title("Do you think there are enough trash bins in campus?")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(
    reading_of_types,
    labels=None,
    autopct=custom_autopct,
    startangle=90,
)
plt.title("How often do you read what type of recycle bin you are using?")
plt.legend(wedges, reading_of_types.index, title="Answers", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 8))
plt.pie(struggling, labels=struggling.index, autopct='%1.1f%%', startangle=90)
plt.title("Have you ever struggled to find a trash bin when you needed one?")
plt.tight_layout()
plt.show()


wrapped_labels = []
for label in changing.index:
    wrapped_labels.append("\n".join(wrap(label, width=75)))
fig, ax = plt.subplots(figsize=(10, 12))
bars = plt.barh(wrapped_labels, changing.values)
for bar in bars:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height() / 2, f"{int(width)}", va='center', fontsize=10)
ax.set_title("How would you change the design of trash bins?")
ax.invert_yaxis()
plt.tight_layout()
plt.show()

column_name = "In which areas on campus would you like to see trash bins?"
df_filtered = df.dropna(subset=[column_name])
all_areas = []
for row in df_filtered[column_name]:
    splitted = [item.strip() for item in row.split(",")]
    all_areas.extend(splitted)
areas_series = pd.Series(all_areas).value_counts()
plt.figure(figsize=(10, 6))
plt.bar(areas_series.index, areas_series.values)
plt.title("In which areas on campus would you like to see trash bins?")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

column_name = "What do you think is the biggest issue with trash bins?"
df_filtered = df.dropna(subset=[column_name])
all_areas = []
for row in df_filtered[column_name]:
    splitted = [item.strip() for item in row.split(",")]
    all_areas.extend(splitted)
areas_series = pd.Series(all_areas).value_counts()
mask = areas_series > 1
main_series = areas_series[mask].copy()
others_sum = areas_series[~mask].sum()
if others_sum > 0:
    main_series["Other"] = others_sum
wrapped_labels = ["\n".join(wrap(label, width=60)) for label in main_series.index]
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(wrapped_labels, main_series.values)
for bar in bars:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height() / 2, f"{int(width)}", va='center', fontsize=10)
ax.set_title("What do you think is the biggest issue with trash bins?")
ax.invert_yaxis()
plt.tight_layout()
plt.show()
