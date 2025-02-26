import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.preprocessing import LabelEncoder

# Load McDonalds dataset (replace with actual data loading method)
data = pd.DataFrame({
    "yummy": np.random.choice(["Yes", "No"], size=1453),
    "convenient": np.random.choice(["Yes", "No"], size=1453),
    "spicy": np.random.choice(["Yes", "No"], size=1453),
    "fattening": np.random.choice(["Yes", "No"], size=1453),
    "greasy": np.random.choice(["Yes", "No"], size=1453),
    "fast": np.random.choice(["Yes", "No"], size=1453),
    "cheap": np.random.choice(["Yes", "No"], size=1453),
    "tasty": np.random.choice(["Yes", "No"], size=1453),
    "expensive": np.random.choice(["Yes", "No"], size=1453),
    "healthy": np.random.choice(["Yes", "No"], size=1453),
    "disgusting": np.random.choice(["Yes", "No"], size=1453),
    "like": np.random.randint(-5, 6, size=1453),
    "age": np.random.randint(18, 65, size=1453),
    "visit_frequency": np.random.choice(["Rarely", "Occasionally", "Frequently"], size=1453),
    "gender": np.random.choice(["Male", "Female"], size=1453)
})

# Convert categorical values to numeric
label_encoder = LabelEncoder()
categorical_columns = ["yummy", "convenient", "spicy", "fattening", "greasy", "fast", "cheap", "tasty", "expensive", "healthy", "disgusting", "visit_frequency", "gender"]
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Step 4: Exploring Data - PCA for perceptual mapping
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data.iloc[:, :-3])
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5, c='grey')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Perceptual Map of McDonalds Attributes")
plt.show()

# Step 5: Extracting Segments using k-means clustering
kmeans = KMeans(n_clusters=4, random_state=1234, n_init=10)
data['segment'] = kmeans.fit_predict(data.iloc[:, :-3])

# Step 6: Profiling Segments - Bar Chart Representation
segment_profiles = data.groupby("segment").mean()
segment_profiles.iloc[:, :-3].plot(kind='bar', figsize=(12, 6))
plt.title("Segment Profiles")
plt.ylabel("Mean Value of Attributes")
plt.xlabel("Segment Number")
plt.show()

# Step 7: Describing Segments - Mosaic Plot
contingency_table = pd.crosstab(data['segment'], data['like'])
mosaic(contingency_table.stack(), title='Segment vs Like/Dislike McDonalds')
plt.show()

# Step 8: Selecting Target Segments - Scatter Plot with Bubble Size for Gender Distribution
visit_freq = data.groupby("segment")['visit_frequency'].mean()
like_level = data.groupby("segment")['like'].mean()
bubble_size = data.groupby("segment")['gender'].mean() * 500
plt.scatter(visit_freq, like_level, s=bubble_size, alpha=0.5)
plt.xlabel("Visit Frequency")
plt.ylabel("Liking Level")
plt.title("Segment Evaluation Plot for McDonalds")
plt.show()

# Step 9: Customizing the Marketing Mix (suggested strategies based on cluster analysis)
def marketing_strategy(segment):
    if segment == 3:
        return "Introduce a budget-friendly menu"
    elif segment == 4:
        return "Emphasize quality and taste in promotions"
    else:
        return "Improve brand perception through targeted campaigns"

strategies = {seg: marketing_strategy(seg) for seg in range(4)}
print(strategies)
# market_segmentation
