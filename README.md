# Dashboard App for Querying BabyCenter Data

This dashboard allows users to update a query form, submit queries to the BabyCenter Database, and then choose from several types of Natural Language Processing (NLP) analyses to extract insights from the data. The analyses available include:

1. **N-Gram Analysis**  
2. **Sentiment Analysis**  
3. **Topic Modeling**  
4. **Retrieval Augmented Generation (RAG)**  


## Setup

To use this app, you need to have python3.11, be on the UVM VPN, and have access to the Computational Story Lab Gitlab Organization.

First, clone this project into your working directory

```
git clone git@github.com:Jayman391/storykit.git
```

Next, we need to clone the BabyCenterDB Package and build the current distribution. https://gitlab.com/compstorylab/babycenterdb    

```
git clone git@gitlab.com:compstorylab/babycenterdb.git
```

Read the documentation for babycenterdb on how to build the package.

After the package is built, change to the storykit directory, create a virtual environment, install the babycenterdb package, and then install the rest of the dependencies

```
cd storykit
python3.11 -m venv venv
source venv/bin/activate
pip install ../<path to BabyCenterDB package>/babycenterdb/dist/babycenterdb-0.1.0-py3-none-any.whl
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Finally, we need to modify the shifterator packages source code to work with python 3.11 

```
python monkeypatch_shifterator.py
```

If you want to enable RAG, you will need to create an openai api key, and export the environment variable

```
export OPENAI_API_KEY="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"
```

Finally, everything is setup! Now you can run 

```
python app.py
```

and view the dashboard at http://0.0.0.0:8050/

Below is an overview of each analysis type and the techniques used.

---

## 1. N-Gram Analysis

N-Gram Analysis breaks down the queried text data into individual n-grams over time. An **n-gram** is a contiguous sequence of *n* words from a given text. This analysis provides insights into the frequency and evolution of specific phrases or combinations of words.

### Key Features:
- **Customizable N-Gram Selection:** Users can specify the number of words in an n-gram (e.g., 2-gram for bigrams such as "covid vaccine").
- **Time Series Analysis:** Display how the frequency of these n-grams changes over time.
- **Interactive Graphs:** Users can interact with the time series plot to inspect specific dates or trends in the queried data.

### Use Case:
- **Tracking Trends:** For instance, by searching for the 2-gram "covid vaccine," users can observe the evolution of discussions or mentions around the topic as public sentiment and information change over time.

---

## 2. Sentiment Analysis

Sentiment Analysis utilizes labeled data collected from Amazon Mechanical Turk workers to assign sentiment scores to the top 10,000 most commonly used English words. This analysis examines the sentiment of the text data on a day-by-day basis, offering insights into public mood and emotional shifts.

### Key Features:
- **Daily Sentiment Calculation:** Compute the average sentiment of all words (or n-grams) for each day.
- **Time Series Sentiment Plot:** Visualize the sentiment score over time to identify trends and shifts in mood.
- **Interactive Shifterator Graph:** On selecting a specific date, the dashboard displays a "shifterator" graph. This graph highlights the words that contributed most significantly to the change in overall sentiment when that day is compared to the dataset's average sentiment.
  
### Use Case:
- **Emotional Analysis:** Determine how external events or news might impact the sentiment of the conversation in the BabyCenter community over time.

---

## 3. Topic Modeling

Topic Modeling is an unsupervised learning technique that helps in discovering abstract topics within large collections of text data. In this dashboard, topic modeling is enhanced with several advanced visualization and analysis techniques.

### Key Features:
- **Document Visualization Using Dimensionality Reduction:**  
  - **Text Embeddings:** Convert text documents into high-dimensional embedding vectors.
  - **Dimensionality Reduction:** Apply techniques like t-SNE or UMAP to reduce the high-dimensional space to 2D or 3D for easy visualization. This allows users to see clusters of documents that share similar topics.
  
- **Hierarchical Tree Construction Based on Topic Distance:**  
  - **Tree Structure:** Create a hierarchical tree (dendrogram) that shows the relationships between topics based on the distance between them in the embedding space.
  - **Topic Proximity:** Topics that are closer together in the tree are more similar in content, helping users understand the structure and subtopics within the overall dataset.
  
- **Heatmap of Cosine Similarity Scores Between Topics:**  
  - **Cosine Similarity:** Calculate cosine similarity scores between topic vectors to measure how closely related different topics are.
  - **Heatmap Visualization:** Display these scores in a heatmap, providing a clear visual representation of topic correlations. Darker or more intense colors in the heatmap indicate higher similarity.

### Use Case:
- **Insight Discovery:** Allow users to visually explore the landscape of topics in the BabyCenter data, identify clusters of related topics, and understand the evolution of discussions over time.

---

## 4. Retrieval Augmented Generation (RAG)

Retrieval Augmented Generation (RAG) integrates the strengths of traditional retrieval-based search with large language models (LLMs) to enhance response quality. This approach ensures that responses are both contextually relevant and enriched with detailed information from the queried data.

### Key Features:
- **LLMs for Search:**
  - **Contextual Understanding:** Utilize state-of-the-art LLMs to understand the context of user queries and retrieve the most relevant documents from the BabyCenter Database.
  - **Natural Language Processing:** The LLM processes the retrieved documents to generate comprehensive and context-aware responses.
  
- **Improved Response Quality through Retrieval Augmentation:**
  - **Hybrid Approach:** Combine retrieved text passages with LLM-generated content. The LLM uses the retrieved context to generate answers that are more accurate and relevant.
  - **Dynamic Querying:** The system can perform iterative queries, where the initial retrieval informs further refinement of the search results, leading to a more precise and well-rounded answer.
  
- **Enhanced User Experience:**
  - **Search Precision:** By leveraging retrieval augmentation, users receive detailed responses that incorporate real-time data from the BabyCenter Database.
  - **Comprehensive Insights:** The system can answer complex queries by referencing multiple sources of information, making it a powerful tool for users needing in-depth analysis.

### Use Case:
- **Expert-Level Responses:** When a user poses a complex query that requires nuanced understanding or cross-referencing multiple data points, RAG ensures that the response is not only accurate but also enriched with context from the latest data, ultimately providing a more reliable and detailed answer.

---

## User Workflow Summary

1. **Query Submission:**  
   - The user updates the query form and submits a query to the BabyCenter Database.
   
2. **Data Retrieval:**  
   - The system fetches relevant data based on the query.

3. **Analysis Selection:**  
   - The user selects one of the analyses (N-Gram, Sentiment, Topic Modeling, or RAG) to perform on the queried data.

4. **Results Visualization:**  
   - Each analysis returns visual and interactive outputs (time series graphs, shifterator graphs, dimensionality reduction plots, hierarchical trees, and heatmaps) that provide deep insights into the data.

5. **Exploration and Iteration:**  
   - Users can interact with the visualizations, refine their queries, and run additional analyses as needed to further explore the data and derive meaningful conclusions.

---

This dashboard app is a powerful tool for researchers, analysts, and community managers interested in understanding trends, sentiment, topics, and complex query responses within the BabyCenter data. By combining multiple NLP techniques and advanced visualization methods, the application provides a comprehensive and user-friendly interface for deep data analysis.


