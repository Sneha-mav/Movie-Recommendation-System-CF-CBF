# Movie-Recommendation-System-CF-CBF

A web-based movie recommendation engine built using **Streamlit**, implementing both **Content-Based Filtering** and **Collaborative Filtering (KNN)**.

- **Content-Based Filtering**: Computes cosine similarity between movie metadata to suggest similar titles.
- **Collaborative Filtering**: Utilizes user-item matrix and K-Nearest Neighbors (KNN) for personalized recommendations.

## Datasets
- https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?select=tmdb_5000_movies ( credits, movies )
- https://www.kaggle.com/datasets/aayushsoni4/tmdb-6000-movie-dataset-with-ratings ( ratings )

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Sneha-mav/Movie-Recommendation-System-CF-CBF.git
cd Movie-Recommendation-System-CF-CBF
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Configure Environment Variables
Create a .env file and add your TMDB Bearer Token:
```bash
TMDB_API_KEY=your_token
```
### 4. Run the Application
```cmd
streamlit run app.py
```
