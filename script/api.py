import requests

API_KEY = " "

TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_VIDEO_URL = "https://api.themoviedb.org/3/movie/{}/videos"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"


# ---------------------------
# FUNCTIONS
# ---------------------------
def get_movie_data(movie_name):
    """Fetch movie ID + poster from TMDB"""
    params = {"api_key": API_KEY, "query": movie_name}
    res = requests.get(TMDB_SEARCH_URL, params=params).json()

    if not res.get("results"):
        return None

    movie = res["results"][0]
    return movie


def get_trailer(movie_id):
    """Fetch YouTube trailer"""
    url = TMDB_VIDEO_URL.format(movie_id)
    params = {"api_key": API_KEY}
    res = requests.get(url, params=params).json()

    for video in res.get("results", []):
        if video["site"] == "YouTube" and video["type"] == "Trailer":
            return f"https://www.youtube.com/watch?v={video['key']}"

    return None
