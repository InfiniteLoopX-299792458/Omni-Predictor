import requests

def get_market_sentiment():
    """Fetches the official Crypto Fear & Greed Index to gauge market panic/hype."""
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        value = int(data['data'][0]['value'])
        classification = data['data'][0]['value_classification']
        
        return {
            "score": value,
            "status": classification
        }
    except Exception as e:
        # Fallback if API fails
        return {
            "score": 50,
            "status": "Neutral"
        }