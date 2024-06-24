import requests
import json
from bs4 import BeautifulSoup
from langchain.schema import Document

def google_custom_search(query, api_key, cx):
    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": cx,
        "num": 5,  # Number of results you want to retrieve
        "excludeTerms": "site:youtube.com",  # Exclude YouTube videos
        "sort": "date:20250101:20200101"
    }
    response = requests.get(base_url, params=params)
    links = []
    if response.status_code == 200:
        data = response.json()
        if 'items' in data:
            for item in data['items']:
                # print(item['title'])
                # print(item['link'])
                # print(item['snippet'])
                # print()

                links.append(item)
        else:
            print("No results found.")
    else:
        print("Error:", response.status_code)
    
    return links


def get_website_text(url):
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:76.0) Gecko/20100101 Firefox/76.0'}
    session = requests.Session()
    response = session.get(url, timeout=30, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error in fetching data from {url}: Status Code {response.status_code}")
    soup = BeautifulSoup(response.content, 'html.parser')
    include_line = lambda line: not (line.isspace() or line == "")
    return "\n".join([line for line in soup.get_text().split("\n") if include_line(line)])


def load_langchain_documents(links):
    documents = []
    for link in links:
        try:
            website_text = get_website_text(link["link"])
            document = Document(page_content=website_text)
            documents.append(document)
        except Exception as e:
            continue
    return documents


def retrieve_relevant_documents(query, api_key_file):
    with open(api_key_file, "r") as f:
        api_keys = json.load(f)
        search_api_key = api_keys["google"]["api_key"]
        search_engine_id = api_keys["google"]["search_engine_id"]
    
    print("Running Google Search")
    links = google_custom_search(query, search_api_key, search_engine_id)

    print("Loading results into Langchain Documents")
    documents = load_langchain_documents(links)
    return documents, links
