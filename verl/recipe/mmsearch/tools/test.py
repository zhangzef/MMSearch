from ddgs import DDGS
import requests
from load_url import save_url_content_as_image

results = DDGS().text("python programming", max_results=5)
save_url_content_as_image(results[1]["href"], "output.png")
# print(results)


# jina_url = "https://r.jina.ai/"
# url = jina_url + results[1]["href"]
# print(url)
# # headers = {
# #     "Authorization": "Bearer jina_923869d5ccc844d99db38119ab698e972cqA3F1Qc2tiOYhcO2mr3aWEzAbF"
# # }

# response = requests.get(url, timeout=20)

# print(response.text)
# with open("test.md", "w") as f:
#     f.write(response.text)
