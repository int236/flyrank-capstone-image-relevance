import os, json, httpx

c = httpx.Client(base_url="http://127.0.0.1:8000")

# 1. register every image in the images/ folder
for fname in os.listdir("images"):
    r = c.post("/images", json={"path": f"images/{fname}"})
    print(r.status_code, fname)

# 2. register your blog posts
posts = json.load(open("posts.json"))  # paste the 7 posts array into posts.json
for p in posts:
    r = c.post("/posts", json=p)
    print(r.status_code, p["title"])