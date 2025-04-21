from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


plt.rcParams["font.family"] = "AppleGothic"

model = SentenceTransformer("jhgan/ko-sroberta-multitask")

semantically_similar = [
    "나는 아침에 커피를 마셨다.",
    "아침에 마신 커피가 정말 맛있었다.",
    "카페에서 마신 커피가 인상 깊었다.",
    "하루를 커피로 시작하는 걸 좋아한다.",
    "모닝커피 없이는 하루를 시작할 수 없다.",
]

visually_similar = [
    "나는 오늘 점심에 김밥을 먹었다.",
    "나는 오늘 점심에 김치찌개를 먹었다.",
    "나는 오늘 점심에 떡볶이를 먹었다.",
    "나는 오늘 점심에 비빔밥을 먹었다.",
    "나는 오늘 점심에 불고기를 먹었다.",
]

sentences = semantically_similar + visually_similar
labels = (["semantic"] * len(semantically_similar)) + (
    ["visual"] * len(visually_similar)
)


embeddings = model.encode(sentences, convert_to_tensor=True)
cos_sim = util.cos_sim(embeddings, embeddings).cpu()

plt.figure(figsize=(12, 10))
sns.heatmap(
    cos_sim,
    annot=True,
    xticklabels=sentences,
    yticklabels=sentences,
    cmap="YlGnBu",
    fmt=".2f",
    linewidths=0.5,
    cbar=True,
)
plt.xticks(rotation=45, ha="right")
plt.title("코사인 유사도 히트맵", fontsize=16)
plt.tight_layout()
plt.show()
