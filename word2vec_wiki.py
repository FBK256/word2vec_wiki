import re
from gensim.models import Word2Vec
import logging
import MeCab
import gc

# MeCabを使用して日本語のテキストをトークン化
def tokenize_japanese_text(text):
    tagger = MeCab.Tagger("-Owakati")
    tokens = tagger.parse(text).split()
    return tokens

# XMLファイルからテキストを抽出
def extract_text_from_xml(xml_file):
    with open(xml_file, 'r', encoding='utf-8') as f:
        xml_data = f.read()
    # タグ内のテキストを抽出
    print("get Text")
    text = re.findall(r'>([^<]+)<', xml_data)
    return ''.join(text)

# XMLファイルからテキストを抽出
xml_file = './jawiki-latest-pages-articles.xml'
text = extract_text_from_xml(xml_file)
with open("./step1.txt", "w", encoding='utf-8') as t:
    t.write(text)
print("抽出完了")
gc.collect()
# テキストをトークン化
tokens = tokenize_japanese_text(text)
with open("./step2.txt", "w", encoding='utf-8') as t:
    t.write(tokens)
print("トークン化完了")
gc.collect()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Word2Vecモデルの学習
model = Word2Vec([tokens], vector_size=100, window=5, min_count=1, workers=4)
gc.collect()
# 学習済みモデルの保存
model.save('./word2vec_model_japanese.bin', binary=True)