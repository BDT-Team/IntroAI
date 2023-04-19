import re
from underthesea import sent_tokenize, word_tokenize, text_normalize

a = "💥 Sẵn tại cửa hàng"
a1 = "Mình cũng shop bán hàng bình giữ nhiệt, nên ai có nhu cầu liên hệ shopppp nha 🥰😅"
a2 = "Ship nhanh, đúng hàng, date xa, ăn ngon, săn sale rẻ, mốt mua tiếp :))."

def delete_emojies(text):
  # Ref: https://gist.github.com/Alex-Just/e86110836f3f93fe7932290526529cd1#gistcomment-3208085
  # Ref: https://en.wikipedia.org/wiki/Unicode_block
  EMOJI_PATTERN = re.compile(
    "(["
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "])"
  )
  text = re.sub(EMOJI_PATTERN, r'', text)
  return text.strip()

def remove_punct(text):
    list_text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(list_text.split())

def remove_stopword(text, list_stopword):
    return " ".join([word for word in text.split() if word not in list_stopword])

with open("vietnamese-stopwords-dash.txt", "r", encoding="utf-8") as f:
    list_stopword = [line.strip() for line in f.readlines()]

def replace_datetime(text):
    list_text = re.sub(r'(?:\d{1,2}[:\/,.-]){1,2}\d{2,4}', ' datetime ', text)
    return ' '.join(list_text.split())

def replace_phone(text):
    list_text = re.sub(r'(84|\d{3,5})([\.-])*(\d{3,5})([\.-])*(\d{3,5})([\.-])*', ' spam_phone ', text)
    return ' '.join(list_text.split())

def replace_link(text):
    list_text = re.sub(r'(https|http)?:\/\/([\w\.\/\?\=\&\%])*\b', ' spam_link ', text)
    return ' '.join(list_text.split())

def pipeline(text, list_stopword):
    text = delete_emojies(text)         # delete emojies

    text = word_tokenize(text, format="text")       # word segmentation   
    text = text_normalize(text).lower()         # word normalization and lower
    
    text = replace_phone(text)
    text = replace_datetime(text) 
    text = replace_link(text)  
    
    
    text = remove_punct(text)       
    text = remove_stopword(text, list_stopword)
    
    text = text.replace(" _", " ")
    return text
    
print(pipeline("Chànggggg trai em mng 9X Quảng Trị khởi nghiệp từ nấm sò 17.30.2022. ở số 1530975259765. Em http://url.com/bla1/blah1/", list_stopword))
