import os
import random
import re

#import emoji
from IPython.display import display, HTML
#from soynlp.normalizer import repeat_normalize

base_pattern = re.compile(r'[^ .,?!/@$%~％·∼()\x20-\x7Fㄱ-ㅣ가-힣]+')
url_pattern = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

pattern1 = re.compile(r'\(.{0,20}\)')
pattern2 = re.compile(r'\{.{0,20}\}')
pattern3 = re.compile(r'\[.{0,20}\]')
pattern4 = re.compile(r'\.{4,30}')
pattern5 = re.compile(r',{2,30}')
pattern6 = re.compile(r'-{2,30}')
pattern7 = re.compile(r'\"')
pattern8 = re.compile(r'\'')
pattern9 = re.compile(r'\s{2,20}')
pattern10 = re.compile(r'\<.{0,20}\>')

def clean_line(x):
    x = base_pattern.sub(' ', x)
    x = url_pattern.sub('', x)
    #x = repeat_normalize(x, num_repeats=2)
    x = pattern1.sub('', x)
    x = pattern2.sub('', x)
    x = pattern3.sub('', x)
    x = pattern4.sub('.', x)
    x = pattern5.sub(',', x)
    x = pattern6.sub(',', x)
    x = pattern7.sub('', x)
    x = pattern8.sub('', x)
    x = pattern9.sub(' ', x)
    x = pattern10.sub('', x) 
    return x


def get_token_length(text):
    token_length = 0
    for sentence in text:
        token_length += len(sentence.split(" "))
    return token_length


# pattern.sub() 적용시 오류 발생하는 경우가 있음. 아마도 특수/외국어 문자?
# => 오류 발생시 0.0을 반환하는 것으로 수정
korean_ratio_pattern = re.compile(f'[^가-힣]')
def get_korean_ratio(text):
    try:
        org_length = len(text)
        new_text = korean_ratio_pattern.sub('', text)
        new_length = len(new_text)
        #print(f"org_text: {text}\n new_text: {new_text}")
        return new_length/org_length
    except:
        return 0.0
    
space_ratio_pattern = re.compile(f'\W')    
def get_space_ratio(text):
    try:
        org_length = len(text)
        new_text = space_ratio_pattern.sub('', text)
        new_length = len(new_text)
        #print(f"org_text: {text}\n new_text: {new_text}")
        return new_length/org_length
    except:
        return 0.0

    
def check_text_ratio(text):
    count = 0
    out = re.findall("[마사지|맛사지|콜걸|안마|카지노|커플|모텔]*", text)
    for item in out:
        if item != '':
            count += 1
    return count/len(text.split())


def check_dup_words(text):
    lst = text.split()
    st = set(lst)
    return len(st)/len(lst)
    

def parse_paths(folder, ext=".txt"):
    result = []
    for current, dirs, files in os.walk(folder):
        if len(files) > 0:
            result.extend([os.path.join(current, file) for file in files if file.endswith(ext)])        
    return result


def show_random_elements(df, num_examples=10):
    assert num_examples <= len(df)
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(df) - 1)
        while pick in picks:
            pick = random.randint(0, len(df) - 1)
        picks.append(pick)

    display(HTML(df.loc[picks].to_html()))


def filter_elements(df, low_limit, upper_limit):
    tmp = df.loc[df.sent_length > low_limit]
    tmp = tmp.loc[tmp.sent_length < upper_limit]
    return tmp.reset_index()


def drop_elements(df, pattern):
    tmp = df.loc[df.sentence.str.find(pattern) <0]
    return tmp.reset_index()


def save_file(path, src):
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)

