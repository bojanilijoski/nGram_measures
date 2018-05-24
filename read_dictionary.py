import codecs
import os
from tqdm import tqdm
from nltk import bigrams
from collections import Counter
from scipy import spatial
from pandas import *
import operator
from random import randint
from random import shuffle
import itertools
import numpy as np
from lingpy.compare.strings import dice
from lingpy.compare.strings import jcd
from lingpy.compare.strings import lcs
from lingpy.compare.strings import ldn
from lingpy.compare.strings import bisim1
from lingpy.compare.strings import bisim2
from lingpy.compare.strings import bisim3
from lingpy.compare.strings import trisim1
from lingpy.compare.strings import trisim2
from lingpy.compare.strings import xdice
from lingpy.compare.strings import xxdice
import wikipedia


alphabet_mk = " -’'‘абвгдѓежзѕијклљмнњопрстќуфхцчџш"

def read_dictionary():
    dict = {}
    filepath = "wfl-mk.tbl"
    t = tqdm(total=os.path.getsize(filepath))
    with codecs.open(filepath, encoding='utf-8')as f:
        for line in f:
            t.update(len(line.encode('utf-8')))
            s = line.split("\t")
            dict[s[0]] = s[1]
    return dict

def dice_similarity(seq1, seq2):
    seq_inter = Counter(seq1) & Counter(seq2)
    seq_inter = list(seq_inter.elements())
    return (2*len(seq_inter))/(len(seq1)+len(seq2))

def jaccard_similarity(seq1, seq2):
    seq_inter = Counter(seq1) & Counter(seq2)
    seq_inter = list(seq_inter.elements())
    return (len(seq_inter))/((len(seq1)+len(seq2))-len(seq_inter))

def overlap_similarity(seq1, seq2):
    seq_inter = Counter(seq1) & Counter(seq2)
    seq_inter = list(seq_inter.elements())
    return (len(seq_inter))/(min(len(seq1),len(seq2)))

def cosine_similarity(seq1, seq2):
    return spatial.distance.cosine(seq1, seq2)

def closeness(seq1, seq2, par_power=1):
    different_ngrams = list(set().union(seq1, seq2))
    closeness = 0
    for t in different_ngrams:
        indices_seq1 = [i for i, x in enumerate(seq1) if x == t]
        indices_seq2 = [i for i, x in enumerate(seq2) if x == t]
        if len(indices_seq1) == 0 or len(indices_seq2) == 0:
            continue
        diff_list = []
        if len(indices_seq1) == len(indices_seq2):
            diff_list = list(map(abs,map(operator.sub, indices_seq1, indices_seq2)))
        else:
            if len(indices_seq1) > len(indices_seq2):
                indices_seq1, indices_seq2 = indices_seq2, indices_seq1
            # create matrix od distances between all ngrams
            diff_matrix = np.zeros((len(indices_seq1),len(indices_seq2)))
            for i in range(len(indices_seq1)):
                for j in range(len(indices_seq2)):
                    diff_matrix[i][j] = abs(indices_seq1[i] - indices_seq2[j])

            perm = list(itertools.permutations(list(np.linspace(0,len(indices_seq2)-1,len(indices_seq2))), len(indices_seq1)))
            #print(diff_matrix)
            #print(perm)
            best = [1000]
            for p in perm:
                tmp_list = []
                for i in range(len(p)):
                    #print("p: "+str(i)+str(int(p[i])))
                    tmp_list.append(diff_matrix[i][int(p[i])])
                if sum(best) > sum(tmp_list):
                    best = tmp_list
            diff_list = best
        # print(str(t)+" "+str(diff_list))
        for dx in diff_list:
            if dx == 0:
                closeness += 1
            else:
                closeness += (1/(dx+1))**par_power
    return closeness

def pos_sim_min(seq1, seq2):
    return closeness(seq1, seq2)/min(len(seq1),len(seq2))

def pos_sim_union(seq1, seq2):
    seq_union = Counter(seq1) | Counter(seq2)
    seq_union = list(seq_union.elements())
    return closeness(seq1, seq2)/len(seq_union)

def pos_sim_union_07(seq1, seq2):
    seq_union = Counter(seq1) | Counter(seq2)
    seq_union = list(seq_union.elements())
    return closeness(seq1, seq2, 0.7) / len(seq_union)

def pos_sim_union_05(seq1, seq2):
    seq_union = Counter(seq1) | Counter(seq2)
    seq_union = list(seq_union.elements())
    return closeness(seq1, seq2, 0.5) / len(seq_union)

def dist_sim(seq1, seq2, bigram_matrix):
    seq_inter = Counter(seq1) & Counter(seq2)
    seq_inter = list(seq_inter.elements())
    seq_union = Counter(seq1) | Counter(seq2)
    seq_union = list(seq_union.elements())
    inter_sum = 0
    try:
        for t in seq_inter:
            if t[0] == ' ':
                inter_sum += bigram_matrix['blank'][t[1]]
                continue
            if t[1] == ' ':
                inter_sum += bigram_matrix[t[0]]['blank']
                continue
            inter_sum += bigram_matrix[t[0]][t[1]]

        union_sum = 0
        for t in seq_union:
            if t[0] == ' ':
                union_sum += bigram_matrix['blank'][t[1]]
                continue
            if t[1] == ' ':
                union_sum += bigram_matrix[t[0]]['blank']
                continue
            union_sum += bigram_matrix[t[0]][t[1]]

    except:
        return -1
    return inter_sum/union_sum

def bi_lcs(seq1, seq2):
    count = Counter()
    print(seq1)
    print(seq2)
    for i, t in enumerate(seq1):
        indices = [j-i for j, x in enumerate(seq2) if x == t]
        count += Counter(indices)
    print(count)
    return 0 if len(count) == 0 else count.most_common(1)[0][1] /  max(len(seq1),len(seq2))


def initialize_bigram_dictionary():
    row_dict = {}
    for i in alphabet_mk:
        row_dict[i] = 0
    bigram_dict = {}
    for i in alphabet_mk:
        bigram_dict[i] = row_dict

    return bigram_dict

def create_bigram_dictionary():
    # words_dict = read_dictionary()
    bigram_dict = initialize_bigram_dictionary()
    for k in ["тест", "бојан", "супер", "јате", "пејам"]:#words_dict.keys():
        k = k.replace('\ufeff', '').lower().strip()
        print(k)
        for t in list(bigrams(" "+k+" ")):
            if t[0] in bigram_dict.keys() and t[1] in bigram_dict.keys():
                r = bigram_dict[t[0]]
                v = r[t[1]]
                r[t[1]] = v+1
    return bigram_dict

def create_bigram_matrix():
    words_dict = read_dictionary()
    df_ = DataFrame(columns=list(alphabet_mk), index=range(0, len(alphabet_mk)))
    df_ = df_.rename({i: alphabet_mk[i] for i in range(len(alphabet_mk))})
    df_ = df_.fillna(0)

    for k in words_dict.keys():
        k = k.replace('\ufeff', '').lower().strip()
        print(k)
        for t in list(bigrams(" "+k+" ")):
            if t[0] in list(df_.columns.values) and t[1] in list(df_.columns.values):
                v = df_[t[1]][t[0]]
                df_[t[1]][t[0]] = v + 1

    return df_


def twod_dic_to_pandas(dict):
    return DataFrame(dict).T.fillna(0)


def read_bigram_matrix():
    return read_csv("bigram_matrix.tsv", sep='\t', index_col=0)


def create_wiki_dictionary():
    word_dict = read_dictionary()
    list_themes = ['економија', 'спорт', 'историја', 'географија', 'политика', 'матаматика', 'инфроматика', 'свет', 'физика', 'хемија', 'биологија', 'живот', 'здравје', 'технологија']
    wikipedia.set_lang("mk")
    words = []
    for theme in list_themes:
        page_names = wikipedia.search(theme, results=7)
        for name in page_names:
            print(name)
            try:
                page = wikipedia.page(name)
            except:
                print("page not found")
                continue
            page = page.content
            for word in page.split(' '):
                word = word.lower()
                if word not in words and len(word) > 3:
                    if word in word_dict.keys():
                        words.append(word.lower())

    shuffle(words)
    return words


def generate_test_cases():
    fp = codecs.open("test_cases_tuples_wiki.tsv", "w", "utf-8")
    word_dict = read_dictionary()
    size = len(word_dict)
    print(size)
    n = 10000
    positive = 0
    test_words_tupes = []
    values = create_wiki_dictionary()#list(set(word_dict.values()))
    values_size = len(values)
    #  positive
    i = 0
    while i < n/2:
        try:
            random_value = randint(0, values_size)
            lemma = values[random_value]
            words = [key for key, value in word_dict.items() if value == lemma]
            if len(words) == 0:
                continue
            print(str(i))
            rand1 = randint(0, len(words)-1)
            rand2 = randint(0, len(words)-1)
            if len(words) <= 1:
                continue
            while rand1 == rand2:
                rand2 = randint(0, len(words)-1)
            t = (words[rand1], words[rand2], 1)
            if t not in test_words_tupes:
                positive += 1
                test_words_tupes.append(t)
                print(t)
                fp.write(t[0] + "\t" + t[1] + "\t" + str(t[2]) + "\n")
                i += 1
        except:
            i += 1
    #  negative
    i = 0
    while i < n/2:
        print(str(i))
        try:
            rand1 = i #randint(0, size)
            i += 1
            rand2 = i #randint(0, size)
            #if rand1 == rand2:
            #    continue
            key1 = values[rand1]#list(word_dict.keys())[rand1]
            key2 = values[rand2]#list(word_dict.keys())[rand2]
            sim = 1 if word_dict[key1] == word_dict[key2] else 0
            t = (key1, key2, sim)
            if t not in test_words_tupes:
                positive += sim
                test_words_tupes.append(t)
                fp.write(t[0]+"\t"+t[1]+"\t"+str(t[2])+"\n")
                print(t)
                i += 1
        except:
            i += 1
    print("positive ", positive)
    fp.close()

def run_test_cases():
    filepath = "test_cases_tuples_wiki.tsv"
    fw = codecs.open("test_cases_tuples_results_wiki_3.tsv", "w", "utf-8")
    # t = tqdm(total=os.path.getsize(filepath))
    i = 0
    fw.write("word1\tword2\tsimilarity\tpos_sim_union\tbi_lcs\tpos_sim_union_05\tdice\tjcd\tlcs\tldn\tbisim1\t"
             "bisim2\tbisim3\ttrisim1\txdice\txxdice\n")
    bigram_matrix = read_bigram_matrix()
    with codecs.open(filepath, encoding='utf-8')as f:
        for line in f:
            print(i)
            i += 1
            # t.update(len(line.encode('utf-8')))
            s = line.split("\t")
            s0 = " " + s[0] + " "
            s1 = " " + s[1] + " "
            fw.write(s0+"\t"+s1+"\t"+s[2].strip()+"\t"
                     + str(pos_sim_union(list(bigrams(s0)), list(bigrams(s1))))+"\t"
                     + str(bi_lcs(list(bigrams(s0)), list(bigrams(s1)))) + "\t"
                     + str(pos_sim_union_05(list(bigrams(s0)), list(bigrams(s1)))) + "\t"
                     + str(1-dice(s0, s1)) + "\t"
                     + str(1-jcd(s0, s1)) + "\t"
                     + str(1-lcs(s0, s1)) + "\t"
                     + str(1-ldn(s0, s1)) + "\t"
                     + str(1-bisim1(s0, s1)) + "\t"
                     + str(1-bisim2(s0, s1)) + "\t"
                     + str(1-bisim3(s0, s1)) + "\t"
                     + str(1-trisim1(s0, s1)) + "\t"
                     + str(1-xdice(s0, s1)) + "\t"
                     + str(1-xxdice(s0, s1)) + "\t"
                     + "\n")


#create_wiki_dictionary()
run_test_cases()


#s1 = " превозено "
#s2 = " повозната "
#bigram_matrix = read_bigram_matrix()

#print(1-lcs("abababa","aaaa"))
#print(bi_lcs(list(bigrams(s1)), list(bigrams(s2))))

'''
print(pos_sim_union(list(bigrams(s1)), list(bigrams(s2))))
print(pos_sim_union_2(list(bigrams(s1)), list(bigrams(s2))))
print(pos_sim_union_05(list(bigrams(s1)), list(bigrams(s2))))
print(dist_sim(list(bigrams(s1)), list(bigrams(s2)), bigram_matrix))
print(1-dice(s1, s2))
print(1-xdice(s1, s2))
print(1-xxdice(s1, s2))

'''



#generate_test_cases()
#df = create_bigram_matrix()
#with option_context('display.max_rows', None, 'display.max_columns', None):
#    print(df)

'''
print(dice_similarity(list(bigrams(" ababa ")), list(bigrams(" babab "))))
print(jaccard_similarity(list(bigrams(" ababa ")), list(bigrams(" babab "))))
print(overlap_similarity(list(bigrams(" ababa ")), list(bigrams(" babab "))))
'''

#print(twod_dic_to_pandas(create_bigram_dictionary()))
'''
df_ = DataFrame(columns=list(alphabet_mk), index=range(0,len(alphabet_mk)))
df_ = df_.rename({i:alphabet_mk[i] for i in range(len(alphabet_mk))})
df_ = df_.fillna(0)
df_['б']['ш'] += 1
print(df_['б']['ш'])
'''
