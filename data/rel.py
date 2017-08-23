#-*- coding: UTF-8 -*-
import re
import pickle
def generate_rel():
  relation_pairs = {}
  relation_names = {}
  with open('rel',encoding='utf-8') as file:
    for line in file.readlines():
      content = line.strip()
      if len(content)>0:
        sections = re.sub(r'[ ]+', ' ', content).split(' ')
        rel_name = sections[0]
        arg1 = sections[1][:-1].split(':')[1]
        arg2 = sections[2].split(':')[1]
        if relation_pairs.get(arg1) == None:
          relation_pairs[arg1] = [arg2]
        else:
          relation_pairs[arg1].append(arg2)
        relation_names[arg1+':'+arg2] = rel_name
  with open('rel_pairs','wb') as pairs_file:
    pickle.dump(relation_pairs,pairs_file)
  with open('rel_names','wb') as names_file:
    pickle.dump(relation_names,names_file)

generate_rel()