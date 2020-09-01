document = [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

minSup = 2

dict_1 = {}

for group in document:
  for item in group:
    if item in dict_1:
      dict_1[item] += 1
    else:
      dict_1[item] = 1

print(dict_1)

list_1 = []

for key ,value in dict_1.items():
  if value >= minSup:
    list_1.append([key])

print(list_1)

list_1 = sorted(list_1)
list_1_2 = list_1.copy()
list_1_3 = list_1.copy()

while len(list_1_2) > 0:
  new_list = []
  for idx1 in range(len(list_1_3)):
    item1 = list_1_3[idx1]
    for idx2 in range(idx1 + 1, len(list_1_3)):
      item2 = list_1_3[idx2]

      if (item1[:-1] == item2[:-1]) and (item1[-1] < item2[-1]):
        item = item1 + [item2[-1]]
        new_list.append(item)

  list_1_2 = []
  for item in new_list:
    count = 0
    for group in document:
      check = True
      for idx in item:
        if idx not in group:
          check = False
      if check:
        count += 1
    if count >= minSup:
      list_1_2.append(item)

  print(list_1_2)
  list_1_3 = list_1_2.copy()