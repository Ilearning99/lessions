# def start_K(s):
# 	if len(s)<1:
# 		return False
# 	return s[0]=='K'

# print(start_K("Karl"))
# print(start_K("Abe"))
# print(start_K(""))

# def middle(a, b, c):
# 	if b>=a>=c or c>=a>=b:
# 		return a
# 	if a>=b>=c or c>=b>=a:
# 		return b
# 	if a>=c>=b or b>=c>=a:
# 		return c

# print(middle('cork', 'apple', 'basil'))
# print(middle('Apple', 'applE', 'aPple'))

# def count_x(s):
# 	count=0
# 	for x in s:
# 		if x == 'x':
# 			count+=1
# 	return count

# def count_ch(s,x):
# 	count=0
# 	for c in s:
# 		if c == x:
# 			count+=1
# 	return count

# print(count_x("x"))
# print(count_x("oxen and foxen all live in boxen"))
# print(count_x("that letter isn't here"))

# print(count_ch("the goofy doom of the balloon goons", "o"))
# print(count_ch("papa pony and the parcel post problem", "p"))
# print(count_ch("this bunch of words has no", "e"))

# def starts_with(target, string):
# 	n = len(target)
# 	m = len(string)
# 	if n > m:
# 		return False
# 	return target == string[:n]

# # print(starts_with('bob','bob newby'))
# # print(starts_with('bill','electric bill'))

# def is_substring(target, string):
# 	n = len(string)
# 	m = len(target)
# 	for i in range(n-m+1):
# 		if target == string[i:i+m]:
# 			return True
# 	return False

# print(is_substring('bad','abracadabra'))
# print(is_substring('dab','abracadabra'))
# print(is_substring('pony','pony'))
# print(is_substring('','balloon'))
# print(is_substring('balloon',''))

# def count_substring(string, target):
# 	index = 0
# 	count = 0
# 	while index < len(string) - len(target) + 1:
# 		if string[index:index+len(target)] == target:
# 			count += 1
# 			index = index + len(target)
# 		else:
# 			index += 1
# 	return count
# print(count_substring('love, love, love, all you need is love', 'love'))

# def locate_first(target, string):
# 	index = 0
# 	while index < len(string) - len(target) + 1:
# 		if string[index:index+len(target)] == target:
# 			return index
# 		index += 1
# 	return -1

# print(locate_first('ook','cookbook'))

# def locate_all(string, target):
# 	index = 0
# 	pos = []
# 	while index < len(string) - len(target) + 1:
# 		if  string[index:index+len(target)] == target:
# 			pos.append(index)
# 		index += 1
# 	return pos
# print(locate_first('base','all your bass are belong to us'))

# print(locate_all('cookbook','ook'))
# print(locate_all('yesyesyes','yes'))
# print(locate_all('the upside down','barb'))

# def breakify(lines):
# 	return "<br>".join(lines)
# lines = ["Haiku frogs in snow",
#          "A limerick came from Nantucket",
#          "Tetrametric drum-beats thrumming, Hiawathianic rhythm."]
# print(breakify(lines))

# def extract_place(name):
# 	index = 0
# 	begin = -1
# 	end = -1
# 	while index < len(name):
# 		if name[index] == '_':
# 			if begin == -1:
# 				begin = index
# 			else:
# 				end = index
# 				break
# 		index += 1
# 	return name[begin+1:end]
# print(extract_place("2018-06-06_MountainView_16:20:00.jpg"))