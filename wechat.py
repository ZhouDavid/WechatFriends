# -*- coding: gbk -*-
# import itchat


# def get_sex_proportion(friends):
# 	total_friends_num = len(friends)
# 	male_num = 0
# 	female_num = 0
# 	other_num = 0
# 	for fr in friends:
# 		if fr['Sex'] == 1:
# 			male_num += 1
# 		elif fr['Sex'] == 2:
# 			female_num += 1
# 		else:
# 			other_num += 1
# 	print('male:{}, female:{}, others:{}, total:{}'.format(male_num,female_num,other_num,total_friends_num))
# 	return (male_num/total_friends_num,female_num/total_friends_num,other_num/total_friends_num)


# itchat.auto_login(hotReload=True)
# friends = itchat.get_friends(update=True)
# (p1,p2,p3) = get_sex_proportion(friends[1:])
# print("proportion:{},{},{}".format(p1,p2,p3))


# 首先需要安装itchat等包：pip install itchat
import itchat  # itchat documentation -- https://itchat.readthedocs.io/zh/latest/api/
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re
from wordcloud import WordCloud, ImageColorGenerator
import PIL.Image as Image
import jieba  # chinese word segementation tool
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='DroidSansFallbackFull.ttf', size=14)  # load font

# 登录自己的微信。过程中会生产一个登录二维码，扫码之后即可登录。
# login, default a QR code will be generated, scan for login
itchat.auto_login(hotReload = True)

# 把自己好友的相关信息爬下来
friends = itchat.get_friends(update=True)[0:]  # get all friends

# get male-female-ratio
def get_male_female_count(friends):
    male = 0
    female = 0
    others = 0
    for friend in friends:
        # ”性别“是存放在一个字典里面的，key 是”Sex“
        # 男性值为 1，女性为 2，其他是不明性别的（就是没有填的）
        sex = friend['Sex']
        if sex == 1:
            male += 1
        elif sex == 2:
            female += 1
        else:
            others += 1
    return male, female, others

# friend[0]是自己的信息，所以从friend[1]开始
male, female, others = get_male_female_count(friends[1:])

# 微信好友数量
total = len(friends[1:])
# 打印出好友性别比例
print('男性数量: {:d}, ratio: {:.4f}'.format(male, male / float(total)))
print('女性数量: {:d}, ratio: {:.4f}'.format(female, female / float(total)))
print('其他数量: {:d}, ratio: {:.4f}'.format(others, others / float(total)))
print('total: {:d}'.format(total))

# 把好友性别数据画成图
# plot male-female-ratio
index = np.arange(3)
genders = (male, female, others)
bar_width = 0.35
plt.figure(figsize=(14, 7))
plt.bar(index, genders, bar_width, alpha=0.6, color='rgb')
plt.xlabel('Gender', fontsize=16)  
plt.ylabel('Population', fontsize=16)
plt.title('Male-Female Population', fontsize=18)  
plt.xticks(index, ('Male', 'Female', 'Other'), fontsize=14, rotation=20)
plt.ylim(0,1000)
for idx, gender in zip(index, genders):
    plt.text(idx, gender + 0.1, '%.0f' % gender, ha='center', va='bottom', fontsize=14, color='black')
plt.show()

# 将数据导入到DataFrame，并筛选出微信名、备注名、性别、省份、城市、个性签名五个字段
# extract the variables: NickName, Sex, City, Province, Signature
def get_features(friends):
    features = []
    for friend in friends:
        feature = {'NickName': friend['NickName'], 'Sex': friend['Sex'], 'City': friend['City'], 
                  'Province': friend['Province'], 'Signature': friend['Signature']}
        features.append(feature)
    return pd.DataFrame(features)


features = get_features(friends[1:])


# 微信好友地域分布分析
# 根据省份、城市进行数据的分组和聚合，选择排名前二十的
locations = features.loc[:, ['Province', 'City']]  # get location columns
locations = locations[locations['Province'] != '']  # clean empty city or province records
data = locations.groupby(['Province', 'City']).size().unstack()  # group by and count
count_subset = data.take(data.sum(1).argsort())[-20:]  # obtain the 20 highest data

# plot
subset_plot = count_subset.plot(kind='bar', stacked=True, figsize=(24, 24))

# set fonts
xtick_labels = subset_plot.get_xticklabels()
for label in xtick_labels: 
    label.set_fontproperties(font)
legend_labels = subset_plot.legend().texts
for label in legend_labels:
    label.set_fontproperties(font)
    label.set_fontsize(10)

plt.xlabel('Province', fontsize=20)
plt.ylabel('Number', fontsize=20)
plt.show()

sigature_list = []
for signature in features['Signature']:
    # 先替换掉emoji、span、class 等等这些无关紧要的词
    signature = signature.strip().replace('span', '').replace('class', '').replace('emoji', '')
    # 还有类似<>/= 之类的符号，也需要写个简单的正则替换掉
    # re.compile(ur'[^a-zA-Z0-9\u4e00-\u9fa5 ]').sub('', signature)
    signature = re.compile('1f\d+\w*|[<>/=]').sub('', signature)
    if (len(signature) > 0):
        sigature_list.append(signature)
# 再把所有拼起来，得到 text 字串
text = " ".join(sigature_list)


wordlist = jieba.cut(text, cut_all=True)
words = " ".join(wordlist)

# 微信好友个性签名的词云统计
coloring = np.array(Image.open('avatar.jpg'))
wc = WordCloud(background_color='white', max_words=2000, mask=coloring, max_font_size=60, random_state=42, 
               font_path='DroidSansFallbackFull.ttf', scale=2).generate(words)
image_color = ImageColorGenerator(coloring)
plt.figure(figsize=(32, 16))
# plt.imshow(wc.recolor(color_func=image_color))
plt.imshow(wc)
plt.axis('off')
plt.show()
