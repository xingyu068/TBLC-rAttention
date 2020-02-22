# -*- coding: utf-8 -*-
"""
Created on Sun May  5 13:50:09 2019
@author: xingyu

"""

from __future__ import print_function
import os
import tensorflow as tf
import tensorflow.contrib.keras as kr
from model_TBLC_rAttention import TBLC_rAttention_config , TBLC_rAttention
from data.cnews_loader import read_category, read_vocab
import matplotlib.pyplot as plt
import jieba
import jieba.analyse
import numpy as np
import codecs
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator  
import EmotionalComputing



try:
    bool(type(unicode))
except NameError:
    unicode = str


base_dir = 'helper/data/cnews' 
train_dir = os.path.join(base_dir, 'train_999.txt') 
train_dir999 = os.path.join(base_dir, 'sum_999.txt') 
test_dir = os.path.join(base_dir, 'test_999.txt')  
val_dir = os.path.join(base_dir, 'val_999.txt')     
vocab_dir = os.path.join(base_dir, 'vocab_999.txt') 
save_dir = 'checkpoints/emotion_TBLC_rAttention'                     
save_path = os.path.join(save_dir, 'best_validation') 



class TBLC_rAttentionModel:

    def __init__(self):
        self.config = TBLC_rAttention_config
        self.categories, self.cat_to_id = read_category()    
        self.words, self.word_to_id = read_vocab(vocab_dir) 
        self.config.vocab_size = len(self.words)
        self.model = TBLC_rAttention(self.config)         

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  


    def predict(self, message):
       
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id] 
        
        feed_dict = {            
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)

        handler = EmotionalComputing.EmotionalComputing()   
        score = handler.detect(message)      
        if y_pred_cls == [2]:
            score = -score
        if y_pred_cls == [1]: 
            score = 0.0

        print('Forecast used data:',message)
        print('Forecast Result:', y_pred_cls, '   ', score, '   ', end="")
        
        return self.categories[y_pred_cls[0]]
        
    

def wordcloudtest(wordcloudbase,wordcloudname='wordcloud'):
    print(10*'=','tain_wordcloud',10*'=')
    backgroud_Image = plt.imread('矩形2.jpg')

    wc = WordCloud(
        background_color='white', 
        mask=backgroud_Image,
        font_path='D:/pythontraining/jinyong/jinyong15/simhei.ttf',
        max_words=2000, 
        stopwords=STOPWORDS,
        max_font_size=150,
        random_state=30)    
    
    text=codecs.open(wordcloudbase,'rb','UTF-8').read()          
    tags=jieba.analyse.extract_tags(text,topK=100,withWeight=True)
    tf=dict((a[0],a[1]) for a in tags)
    wc.generate_from_frequencies(tf)
    img_colors = ImageColorGenerator(backgroud_Image)
    plt.figure(num=None,figsize=(12,10),facecolor='w',edgecolor='k')

    plt.imshow(wc)
    plt.axis('off')
    plt.show()
    wc.to_file(wordcloudname+'.png')



if __name__ == '__main__':
    print(10*'=','TBLC_rAttention:test',10*'=')
    TBLC_rAttention_model = TBLC_rAttentionModel()  
    test_demo = ['在门口药店买了一堆云南贵州那边的杂牌子，吃了不管用，我又从京东买了这个999，真神奇，第二天几天了，坚决好评，以后感冒我只买999。',
                 '多次买了，有点感冒吃一袋就对了，比吃西药好，我们常备的999感冒灵颗粒，京东大药房值得点赞，发货速度快！',
                 '京东快递一如既往地快，以后还会光顾，好评哦！',
                 '不错啊，速度超快的，一天就到了，在其他地方买的至少要三天啊，谢谢老板还送了漂亮的杯子，很满意哦。',
                 '质量很好，送货速度，包装完整。',
                 '第一次在网上买药，回来就查了下，是正品！京东比药店便宜好几块钱呢！速度也快，第二天就到了，家庭必备药，以后就选京东了。',
                 '一直以来都是用999的产品，大品牌值得信赖，中药成分更安全放心。',
                 '送货很快，是正品，感冒的时候一直喝这个，现在特价，也帮亲戚拍几盒备着。',
                 '家里常备药，省的关键时刻弄药吃。还是比较喜欢用999。',
                 '自营店的药品已经买过好几次了，比外面药店便宜，感冒药家庭常备。',
                 '',
                 '',
                 '',
                 '图方便 性价比越来越一般了。',
                 '效果一般，依旧感冒，还是多喝热水吧。',
                 '到货时间长，效果不是很理想！',
                 '还好吧，先试试看等后续追评。',
                 '感觉这次买的不起作用，喝了没啥用。',
                 '还可以吧，一般般了。',
                 '效果一般吧，吃了两天感觉没什么太大作用……',
                 '不知道什么原因，感冒的时候喝上也不起作用。',
                 '药效很不错，就是买了第二天就降价，还不给价保，很不合理！！！',
                 '药很满意就是直营店快递也太慢了正常我們这里京东自营店都是一天到了感冒了等这个药等严重了。',
                 '一般般吧，效果也不是很明显。',
                 '',
                 '',
                 '',
                 '大部分都是漏气的。',
                 '大家不要买京东会员，简直巨坑！！欺侮人！！愤怒中！买了以后领不到券，打电电给客服全部统一台词不处理问题，只会和我们说官话，还说因为我以前享受过了，你家京东是一锤子买-卖？老客户不能享受？客服只会反复说领券是概率问题，那么我一张领不到的属于什么概率啊？还说让我原价购买，你199减100的券，别人花100，你意思让我花*？还说以前能领券是新注册用户会给，建议我重新注册！说重新注册就能重新注册？白条小金库开了绑定我的身份了那我办的会员你给退不？硬是说我的号没被黑，没任何问题，我第一没有代买，第二基本没拒收过，就是拿我家里人号同样抢了券，然后家人号都变黑号，我朋友号随便领券，我的什么都领不了！！谁家没两个手机？我办了两年会员，现在任何券领不到！问客服不承认京东会黑号，会员就是坑钱的，老客户黑，新客户才能领券，大家以后都别办会员，因为需要总换号！东西大部分都提高价格，只有领券才能真的，很多都是都可以半价！领不到券，不懂的就只能买原价，我直老老实实买东西，结果坑客户，损害消费者权益，已经决定去投诉，投诉到底！别的地方都是老客户有好的待遇，只有你们京东专坑老用户。',
                 '第一次在京东买药 感觉不像正品 泡出来的颜色不如实体的浓,效果不好。',
                 '包装太差，都有洞，可以看见里面的东西。',
                 '特别特别差，大家千万不要买，一点效果没有，我奶吃完还刺激胃，胃疼。',
                 '看哈你们发过来的药，像是垃圾堆里扒出来的一样，为什么会有这样的东西发出来。',
                 '不是吧，买个感冒药还是被拆过的，这谁敢喝啊，京东大药房真烂。',
                 '这药全部被打开了，气死人?',
                 '效果很一般喝完一盒了额没啥用不如一起买的板蓝根有用。',
                 '药到的时候我想我已经死了！']
    
    for i in test_demo:
        print(TBLC_rAttention_model.predict(i))
         
    wordcloudtest(train_dir999,'999感冒灵词云')        


       
