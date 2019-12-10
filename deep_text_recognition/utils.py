# -*- coding: utf-8 -*-

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CTCLabelConverter(object):
    """ テキストラベルとテキストインデックス間の変換 """

    def __init__(self, character):
        # character（str) : 可能な文字のセット。
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # 注：CTCLossで必要な「'blank'」トークンには0が予約されています
            self.dict[char] = i + 1

        self.character = ['[blank]'] + dict_character  # CTCLossのダミー['blank']トークン（インデックス0）

    def encode(self, text, batch_max_length=25):
        """text-labelをtext-indexに変換します。
         入力：
             text：各画像のテキストラベル。 [batch_size]
         出力：
             text：CTCLossの連結テキストインデックス。
                     [sum（text_lengths）] = [text_index_0 + text_index_1 + ... + text_index_（n-1）]
             length：各テキストの長さ。 [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, text_index, length):
        """ テキストインデックスをテキストラベルに変換します。 """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # 繰り返し文字と空白を削除します。
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts


class AttnLabelConverter(object):
    """ テキストラベルとテキストインデックス間の変換 """

    def __init__(self, character):
        # character (str): 可能な文字のセット。
        # [GO]アテンションデコーダーの開始トークン。 [s]文末トークン。
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ テキストラベルをテキストインデックスに変換します。
        input:
         入力：
             text：各画像のテキストラベル。 [batch_size]
             batch_max_length：バッチ内のテキストラベルの最大長。 デフォルトでは25
         出力：
             text：アテンションデコーダーの入力。 [batch_size x（max_length + 2）] [GO]トークンに対して+1、[s]トークンに対して+1。
                   text [:, 0]は[GO]トークンであり、テキストは[s]トークンの後に[GO]トークンが埋め込まれます。
             length：[s]トークンもカウントするアテンションデコーダーの出力の長さ。 [3、7、....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # 文の終わりの[s]に対して+1
        # batch_max_length = max（length）＃これはマルチGPU設定には使用できません
        batch_max_length += 1
        # 最初のステップで[GO]の追加+1。 batch_textには、[s]トークンの後に[GO]トークンが埋め込まれます。
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ テキストインデックスをテキストラベルに変換します。 """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class Averager(object):
    """ 損失平均に使用されるtorch.Tensorの平均を計算します。 """

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res