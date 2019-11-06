# -*- coding: utf-8 -*-

import os
import time
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
from nltk.metrics.distance import edit_distance

from utils import CTCLabelConverter, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate
from model import Model
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

def benchmark_all_eval(model, criterion, converter, opt, calculate_infer_time=False):
    #eval_data_list = ['IIIT5k_3000', 'SVT', 'IC03_860', 'IC03_867', 'IC13_857',
    #                 'IC13_1015', 'IC15_1811', 'IC15_2077', 'SVTP', 'CUTE80']

    eval_data_list = ['origin']

    if calculate_infer_time:
        evaluation_batch_size = 1  # 画像ごとのGPU推論時間を計算するには、batch_sizeを1にする必要があります。
    else:
        evaluation_batch_size = opt.batch_size

    list_accuracy = []
    total_forward_time = 0
    total_evaluation_data_number = 0
    total_correct_number = 0
    print('-' * 80)
    for eval_data in eval_data_list:
        eval_data_path = os.path.join(opt.eval_data, eval_data)
        AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        eval_data = hierarchical_dataset(root=eval_data_path, opt=opt)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)

        _, accuracy_by_best_model, norm_ED_by_best_model, PRED, GT, infer_time, length_of_data = validation(
            model, criterion, evaluation_loader, converter, opt)
        list_accuracy.append(f'{accuracy_by_best_model:0.3f}')
        total_forward_time += infer_time
        total_evaluation_data_number += len(eval_data)
        total_correct_number += accuracy_by_best_model * length_of_data
        print('Acc %0.3f\t normalized_ED %0.3f' % (accuracy_by_best_model, norm_ED_by_best_model))
        print('-' * 80)

    distance_list = []
    for p, g in zip(PRED, GT):
        distance = edit_distance(p, g) / max([len(p), len(g)])
        distance_list(distance)
        print('-*' * 45)
        print('{}番目'.format(len(distance_list)))
        print('PRED:{}\n GT:{}\n edit_dis:{}\n'.format(p, g, distance))
    print('NORM: {}'.format(1 - (sum(distance_list) / len(distance_list))))
    averaged_forward_time = total_forward_time / total_evaluation_data_number * 1000
    total_accuracy = total_correct_number / total_evaluation_data_number
    params_num = sum([np.prod(p.size()) for p in model.parameters()])

    evaluation_log = 'accuracy: '
    for name, accuracy in zip(eval_data_list, list_accuracy):
        evaluation_log += f'{name}: {accuracy}\t'
    evaluation_log += f'total_accuracy: {total_accuracy:0.3f}\t'
    evaluation_log += f'averaged_infer_time: {averaged_forward_time:0.3f}\t# parameters: {params_num/1e6:0.3f}'
    print(evaluation_log)
    with open(f'./result/{opt.experiment_name}/log_all_evaluation.txt', 'a') as log:
        log.write(evaluation_log + '\n')

    return None


def validation(model, criterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # 最大長予測用
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

        start_time = time.time()
        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred).log_softmax(2)
            forward_time = time.time() - start_time


            # CTCデコーダーの検証損失を計算します。
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.permute(1, 0, 2)  # CTCloss形式を使用するには

            # ctc_lossの問題を回避するため、ctc_lossの計算でcudnnを無効にしました
            # https://github.com/jpuigcerver/PyLaia/issues/16
            torch.backends.cudnn.enabled = False
            cost = criterion(preds, text_for_loss, preds_size, length_for_loss)
            torch.backends.cudnn.enabled = True

            # 最大確率を選択し、インデックスを文字にデコードします
            _, preds_index = preds.max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)

        else:
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time.time() - start_time

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            # 最大確率を選択し、インデックスを文字にデコードします
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # 精度を計算します。
        for pred, gt in zip(preds_str, labels):
            if 'Attn' in opt.Prediction:
                pred = pred[:pred.find('[s]')]  # 文の終わりトークン（[s]）の後の剪定
                gt = gt[:gt.find('[s]')]

            if pred == gt:
                n_correct += 1
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)

    accuracy = n_correct / float(length_of_data) * 100

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, labels, infer_time, length_of_data


def test(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location='cpu'))
    opt.experiment_name = '_'.join(opt.saved_model.split('/')[1:])
    # print(model)

    """ keep evaluation model and result logs """
    os.makedirs(f'./result/{opt.experiment_name}', exist_ok=True)
    os.system(f'cp {opt.saved_model} ./result/{opt.experiment_name}/')

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    """ evaluation """
    model.eval()
    with torch.no_grad():
        if opt.benchmark_all_eval:  # evaluation with 10 benchmark evaluation datasets
            benchmark_all_eval(model, criterion, converter, opt)
        else:
            AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
            eval_data = hierarchical_dataset(root=opt.eval_data, opt=opt)
            evaluation_loader = torch.utils.data.DataLoader(
                eval_data, batch_size=opt.batch_size,
                shuffle=False,
                num_workers=int(opt.workers),
                collate_fn=AlignCollate_evaluation, pin_memory=True)
            _, accuracy_by_best_model, _, _, _, _, _ = validation(
                model, criterion, evaluation_loader, converter, opt)

            print(accuracy_by_best_model)
            with open('./result/{0}/log_evaluation.txt'.format(opt.experiment_name), 'a') as log:
                log.write(str(accuracy_by_best_model) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', required=True, help='path to evaluation dataset')
    parser.add_argument('--benchmark_all_eval', action='store_true', help='evaluate 10 benchmark evaluation datasets')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    #parser.add_argument('--character', type=str, default='./japanese_wod/japanese_wod.txt', help='character label')
    parser.add_argument('--character', type=str,default='あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわゐうゑをんアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヰウヱヲンゔがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽヴガギグゲゴザジズゼゾダヂヅデドバビブベボヷヸヹヺパピプペポぁぃぅぇぉっゃゅょゎァィゥェォヵㇰヶㇱㇲッㇳㇴㇵㇶㇷㇸㇹㇺャュョㇻㇼㇽㇾㇿヮ亜哀挨愛曖悪握圧扱宛嵐安案暗以衣位囲医依委威為畏胃尉異移萎偉椅彙意違維慰遺緯域育一壱逸茨芋引印因咽姻員院淫陰飲隠韻右宇羽雨唄鬱畝浦運雲永泳英映栄営詠影鋭衛易疫益液駅悦越謁閲円延沿炎怨宴媛援園煙猿遠鉛塩演縁艶汚王凹央応往押旺欧殴桜翁奥横岡屋億憶臆虞乙俺卸音恩温穏下化火加可仮何花佳価果河苛科架夏家荷華菓貨渦過嫁暇禍靴寡歌箇稼課蚊牙瓦我画芽賀雅餓介回灰会快戒改怪拐悔海界皆械絵開階塊楷解潰壊懐諧貝外劾害崖涯街慨蓋該概骸垣柿各角拡革格核殻郭覚較隔閣確獲嚇穫学岳楽額顎掛潟括活喝渇割葛滑褐轄且株釜鎌刈干刊甘汗缶完肝官冠巻看陥乾勘患貫寒喚堪換敢棺款間閑勧寛幹感漢慣管関歓監緩憾還館環簡観韓艦鑑丸含岸岩玩眼頑顔願企伎危机気岐希忌汽奇祈季紀軌既記起飢鬼帰基寄規亀喜幾揮期棋貴棄毀旗器畿輝機騎技宜偽欺義疑儀戯擬犠議菊吉喫詰却客脚逆虐九久及弓丘旧休吸朽臼求究泣急級糾宮救球給嗅窮牛去巨居拒拠挙虚許距魚御漁凶共叫狂京享供協況峡挟狭恐恭胸脅強教郷境橋矯鏡競響驚仰暁業凝曲局極玉巾斤均近金菌勤琴筋僅禁緊錦謹襟吟銀区句苦駆具惧愚空偶遇隅串屈掘窟熊繰君訓勲薫軍郡群兄刑形系径茎係型契計恵啓掲渓経蛍敬景軽傾携継詣慶憬稽憩警鶏芸迎鯨隙劇撃激桁欠穴血決結傑潔月犬件見券肩建研県倹兼剣拳軒健険圏堅検嫌献絹遣権憲賢謙鍵繭顕験懸元幻玄言弦限原現舷減源厳己戸古呼固股虎孤弧故枯個庫湖雇誇鼓錮顧五互午呉後娯悟碁語誤護口工公勾孔功巧広甲交光向后好江考行坑孝抗攻更効幸拘肯侯厚恒洪皇紅荒郊香候校耕航貢降高康控梗黄喉慌港硬絞項溝鉱構綱酵稿興衡鋼講購乞号合拷剛傲豪克告谷刻国黒穀酷獄骨駒込頃今困昆恨根婚混痕紺魂墾懇左佐沙査砂唆差詐鎖座挫才再災妻采砕宰栽彩採済祭斎細菜最裁債催塞歳載際埼在材剤財罪崎作削昨柵索策酢搾錯咲冊札刷刹拶殺察撮擦雑皿三山参桟蚕惨産傘散算酸賛残斬暫士子支止氏仕史司四市矢旨死糸至伺志私使刺始姉枝祉肢姿思指施師恣紙脂視紫詞歯嗣試詩資飼誌雌摯賜諮示字寺次耳自似児事侍治持時滋慈辞磁餌璽鹿式識軸七失室疾執湿嫉漆質実芝写社車舎者射捨赦斜煮遮謝邪蛇尺借酌釈爵若弱寂手主守朱取狩首殊珠酒腫種趣寿受呪授需儒樹収囚州舟秀周宗拾秋臭修袖終羞習週就衆集愁酬醜蹴襲十汁充住柔重従渋銃獣縦叔祝宿淑粛縮塾熟出述術俊春瞬旬巡盾准殉純循順準潤遵処初所書庶暑署緒諸女如助序叙徐除小升少召匠床抄肖尚招承昇松沼昭宵将消症祥称笑唱商渉章紹訟勝掌晶焼焦硝粧詔証象傷奨照詳彰障憧衝賞償礁鐘上丈冗条状乗城浄剰常情場畳蒸縄壌嬢錠譲醸色拭食植殖飾触嘱織職辱尻心申伸臣芯身辛侵信津神唇娠振浸真針深紳進森診寝慎新審震薪親人刃仁尽迅甚陣尋腎須図水吹垂炊帥粋衰推酔遂睡穂随髄枢崇数据杉裾寸瀬是井世正生成西声制姓征性青斉政星牲省凄逝清盛婿晴勢聖誠精製誓静請整醒税夕斥石赤昔析席脊隻惜戚責跡積績籍切折拙窃接設雪摂節説舌絶千川仙占先宣専泉浅洗染扇栓旋船戦煎羨腺詮践箋銭潜線遷選薦繊鮮全前善然禅漸膳繕狙阻祖租素措粗組疎訴塑遡礎双壮早争走奏相荘草送倉捜挿桑巣掃曹曽爽窓創喪痩葬装僧想層総遭槽踪操燥霜騒藻造像増憎蔵贈臓即束足促則息捉速側測俗族属賊続卒率存村孫尊損遜他多汰打妥唾堕惰駄太対体耐待怠胎退帯泰堆袋逮替貸隊滞態戴大代台第題滝宅択沢卓拓託濯諾濁但達脱奪棚誰丹旦担単炭胆探淡短嘆端綻誕鍛団男段断弾暖談壇地池知値恥致遅痴稚置緻竹畜逐蓄築秩窒茶着嫡中仲虫沖宙忠抽注昼柱衷酎鋳駐著貯丁弔庁兆町長挑帳張彫眺釣頂鳥朝貼超腸跳徴嘲潮澄調聴懲直勅捗沈珍朕陳賃鎮追椎墜通痛塚漬坪爪鶴低呈廷弟定底抵邸亭貞帝訂庭逓停偵堤提程艇締諦泥的笛摘滴適敵溺迭哲鉄徹撤天典店点展添転塡田伝殿電斗吐妬徒途都渡塗賭土奴努度怒刀冬灯当投豆東到逃倒凍唐島桃討透党悼盗陶塔搭棟湯痘登答等筒統稲踏糖頭謄藤闘騰同洞胴動堂童道働銅導瞳峠匿特得督徳篤毒独読栃凸突届屯豚頓貪鈍曇丼那奈内梨謎鍋南軟難二尼弐匂肉虹日入乳尿任妊忍認寧熱年念捻粘燃悩納能脳農濃把波派破覇馬婆罵拝杯背肺俳配排敗廃輩売倍梅培陪媒買賠白伯拍泊迫剝舶博薄麦漠縛爆箱箸畑肌八鉢発髪伐抜罰閥反半氾犯帆汎伴判坂阪板版班畔般販斑飯搬煩頒範繁藩晩番蛮盤比皮妃否批彼披肥非卑飛疲秘被悲扉費碑罷避尾眉美備微鼻膝肘匹必泌筆姫百氷表俵票評漂標苗秒病描猫品浜貧賓頻敏瓶不夫父付布扶府怖阜附訃負赴浮婦符富普腐敷膚賦譜侮武部舞封風伏服副幅復福腹複覆払沸仏物粉紛雰噴墳憤奮分文聞丙平兵併並柄陛閉塀幣弊蔽餅米壁璧癖別蔑片辺返変偏遍編弁便勉歩保哺捕補舗母募墓慕暮簿方包芳邦奉宝抱放法泡胞俸倣峰砲崩訪報蜂豊飽褒縫亡乏忙坊妨忘防房肪某冒剖紡望傍帽棒貿貌暴膨謀頰北木朴牧睦僕墨撲没勃堀本奔翻凡盆麻摩磨魔毎妹枚昧埋幕膜枕又末抹万満慢漫未味魅岬密蜜脈妙民眠矛務無夢霧娘名命明迷冥盟銘鳴滅免面綿麺茂模毛妄盲耗猛網目黙門紋問冶夜野弥厄役約訳薬躍闇由油喩愉諭輸癒唯友有勇幽悠郵湧猶裕遊雄誘憂融優与予余誉預幼用羊妖洋要容庸揚揺葉陽溶腰様瘍踊窯養擁謡曜抑沃浴欲翌翼拉裸羅来雷頼絡落酪辣乱卵覧濫藍欄吏利里理痢裏履璃離陸立律慄略柳流留竜粒隆硫侶旅虜慮了両良料涼猟陵量僚領寮療瞭糧力緑林厘倫輪隣臨瑠涙累塁類令礼冷励戻例鈴零霊隷齢麗暦歴列劣烈裂恋連廉練錬呂炉賂路露老労弄郎朗浪廊楼漏籠六録麓論和話賄脇惑枠湾腕丑丞乃之乎也云亘些亦亥亨亮仔伊伍伽佃佑伶侃侑俄俠俣俐倭俱倦倖偲傭儲允兎兜其冴凌凜凧凪凰凱函劉劫勁勺勿匁匡廿卜卯卿厨厩叉叡叢叶只吾吞吻哉哨啄哩喬喧喰喋嘩嘉嘗噌噂圃圭坐尭坦埴堰堺堵塙壕壬夷奄奎套娃姪姥娩嬉孟宏宋宕宥寅寓寵尖尤屑峨峻崚嵯嵩嶺巌巫已巳巴巷巽帖幌幡庄庇庚庵廟廻弘弛彗彦彪彬徠忽怜恢恰恕悌惟惚悉惇惹惺惣慧憐戊或戟托按挺挽掬捲捷捺捧掠揃摑摺撒撰撞播撫擢孜敦斐斡斧斯於旭昂昊昏昌昴晏晃晒晋晟晦晨智暉暢曙曝曳朋朔杏杖杜李杭杵杷枇柑柴柘柊柏柾柚桧栞桔桂栖桐栗梧梓梢梛梯桶梶椛梁棲椋椀楯楚楕椿楠楓椰楢楊榎樺榊榛槙槍槌樫槻樟樋橘樽橙檎檀櫂櫛櫓欣欽歎此殆毅毘毬汀汝汐汲沌沓沫洸洲洵洛浩浬淵淳渚淀淋渥湘湊湛溢滉溜漱漕漣澪濡瀕灘灸灼烏焰焚煌煤煉熙燕燎燦燭燿爾牒牟牡牽犀狼猪獅玖珂珈珊珀玲琢琉瑛琥琶琵琳瑚瑞瑶瑳瓜瓢甥甫畠畢疋疏皐皓眸瞥矩砦砥砧硯碓碗碩碧磐磯祇祢祐祷禄禎禽禾秦秤稀稔稟稜穣穹穿窄窪窺竣竪竺竿笈笹笙笠筈筑箕箔篇篠簞簾籾粥粟糊紘紗紐絃紬絆絢綺綜綴緋綾綸縞徽繫繡纂纏羚翔翠耀而耶耽聡肇肋肴胤胡脩腔脹膏臥舜舵芥芹芭芙芦苑茄苔苺茅茉茸茜莞荻莫莉菅菫菖萄菩萌萊菱葦葵萱葺萩董葡蓑蒔蒐蒼蒲蒙蓉蓮蔭蔣蔦蓬蔓蕎蕨蕉蕃蕪薙蕾蕗藁薩蘇蘭蝦蝶螺蟬蟹蠟衿袈袴裡裟裳襖訊訣註詢詫誼諏諄諒謂諺讃豹貰賑赳跨蹄蹟輔輯輿轟辰辻迂迄辿迪迦這逞逗逢遥遁遼邑祁郁鄭酉醇醐醍醬釉釘釧銑鋒鋸錘錐錆錫鍬鎧閃閏閤阿陀隈隼雀雁雛雫霞靖鞄鞍鞘鞠鞭頁頌頗顚颯饗馨馴馳駕駿驍魁魯鮎鯉鯛鰯鱒鱗鳩鳶鳳鴨鴻鵜鵬鷗鷲鷺鷹麒麟麿黎黛鼎0123456789。、ー～`"゜″()abcdfghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ龍澤嶋魏國癌應趙叩剥牝倶輌廠吊曾填鰭條揆聘蒋學藝嚢袁裔曰讐遽諱灌蕭姦桓渕榴鐵綬塵廣牌圓掴萬濱謳梱嘘歪頸眞嶽隋脆翅盧于舘焉憑諜棘牢峙尹醤騨壺痺噛譚扮狗瀧諡襄狐崔仇扁筐勒漑蜀諫團燈罹咸聯冨渠糞嗜櫻苻擲隕鍾輛胚叱蔡靭禰檜揖崗罠跋叛囃苫祠礫孵騙煽娼鄧埠荊嶼妓薇麹爬浙揶揄濠淮饉蛋薔屏侠膠雍馮薨誅矮渤墟榜髭渾鉾攘噺俯殷宍站疹煕篭寇彭腱做嘴冪訛剃堯會腿堡匈彷妾曼姜豫賈衙瞑餐傅爺陝疆疇弼壽銚耆隘鼠邊鹵莢梵洩殲泄夭窩彌蘆轢齋捏頬縣酋虔萼鍼拗姚荼蝋邱范鰓覗藪﨑聚珪喘佛實繍熾傀穆艤娶儡冀址炸遙粕冑掻汪弩璋炒僑閔咎橿飴躇贅躊厥盃鉤呆塹肛韋傳怯愕濾拿贄宦瀋裴鸞奢阮屠嘔揉翰鹸閻婁祓槇浚淘苞鮫姑蝕懿荏掩洒茹薛莱嵌膵狸蔚戮錨閘甕冤掟濤逼鴎彿靱蕩賤絨榮拮趾咬鑽篆軋臀瘤祚桿穎稙衍竈鯖蕊砺糠溥旛僭垢嬰蜘邉厭髙凛掾瓊蛛蹊躯饅甑羌廓與鄴漿麾嚆苅涌攣邇戎荀紆滸圀讒哈ा韮顆卦惠綏潭撥姶鋏塘椒膿靡楔廬佩幇蛭槃狛煥頚簒壷媚來枡乖證吠岑隧亢鉉腋脛穢游并贋擾咳蟻贖饒兗猥禹毫聾祟疼薗舒埔睨畷囁竇袂鮭艘抒鞆膣誦邵崑圳丕棍膀鍮胱捌霖瑜謗屍隴閩毯諌謬獨櫃什鰐爛禿韶蟄莽旱氣竄筏誹蛹蛙魃燻樊藏顛疱驤磔賽邁餃餡恪輻攪猷悶轍咄霍瘡岱唖糎芒雉讀撹帛駁蜷渭檻幟冉菟當赫欅冲吃碍髷鵠笏狄鏑戌俘墺鏃臧驃咋檣眷鶯漉鞨徊埃碇徘繹禮靺耿邯甦聰兌茲徨婢燐煬眩樂嗚糟箚髑拌髏炳脾桝凋拼俟辟渫盈斂鍔㎡稠坡貶斌賣鯱邨躙箪懺槐㎏攀粲孚趨寶兪熹號筵總瞞嬪閾戍齊朧匐羲楡鎬焙泗龐潁槿庾哥鐸鰹薮琅瀑蹂舛囮奸焔瞰琨卍孕逍訶秉鴉滬匍艾諍鮑箒蝉粍翟晁帷呟蘂稷臍蠣禧憚汾鋤溪槓瞼攸涜澳齧鰻衞奘迹饌奠坤檗菰洙埜鎚舐礦舅捩睿仄僻偕龕剪檄聳瑕儁鬘疽燁尸咀簗奕涵崙贔瀉蒜瀞屓雙闍崋滓廈晰德蛾羹葱澱鈔疵歿褻涛滄臺匪婉闕宸蛸哭鋲奚齟滲齬鄒郢虻踵翊闊臂躁喀箭蓼棠祺寨亞邢剌痒琲杓緬綽廖諷隗飫怡聲澎烙槨咤霸匝叟澁顒盪鈞棹瑤逅邂狡逵斛่瀾旁攫佇鄲仗杣柩鬚嚥鉦薊潼臘扈黌萇痍謐襷兒閭娥矧卉痰沛藉撚棗謨沐橇弗燮涸篩猾蟲ุ沂幀癬夙竟冏昶肆訥癸頴笥褶嬌炙柯腓侈體皺嗟巖寳愍魄杢瑪褌韜繇葭黔跪籃疸齎猜茗拵鞮靳紇艀滕藺恫禊้瑩์翳籤皝彧淹經咫几嬬椙梟苟葯份禦嚼曄燼疣粤瞻屁篁偈刎躬咩́蝗餘礒咥猗陞將畦冰橈鑿翡褚敲鴈菴芬瓚舍漳芻瑾譙睾炯瓘弉濬弋袍驕嵜凉斃衾鐙酊襞邀蝠甄沮樵瀛唸杞螢關礪寔剱呵駱蘊截枷琦徂磾鄙爲蜃狽炮碕肱賁廼絲郝犍駈楳昱吼羯鮒贛鄂蒴涇歆嶷筧匯邕獏偃粂猴蝙獰銛邏舩烹淆壹掖鵡蜻銕滎鬆詛酩摸搗舫暈鹽囂㎜蛯嚮霰擱艮胥捐蟠絣琮鈎欒沽聶廂咆戔憺筍頡賎簀錚渟鞋廆戈厲綝秣谿苧濮鯰蝸俎邳罕燧戰楮嘯甜琛蟇鉋癇樅渣鑼蛉哮佼髴頽嶌儚劭驪髣磋皖鈿彝鴦蝿栢娑鑓譴敖軾籌圖筰篋暐猩喇倚甌誣醗厦砒撼躓褥發泓毓鰍軛鵄頤愈顥匙敞鞅苓臚熈郞僊薯麩顗雹馥恂辯輳痔訌翫彊滇鄱匣剽澗蛤銜龍澤嶋魏國癌應趙叩剥牝倶輌廠吊曾填鰭條揆聘蒋學藝嚢袁裔曰讐遽諱灌蕭姦桓渕榴鐵綬塵廣牌圓掴萬濱謳梱嘘歪頸眞嶽隋脆翅盧于舘焉憑諜棘牢峙尹醤騨壺痺噛譚扮狗瀧諡襄狐崔仇扁筐勒漑蜀諫團燈罹咸聯冨渠糞嗜櫻苻擲隕鍾輛胚叱蔡靭禰檜揖崗罠跋叛囃苫祠礫孵騙煽娼鄧埠荊嶼妓薇麹爬浙揶揄濠淮饉蛋薔屏侠膠雍馮薨誅矮渤墟榜髭渾鉾攘噺俯殷宍站疹煕篭寇彭腱做嘴冪訛剃堯會腿堡匈彷妾曼姜豫賈衙瞑餐傅爺陝疆疇弼壽銚耆隘鼠邊鹵莢梵洩殲泄夭窩彌蘆轢齋捏頬縣酋虔萼鍼拗姚荼蝋邱范鰓覗藪﨑聚珪喘佛實繍熾傀穆艤娶儡冀址炸遙粕冑掻汪弩璋炒僑閔咎橿飴躇贅躊厥盃鉤呆塹肛韋傳怯愕濾拿贄宦瀋裴鸞奢阮屠嘔揉翰鹸閻婁祓槇浚淘苞鮫姑蝕懿荏掩洒茹薛莱嵌膵狸蔚戮錨閘甕冤掟濤逼鴎彿靱蕩賤絨榮拮趾咬鑽篆軋臀瘤祚桿穎稙衍竈鯖蕊砺糠溥旛僭垢嬰蜘邉厭髙凛掾瓊蛛蹊躯饅甑羌廓與鄴漿麾嚆苅涌攣邇戎荀紆滸圀讒哈ा韮顆卦惠綏潭撥姶鋏塘椒膿靡楔廬佩幇蛭槃狛煥頚簒壷媚來枡乖證吠岑隧亢鉉腋脛穢游并贋擾咳蟻贖饒兗猥禹毫聾祟疼薗舒埔睨畷囁竇袂鮭艘抒鞆膣誦邵崑圳丕棍膀鍮胱捌霖瑜謗屍隴閩毯諌謬獨櫃什鰐爛禿韶蟄莽旱氣竄筏誹蛹蛙魃燻樊藏顛疱驤磔賽邁餃餡恪輻攪猷悶轍咄霍瘡岱唖糎芒雉讀撹帛駁蜷渭檻幟冉菟當赫欅冲吃碍髷鵠笏狄鏑戌俘墺鏃臧驃咋檣眷鶯漉鞨徊埃碇徘繹禮靺耿邯甦聰兌茲徨婢燐煬眩樂嗚糟箚髑拌髏炳脾桝凋拼俟辟渫盈斂鍔㎡稠坡貶斌賣鯱邨躙箪懺槐㎏攀粲孚趨寶兪熹號筵總瞞嬪閾戍齊朧匐羲楡鎬焙泗龐潁槿庾哥鐸鰹薮琅瀑蹂舛囮奸焔瞰琨卍孕逍訶秉鴉滬匍艾諍鮑箒蝉粍翟晁帷呟蘂稷臍蠣禧憚汾鋤溪槓瞼攸涜澳齧鰻衞奘迹饌奠坤檗菰洙埜鎚舐礦舅捩睿仄僻偕龕剪檄聳瑕儁鬘疽燁尸咀簗奕涵崙贔瀉蒜瀞屓雙闍崋滓廈晰德蛾羹葱澱鈔疵歿褻涛滄臺匪婉闕宸蛸哭鋲奚齟滲齬鄒郢虻踵翊闊臂躁喀箭蓼棠祺寨亞邢剌痒琲杓緬綽廖諷隗飫怡聲澎烙槨咤霸匝叟澁顒盪鈞棹瑤逅邂狡逵斛่瀾旁攫佇鄲仗杣柩鬚嚥鉦薊潼臘扈黌萇痍謐襷兒閭娥矧卉痰沛藉撚棗謨沐橇弗燮涸篩猾蟲ุ沂幀癬夙竟冏昶肆訥癸頴笥褶嬌炙柯腓侈體皺嗟巖寳愍魄杢瑪褌韜繇葭黔跪籃疸齎猜茗拵鞮靳紇艀滕藺恫禊้瑩์翳籤皝彧淹經咫几嬬椙梟苟葯份禦嚼曄燼疣粤瞻屁篁偈刎躬咩́蝗餘礒咥猗陞將畦冰橈鑿翡褚敲鴈菴芬瓚舍漳芻瑾譙睾炯瓘弉濬弋袍驕嵜凉斃衾鐙酊襞邀蝠甄沮樵瀛唸杞螢關礪寔剱呵駱蘊截枷琦徂磾鄙爲蜃狽炮碕肱賁廼絲郝犍駈楳昱吼羯鮒贛鄂蒴涇歆嶷筧匯邕獏偃粂猴蝙獰銛邏舩烹淆壹掖鵡蜻銕滎鬆詛酩摸搗舫暈鹽囂㎜蛯嚮霰擱艮胥捐蟠絣琮鈎欒沽聶廂咆戔憺筍頡賎簀錚渟鞋廆戈厲綝秣谿苧濮鯰蝸俎邳罕燧戰楮嘯甜琛蟇鉋癇樅渣鑼蛉哮佼髴頽嶌儚劭驪髣磋皖鈿彝鴦蝿栢娑鑓譴敖軾籌圖筰篋暐猩喇倚甌誣醗厦砒撼躓褥發泓毓鰍軛鵄頤愈顥匙敞鞅苓臚熈郞僊薯麩顗雹馥恂辯輳痔訌翫彊滇鄱匣剽澗蛤銜',help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        #opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
        with open("./japanese_word/japanese_word.txt", "r") as ja:
            opt.character = ja.read()

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    test(opt)