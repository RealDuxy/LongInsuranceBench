import re
import string

import jieba
# from fuzzywuzzy import fuzz
import difflib

from typing import List
from collections import Counter
from rouge import Rouge
import sacrebleu

metric = sacrebleu.metrics.TER()

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def count_product_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def retrieval_zh_score(prediction, ground_truth, **kwargs):
    pattern = r'段落(\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)

def retrieval_product_zh_score(prediction, ground_truth, **kwargs):
    score = 0
    if ground_truth in prediction:
        score += 1
    return score


# def code_sim_score(prediction, ground_truth, **kwargs):
#     all_lines = prediction.lstrip('\n').split('\n')
#     prediction = ""
#     for line in all_lines:
#         if ('`' not in line) and ('#' not in line) and ('//' not in line):
#             prediction = line
#             break
#     return (fuzz.ratio(prediction, ground_truth) / 100)

def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        score = (1.0 / len(em_match_list))
    else:
        score = 0.0
    return score
    
def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]

def rouge_zh_score(prediction, ground_truth, **kwargs):
    if isinstance(prediction, float):
        return 0
    prediction = prediction.replace("\"", "'").replace("\n", '').replace(" ", '')
    ground_truth = ground_truth.replace("\"", "'").replace("\n", '').replace(" ", '')
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False))) 
    score = rouge_score(prediction, ground_truth)
    return score

def translation_edit_distance(prediction, ground_truth, **kwargs):
    if isinstance(prediction, float):
        return 0
    prediction = prediction.replace("\"", "'").replace("\n", '').replace(" ", '')
    ground_truth = ground_truth.replace("\"", "'").replace("\n", '').replace(" ", '')
    # prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    # ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    score = metric.corpus_score(prediction, [ground_truth]).score
    return score / 100

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)

if __name__ == '__main__':
    pred = "```json\n{\n  \"产品代码\": \"1730\",\n  \"产品名称\": \"平安福满分（2023）两全保险\",\n  \"发布时间\": \"2023-07-27\",\n  \"产品特色\": \"满期给付生存金，为未来储备一笔资金\\n保险期满时生存可领取生存金，满足家庭生活所需\\n身故保障延续爱\\n保险期内不幸身故，身故保险金守护家人生活\",\n  \"保险责任\": \"满期生存保险金...身故保险金...\",\n  \"其他权益\": \"保单贷款...\",\n  \"责任免除\": \"因下列情形...导致被保险人身故的，我们不承担给付保险金的责任...\",\n  \"其他免责条款\": \"除以上责任免除外，合同中还有一些免除保险人责任的情况...\",\n  \"犹豫期及合同解除（退保）\": \"犹豫期...退保金及合同解除（退保）风险...\",\n  \"投保范围\": \"投保年龄为18周岁至55周岁。\",\n  \"保险期间\": \"自合同生效时起至被保险人60/70/80周岁的保单周年日零时止\",\n  \"交费方式\": \"可选年交等。\"\n}\n\n{\n  \"产品代码\": \"583\",\n  \"产品名称\": \"平安附加住院安心18医疗保险\",\n  \"发布时间\": \"2022-01-01\",\n  \"产品特色\": \"补贴住院生活费用，恶性肿瘤住院或手术津贴额外领\",\n  \"保险责任\": \"一般住院医疗津贴...恶性肿瘤住院津贴...住院手术医疗津贴...\",\n  \"责任免除\": \"因下列情形...造成被保险人住院治疗的，我们不承担给付保险金的责任...\",\n  \"其他免责条款\": \"除以上责任免除外，本合同中还有一些免除保险人责任的情况...\",\n  \"犹豫期及合同解除（退保）\": \"犹豫期...退保金及合同解除（退保）风险...\",\n  \"投保范围\": \"3周岁至64周。\",\n  \"保险期间\": \"保险期间为1年，保证续保期间为5年\",\n  \"交费方式\": \"可选年交等。\",\n  \"费率说明\": \"本产品为一年期产品，保费随被保险人年龄增加等变化可能不同。\"\n}\n```"
    target = "{'产品代码': '1730', '产品名称': '平安福满分（2023）两全保险', '发布时间': '2023-07-27', '产品特色': '\\uf06c满期给付生存金，为未来储备一笔资金\n保险期满时生存可领取生存金，满足家庭生活所需\n\\uf06c身故保障延续爱\n保险期内不幸身故，身故保险金守护家人生活\n', '保险责任': '\\uf06c满期生存保险金\n若未附加提前给付型重大疾病保险，被保险人于保险期满时仍生存，我们按照约定金额\n给付满期生存保险金，合同终止。\n若附加了提前给付型重大疾病保险，被保险人于保险期满时仍生存，我们按照保险期间\n届满时本主险合同基本保险金额确定的年交保险费及提前给付型重大疾病保险合同基本保险\n金额确定的年交保险费之和×您与我们约定的交费年期给付满期生存保险金，合同终止。\n\\uf06c身故保险金\n被保险人身故，我们按身故时合同的基本保险金额给付身故保险金，合同终止。\n若附加了提前给付型重大疾病保险，且被保险人发生了符合给付条件的重大疾病，则本\n主险的基本保险金额按提前给付型重大疾病保险约定的重大疾病保险金金额等额减少，本主\n险的各项保险责任及现金价值按减少后的基本保险金额确定。当本主险基本保险金额减少至\n零时，所有保险责任均终止。\n', '其他权益': '\\uf06c保单贷款\n经我们审核同意后您可办理保单贷款。贷款金额不得超过保险合同现金价值的80%扣除\n各项欠款后余额，每次贷款期限最长不超过6个月。当未还贷款本金及利息加上其他欠款达\n到现金价值时，保险合同效力中止。\n', '责任免除': '因下列情形之一导致被保险人身故的，我们不承担给付保险金的责任：\n（1）投保人对被保险人的故意杀害、故意伤害；\n（2）被保险人故意犯罪或者抗拒依法采取的刑事强制措施；\n（3）被保险人自本主险合同成立或者合同效力恢复之日起2年内自杀，但被保险人自杀\n时为无民事行为能力人的除外；\n（4）被保险人服用、吸食或注射毒品；\n（5）被保险人酒后驾驶机动车；\n（6）战争、军事冲突、暴乱或武装叛乱；\n（7）核爆炸、核辐射或核污染。\n发生上述第（1）项情形导致被保险人身故的，合同终止，我们向投保人之外的其他权利\n人退还本主险合同的现金价值，其他权利人为被保险人的继承人。\n发生上述其他情形导致被保险人身故的，合同终止，我们向您退还本主险合同的现金价\n值。\n', '其他免责条款': '除以上责任免除外，合同中还有一些免除保险人责任的情况，详见平安福满分（2023）\n两全保险以下条款中背景突出显示的内容：“4.2保险事故通知”、“5.1犹豫期”、“7.4年龄\n错误”。\n', '犹豫期及合同解除（退保）': '\\uf06c犹豫期：\n自您签收合同之日起，有20日的犹豫期。\n\\uf06c退保金及合同解除（退保）风险：\n合同成立后，您可以申请解除合同。解除合同时，您需要填写申请书，并提供您的保险\n合同及有效身份证件。自我们收到解除合同的书面申请时起，合同终止。\n您在犹豫期内申请解除合同的，我们将无息退还您所支付的全部保险费，合同解除前发\n生的保险事故我们不承担保险责任。\n您在犹豫期后申请解除合同的，退保金为合同的现金价值，自我们收到解除合同的书面\n申请之日起30日内向您退还。\n现金价值指保险单所具有的价值，通常体现为解除合同时，根据精算原理计算的，由我\n们退还的那部分金额。我们在退还现金价值时，如果您有欠交的保险费或其他未还清款项，\n我们会在扣除上述欠款及应付利息后给付。\n考虑到保单平均承担的本公司经营支出、保险责任对应的成本以及客户提前终止保单对\n本公司的影响，我们从您所交的保险费中扣除了相关费用。因此，您在犹豫期后解除合同会\n遭受一定损失。\n解除合同后，您不再享有原有的保障。\n', '投保范围': '投保年龄为18周岁至55周岁。\n', '保险期间': '自合同生效时起至被保险人60/70/80周岁的保单周年日零时止三种，您在投保时可选择\n其中一种。\n', '交费方式': '可选年交等。\n'}\n\n{'产品代码': '583', '产品名称': '平安附加住院安心18医疗保险', '发布时间': '2022-01-01', '产品特色': '\\uf06c补贴住院生活费用，恶性肿瘤住院或手术津贴额外领\n', '保险责任': '\\uf06c一般住院医疗津贴\n被保险人因意外伤害或疾病经医院诊断必须住院治疗，可从每次住院第4日开始按住院\n天数给付“一般住院津贴”，每个保险期内给付天数最多180天。每日住院津贴一档30元，\n二档50元，三档80元。\n\\uf06c恶性肿瘤住院津贴\n被保险人经医院初次发生合同约定的“恶性肿瘤”，从每次住院的第1日开始按天数给付\n“恶性肿瘤住院日额保险金”，每个保险期内给付天数最多180天。每日恶性肿瘤住院津贴一\n档50元，二档80元，三档100元。\n\\uf06c住院手术医疗津贴\n被保险人因疾病或意外伤害导致住院实行手术治疗，根据所实行手术项目给付手术医疗\n津贴。在保险期间内以5000元为限，手术项目详见条款。各等级的手术津贴如下：\n津贴等级(级)\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n津贴金额(元)5000\n4500\n4000\n3500\n3000\n2500\n2000\n1500\n1000\n500\n说明：\n1.医院范围：中华人民共和国境内（港、澳、台地区除外）合法经营的二级以上（含二\n级）的基本医疗保险定点的医院。重症监护病房指经医疗卫生行政主管机关批准，在医院内\n正式设立的重症监护病房。该病房为危重患者提供24小时持续护理及治疗，配备有重症监护\n-\n专科医生、护士以及相应的监护、复苏抢救设备，详见条款。\n', '责任免除': '因下列情形之一造成被保险人住院治疗的，我们不承担给付保险金的责任：\n（1）被保险人故意自伤、自杀；被保险人服用、吸食或注射毒品\n（2）被保险人酒后驾驶机动车；\n（3）被保险人感染艾滋病病毒或患艾滋病期间因疾病导致的；\n（4）怀孕、分娩、避孕及节育手术；\n（5）精神和行为障碍（依照世界卫生组织《疾病和有关健康问题的国际统计分类》第十\n次修订版（ICD-10）确定）；\n（6）先天性畸形、变形和染色体异常；\n（7）本附加险合同生效时或生效后三十天内所患疾病(续保除外)；\n（8）战争、军事冲突、暴乱或武装叛乱；\n（9）核爆炸、核辐射或核污染。\n', '其他免责条款': '除以上责任免除外，本合同中还有一些免除保险人责任的情况，详见平安附加住院安心\n18医疗保险条款“1.4犹豫期”、“2.2保险责任”、“3.2保险事故通知”、“6.2年龄错误”\n及“7释义”中背景突出显示的内容。\n', '犹豫期及合同解除（退保）': '\\uf06c犹豫期：自您签收合同次日起，有20日的犹豫期。\n\\uf06c退保金及合同解除（退保）风险：合同成立后，您可以申请解除合同，解除合同时，\n您需要填写申请书，并提供您的保险合同及有效身份证件。自我们收到您解除合同的书面申\n请时起，合同即被解除。\n您在犹豫期内申请解除合同的，我们将无息退还您支付的全部保费，合同解除前发生的\n保险事故我们不承担保险责任。\n您在犹豫期后申请解除合同的，退保金为合同的现金价值。自我们收到解除合同的书面\n申请之日起30日内向您退还。\n-\n现金价值的计算公式为“保险费×（1-30%）×（1-经过天数/保险期间的天数）”经过\n天数不足1天的按1天计算。“经过天数”是指合同从生效之日至终止之日实际经过的天数。\n考虑到保单平均承担的本公司经营支出、保险责任对应的成本以及客户提前终止保单对\n本公司的影响，我们从您所交的保险费中扣除了相关费用。因此，您在犹豫期后解除合同会\n遭受一定损失。\n解除合同后，您不再享有原有的保障。\n', '投保范围': '3周岁至64周。\n保险期间和续保\n保险期间为1年。\n自本附加险合同的生效日起，5年为一个保证续保期间。保证续保期间内，每一保险期\n间届满之前，若我们未收到您不再继续投保的书面通知，则视作您申请续保，我们将按照以\n下约定续保本附加险合同：\n在保证续保期间内每一保险期间届满时，我们按续保时年龄对应的费率收取保险费，续\n保后的新合同生效。但若于保证续保期间内每一保险期间届满时存在下列情形之一时，本附\n加险合同不再接受续保：\n（1）续保时被保险人年满65周岁；\n（2）主险交费期满或主险已办理减额交清；\n（3）主险合同效力中止。\n每个保证续保期间届满时，若您要继续享有本产品提供的保障，您需要重新投保。\n若保证续保期间届满时，本产品已停止销售，我们不再接受投保申请，但会向您提供投\n保其他保险产品的合理建议。\n', '交费方式': '可选年交等。\n', '费率说明': '-\n本产品为一年期产品，保费随被保险人年龄增加等各项因素的变化可能逐年不同，具体\n数额详见条款中的费率表。\n'}"
    translation_edit_distance(pred,target)
