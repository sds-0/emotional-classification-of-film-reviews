import os
import numpy as np
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import warnings
import csv


def set_chinese_font():
    try:
        font_path = None

        for font in fm.findSystemFonts():
            if 'simhei' in font.lower():
                font_path = font
                break
            elif 'simsun' in font.lower():
                font_path = font
                break
            elif 'microsoft yahei' in font.lower():
                font_path = font
                break
            elif 'stsong' in font.lower():
                font_path = font
                break

        if font_path:
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            print(f"已设置中文字体: {font_prop.get_name()}")
        else:
            plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'STSong']
            plt.rcParams['axes.unicode_minus'] = False
            print("使用预设中文字体")
    except Exception as e:
        print(f"设置中文字体失败: {e}")
        print("图像中的中文可能无法正确显示")


set_chinese_font()

warnings.filterwarnings('ignore')


class Config:
    def __init__(self):
        self.data_dir = r"C:\Users\19658\Desktop\数据挖掘项目实践\ChnSentiCorp[dataset 1]\ChnSentiCorp"

        self.max_features = 5000  # TF-IDF特征数量
        self.max_depth = 30  # 决策树最大深度
        self.min_samples_split = 10  # 决策树节点最小分割样本数
        self.min_samples_leaf = 5  # 决策树叶节点最小样本数

        self.log_dir = './log'
        self.model_save_path = './decision_tree_model.pkl'


config = Config()

if not os.path.exists(config.log_dir):
    os.makedirs(config.log_dir)


def load_tsv_file(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f, delimiter='\t')

            headers = next(reader)
            num_columns = len(headers)

            if num_columns == 2:
                label_idx = 0
                text_idx = 1
            elif num_columns >= 3:
                label_idx = 1
                text_idx = 2
            else:
                raise ValueError(f"文件列数不足: {file_path} (只有{num_columns}列)")

            for row in reader:
                if len(row) < num_columns:
                    continue

                try:
                    label = int(row[label_idx])
                    text = row[text_idx].strip()
                    if text:
                        data.append((label, text))
                except ValueError:
                    continue
    except Exception as e:
        print(f"读取文件时出错: {file_path}")
        print(f"错误信息: {e}")

    return data


print("加载数据集...")
train_data = load_tsv_file(os.path.join(config.data_dir, 'train.tsv'))
dev_data = load_tsv_file(os.path.join(config.data_dir, 'dev.tsv'))
test_data = load_tsv_file(os.path.join(config.data_dir, 'test.tsv'))

all_data = train_data + dev_data

print(f"训练+验证集大小: {len(all_data)}")
print(f"测试集大小: {len(test_data)}")


def preprocess_text(text):
    text = re.sub(r'[^\u4e00-\u9fa5]', ' ', text)
    words = jieba.cut(text)
    stop_words = set(
        ['的', '了', '和', '在', '是', '我', '有', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说',
         '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '这样', '那样', '我们', '你们', '他们'])
    words = [word for word in words if len(word) > 1 and word not in stop_words]
    return ' '.join(words)


print("预处理文本数据...")
all_texts = [preprocess_text(text) for label, text in all_data]
all_labels = [label for label, text in all_data]

test_texts = [preprocess_text(text) for label, text in test_data]
test_labels = [label for label, text in test_data]

print("创建TF-IDF特征...")
vectorizer = TfidfVectorizer(max_features=config.max_features)
X = vectorizer.fit_transform(all_texts)
y = np.array(all_labels)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2023
)

print(f"训练集特征形状: {X_train.shape}")
print(f"验证集特征形状: {X_val.shape}")

print("创建决策树模型...")
dt_model = DecisionTreeClassifier(
    max_depth=config.max_depth,
    min_samples_split=config.min_samples_split,
    min_samples_leaf=config.min_samples_leaf,
    random_state=2023
)

print("开始训练模型...")
dt_model.fit(X_train, y_train)


def evaluate(model, X, y):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')
    return acc, f1


val_acc, val_f1 = evaluate(dt_model, X_val, y_val)
print(f"验证集准确率: {val_acc:.4f}, 验证集F1分数: {val_f1:.4f}")

print("在测试集上评估模型...")
X_test = vectorizer.transform(test_texts)
y_test = np.array(test_labels)
test_acc, test_f1 = evaluate(dt_model, X_test, y_test)
print("\n" + "=" * 60)
print(f"测试集准确率: {test_acc:.4f}")
print(f"测试集F1分数: {test_f1:.4f}")
print("=" * 60)

print("可视化特征重要性...")
feature_importances = dt_model.feature_importances_
feature_names = vectorizer.get_feature_names_out()

top_n = 20
top_indices = feature_importances.argsort()[-top_n:][::-1]
top_features = [feature_names[i] for i in top_indices]
top_importances = feature_importances[top_indices]

plt.figure(figsize=(12, 8))
plt.barh(top_features, top_importances, color='skyblue')
plt.xlabel('特征重要性')
plt.title('决策树模型 - 前20个重要特征')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(config.log_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
print(f"特征重要性图已保存至: {os.path.join(config.log_dir, 'feature_importance.png')}")
plt.show()


def predict_sentiment(model, vectorizer, text):
    preprocessed_text = preprocess_text(text)
    text_vector = vectorizer.transform([preprocessed_text])
    proba = model.predict_proba(text_vector)[0]
    pred = model.predict(text_vector)[0]
    confidence = proba[pred]
    sentiment = "正向" if pred == 1 else "负向"
    return sentiment, confidence, proba


examples = [
    "酒店环境很好，服务态度也很不错，下次还会来住",
    "房间太小了，而且卫生条件很差，完全不值这个价格",
    "设施比较陈旧，但是位置很方便，总体还可以",
    "早餐种类太少，房间隔音效果差，不会再来了",
    "性价比很高，交通便利，推荐给大家"
]

print("\n情感预测示例:")
for text in examples:
    sentiment, confidence, probs = predict_sentiment(dt_model, vectorizer, text)
    print(f"文本: {text}")
    print(f"情感: {sentiment} (置信度: {confidence:.4f}), 概率分布: [负向: {probs[0]:.4f}, 正向: {probs[1]:.4f}]")
    print("-" * 80)

print("\n实验总结:")
print(f"测试集准确率: {test_acc:.4f}")
print(f"测试集F1分数: {test_f1:.4f}")
print(f"验证集准确率: {val_acc:.4f}")