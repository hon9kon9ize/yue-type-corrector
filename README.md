# Cantonese Typo Corrector 廣東話錯字糾正器

簡單嘅腳本，用嚟糾正廣東話嘅錯字。用咗 BERT fill-mask 模型去預測正確嘅字。

例子：「你呢D嘢太壞㗎！」句中嘅「D」係「啲」嘅常見同音字寫法。腳本會將佢改正做「你呢啲嘢太壞㗎！」。

腳本入面定義咗一個常見嘅廣東話錯字表，有兩種糾正方法：

1.) 用 BERT fill-mask 模型去預測正確字，用於需要上下文嘅錯字。
2.) 用簡單嘅正則表達式，用於可以用簡單規則糾正嘅錯字。

===

A simple script to correct Cantonese typos. It is used BERT fill-mask model to predict the correct word.

Here is the example:「你呢D嘢太壞㗎！」the word "D" in the sentence is a common homophonic writing of "啲". The script will correct it to「你呢啲嘢太壞㗎！」。

We have defined a list of common Cantonese typos in the script, there two type of corrections:

1.) BERT fill-mask model as a corrector, for those typos that relied on context to correct.
2.) Simple regular expression, for those typos that can be corrected by simple rules.


```python
register_mask_typo("番", "返")
register_mask_typo("翻", "返")
register_mask_typo("黎", "嚟")
register_mask_typo("左", "咗")
...
register_regular_typo("([^a-z])o敢", "\1噉")
register_regular_typo("([^a-z])o左", "\1咗")
register_regular_typo("噖晚", "琴晚")
```

## How's BERT fill-mask model works?

The script uses the fill-mask model to predict the correct word. The model is trained to predict the masked token in a sentence. For example, in the sentence "你呢D嘢太壞㗎！", the D in the sentence would be masked as "你呢[MASK]嘢太壞㗎！". The model will predict the probability of the candidate words for the masked token. The script will choose the word with the highest probability as the correct word, in this case, "啲".

呢個腳本用 BERT fill-mask 模型去預測正確字。呢個模型係訓練去預測句子中嘅被遮掩嘅字。例如，句子「你呢D嘢太壞㗎！」中嘅 D 會被遮掩做「你呢[MASK]嘢太壞㗎！」。模型會預測候選字嘅機率，腳本會選擇機率最高嘅字作為正確字，喺呢個例子中係「啲」。

Here is how to define the candidate words for the fill-mask model:

```python
register_regular_typo("Ｄ", "啲")
```

## Usage 使用方法

```bash
git clone https://github.com/hon9kon9ize/yue-type-corrector
cd yue-type-corrector
pip install -r requirements.txt
```

```bash
python corrector.py \
  --file_path your_csv_file.csv \
  --output_path output.csv \
  [--column_name text]
```