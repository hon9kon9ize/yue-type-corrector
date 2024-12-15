import pandas as pd
from transformers import pipeline
from tqdm.auto import tqdm
from typing import List
import fire
from torch.utils.data import Dataset

# Use Bert's fill-mask model to fix typos

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = pipeline("fill-mask", model="hon9kon9ize/bert-base-cantonese", device=device)


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


class Typo:
    def __init__(self, typo, correct):
        self.typo = typo
        self.correct = correct

    def fix(self, lines: pd.Series) -> pd.Series:
        raise NotImplementedError


class RegularTypo(Typo):
    def __init__(self, typo, correct):
        self.typo = typo
        self.correct = correct

    def fix(self, lines: pd.Series) -> pd.Series:
        return lines.str.replace(self.typo, self.correct)


class MaskTypo(Typo):
    def __init__(
        self,
        typo,
        correct,
        length_limit=460,
        batch_size=64,
    ):
        self.typo = typo
        self.correct = correct
        self.length_limit = length_limit
        self.delimiters = ["。", "！", "？", "；", "，", "\n", "⋯", "…", "?", "!"]
        self.batch_size = batch_size

    # return the score of the typo and the correct word
    def _get_typo_score(self, outputs) -> List[float]:
        if isinstance(outputs[0], list):
            my_scores = []
            # multiple target words
            for j in range(len(outputs)):
                scores = self._get_typo_score(outputs[j])
                my_scores += scores

            return my_scores

        src_outputs = list(filter(lambda x: x["token_str"] == self.typo, outputs))
        tgt_outputs = list(filter(lambda x: x["token_str"] == self.correct, outputs))

        if len(src_outputs) != 1 or len(tgt_outputs) != 1:
            return [1]

        src_output = src_outputs[0] if src_outputs else None
        tgt_output = tgt_outputs[0] if tgt_outputs else None
        src_score = src_output["score"]
        tgt_score = tgt_output["score"]

        return [tgt_score - src_score]

    # fix the typo in the Panda Series
    def fix(self, series: pd.Series) -> pd.Series:
        lines = series.copy()
        return pd.Series(self._fix(lines.to_list()), index=lines.index)

    def _fix(self, lines: List[str]):
        fixed_lines = lines.copy()
        items = []

        for line_index, line in enumerate(lines):
            occurrence = [
                i for i, c in enumerate(line) if c.lower() == self.typo.lower()
            ]

            # iterate the line from middle of the index to left and right every time
            for idx in occurrence:
                left = new_left = idx
                right = new_right = idx
                while (
                    (right - left) < self.length_limit
                    and left - 1 > 0
                    and right + 1 < len(line)
                ):
                    if line[left - 1] not in self.delimiters and left - 1 > 0:
                        new_left = left - 1
                    if line[right + 1] not in self.delimiters and right < len(line):
                        new_right = right + 1
                    if new_left == left and new_right == right:
                        break
                    left = new_left
                    right = new_right
                sentence = line[left:right]

                if len(sentence) + 1 > self.length_limit:
                    # print(sentence, len(sentence))
                    # raise ValueError("Sentence too long")
                    continue

                if len(sentence) > 0:
                    # replace the index of the typo char to [MASK]
                    sentence = (
                        sentence[: idx - left] + "[MASK]" + sentence[idx - left + 1 :]
                    )
                    # replace the other occurence of the typo char to [UNK], so that the model can predict the correct word without bias
                    sentence = sentence.replace(self.typo, "[UNK]").replace(
                        self.typo, "[UNK]"
                    )
                    sentence = (
                        "；" + sentence
                    )  # add a delimiter to the beginning of the sentence to prevent prediction bias of the first word
                    items.append((line_index, idx, (left, right), sentence))

        if len(items) == 0:
            return fixed_lines

        # get the score of the typo and the correct word
        ds = ListDataset([sentence[3] for sentence in items])
        outputs = []

        for out in tqdm(
            pipe(ds, targets=[self.typo, self.correct], batch_size=self.batch_size),
            total=len(ds),
            desc=f"{self.typo} -> {self.correct}",
            leave=False,
        ):
            outputs.append(out)

        scores = self._get_typo_score(outputs)

        for item, score in zip(items, scores):
            if score > 0:
                item_index, typo_idx, (left, right), _ = item
                corrected_sentence = fixed_lines[item_index]
                corrected_sentence = (
                    corrected_sentence[:typo_idx]
                    + self.correct
                    + corrected_sentence[typo_idx + 1 :]
                )
                fixed_lines[item_index] = corrected_sentence

        return fixed_lines


regular_typo_items = []
mask_typo_items = []


def register_regular_typo(typo, correct):
    regular_typo_items.append(RegularTypo(typo, correct))


def register_mask_typo(typo, correct):
    mask_typo_items.append(MaskTypo(typo, correct))


register_regular_typo("傾計", "傾偈")
register_regular_typo("訓覺", "瞓覺")
register_regular_typo("戇鳩", "戇𨳊")
register_regular_typo("on9", "戇鳩")
register_regular_typo("on居", "戇居")
register_regular_typo("潤9[我佢你]", r"潤鳩\1")
register_regular_typo("鐘意", "鍾意")
register_regular_typo("揾", "搵")
register_regular_typo("哩個", "呢個")
register_regular_typo("丫([喇嗱嘛麻？！~?!，])", "吖\1")
register_regular_typo("([啦吖㗎])麻", "\1嘛")
register_regular_typo("([咪哇哎])丫", "\1吖")
register_regular_typo("Ｄ", "啲")
register_regular_typo(
    r"([多咗呢嗰邊佢我你畀借還嚟出入緊鬆實較大細深淺晏常差依哋返面講係夜乜先過己晒用下耐頭靜間])d([^\\w\\d\\s])",
    r"\1啲\2",
)
register_regular_typo(r"([^\w\d\s])D([咩嘢])", r"\1啲\2")
register_regular_typo("[果嗰][d啲]", "嗰啲")
register_regular_typo(
    r"([你佢我俾畀嗰有某衰好細勁易大高一返番真少])D([^\\w])", r"\1啲\2"
)
register_regular_typo("([^a-z])o敢", "\1噉")
register_regular_typo("([^a-z])o左", "\1咗")
register_regular_typo("([^a-z])o甘", "\1咁")
register_regular_typo("([^a-z])o既", "\1嘅")
register_regular_typo("([^a-z])o地", "\1哋")
register_regular_typo("([^a-z])o啱", "\1啱")
register_regular_typo("([^a-z])o刺", "\1喇")
register_regular_typo("([^a-z])o個", "\1嗰")
register_regular_typo("([^a-z])o拉", "\1啦")
register_regular_typo("([^a-z])o拿", "\1嗱")
register_regular_typo("([^a-z])o野", "\1嘢")
register_regular_typo("([^a-z])o架", "\1㗎")
register_regular_typo("([^a-z])o黎", "\1嚟")
register_regular_typo("([^a-z])o吾", "\1唔")
register_regular_typo("拿拿([淋林聲臨])", "嗱嗱$1")
register_regular_typo("嗱嗱[淋林]", "嗱嗱臨")
register_regular_typo("大嗱嗱", "大拿拿")
register_regular_typo("嚟拿[?？]", "嚟嗱？")
register_regular_typo("裡", "裏")
register_regular_typo("o丫", "吖")
register_regular_typo("㞗", "𨳊")
register_regular_typo("嗮", "晒")
register_regular_typo("尐", "啲")
register_regular_typo("揾", "搵")
register_regular_typo("噖晚", "琴晚")
register_regular_typo("噚晚", "尋晚")
register_regular_typo("噖日", "琴日")
register_regular_typo("岩岩", "啱啱")
register_regular_typo("噚日", "尋日")
register_regular_typo("[撲扑]街", "仆街")
register_regular_typo("[痴癡黐][綫線]", "黐線")
register_regular_typo("[痴癡黐][撚𠹌能][綫線]", "黐撚線")
register_mask_typo("番", "返")
register_mask_typo("翻", "返")
register_mask_typo("黎", "嚟")
register_mask_typo("左", "咗")
register_mask_typo("佐", "咗")
register_mask_typo("遮", "即")
register_mask_typo("哩", "呢")
register_mask_typo("姐", "即")
register_mask_typo("姐", "啫")
register_mask_typo("姐", "啫")
register_mask_typo("吓", "下")
register_mask_typo("岩", "啱")
register_mask_typo("果", "嗰")
register_mask_typo("攪", "搞")
register_mask_typo("既", "嘅")
register_mask_typo("比", "俾")
register_mask_typo("奶", "舐")
register_mask_typo("丫", "啊")
register_mask_typo("丫", "吖")
register_mask_typo("著", "着")
register_mask_typo("訓", "瞓")
register_mask_typo("嫁", "㗎")
register_mask_typo("曬", "晒")
register_mask_typo("甘", "噉")
register_mask_typo("噤", "噉")
register_mask_typo("哩", "匿")
register_mask_typo("到", "度")
register_mask_typo("地", "哋")
register_mask_typo("野", "嘢")
register_mask_typo("駛", "使")
register_mask_typo("洗", "使")
register_mask_typo("俾", "畀")
register_mask_typo("畀", "俾")
register_mask_typo("咁", "噉")
register_mask_typo("架", "㗎")
register_mask_typo("拿", "嗱")

# register_mask_typo('d', '啲', [r'[^\w]d[^\w]']) # Bert model has bias towards Chinese characters
# register_mask_typo('Ｄ', '啲') # Ｄ is OOV


def fix_typo(df: pd.DataFrame, column_name: str):
    for regular_typo in tqdm(regular_typo_items, desc="Regular typo correction"):
        df[column_name] = regular_typo.fix(df[column_name])

    for mask_typo in tqdm(mask_typo_items, desc="Bert typo correction"):
        df[column_name] = mask_typo.fix(df[column_name])

    # post process
    df[column_name] = df[column_name].str.replace(r"噉(cheap)", r"咁\1", regex=True)

    return df


def main(
    file_path: str,
    output_path: str,
    column_name="text",
):
    ext = file_path.split(".")[-1]

    # read
    if "csv" in ext or ext == "tsv" or ext == "txt":
        df = pd.read_csv(file_path)
    elif ext == "json":
        df = pd.read_json(file_path)
    else:
        raise ValueError("file format not supported")

    if column_name not in df.columns:
        raise ValueError("text column not found in json file")

    df = df.dropna(subset=[column_name])

    fix_typo(df, column_name)

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # test_df = pd.DataFrame(
    #     [
    #         {
    #             "text": "岩岩「仲有隻雞……有……」預先一日準備好既雞，Ｄ香料既味已經入晒隻雞到。一番賞"
    #         }
    #     ]
    # )
    # fixed_df = fix_typo(test_df, "text")

    # print(fixed_df["text"].values[0])

    fire.Fire(main)
