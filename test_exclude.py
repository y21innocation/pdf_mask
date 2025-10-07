#!/usr/bin/env python3
import re

# テスト用の除外パターン
EXCLUDE_KEYWORDS_PATTERN = re.compile(
    r"(資\s*本\s*金|売\s*上\s*高|売\s*上|資\s*産|負\s*債|純\s*資\s*産|時\s*価\s*総\s*額|企\s*業\s*価\s*値|株\s*式\s*数|株\s*価|取\s*引\s*額|契\s*約\s*金\s*額|投\s*資\s*額|融\s*資\s*額|借\s*入\s*金|預\s*金|残\s*高|口\s*座|従\s*業\s*員\s*数|設\s*立|創\s*業|上\s*場)",
    re.IGNORECASE
)

# テストケース
test_lines = [
    "資本金:300万円 売上高:非公開 従業員数:5名",
    "資本金:1,000万円 売上高:非公開 従業員数:44名", 
    "資本金:1億円 売上高:非公開 従業員数:10名",
    "資本金:100万円 売上高:非公開 従業員数:5名",
    "資本金:5億1,979万円 売上高:8億872万円 従業員数:50名 東証グロース",
    "直近の年収 500万〜600万円",
    "希望年収 400万円以上",
    "女性 / 30歳 / 東京都 / 500万〜600万円"
]

print("除外パターンテスト:")
for line in test_lines:
    match = EXCLUDE_KEYWORDS_PATTERN.search(line)
    status = "SKIP" if match else "MASK"
    print(f"{status}: {line}")