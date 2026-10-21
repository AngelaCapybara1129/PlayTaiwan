"""只隔離 location_codename 的提示詞；保留所有驗證與其他欄位生成。"""
from pathlib import Path
from datetime import datetime
import ast
import shutil

path = Path('services/node_text_generator.py')
source = path.read_text(encoding='utf-8-sig')
marker = '[代號獨立命名]'
if marker in source:
    raise SystemExit('代號獨立命名已安裝，未重複修改。')
if '章節純文字 v2' not in source:
    raise SystemExit('目前不是預期的章節 v2，未修改檔案。')
tree = ast.parse(source)
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_node_text_batch')
matches = [n for n in ast.walk(function) if isinstance(n, ast.Assign)
           and any(isinstance(t, ast.Name) and t.id == 'raw' for t in n.targets)
           and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
           and n.value.func.id == 'generate_chat_text']
if len(matches) != 1:
    raise SystemExit('無法唯一定位生成呼叫，未修改。')
node = matches[0]
keyword = next(k for k in node.value.keywords if k.arg == 'system_prompt')
if not isinstance(keyword.value, ast.Name) or keyword.value.id != 'SYSTEM':
    raise SystemExit('系統提示詞形狀與預期不同，未修改。')
lines = source.splitlines(keepends=True)
starts = [0]
for line in lines:
    starts.append(starts[-1] + len(line))
def offset(line, col):
    return starts[line - 1] + len(lines[line - 1].encode('utf-8')[:col].decode('utf-8'))
indent = lines[node.lineno - 1][:node.col_offset]
block = [
    'active_system = SYSTEM',
    "if field == 'location_codename':",
    "    active_system = '你是中文短名稱產生器，不是故事作家。只輸出二至五個漢字，不加標點、欄位名稱、說明或故事。不得使用真實地名或設施名稱。'",
    "    prompt = ('替自然觀察遊戲取一個抽象代號。只輸出二至五個漢字。'",
    "              + '\\n主題參考：' + data.get('sn_title', '')[:12]",
    "              + '\\n請只回答名稱：')",
    "    if reason:",
    "        prompt += '\\n上次不符合格式，這次不要寫故事。'",
    "    print('[代號獨立命名] 已移除故事、角色與景點長文，只保留短主題')",
]
block = [line.replace(chr(92)*2,chr(92)) for line in block]
insertion = ''.join(indent + line + '\n' for line in block)
insertion = insertion.replace(chr(92)+'n', chr(10)) if False else insertion
# 插入的程式行分隔使用真正換行；程式字串內的跳脫保持不變。
insertion = ''.join(indent + line + chr(10) for line in block)
a = offset(keyword.value.lineno, keyword.value.col_offset)
b = offset(keyword.value.end_lineno, keyword.value.end_col_offset)
edits = [(a,b,'active_system'),(starts[node.lineno - 1],starts[node.lineno - 1],insertion)]
fixed = source
for start,end,value in sorted(edits,reverse=True):
    fixed = fixed[:start] + value + fixed[end:]
compile(fixed,str(path),'exec')
stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
backup = path.with_name(path.name + '.' + stamp + '.bak')
shutil.copy2(path,backup)
path.write_text(fixed,encoding='utf-8')
print(f'已備份：{backup.name}')
print(f'已修改：{path}')
print('只改代號生成的提示詞；二至五個漢字驗證與其餘欄位均保留。')