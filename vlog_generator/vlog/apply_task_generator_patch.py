"""在專案根目錄執行；備份後切換任務層生成器，不降低驗證標準。"""
from pathlib import Path
from datetime import datetime
import ast
import shutil

path = Path('services/story_tasks_service.py')
helper = Path(__file__).resolve().with_name('task_json_generator.py')
helper_target = Path('services/task_json_generator.py')
if not helper.is_file():
    raise SystemExit('請將兩份附件放在同一資料夾，再執行本程式。')
source = path.read_text(encoding='utf-8-sig')
helper_source = helper.read_text(encoding='utf-8-sig')
compile(helper_source, str(helper_target), 'exec')
tree = ast.parse(source)
lines = source.splitlines(keepends=True)
starts = [0]
for line in lines:
    starts.append(starts[-1] + len(line))
def offset(line, column):
    return starts[line - 1] + len(lines[line - 1].encode('utf-8')[:column].decode('utf-8'))
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
many = next(node for node in functions if node.name == '_many')
if any(arg.arg == 'generator' for arg in many.args.args):
    raise SystemExit('已偵測到 generator 參數，未再次修改。請確認先前的修改結果。')
header = 'def _many(jobs, max_tokens, builder, layer):'
if source.count(header) != 1:
    raise SystemExit('_many 宣告與預期不同，未修改任何檔案。')
calls = [node for node in ast.walk(many) if isinstance(node, ast.Call)
         and isinstance(node.func, ast.Name) and node.func.id == '_batch_generate']
if len(calls) != 1:
    raise SystemExit('無法唯一定位批次呼叫，未修改任何檔案。')
task_calls = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == 'tasks' for target in node.targets)
              and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Name) and node.value.func.id == '_many']
if len(task_calls) != 1:
    raise SystemExit('無法唯一定位 tasks = _many(...)，未修改任何檔案。')
call = task_calls[0]
if any(keyword.arg == 'generator' for keyword in call.keywords):
    raise SystemExit('任務層已指定生成器，未修改任何檔案。')
header_start = source.index(header)
func = calls[0].func
close = offset(call.end_lineno, call.end_col_offset) - 1
prefix = '' if source[:close].rstrip().endswith(',') else ','
edits = [
    (header_start, header_start + len(header), 'def _many(jobs, max_tokens, builder, layer, generator=None):'),
    (offset(func.lineno, func.col_offset), offset(func.end_lineno, func.end_col_offset), '(generator or _batch_generate)'),
    (close, close, prefix + ' generator=generate_task_batch'),
]
if 'from services.task_json_generator import generate_task_batch' not in source:
    pos = starts[functions[0].lineno - 1]
    edits.append((pos, pos, 'from services.task_json_generator import generate_task_batch' + chr(10) * 3))
fixed = source
for start, end, value in sorted(edits, reverse=True):
    fixed = fixed[:start] + value + fixed[end:]
compile(fixed, str(path), 'exec')
if helper_target.exists() and helper_target.read_text(encoding='utf-8-sig') != helper_source:
    raise SystemExit('services/task_json_generator.py 已存在且內容不同，未覆寫。')
stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
backup = path.with_name(path.name + '.' + stamp + '.bak')
shutil.copy2(path, backup)
helper_target.write_text(helper_source, encoding='utf-8')
path.write_text(fixed, encoding='utf-8')
print(f'已備份：{backup.name}')
print('已切換任務層至專用生成器；故事層、章節層與 _build_task 驗證不變。')