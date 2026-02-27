# -*- coding: utf-8 -*-
import subprocess
import sys
import os
import json
from datetime import datetime

# 忽略列表文件
IGNORED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.sync_ignored.json')

# 开启Windows ANSI颜色支持
os.system('')

# 配置颜色
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_color(text, color=Colors.ENDC):
    print(f"{color}{text}{Colors.ENDC}")

def format_commit_date(date_str):
    """格式化提交日期：当年显示月-日，往年显示年-月-日"""
    try:
        # 解析 ISO 格式日期 (2024-12-05 10:30:00 +0800)
        date_part = date_str.split()[0]  # 取 2024-12-05
        commit_date = datetime.strptime(date_part, '%Y-%m-%d')
        current_year = datetime.now().year
        
        if commit_date.year == current_year:
            return commit_date.strftime('%m-%d')  # 当年: 12-05
        else:
            return commit_date.strftime('%Y-%m-%d')  # 往年: 2024-12-05
    except:
        return date_str  # 解析失败返回原始值

def load_ignored_commits():
    """加载已忽略的提交列表"""
    if os.path.exists(IGNORED_FILE):
        try:
            with open(IGNORED_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return []

def save_ignored_commits(ignored_list):
    """保存已忽略的提交列表"""
    with open(IGNORED_FILE, 'w', encoding='utf-8') as f:
        json.dump(ignored_list, f, ensure_ascii=False, indent=2)

def add_to_ignored(commit_hash, commit_msg):
    """添加提交到忽略列表"""
    ignored = load_ignored_commits()
    if commit_hash not in [c['hash'] for c in ignored]:
        ignored.append({'hash': commit_hash, 'msg': commit_msg})
        save_ignored_commits(ignored)
        return True
    return False

def run_cmd(cmd, check=True, capture=True):
    try:
        if capture:
            result = subprocess.run(cmd, shell=True, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
            return result.stdout.strip()
        else:
            result = subprocess.run(cmd, shell=True, check=check)
            return None
    except subprocess.CalledProcessError as e:
        if check:
            print_color(f"\n[错误] 执行命令失败: {cmd}", Colors.RED)
            if capture:
                print_color(e.stderr, Colors.RED)
            sys.exit(1)
        return None

def get_local_commit_messages():
    """获取本地分支的所有提交信息（用于智能匹配）"""
    output = run_cmd('git log --pretty=format:"%s" -100', check=False)
    if output:
        return [msg.strip() for msg in output.splitlines()]
    return []

def is_commit_applied(commit_msg, local_messages):
    """检查提交是否已通过 cherry-pick 应用（通过提交信息匹配）"""
    # 精确匹配
    if commit_msg in local_messages:
        return True
    # 模糊匹配：去掉前缀后匹配（如 "feat: xxx" 匹配 "xxx"）
    commit_core = commit_msg.split(':', 1)[-1].strip() if ':' in commit_msg else commit_msg
    for local_msg in local_messages:
        local_core = local_msg.split(':', 1)[-1].strip() if ':' in local_msg else local_msg
        if commit_core == local_core:
            return True
    return False

def get_commits(upstream="origin", branch="main"):
    # 获取当前分支
    current = run_cmd("git branch --show-current")
    
    # 显示检查信息
    print_color(f"\n📍 当前分支: {current}", Colors.CYAN)
    print_color(f"🔍 检查目标: {upstream}/{branch}", Colors.CYAN)
    
    # Fetch
    print_color(f"\n正在获取远程更新...", Colors.CYAN)
    run_cmd(f"git fetch {upstream}", check=False, capture=False)
    
    # 获取当前分支最新提交
    current_hash = run_cmd("git rev-parse --short HEAD", check=False)
    current_msg = run_cmd('git log -1 --pretty=format:"%s"', check=False)
    print_color(f"✅ 当前版本: {current_hash} - {current_msg}", Colors.GREEN)
    
    # 获取本地提交信息（用于智能匹配）
    local_messages = get_local_commit_messages()
    
    # 获取差异提交
    # 格式: hash|author|date|message（使用 ISO 格式日期）
    log_cmd = f'git log {current}..{upstream}/{branch} --pretty=format:"%h|%an|%ci|%s"'
    output = run_cmd(log_cmd, check=False)
    
    commits = []
    auto_ignored = []  # 自动识别为已应用的提交
    ignored = load_ignored_commits()
    ignored_hashes = [c['hash'] for c in ignored]
    ignored_msgs = [c['msg'] for c in ignored]
    
    if output:
        for line in output.splitlines():
            parts = line.split('|', 3)
            if len(parts) == 4:
                commit_hash = parts[0]
                commit_msg = parts[3]
                
                # 1. 检查是否在手动忽略列表中（hash 或 msg 匹配）
                if commit_hash in ignored_hashes or commit_msg in ignored_msgs:
                    continue
                
                # 2. 智能检测：检查提交信息是否已存在于本地（cherry-pick 后 hash 会变但 msg 不变）
                if is_commit_applied(commit_msg, local_messages):
                    auto_ignored.append({'hash': commit_hash, 'msg': commit_msg})
                    continue
                
                commits.append({
                    'hash': commit_hash,
                    'author': parts[1],
                    'date': format_commit_date(parts[2]),  # 格式化日期
                    'msg': commit_msg
                })
    
    # 显示自动识别的已应用提交
    if auto_ignored:
        print_color(f"\n🔍 智能识别：{len(auto_ignored)} 个提交已通过 cherry-pick 应用:", Colors.BLUE)
        for c in auto_ignored:
            print(f"  {Colors.BLUE}✓{Colors.ENDC} {c['hash']} - {c['msg']}")
    
    return commits, current, ignored

def resolve_conflict():
    print_color("\n⚠️ 检测到合并冲突！", Colors.RED)
    print_color("请在另一个终端手动解决冲突：", Colors.YELLOW)
    print("1. 编辑冲突文件解决冲突")
    print("2. git add .")
    print("3. git cherry-pick --continue")
    print_color("\n或者选择放弃此提交：", Colors.YELLOW)
    print("git cherry-pick --abort")
    
    while True:
        choice = input("\n冲突解决了吗？(y=已解决继续 / a=放弃此提交): ").lower().strip()
        if choice == 'y':
            try:
                # 尝试继续，如果用户已经commit了可能会报错，如果是add了会提交
                # 先检查是否正在cherry-pick
                if os.path.exists(".git/CHERRY_PICK_HEAD"):
                    subprocess.run("git cherry-pick --continue", shell=True, check=True)
                return True
            except subprocess.CalledProcessError:
                print_color("无法继续，请确认冲突已解决并已暂存(git add)", Colors.RED)
        elif choice == 'a':
            subprocess.run("git cherry-pick --abort", shell=True)
            return False

def main():
    print_color("\n=== 选择性同步主项目工具 (Python版) ===", Colors.HEADER)
    
    upstream = "upstream"
    private_remote = "origin"
    
    # 检查Git
    run_cmd("git --version")
    
    # 获取待同步提交
    commits, current_branch, ignored = get_commits(upstream, "main")
    
    # 显示已忽略的提交
    if ignored:
        print_color(f"\n📝 已忽略 {len(ignored)} 个提交:", Colors.BLUE)
        for c in ignored:
            print(f"  {Colors.BLUE}✓{Colors.ENDC} {c['hash']} - {c['msg']}")
    
    if not commits:
        print_color("\n✅ 当前分支已是最新，无需同步。", Colors.GREEN)
        print_color("✅ 本地仓库已包含主项目的所有提交。", Colors.GREEN)
        input("按任意键退出...")
        return

    print_color(f"\n⚠️ 发现 {len(commits)} 个未应用的提交 (显示顺序: 新 -> 旧):", Colors.YELLOW)
    print_color("这些提交存在于主项目但尚未应用到当前分支", Colors.YELLOW)
    print("-" * 60)
    
    # 显示提交列表
    for i, c in enumerate(commits):
        index = f"[{i+1}]".ljust(5)
        print(f"{Colors.YELLOW}{index}{Colors.ENDC} {Colors.BLUE}({c['date']}){Colors.ENDC} {Colors.GREEN}{c['hash']}{Colors.ENDC} - {c['msg']}")
    print("-" * 60)

    print("\n功能选项:")
    print("  输入数字 (例如 1) 单个应用")
    print("  输入范围 (例如 1-3) 批量应用")
    print("  输入列表 (例如 1,3,5) 组合应用")
    print("  i+数字/范围/列表 (例如 i1, i1-3, i1,3,5) 标记为已应用（忽略）")
    print("  q 退出")
    
    selection = input("\n请选择要应用的提交: ").strip()
    if selection.lower() == 'q':
        return
    
    # 处理忽略命令
    if selection.lower().startswith('i'):
        ignore_selection = selection[1:].strip()
        ignore_indices = []
        try:
            parts = ignore_selection.replace('，', ',').split(',')
            for part in parts:
                part = part.strip()
                if not part: continue
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    if start > end: start, end = end, start
                    ignore_indices.extend(range(start-1, end))
                else:
                    ignore_indices.append(int(part)-1)
            
            # 过滤无效索引并去重
            ignore_indices = sorted(list(set([i for i in ignore_indices if 0 <= i < len(commits)])))
            
            if ignore_indices:
                ignored_count = 0
                for idx in ignore_indices:
                    c = commits[idx]
                    if add_to_ignored(c['hash'], c['msg']):
                        print_color(f"✅ {c['hash']} - {c['msg']}", Colors.GREEN)
                        ignored_count += 1
                print_color(f"\n已标记 {ignored_count} 个提交为已应用，下次运行时不再显示", Colors.CYAN)
            else:
                print_color("\n❌ 未选择有效提交", Colors.RED)
        except ValueError:
            print_color("\n❌ 输入格式错误，例如: i1 或 i1-3 或 i1,3,5", Colors.RED)
        input("\n按任意键继续...")
        return main()  # 重新运行

    # 解析选择
    selected_indices = []
    try:
        parts = selection.replace('，', ',').split(',') # 支持中文逗号
        for part in parts:
            part = part.strip()
            if not part: continue
            if '-' in part:
                start, end = map(int, part.split('-'))
                # 确保范围正确
                if start > end: start, end = end, start
                selected_indices.extend(range(start-1, end))
            else:
                selected_indices.append(int(part)-1)
    except ValueError:
        print_color("❌ 输入格式错误，请输入数字", Colors.RED)
        return

    # 过滤无效索引并去重
    selected_indices = sorted(list(set([i for i in selected_indices if 0 <= i < len(commits)])))
    
    if not selected_indices:
        print_color("❌ 未选择有效提交", Colors.YELLOW)
        return

    # 关键：按时间正序应用（从旧到新），即列表索引从大到小
    # git log显示的是最新的在前面（索引0），最旧的在后面（索引N）
    # 为了避免依赖问题，应该先应用旧的
    selected_indices.sort(reverse=True)

    print_color(f"\n即将应用 {len(selected_indices)} 个提交...", Colors.CYAN)
    print_color("应用顺序: 从旧到新（避免依赖问题）", Colors.BLUE)
    
    success_list = []
    failed_list = []
    
    for idx in selected_indices:
        c = commits[idx]
        print_color(f"\n正在应用 [{idx+1}]: {c['hash']} - {c['msg']}", Colors.BLUE)
        
        try:
            subprocess.run(f"git cherry-pick {c['hash']}", shell=True, check=True)
            print_color("✅ 成功!", Colors.GREEN)
            success_list.append(c)
        except subprocess.CalledProcessError:
            if resolve_conflict():
                print_color("✅ 成功解决并应用!", Colors.GREEN)
                success_list.append(c)
            else:
                print_color(f"⚠️ 跳过 {c['hash']}", Colors.YELLOW)
                failed_list.append(c)
    
    # 显示详细总结
    print_color("\n" + "="*70, Colors.HEADER)
    print_color("📊 应用结果总结", Colors.HEADER)
    print_color("="*70, Colors.HEADER)
    
    print_color(f"\n✅ 成功应用: {len(success_list)}/{len(selected_indices)} 个提交", Colors.GREEN)
    if success_list:
        for c in success_list:
            print(f"  ✓ {Colors.GREEN}{c['hash']}{Colors.ENDC} - {c['msg']}")
    
    if failed_list:
        print_color(f"\n❌ 失败/跳过: {len(failed_list)}/{len(selected_indices)} 个提交", Colors.RED)
        for c in failed_list:
            print(f"  ✗ {Colors.RED}{c['hash']}{Colors.ENDC} - {c['msg']}")
        print_color("\n💡 提示: 失败的提交可以稍后单独重试", Colors.YELLOW)
    else:
        print_color("\n🎉 所有选择的提交都已成功应用！", Colors.GREEN)
    
    print_color("\n" + "="*70, Colors.HEADER)
    
    # 推送提示
    current = run_cmd("git branch --show-current")
    print_color(f"\n当前分支: {current}", Colors.CYAN)
    push = input(f"是否推送到私人仓库 ({private_remote})? (y/n): ").lower().strip()
    if push == 'y':
        print_color("正在推送...", Colors.CYAN)
        subprocess.run(f"git push {private_remote} {current}", shell=True)
        print_color("✅ 推送完成", Colors.GREEN)
    
    input("\n按任意键退出...")

if __name__ == "__main__":
    main()
