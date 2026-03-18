#!/usr/bin/env python3
"""
AI人格解构者 - 完整测试脚本
"""

import os
import sys
import time


def resolve_model_path():
    """解析可用的Qwen模型路径。"""
    candidates = [
        os.getenv("QWEN_MODEL_PATH"),
        os.getenv("MODEL_PATH"),
        "./Qwen",
        "./qwen/Qwen3-8B",
        "/workspace/qwen/Qwen3-8B",
        "/workspace/Qwen",
    ]

    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return os.path.abspath(candidate)

    return None

def test_document_reader():
    """测试文档读取器"""
    print("=" * 60)
    print("测试1：文档读取器")
    print("=" * 60)
    
    # 导入文档读取器
    sys.path.append('.')
    from document_reader import test_reader
    
    content = test_reader()
    return content is not None

def test_character_analyzer():
    """测试角色分析器"""
    print("\n" + "=" * 60)
    print("测试2：角色人格分析器")
    print("=" * 60)
    
    model_path = resolve_model_path()
    if not model_path:
        print("警告：未找到可用Qwen模型路径")
        print("请设置 QWEN_MODEL_PATH，或将模型放在 ./Qwen / /workspace/qwen/Qwen3-8B")
        return False

    os.environ["QWEN_MODEL_PATH"] = model_path
    print(f"检测到模型路径: {model_path}")
    
    try:
        # 导入角色分析器
        from character_analyzer import test_analyzer
        return bool(test_analyzer())
    except Exception as e:
        print(f"测试失败: {e}")
        return False

def test_dialogue_system():
    """测试对话系统"""
    print("\n" + "=" * 60)
    print("测试3：对话系统")
    print("=" * 60)
    
    # 检查角色档案
    profile_file = "character_profile.json"
    if not os.path.exists(profile_file):
        print(f"警告：角色档案 {profile_file} 不存在")
        print("请先运行测试2创建角色档案")
        return False
    
    try:
        # 导入对话系统
        from dialogue_system import test_dialogue
        print("即将进入真实对话模式，输入 '退出' 或 'quit' 结束。")
        try:
            return bool(test_dialogue(interactive=True))
        except TypeError as te:
            # 兼容旧版 test_dialogue() 无 interactive 参数的情况
            if "unexpected keyword argument 'interactive'" in str(te):
                print("检测到旧版对话测试函数，自动切换为兼容调用。")
                return bool(test_dialogue())
            raise
    except Exception as e:
        print(f"测试失败: {e}")
        return False

def quick_demo():
    """快速演示"""
    print("\n" + "=" * 60)
    print("快速演示：完整流程")
    print("=" * 60)
    
    # 创建示例文本
    sample_text = """ 林风站在窗前，雨滴敲打着玻璃。他推了推金丝眼镜，这是他在思考案件时的习惯动作。
"事情不像表面那么简单，"他低声自语，手指无意识地敲打着窗台。
助手小李推门进来："林侦探，有新线索了。"
林风转身，眼镜后的目光锐利如鹰："说。"
"被害人昨晚见过一个穿灰色风衣的男人。"小李递过一张模糊的照片。
林风接过照片，仔细端详：“灰色风衣...又是他。” """
    
    # 保存示例文本
    demo_file = "demo_text.txt"
    with open(demo_file, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    print(f"已创建演示文件: {demo_file}")
    print(f"示例文本长度: {len(sample_text)} 字符")
    
    # 模拟分析过程
    print("\n模拟角色分析...")
    print("角色姓名: 林风")
    print("身份: 侦探")
    print("核心性格: 冷静, 敏锐, 专注")
    print("语言风格: 简洁, 直接")
    print("标志性动作: 推眼镜, 敲打手指")
    
    # 模拟对话
    print("\n模拟对话演示:")
    print("-" * 40)
    print("你: 林侦探，这个案子你怎么看？")
    print("林风: （推了推眼镜，目光停留在照片上）凶手很聪明，但留下了破绽。")
    print("\n你: 什么破绽？")
    print("林风: （手指轻轻敲打桌面）灰色风衣上的污渍，和现场发现的完全一致。")
    print("-" * 40)
    
    # 清理
    if os.path.exists(demo_file):
        os.remove(demo_file)
    
    return True

def check_environment():
    """检查环境"""
    print("=" * 60)
    print("环境检查")
    print("=" * 60)
    
    checks = {
        "Python版本": sys.version.split()[0],
        "当前目录": os.getcwd(),
        "虚拟环境": "myenv" if "myenv" in sys.prefix else "未检测到myenv虚拟环境",
    }
    
    for key, value in checks.items():
        print(f"{key}: {value}")
    
    # 检查关键文件
    print("\n文件检查:")
    model_path = resolve_model_path()
    files_to_check = [
        ("document_reader.py", "文档读取器"),
        ("character_analyzer.py", "角色分析器"),
        ("dialogue_system.py", "对话系统"),
    ]
    
    all_exists = True
    for file, description in files_to_check:
        exists = os.path.exists(file)
        status = "✓" if exists else "✗"
        print(f"  {status} {description}: {file}")
        if not exists:
            all_exists = False

    model_status = "✓" if model_path else "✗"
    model_desc = model_path if model_path else "未找到（可设置 QWEN_MODEL_PATH）"
    print(f"  {model_status} Qwen模型: {model_desc}")
    
    return all_exists

def main():
    """主测试函数"""
    print("AI人格解构者 - 系统测试")
    print("=" * 60)
    
    # 检查环境
    if not check_environment():
        print("\n警告：部分文件缺失，某些测试可能失败")
    
    # 询问用户要运行哪些测试
    print("\n请选择测试模式:")
    print("1. 完整测试（所有功能）")
    print("2. 快速演示（不依赖模型）")
    print("3. 仅检查环境")
    
    choice = input("\n请输入选择 (1-3): ").strip()
    
    if choice == "1":
        # 完整测试
        results = []
        
        print("\n开始完整测试...")
        time.sleep(1)
        
        # 测试1: 文档读取
        results.append(("文档读取器", test_document_reader()))
        time.sleep(1)
        
        # 测试2: 角色分析
        results.append(("角色分析器", test_character_analyzer()))
        time.sleep(1)
        
        # 测试3: 对话系统
        print("\n注意：对话系统测试需要加载模型，可能需要较长时间...")
        results.append(("对话系统", test_dialogue_system()))
        
        # 显示结果
        print("\n" + "=" * 60)
        print("测试结果")
        print("=" * 60)
        
        all_passed = True
        for test_name, passed in results:
            status = "通过 ✓" if passed else "失败 ✗"
            print(f"{test_name}: {status}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n所有测试通过！系统可以正常运行。")
        else:
            print("\n部分测试失败，请检查上述错误信息。")
            
    elif choice == "2":
        # 快速演示
        quick_demo()
        print("\n演示完成！")
        
    elif choice == "3":
        # 仅检查环境
        print("\n环境检查完成。")
    else:
        print("无效选择，程序退出。")


if __name__ == "__main__":
    main()