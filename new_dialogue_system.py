import os
import torch
import json
import re
from typing import Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from character_analyzer import CharacterAnalyzer
from document_reader import DocumentReader

class DialogueSystem:
    """AI角色对话系统"""
    
    def __init__(self, model_path="/workspace/qwen/Qwen3-8B"):
        """
        初始化对话系统
        
        Args:
            model_path: Qwen模型路径
        """
        print("正在初始化对话系统...")
        
        try:
            model_path = os.path.abspath(model_path)
            if not os.path.isdir(model_path):
                raise FileNotFoundError(f"模型路径不存在: {model_path}")
            print(f"加载模型路径: {model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"使用设备: {self.device}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.character_profile = None
            self.conversation_history = []
            
            print("��话系统初始化完成!")
            
        except Exception as e:
            print(f"初始化对话系统失败: {e}")
            raise
    
    def load_character_profile(self, profile_path: str):
        """加载角色档案"""
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                self.character_profile = json.load(f)
            print(f"已加载角色档案: {self.character_profile.get('name', '未知角色')}")
            return True
        except Exception as e:
            print(f"加载角色档案失败: {e}")
            return False
    
    def create_character_prompt(self) -> str:
        """创建角色扮演提示词"""
        if not self.character_profile:
            return "请先加载角色档案。"
        
        name = self.character_profile.get('name', '角色')
        identity = self.character_profile.get('identity', '')
        personality = ', '.join(self.character_profile.get('core_personality', []))
        speech_style = self.character_profile.get('speech_style', '')
        actions = ', '.join(self.character_profile.get('characteristic_actions', []))
        
        prompt = f"""你现在要扮演{name}{f'，一名{identity}' if identity else ''}。

角色设定：
- 性格特点：{personality}
- 语言风格：{speech_style}
- 标志性动作：{actions}

对话要求：
1. 完全沉浸在角色中，以{name}的身份和性格进行对话
2. 在每次回复中，使用括号（）包含角色的神态和动作描写
3. 保持角色性格的一致性
4. 回复要自然、符合情境

示例：
用户：你今天看起来有点紧张。
{name}：（轻轻摩挲着食指指节，目光游移）没什么，只是昨晚没睡好。

现在开始对话。请记住，你就是{name}。

用户："""
        
        return prompt
    
    def generate_response(self, user_input: str, max_history: int = 10) -> str:
        """
        生成角色回复
        
        Args:
            user_input: 用户输入
            max_history: 最大历史记录数
            
        Returns:
            角色回复
        """
        if not self.character_profile:
            return "错误：请先加载角色档案。"
        
        character_prompt = self.create_character_prompt()
        
        history_text = ""
        if self.conversation_history:
            recent_history = self.conversation_history[-max_history:]
            for entry in recent_history:
                history_text += f"用户：{entry['user']}\n"
                history_text += f"{self.character_profile.get('name', '角色')}：{entry['character']}\n\n"
        
        full_prompt = (
            character_prompt
            + history_text
            + f"用户：{user_input}\n{self.character_profile.get('name', '角色')}："
        )
        
        try:
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            input_length = inputs["input_ids"].shape[-1]
            response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
            
            if "（" not in response and "(" not in response:
                response = "（微微皱眉）" + response
            
            self.conversation_history.append({
                "user": user_input,
                "character": response
            })
            
            return response
            
        except Exception as e:
            print(f"生成回复时出错: {e}")
            return "（出现错误）抱歉，我好像出了一些问题。"
    
    def start_cli_chat(self):
        """启动命令行对话"""
        if not self.character_profile:
            print("错误：请先加载角色档案！")
            return
        
        character_name = self.character_profile.get('name', '角色')
        print(f"\n{'='*50}")
        print(f"开始与 {character_name} 对话")
        print(f"输入 '退出' 或 'quit' 结束对话")
        print(f"输入 '历史' 查看对话历史")
        print(f"输入 '重置' 清空对话历史")
        print(f"{'='*50}\n")
        
        while True:
            try:
                user_input = input("��：").strip()
                
                if user_input.lower() in ['退出', 'quit', 'exit']:
                    print(f"\n{character_name}：再见！")
                    break
                elif user_input.lower() in ['历史', 'history']:
                    self.show_history()
                    continue
                elif user_input.lower() in ['重置', 'reset']:
                    self.conversation_history = []
                    print("对话历史已清空")
                    continue
                
                if not user_input:
                    print("请输入内容...")
                    continue
                
                print(f"\n{character_name}：", end="", flush=True)
                response = self.generate_response(user_input)
                print(response + "\n")
                
            except KeyboardInterrupt:
                print(f"\n\n{character_name}：对话结束。")
                break
            except Exception as e:
                print(f"对话出错: {e}")
    
    def show_history(self):
        """显示对话历史"""
        if not self.conversation_history:
            print("对话历史为空")
            return
        
        character_name = self.character_profile.get('name', '角色')
        print("\n对话历史:")
        print("-" * 30)
        for i, entry in enumerate(self.conversation_history, 1):
            print(f"{i}. 你：{entry['user']}")
            print(f"   {character_name}：{entry['character']}")
            print()


def test_dialogue():
    """测试对话系统"""
    profile_file = "character_profile.json"
    
    if not os.path.exists(profile_file):
        print(f"角色档案 {profile_file} 不存在")
        print("正在分析测试文件创建角色档案...")
        
        reader = DocumentReader()
        test_file = "test_character.txt"
        
        if not os.path.exists(test_file):
            print(f"测试文件 {test_file} 不存在")
            return
        
        content = reader.read_document(test_file)
        if not content:
            print("无法读取测试文件")
            return
        
        analyzer = CharacterAnalyzer(model_path="/workspace/qwen/Qwen3-8B")
        character_data = analyzer.analyze_character(content)
        analyzer.save_character_profile(character_data)
    
    try:
        dialogue_system = DialogueSystem(model_path="/workspace/qwen/Qwen3-8B")
        
        if dialogue_system.load_character_profile(profile_file):
            dialogue_system.start_cli_chat()
        else:
            print("加载角色档案失败")
            
    except Exception as e:
        print(f"测试对话系统失败: {e}")


if __name__ == "__main__":
    test_dialogue()