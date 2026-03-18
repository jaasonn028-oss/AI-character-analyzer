import os
import torch
import json
import re
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from character_analyzer import CharacterAnalyzer
from document_reader import DocumentReader

class DialogueSystem:
    """AI角色对话系统"""
    
    def __init__(self, model_path: str = None):
        """
        初始化对话系统
        
        Args:
            model_path: Qwen模型路径
        """
        print("正在初始化对话系统...")
        
        try:
            model_path = self._resolve_model_path(model_path)
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
            
            print("对话系统初始化完成!")
            
        except Exception as e:
            print(f"初始化对话系统失败: {e}")
            raise

    def _resolve_model_path(self, model_path: str = None) -> str:
        """解析模型路径，支持显式参数或环境变量。"""
        default_candidates = [
            "./Qwen",
            "./qwen/Qwen3-8B",
            "/workspace/qwen/Qwen3-8B",
            "/workspace/Qwen"
        ]

        candidates = [
            model_path,
            os.getenv("QWEN_MODEL_PATH"),
            os.getenv("MODEL_PATH")
        ] + default_candidates

        for candidate in candidates:
            if candidate:
                abs_path = os.path.abspath(candidate)
                if os.path.isdir(abs_path):
                    return abs_path

        raise FileNotFoundError(
            "未找到可用模型路径。请传入 model_path 参数，或设置环境变量 QWEN_MODEL_PATH，"
            "或将模型放在 ./Qwen、./qwen/Qwen3-8B、/workspace/qwen/Qwen3-8B。"
        )
    
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
        """创建角色扮演系统提示词"""
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
5. 严禁代替用户发言，不要生成“用户：”或“你：”开头的内容
6. 禁止输出思考过程、推理过程或任何类似<think>标签内容

示例：
用户：你今天看起来有点紧张。
{name}：（轻轻摩挲着食指指节，目光游移）没什么，只是昨晚没睡好。

现在开始对话。请记住，你就是{name}。"""
        
        return prompt

    def _to_text(self, value) -> str:
        """将任意输入安全转换为可编码字符串。"""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    def _build_messages(self, user_input: str, max_history: int = 10) -> List[dict]:
        """构建chat消息，避免文本拼接引发角色串台。"""
        messages: List[dict] = [{"role": "system", "content": self.create_character_prompt()}]

        if self.conversation_history:
            recent_history = self.conversation_history[-max(1, max_history):]
            for entry in recent_history:
                messages.append({"role": "user", "content": self._to_text(entry.get("user"))})
                messages.append({"role": "assistant", "content": self._to_text(entry.get("character"))})

        messages.append({
            "role": "user",
            "content": self._to_text(user_input)
        })

        return messages

    def _prepare_model_inputs(self, messages: List[dict]) -> dict:
        """通过chat template直接生成模型输入，减少文本二次编码不稳定。"""
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt"
            )
        except TypeError:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )

        if isinstance(encoded, torch.Tensor):
            return {"input_ids": encoded}

        # transformers 常见返回类型：BatchEncoding
        if isinstance(encoded, BatchEncoding):
            return dict(encoded)

        if isinstance(encoded, dict):
            return encoded

        # 兼容具备字典接口但不继承dict的对象
        if hasattr(encoded, "keys") and hasattr(encoded, "__getitem__"):
            return {k: encoded[k] for k in encoded.keys()}

        # 兼容极少数tokenizer返回list[int]的情况
        if isinstance(encoded, list):
            return {"input_ids": torch.tensor([encoded], dtype=torch.long)}

        raise ValueError(f"无法识别的编码结果类型: {type(encoded)}")

    def _apply_chat_template(self, messages: List[dict]) -> str:
        """兼容不同transformers版本的chat template调用。"""
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

    def _get_stop_token_ids(self) -> List[int]:
        """构建停止token集合，减少跨轮次续写。"""
        stop_ids = set()

        if isinstance(self.tokenizer.eos_token_id, int):
            stop_ids.add(self.tokenizer.eos_token_id)

        for token in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"]:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0 and token_id != self.tokenizer.unk_token_id:
                stop_ids.add(token_id)

        return list(stop_ids)

    def _sanitize_response(self, response: str) -> str:
        """清理思考过程、角色标签与意外续写内容。"""
        if not response:
            return ""

        cleaned = response.strip()

        # 去除常见思考标签块
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<\|.*?\|>", "", cleaned)

        # 去除开头角色前缀
        cleaned = re.sub(r"^(assistant|助手|AI|角色)\s*[:：]\s*", "", cleaned, flags=re.IGNORECASE)

        # 若模型继续写到下一轮用户发言，截断
        split_patterns = [
            r"\n\s*用户\s*[:：]",
            r"\n\s*你\s*[:：]",
            r"\n\s*User\s*[:：]",
            r"\n\s*Human\s*[:：]",
            r"\n\s*Q\s*[:：]"
        ]
        split_regex = re.compile("|".join(split_patterns), flags=re.IGNORECASE)
        parts = split_regex.split(cleaned, maxsplit=1)
        cleaned = parts[0].strip()

        return cleaned
    
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

        messages = self._build_messages(user_input=user_input, max_history=max_history)
        
        try:
            inputs = self._prepare_model_inputs(messages)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            stop_token_ids = self._get_stop_token_ids()
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.85,
                    repetition_penalty=1.1,
                    eos_token_id=stop_token_ids if stop_token_ids else self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id
                )
            
            input_length = inputs["input_ids"].shape[-1]
            response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            response = self._sanitize_response(response)

            if not response:
                response = "（短暂沉默后轻轻点头）我在听，你可以再说具体一点。"
            
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
                user_input = input("你：").strip()
                
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


def test_dialogue(interactive: bool = True):
    """测试对话系统。

    Args:
        interactive: True时进入命令行对话；False时仅做单轮烟测。
    """
    profile_file = "character_profile.json"
    
    if not os.path.exists(profile_file):
        print(f"角色档案 {profile_file} 不存在")
        print("正在分析测试文件创建角色档案...")
        
        reader = DocumentReader()
        test_file = "test_character.txt"
        
        if not os.path.exists(test_file):
            print(f"测试文件 {test_file} 不存在")
            return False
        
        content = reader.read_document(test_file)
        if not content:
            print("无法读取测试文件")
            return False
        
        analyzer = CharacterAnalyzer()
        character_data = analyzer.analyze_character(content)
        if "error" in character_data or "raw_response" in character_data:
            print("角色档案生成失败，无法继续对话测试")
            return False
        analyzer.save_character_profile(character_data)
    
    try:
        dialogue_system = DialogueSystem()
        
        if dialogue_system.load_character_profile(profile_file):
            if interactive:
                dialogue_system.start_cli_chat()
                return True

            response = dialogue_system.generate_response("你好，请简单介绍一下你自己。")
            if not response or "错误" in response:
                print("对话系统烟测失败")
                return False
            print("对话系统烟测通过")
            print(f"示例回复: {response}")
            return True
        else:
            print("加载角色档案失败")
            return False
            
    except Exception as e:
        print(f"测试对话系统失败: {e}")
        return False


if __name__ == "__main__":
    test_dialogue()