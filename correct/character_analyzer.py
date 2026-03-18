import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
from typing import Dict, Any, List
from document_reader import DocumentReader

class CharacterAnalyzer:
    """角色人格分析器"""
    
    def __init__(self, model_path: str = None):
        print("正在加载Qwen模型...")
        
        try:
            model_path = self._resolve_model_path(model_path)
            if not os.path.isdir(model_path):
                raise FileNotFoundError(f"模型路径不存在: {model_path}")
            print(f"加载模型路径: {model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"使用设备: {device}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            
            if device == "cpu":
                self.model = self.model.to(device)
            
            self.model.eval()
            print("模型加载完成!")
            
        except Exception as e:
            print(f"加载模型失败: {e}")
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
        """构建停止token集合，减少无关续写。"""
        stop_ids = set()

        if isinstance(self.tokenizer.eos_token_id, int):
            stop_ids.add(self.tokenizer.eos_token_id)

        for token in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"]:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0 and token_id != self.tokenizer.unk_token_id:
                stop_ids.add(token_id)

        return list(stop_ids)

    def _sanitize_generation_text(self, text: str) -> str:
        """移除思考标签与代码块包裹，保留JSON主体。"""
        cleaned = text.strip()
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"```json\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "")
        cleaned = re.sub(r"<\|.*?\|>", "", cleaned)
        return cleaned.strip()
    
    def analyze_character(self, text: str, max_length: int = 2000) -> Dict[str, Any]:
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        # FIX: Use chat template with system/user roles, and disable thinking mode
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个专业的文学分析AI。分析文本中的主要角色，"
                    "只输出严格的JSON，不要输出任何其他内容，不要解释，不要思考过程。\n\n"
                    "输出JSON结构：\n"
                    "{\n"
                    '    "name": "角色姓名",\n'
                    '    "identity": "角色身份/职业",\n'
                    '    "age": "年龄或年龄段",\n'
                    '    "core_personality": ["核心性格特征1", "特征2", "特征3"],\n'
                    '    "speech_style": "语言风格描述",\n'
                    '    "characteristic_actions": ["标志性动作1", "动作2"],\n'
                    '    "emotional_traits": ["情感特征1", "特征2"],\n'
                    '    "background": "背景故事概要",\n'
                    '    "relationships": "重要人际关系",\n'
                    '    "key_quotes": ["代表性对话1", "对话2"]\n'
                    "}"
                )
            },
            {
                "role": "user",
                "content": f"请分析以下文本中的主要角色，只输出JSON：\n\n{text}"
            }
        ]
        
        try:
            text_input = self._apply_chat_template(messages)
            
            inputs = self.tokenizer(
                text_input,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            stop_token_ids = self._get_stop_token_ids()
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=600,
                    do_sample=False,
                    repetition_penalty=1.1,
                    eos_token_id=stop_token_ids if stop_token_ids else self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id
                )
            
            input_length = inputs["input_ids"].shape[-1]
            response = self.tokenizer.decode(
                outputs[0][input_length:], skip_special_tokens=True
            ).strip()
            response = self._sanitize_generation_text(response)
            
            print(f"\n模型原始输出:\n{response}\n")
            
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    json_str = self._clean_json_string(json_str)
                    try:
                        return json.loads(json_str)
                    except:
                        print("无法解析JSON，返回原始输出")
                        return {"raw_response": response}
            else:
                print("未找到JSON格式的输出")
                return {"raw_response": response}
                
        except Exception as e:
            print(f"分析角色时出错: {e}")
            return {"error": str(e)}
    
    def _clean_json_string(self, json_str: str) -> str:
        json_str = json_str.strip()
        if not json_str.startswith('{'):
            start = json_str.find('{')
            if start != -1:
                json_str = json_str[start:]
        if not json_str.endswith('}'):
            end = json_str.rfind('}')
            if end != -1:
                json_str = json_str[:end+1]
        return json_str
    
    def save_character_profile(self, character_data: Dict[str, Any], filename: str = "character_profile.json"):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(character_data, f, ensure_ascii=False, indent=2)
        print(f"角色档案已保存到: {filename}")


def test_analyzer():
    reader = DocumentReader()
    
    test_file = "test_character.txt"
    if not os.path.exists(test_file):
        print(f"测试文件 {test_file} 不存在")
        return False
    
    content = reader.read_document(test_file)
    if not content:
        print("无法读取测试文件内容")
        return False
    
    print("文档内容读取成功!")
    print(f"文档长度: {len(content)} 字符")
    
    try:
        analyzer = CharacterAnalyzer()
        
        print("\n正在分析角色...")
        character_data = analyzer.analyze_character(content)
        
        print("\n角色分析结果:")
        print("=" * 50)
        
        if "error" in character_data:
            print(f"分析出错: {character_data['error']}")
            return False
        elif "raw_response" in character_data:
            print("原始响应:")
            print(character_data['raw_response'])
            return False
        else:
            print(json.dumps(character_data, ensure_ascii=False, indent=2))
            analyzer.save_character_profile(character_data)
            print("\n关键信息摘要:")
            print(f"角色姓名: {character_data.get('name', '未知')}")
            print(f"身份:     {character_data.get('identity', '未知')}")
            print(f"核心性格: {', '.join(character_data.get('core_personality', []))}")
            print(f"语言风格: {character_data.get('speech_style', '未知')}")
            return True
            
    except Exception as e:
        print(f"初始化或分析过程中出错: {e}")
        return False


if __name__ == "__main__":
    test_analyzer()