import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
import os
from typing import Dict, Any
from document_reader import DocumentReader

class CharacterAnalyzer:
    """角色人格分析器"""
    
    def __init__(self, model_path="./models--Qwen--Qwen3-8B"):
        """
        初始化分析器
        
        Args:
            model_path: Qwen模型路径
        """
        print("正在加载Qwen模型...")
        
        try:
            # 加载tokenizer和模型
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
            
            # 使用GPU如果可用
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"使用设备: {device}")
            
            # 加载模型，根据设备调整参数
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            ).to(device)
            
            self.model.eval()  # 设置为评估模式
            print("模型加载完成!")
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("请确保模型路径正确，且已下载Qwen模型")
            raise
    
    def analyze_character(self, text: str, max_length: int = 2000) -> Dict[str, Any]:
        """
        分析文本中的主要角色
        
        Args:
            text: 文本内容
            max_length: 最大文本长度
            
        Returns:
            角色信息字典
        """
        # 截断文本以避免过长
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        # 构建系统提示词
        system_prompt = """你是一个专业的文学分析AI。请仔细阅读以下文本，分析其中的主要角色，并提取角色的关键信息。

分析要求：
1. 找出文本中最突出的主角
2. 分析角色的多维度特征
3. 输出必须是严格的JSON格式

输出JSON结构：
{
    "name": "角色姓名",
    "identity": "角色身份/职业",
    "age": "年龄或年龄段",
    "core_personality": ["核心性格特征1", "特征2", "特征3"],
    "speech_style": "语言风格描述",
    "characteristic_actions": ["标志性动作1", "动作2"],
    "emotional_traits": ["情感特征1", "特征2"],
    "background": "背景故事概要",
    "relationships": "重要人际关系",
    "key_quotes": ["代表性对话1", "对话2"]
}

文本内容：
"""
        
        prompt = system_prompt + text + "\n\n请分析以上文本中的主要角色："
        
        try:
            # 准备模型输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096  # Qwen模型的最大长度
            )
            
            # 将输入移到模型所在设备
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # 生成分析结果
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            # 解码结果
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取JSON部分（从响应中提取）
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                try:
                    character_data = json.loads(json_str)
                    return character_data
                except json.JSONDecodeError:
                    # 如果JSON解析失败，尝试清理字符串
                    json_str = self._clean_json_string(json_str)
                    try:
                        character_data = json.loads(json_str)
                        return character_data
                    except:
                        print("无法解析模型输出的JSON，返回原始输出")
                        return {"raw_response": response}
            else:
                print("未找到JSON格式的输出")
                return {"raw_response": response}
                
        except Exception as e:
            print(f"分析角色时出错: {e}")
            return {"error": str(e)}
    
    def _clean_json_string(self, json_str: str) -> str:
        """清理JSON字符串"""
        # 移除可能的问题字符
        json_str = json_str.strip()
        # 确保以{开始，以}结束
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
        """保存角色档案"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(character_data, f, ensure_ascii=False, indent=2)
        print(f"角色档案已保存到: {filename}")


def test_analyzer():
    """测试角色分析器"""
    # 首先读取测试文档
    reader = DocumentReader()
    
    # 读取测试文件
    test_file = "test_character.txt"
    if not os.path.exists(test_file):
        print(f"测试文件 {test_file} 不存在，请先创建测试文件")
        return
    
    content = reader.read_document(test_file)
    if not content:
        print("无法读取测试文件内容")
        return
    
    print("文档内容读取成功!")
    print(f"文档长度: {len(content)} 字符")
    
    # 初始化分析器
    try:
        analyzer = CharacterAnalyzer(model_path="./Qwen")
        
        # 分析角色
        print("\n正在分析角色...")
        character_data = analyzer.analyze_character(content)
        
        print("\n角色分析结果:")
        print("=" * 50)
        
        if "error" in character_data:
            print(f"分析出错: {character_data['error']}")
        elif "raw_response" in character_data:
            print("原始响应:")
            print(character_data['raw_response'])
        else:
            # 美化输出
            print(json.dumps(character_data, ensure_ascii=False, indent=2))
            
            # 保存档案
            analyzer.save_character_profile(character_data)
            
            # 打印关键信息
            print("\n关键信息摘要:")
            print(f"角色姓名: {character_data.get('name', '未知')}")
            print(f"身份: {character_data.get('identity', '未知')}")
            print(f"核心性格: {', '.join(character_data.get('core_personality', []))}")
            print(f"语言风格: {character_data.get('speech_style', '未知')}")
            
    except Exception as e:
        print(f"初始化或分析过程中出错: {e}")
        print("请确保:")
        print("1. Qwen模型已正确下载到 ./Qwen 目录")
        print("2. 有足够的GPU内存 (至少16GB)")


if __name__ == "__main__":
    test_analyzer()