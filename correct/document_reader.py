import os
from pathlib import Path
import PyPDF2

class DocumentReader:
    """文档读取器类，支持PDF和TXT格式"""
    
    def __init__(self):
        pass
    
    def read_document(self, file_path):
        """
        读取文档内容
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 文档内容
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 根据文件扩展名选择读取方式
        if file_path.suffix.lower() == '.pdf':
            return self._read_pdf(file_path)
        elif file_path.suffix.lower() == '.txt':
            return self._read_txt(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
    
    def _read_pdf(self, pdf_path):
        """读取PDF文件"""
        text = ""
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # 遍历所有页面提取文本
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
            return text
            
        except Exception as e:
            print(f"读取PDF文件时出错: {e}")
            return ""
    
    def _read_txt(self, txt_path):
        """读取TXT文件"""
        try:
            # 尝试多种编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            
            for encoding in encodings:
                try:
                    with open(txt_path, 'r', encoding=encoding) as file:
                        content = file.read()
                    return content
                except UnicodeDecodeError:
                    continue
            
            # 如果所有编码都失败，使用二进制读取
            with open(txt_path, 'rb') as file:
                return file.read().decode('utf-8', errors='ignore')
                
        except Exception as e:
            print(f"读取TXT文件时出错: {e}")
            return ""
    
    def preview_content(self, content, preview_length=500):
        """预览文档内容"""
        if not content:
            return "文档内容为空"
        
        # 确保预览长度不超过内容长度
        preview_len = min(preview_length, len(content))
        preview = content[:preview_len]
        
        # 如果内容被截断，添加省略号
        if len(content) > preview_len:
            preview += "..."
        
        return preview
    
    def get_stats(self, content):
        """获取文档统计信息"""
        if not content:
            return {
                "characters": 0,
                "words": 0,
                "lines": 0,
                "paragraphs": 0
            }
        
        # 字符数
        char_count = len(content)
        
        # 字数（中英文混合）
        word_count = len(content.split())
        
        # 行数
        line_count = len(content.splitlines())
        
        # 段落数（连续空行视为段落分隔）
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        return {
            "characters": char_count,
            "words": word_count,
            "lines": line_count,
            "paragraphs": paragraph_count
        }


def test_reader():
    """测试文档读取器"""
    reader = DocumentReader()
    
    # 测试文件路径 - 根据您的环境调整
    test_file = "test_character.txt"  # 使用之前创建的测试文件
    
    if not os.path.exists(test_file):
        print("测试文件不存在，正在创建示例测试文件...")
        # 创建示例测试文件
        sample_text = """这是一个测试文本，用于验证文档读取功能。
主角名叫林风，是一名年轻的侦探。他性格冷静，观察力敏锐，但有时显得孤僻。
他习惯性地推了推眼镜，这是他在思考时的标志性动作。
在这个雨夜，林风站在案发现场，雨水打湿了他的风衣，但他的目光依然锐利如鹰。"""
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(sample_text)
    
    try:
        # 读取文档
        content = reader.read_document(test_file)
        
        if content:
            # 预览内容
            preview = reader.preview_content(content, 300)
            print("文档预览:")
            print("=" * 50)
            print(preview)
            print("=" * 50)
            
            # 显示统计信息
            stats = reader.get_stats(content)
            print("\n文档统计:")
            print(f"字符数: {stats['characters']}")
            print(f"字数: {stats['words']}")
            print(f"行数: {stats['lines']}")
            print(f"段落数: {stats['paragraphs']}")
            
            # 保存提取的内容
            output_file = "extracted_content.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"\n完整内容已保存到: {output_file}")
            
            return content
        else:
            print("无法读取文档内容")
            return None
            
    except Exception as e:
        print(f"读取文档时出错: {e}")
        return None


if __name__ == "__main__":
    content = test_reader()
    if content:
        print("\n文档读取测试完成!")