from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# message = [
#             {
#                 "role": "user",
#                 "content": 
#                 [
#                     {"type": "text", "text": self.texts[i]} for i in range(len(self.texts))
#                 ]+
#                 [
#                     {"type": "image", "image": self.images[i]} for i in range(len(self.images))
#                 ] +
#                 [
#                     {"type": "text", "text": self.prompt}
#                 ]
#             }
#         ]

class QWen2_5VLInfer:
    '''
    使用QWen2.5VL模型进行支持多组图文示例的多卡推理模型
        message: 参考示例构造
        min_pixels (int, optional)
        max_pixels (int, optional)
        max_new_tokens (int, optional)
    '''
    def __init__(self, 
                 message:list[dict]=[{}],
                 min_pixels = 256*28*28,
                 max_pixels = 512*28*28,
                 max_new_tokens=1028,
                 ):
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.output_text = ''
        self.max_new_tokens = max_new_tokens
        self.message = message
        
    def update(self, **kwargs):
        """
        以**kwargs方式更新推理时的参数
        """
        # Iterate over the keyword arguments and update the attributes
        for key, value in kwargs.items():
            if hasattr(self, key):  # Check if the attribute exists
                setattr(self, key, value)

    def initialize(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
        '''
        用于加载模型权重，需要在初始化后运行
        '''
        # default: Load the model on the available device(s)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, 
            # torch_dtype="auto", 
            device_map="auto",
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            # max_memory={0: "2GiB", 1: "6GiB", 2: "6GiB", 3: "6GiB",4: "6GiB",5: "6GiB",6: "6GiB",7: "6GiB"},
            )

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", 
            min_pixels=self.min_pixels, 
            max_pixels=self.max_pixels,
            )

    def infer(self):
        '''
        运行initialize之后进行推理，如变更参数需要调用update方法
        '''
        messages = self.message
        
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        self.output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return self.output_text

if __name__ == "__main__":
    images = [
        "datasets/twitter1517/images/0001.jpg",
        "datasets/twitter1517/images/0002.jpg",
    ]
    texts = [
        "How Jake Paul is changing the influencer game :",
        "Chris Brown and his crew were kicked off a plane after allegedly hot boxing it",
    ]
    prompt = '''Below are two pictures and two paragraphs of text. Please analyze the emotions of the first set of pictures and text and the second set of pictures and text respectively.'''
    model = QWen2_5VLInfer(images=images,texts=texts,prompt=prompt)
    model.initialize()
    model.infer()
    print(model.output_text)