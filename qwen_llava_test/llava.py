import os
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from accelerate import Accelerator

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

class LlavaInfer:
    '''
    使用llava模型进行支持多组图文示例的多卡推理模型
        images (list[str], optional)
        texts (list[str], optional)
        prompt (str, optional)
        min_pixels (int, optional)
        max_pixels (int, optional)
        max_new_tokens (int, optional)
    '''
    def __init__(self, 
                 images: list[str]=[], 
                 texts: list[str]=[], 
                 labels:list[str]=[],
                 prompt: str="", 
                 min_pixels=256*28*28, 
                 max_pixels=512*28*28, 
                 max_new_tokens=512):
        self.images = images
        self.texts = texts
        self.prompt = prompt
        self.labels = labels
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.output_text = ''


    def update(self, **kwargs):
        """
        以**kwargs方式更新推理时的参数
        """
        # Iterate over the keyword arguments and update the attributes
        for key, value in kwargs.items():
            if hasattr(self, key):  # Check if the attribute exists
                setattr(self, key, value)

    def initialize(self, model_id="./llava-1.5-7b-hf"):
        '''
        用于加载模型权重，需要在初始化后运行
        '''
        self.accelerator = Accelerator()  # Initialize Accelerator

        # Load the model and processor with the Accelerator
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            device_map='auto',  # Automatically distribute model across available GPUs
        )

        self.processor = AutoProcessor.from_pretrained(model_id)

        # Prepare the model for multi-GPU (Accelerator handles this)
        self.model = self.accelerator.prepare(model)

    def infer(self) -> str:
        # Construct messages dynamically
        # messages = [
        #     {
        #         "role": "user",
        #         "content": 
        #         [
        #             {"type": "text", "text": "You are a sentiment analysis expert who can accurately identify the sentiment expressed through both the image and text. "}
        #         ] 
        #         + 
        #         [
        #             {"type": "text", "text": "Here is a demonstration:\n Image:"}
        #         ] 
        #         + 
        #         [
        #             {"type": "image", "image": "demo_image.png"}
        #         ] 
        #         +
        #         [
        #             {"type": "text", "text": "Text:\n The Windows background is on fire in California..."}
        #         ] 
        #         + 
        #         [
        #             {"type": "text", "text": "Answer: The overall sentiment expressed by the combination is negative. Although the picture is just the classic Windows XP wallpaper, which conveys a plastic, sensitive feel, the text expresses a sense of negativity due to concerns about fire hazards on the landscape of the wallpaper. Therefore, the overall sentiment conveyed by the combination should be considered negative."}
        #         ] 
        #         + 
        #         [
        #             {"type": "text", "text": "Please determine the overall sentiment expressed by the following combination (positive, negative, or neutral). "}
        #         ]
        #         +
        #         [
        #             {"type": "text", "text": "Image: \n"}
        #         ]
        #         +
        #         [
        #             {"type": "image", "image": self.images[i]} for i in range(len(self.images))
        #         ]
        #         +
        #         [
        #             {"type": "text", "text": "Text: \n"}
        #         ]
        #         +
        #         [
        #             {"type": "text", "text": self.texts[i]} for i in range(len(self.texts))
        #         ]
        #     }
        # ]
        content = [
            {"type": "text", "text": "You are a sentiment analysis expert who can accurately identify the sentiment expressed through both the image and text. You need to answer the query based on the demostration"},
            {"type": "text", "text": "here is demonstration which you can refer, label option:2 for positive,1 for neutral,0 for negtive"}
        ]
        for i in range(len(self.texts) - 1):
            content.extend([
                {"type": "image", "image": self.images[i]},
                {"type": "text", "text": self.texts[i]},
                {"type": "text", "label": self.labels[i]}
            ])
        content.extend([
            {"type": "text", "text": "**Now here is the query**.Please determine the overall sentiment expressed by the following **query**, choose the answer from (positive, negative, or neutral)."},
            {"type": "image", "image": self.images[-1]},              
            {"type": "text", "text": self.texts[-1]},
        ])

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        print(messages)

        # Apply the conversation template
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        # Open all images
        imgs = [Image.open(image_path).convert("RGB") for image_path in self.images]

        # Process the image and text for model input
        inputs = self.processor(images=imgs, text=prompt, return_tensors='pt').to(self.accelerator.device, torch.float16)

        # Generate the output
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)

        # Decode the output and return as a string
        result = self.processor.decode(output[0], skip_special_tokens=True)
        self.output_text = result
        return self.output_text
    
    
if __name__ == "__main__":
    images = [
        "/data/jyh/multimodal-icl/datasets/twitter1517/images/1466.jpg",
        "/data/jyh/multimodal-icl/datasets/twitter1517/images/2782.jpg",
        "/data/jyh/multimodal-icl/datasets/twitter1517/images/1251.jpg",
        "/data/jyh/multimodal-icl/datasets/twitter1517/images/1264.jpg",
        "/data/jyh/multimodal-icl/datasets/twitter1517/images/2327.jpg"
    ]
    texts = [
        "Com on Leicester let ' s have the dream completed football can still live without 💷 💷 💷",
        "Thanks to Nick Ellis for this : - )",
        "Donald Trump Conspiracy Theories : Eleven Fringe Theories Believed By The Man Who Would . . .",
        "Right . If J . K . Rowling won ' t post this then I will :",
        'Pick of the day : Honest , Alma Tavern Theatre . A new play about a run - down street in Horfield .'

    ]
    prompt = '''Please analyze the emotions of the first set of pictures and text and the second set of pictures and text respectively.'''

    model = LlavaInfer(images=images, texts=texts, prompt=prompt)
    model.initialize()
    model.infer()
    print(model.output_text)
    input()
    model.update(images=images,texts=texts,prompt=prompt)
    model.infer()
    print(model.output_text)


