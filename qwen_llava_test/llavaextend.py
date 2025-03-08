import json
from llava import LlavaInfer
class LlavaInferExtended(LlavaInfer):
    def generate_prompt(self, query, demonstrations):
        """
        Generate a prompt for the model based on the query and demonstrations.
        """
        prompt = "You are a sentiment analysis expert. You can accurately identify the sentiment expressed in the given multimodal message through the image and text pair.sentiment option:\"neutral(1)\", \"positive(2)\", \"negative(0)\"\n"
        
        # Add demonstrations to the prompt
        for i, demo in enumerate(demonstrations):
            demo_text = demo['text'].replace('\ud83d\udcb7', '')  # Clean any emoji or special characters if necessary
            prompt += f"Example {i+1}:\nImage: {demo['image']}\nText: {demo_text}\n sentiment label is: {demo['label']}\n\n"

        # Mask the label in query
        masked_label = "[MASK]" if 'label' in query else None
        
        # Add query to the prompt
        query_text_cleaned = query['text'].replace('\ud83d\udcb7', '')
        prompt += f"Query:\nImage: {query['image']}\nText: \"{query_text_cleaned}\" Based on the Examples, May I ask what the sentiment expressed in the image and text of the query is?\n First of all, you should analyze it carefully based on the examples and display your chain of thought. Finally, you must choose the answer from \"neutral(1)\", \"positive(2)\", \"negative(0)\"."
        
        return prompt
    
    def infer_with_demonstrations(self, query, demonstrations):
        # Generate the prompt using query and demonstrations
        self.prompt = self.generate_prompt(query, demonstrations)
        #print(self.prompt)
        # Update images and texts based on the query and demonstrations
        images = [item['image'] for item in demonstrations] + [query['image']]
        #print(images)
        texts = [item['text'] for item in demonstrations] + [query['text']]
        #print(texts)
        # Update the instance attributes
        self.update(images=images, texts=texts, prompt=self.prompt)
        
        # Call original infer method
        return self.infer()
