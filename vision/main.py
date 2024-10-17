import io
import os
import uuid
import torch
import time

import requests
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, AutoModelForCausalLM
from googletrans import Translator

class MY_Vision:

    def __init__(self, model_folder_path:str, device:str="cuda:0"):
        
        assert model_folder_path, f'model_folder_Path is empty'
        assert device, f'device is empty'
        
        self.model_path = model_folder_path
        self.device = device
        self.model_name = self.extract_last_segment(self.model_path) # '/'로 나누고 마지막 부분을 반환
        
        # 모델명에 맞게 모델 로딩
        if self.model_name.startswith("paligemma"):
            self.load_paligemma()
        elif self.model_name.startswith("Phi"):
            self.load_Phi()
        else:
            print(f"*none:{self.model_path}")
        return
    
    #-----------------------------------------------------
    # 모델 generate 
    def generate(self, image_path:str, prompt:str="Describe the image."):
        assert image_path, f'device is empty'
        assert prompt, f'prompt is empty'
        
        # 모델명에 맞게 모델 generate
        if self.model_name.startswith("paligemma"):
            response=self.generate_paligemma(image_path=image_path, prompt=prompt)
        elif self.model_name.startswith("Phi"):
            response=self.generate_Phi(image_path=image_path, prompt=prompt)
        else:
            response = f"*none"
            print(f"*none:{self.model_name}")
            
        return response
        
    #-----------------------------------------------------
    # 구굴 번역
    def translate_google(self, text:str, source_lang:str="en", target_lang:str="ko"):
        assert text, f'text is empty'
        assert source_lang, f'source_lang is empty'
        assert target_lang, f'target_lang is empty'
        
        translator = Translator()

        start_time = time.time()
        res = translator.translate(text, src=source_lang, dest=target_lang)

        # 소요된 시간을 계산합니다.
        end_time = time.time()
        formatted_elapsed_time = "{:.2f}".format(end_time - start_time)
        print(f'time:{formatted_elapsed_time}')

        print(res)
        print()
        response = res.text.strip('"')
        return response

    #-----------------------------------------------------
    def extract_last_segment(self, model_path:str):
        assert model_path, f'model_path is empty'
        # '/'로 나누고 마지막 부분을 반환
        return model_path.split('/')[-1]

    #-----------------------------------------------------
    # 구글 paligemma 모델 로딩
    def load_paligemma(self):
        start_time = time.time()

        # 로컬 파일에서 모델과 프로세서 불러오기
        dtype = torch.bfloat16

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=self.device,
            revision="bfloat16",
        ).eval()
        processor = AutoProcessor.from_pretrained(self.model_path)

        # 소요된 시간을 계산합니다.
        end_time = time.time()
        formatted_elapsed_time = "{:.2f}".format(end_time - start_time)
        print(f'*time:{formatted_elapsed_time}')
        
        self.model = model
        self.processor = processor
        print(f"*load_paligemma:{self.model_path}")
        return model, processor
    
    #-----------------------------------------------------
    # MS Phi 모델 로딩
    def load_Phi(self):
        start_time = time.time()

        # 로컬 파일에서 모델과 프로세서 불러오기
        model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                     device_map=self.device, 
                                                     trust_remote_code=True, 
                                                     torch_dtype="auto", 
                                                     _attn_implementation='flash_attention_2')
        processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

        # 소요된 시간을 계산합니다.
        end_time = time.time()
        formatted_elapsed_time = "{:.2f}".format(end_time - start_time)
        print(f'*time:{formatted_elapsed_time}')
        
        self.model = model
        self.processor = processor
        print(f"*load_Phi:{self.model_path}")
        return model, processor
    
    #-----------------------------------------------------
    # 이미지 전처리
    def preprocess_image(self, image):
        # [bong][2024-06-28] 
        # .PNG 파일 업로드 하면 에러남.
        # => PNG 파일은 투명도를 포함할 수 있는 RGBA 모드를 사용할 수 있는데, 그러나 JPEG는 투명도를 지원하지 않기 때문에 에러남.
        #=> 따라서 PNG 파일을 JPEG로 변환시 투명 부분을 흰색으로 처리함.
        if image.mode == 'RGBA':
            # 흰색 배경의 새로운 이미지 생성
            background = Image.new("RGB", image.size, (255, 255, 255))
            # 기존 이미지를 배경 이미지에 덮어쓰기
            background.paste(image, (0, 0), image)
            image = background 
            
        return image
    #-----------------------------------------------------
    # 구글 paligemma 이미지 처리
    # => prompt 예시 : What does the image show?, Describe the image.
    def generate_paligemma(self, image_path:str, prompt:str="Describe the image.", temperature:float=0.0, en_ko_trans:bool=True):
        assert image_path, f'image_path is empty'
        assert prompt, f'prompt is empty'
        
        start_time = time.time()

        #url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
        #image = Image.open(requests.get(url, stream=True).raw)
        #image = Image.open(image_path) # 로컬이미지 불러옴
        image = self.preprocess_image(image=Image.open(image_path))
            
        # Instruct the model to create a caption in Spanish
        model_inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation_args = { 
                "max_new_tokens": 500, 
                "temperature": temperature, # 0.0 이면 출력고정 
                "do_sample": False, 
            } 
            
            generation = self.model.generate(**model_inputs, **generation_args)
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            #print(decoded)

        # 영문->한국어 번역인 경우 번역해서 출력함
        if en_ko_trans == True:
            response = self.translate_google(text=decoded)
        else:
            response = decoded
            
        # 소요된 시간을 계산합니다.
        end_time = time.time()
        formatted_elapsed_time = "{:.2f}".format(end_time - start_time)
        print(f'*time:{formatted_elapsed_time}')
        
        return response
    
    #-----------------------------------------------------
    # MS phi 이미지 처리
    # => prompt 예시 : What does the image show?, Describe the image.
    def generate_Phi(self, image_path:str, prompt:str="Describe the image.", temperature:float=0.0, en_ko_trans:bool=True):
        assert image_path, f'image_path is empty'
        assert prompt, f'prompt is empty'
        
        start_time = time.time()

        messages = [ 
            {"role": "user", "content": "<|image_1|>\nWhat is shown in this image?"}, 
            #{"role": "assistant", "content": "The chart displays the percentage of respondents who agree with various statements about their preparedness for meetings. It shows five categories: 'Having clear and pre-defined goals for meetings', 'Knowing where to find the information I need for a meeting', 'Understanding my exact role and responsibilities when I'm invited', 'Having tools to manage admin tasks like note-taking or summarization', and 'Having more focus time to sufficiently prepare for meetings'. Each category has an associated bar indicating the level of agreement, measured on a scale from 0% to 100%."}, 
            #{"role": "user", "content": "Provide insightful questions to spark discussion."} 
            {"role": "user", "content": "Describe the image content in detail."} 
        ] 

        #url = "https://assets-c4akfrf5b4d3f4b7.z01.azurefd.net/assets/2024/04/BMDataViz_661fb89f3845e.png" 
        #image = Image.open(requests.get(url, stream=True).raw) 
        image = self.preprocess_image(image=Image.open(image_path))

        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(prompt, [image], return_tensors="pt").to("cuda:0") 

        generation_args = { 
            "max_new_tokens": 500, 
            "temperature": temperature, # 0.0 이면 출력고정
            "do_sample": False, 
        } 

        generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args) 

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        decoded = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
        
        # 영문->한국어 번역인 경우 번역해서 출력함
        if en_ko_trans == True:
            response = self.translate_google(text=decoded)
        else:
            response = decoded
            
        # 소요된 시간을 계산합니다.
        end_time = time.time()
        formatted_elapsed_time = "{:.2f}".format(end_time - start_time)
        print(f'*time:{formatted_elapsed_time}')
        
        return response
        
