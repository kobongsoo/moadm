import os
import time

#---------------------------------------------------------------
# [bong][2024-10-16] 로컬파일이 이미지인지 체크
# => Pillow 라이브러리(이미지 처리 라이브러리)를 사용해 파일을 열기해서,
# 열리면 이미지 파일로 판단.
from PIL import Image

def is_image_file(file_path):
    try:
        # 파일을 이미지로 열어보기
        with Image.open(file_path) as img:
            # 이미지가 정상적으로 열리면 True 반환
            img.verify()
        return True
    except (IOError, SyntaxError):
        # 이미지가 아니거나 손상된 경우 False 반환
        return False
    
#---------------------------------------------------------------
# 스트림으로 받은 파일이 이미지 type인지 체크
def check_mime_type(mime_type:str):
    assert mime_type, f"mime_type is empty!!"
    
    mime:str = "doc"
    if mime_type.startswith("image/"):
        mime = "img"
    elif mime_type == "text/plain":
        mime = "txt"
        
    return mime

#---------------------------------------------------------------
# 파일경로 얻기.
def getfilePath_doc01(filename:str, file_folder:str, mime_type:str):
    assert filename, f'filename is empty'
    assert file_folder, f'file_folder is empty'
    
    # 원본파일을 저장할 org 폴더 지정
    org_folder = f"{file_folder}/org"
    if not os.path.exists(org_folder):
        os.makedirs(org_folder)
    
    # 추출 및 생성한 text 저장할 폴더 지정
    extract_folder = f"{file_folder}/extract"
    tgtPath = f"{extract_folder}/{filename}"
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)
        
    # =====원본 파일 저장===================
    if mime_type == "txt":  # mime_type이 txt 면 추출하지 않고 extrac 폴더에 그냥 저장
        srcPath = f"{extract_folder}/{filename}"
    else:
        srcPath = f"{org_folder}/{filename}"
        
    # 원본저장파일경로, 추출할파일경로지정
    return srcPath, tgtPath