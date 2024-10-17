import os
import time

#---------------------------------------------------------------
# 문서에서 text 추출 /imageToText 추출 후 파일로저장
def extract_save_doc01(instance:dict, srcPath:str, tgtPath:str, mime_type:str):
    assert srcPath, f'srcPath is empty'
    assert tgtPath, f'tgtPath is empty'
    assert mime_type, f'mime_type is empty'
    
    myutils = instance['myutils']
    myvision = instance['myvision']
    shaai = instance['shaai']
    
    status:int = 0
    imageToText:str = ""
    myutils.log_message(f'[extract_save_doc01] *srcPath:{srcPath}, tgtPath:{tgtPath}, mime_type:{mime_type}')
    
    # 소스파일경로 없으면 2번 에러값 리터 
    if os.path.isfile(srcPath) == False:
        myutils.log_message(f'[extract_save_doc01] srcfile is not exist!!')
        imageToText = "srcfile is not exist!!"
        return 2, imageToText
    
    # =====파일에서 text 추출/저장=========
    # mime_type=='doc'인 경우에만
    if mime_type == "doc":
        myutils.log_message(f'[extract_doc01] *tgtPath:{tgtPath}')
        shaai.extract(srcPath=srcPath, tgtPath=tgtPath)
    # ====================================
    
    # ==이미지는 vision모델로 text 생성/저장==
    if mime_type == "img":
        imageToText = myvision.generate(image_path=srcPath)
        
        # text 생성이 너무 짧은경우 1번더 다른 prompt로 text 생성함.
        if len(imageToText) < 20:
            imageToText += myvision.generate(image_path=srcPath, prompt="What does the image show?")
        
        # imageToText 파일로 저장
        with open(tgtPath, 'w', encoding='utf-8') as file:
            file.write(imageToText)
        
        myutils.log_message(f'[extract_doc01] *myvision.generate:\n{imageToText}')
    # ====================================
    
    # tgtPath 없으면 2번 에러값 리터 
    if os.path.isfile(tgtPath) == False:
        myutils.log_message(f'[extract_save_doc01] tgtPath is not exist!!')
        imageToText = "srcfile is not exist!!"
        return 2, imageToText
    
    return 0, imageToText